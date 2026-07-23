//! Auto-download and update llama-server from GitHub releases.
//!
//! Downloads pre-built binaries from the official ggml-org/llama.cpp releases.
//! Stores them in `%LOCALAPPDATA%/InferenceBridge/bin/`.

use std::path::{Path, PathBuf};
use tauri::Emitter;

use super::process::LlamaProcess;

/// List of recent releases (not just latest) — used to scan for Windows binaries
/// since some releases may not include Windows builds.
const GITHUB_RELEASES_API: &str =
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10";

/// Version file stored alongside the binary.
const VERSION_FILE: &str = "llama-server.version";

/// Patterns that match llama-server binary archives (contain llama-server.exe).
/// `cudart-*` packages are excluded — they only contain CUDA runtime DLLs.
fn server_binary_patterns() -> Vec<&'static str> {
    if cfg!(target_os = "windows") {
        if cfg!(target_arch = "x86_64") {
            vec![
                "llama-b*-bin-win-cuda",     // CUDA build with server binary
                "llama-b*-bin-win-avx2-x64", // CPU/AVX2 build
                "llama-b*-bin-win*x64",      // Any Windows x64 build
            ]
        } else {
            vec!["llama-b*-bin-win-arm64"]
        }
    } else if cfg!(target_os = "macos") {
        vec!["llama-b*-bin-macos-arm64"]
    } else {
        vec!["llama-b*-bin-ubuntu-x64"]
    }
}

/// Patterns for CUDA runtime DLL packages (no llama-server.exe inside).
fn cuda_runtime_patterns() -> Vec<&'static str> {
    vec!["cudart-llama*win*cuda*x64"]
}

/// Get the asset patterns for a specific backend preference.
/// Returns (server_patterns, cuda_runtime_patterns).
pub fn asset_pattern_for(backend: &str) -> String {
    match backend {
        "cuda" => {
            if cfg!(target_os = "windows") && cfg!(target_arch = "x86_64") {
                "llama-b*-bin-win-cuda|llama-b*-bin-win*x64".to_string()
            } else {
                server_binary_patterns().first().unwrap_or(&"").to_string()
            }
        }
        "cpu" | "avx2" => {
            if cfg!(target_os = "windows") {
                "llama-b*-bin-win-avx2-x64".to_string()
            } else {
                "llama-b*-bin-ubuntu-x64".to_string()
            }
        }
        _ => server_binary_patterns().first().unwrap_or(&"").to_string(),
    }
}

/// Find a release asset containing llama-server for the given pattern.
/// Scans up to 10 recent releases to handle releases that don't include Windows binaries.
/// Pattern can be pipe-separated to try multiple patterns in order.
/// Returns (tag, download_url, size).
pub async fn find_release_asset(pattern: &str) -> anyhow::Result<(String, String, u64)> {
    let client = reqwest::Client::builder()
        .user_agent("InferenceBridge/0.1")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let releases: Vec<GithubRelease> = client.get(GITHUB_RELEASES_API).send().await?.json().await?;

    let patterns: Vec<&str> = pattern.split('|').map(|p| p.trim()).collect();

    for release in &releases {
        for p in &patterns {
            if let Some(asset) = find_matching_asset(&release.assets, p) {
                tracing::info!(
                    version = %release.tag_name,
                    asset = %asset.name,
                    "Found matching server binary"
                );
                return Ok((
                    release.tag_name.clone(),
                    asset.browser_download_url.clone(),
                    asset.size,
                ));
            }
        }
    }

    // Collect all asset names from the first release for the error message
    let names: Vec<&str> = releases
        .first()
        .map(|r| r.assets.iter().map(|a| a.name.as_str()).collect())
        .unwrap_or_default();
    let releases_checked: Vec<&str> = releases.iter().map(|r| r.tag_name.as_str()).collect();
    Err(anyhow::anyhow!(
        "No asset matching pattern '{pattern}' found in {len} releases ({releases_checked:?}).\n\
         Latest assets: {names:?}",
        len = releases.len()
    ))
}

/// Find the CUDA runtime DLLs package from the latest release (if available).
/// Returns (tag, download_url, size) or None.
pub async fn find_cuda_runtime_asset() -> anyhow::Result<Option<(String, String, u64)>> {
    let client = reqwest::Client::builder()
        .user_agent("InferenceBridge/0.1")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let releases: Vec<GithubRelease> = client.get(GITHUB_RELEASES_API).send().await?.json().await?;

    let patterns = cuda_runtime_patterns();

    for release in &releases {
        for p in &patterns {
            if let Some(asset) = find_matching_asset(&release.assets, p) {
                return Ok(Some((
                    release.tag_name.clone(),
                    asset.browser_download_url.clone(),
                    asset.size,
                )));
            }
        }
    }

    Ok(None)
}

/// Check if we have a managed binary and it's up-to-date.
pub fn current_version() -> Option<String> {
    let dir = LlamaProcess::managed_binary_dir();
    let version_path = dir.join(VERSION_FILE);
    std::fs::read_to_string(version_path)
        .ok()
        .map(|s| s.trim().to_string())
}

fn llama_build_number(version: &str) -> Option<u64> {
    let trimmed = version.trim();
    let digits = trimmed
        .strip_prefix('b')
        .or_else(|| trimmed.strip_prefix('B'))?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Returns true only when `candidate` is newer than the installed release.
/// llama.cpp stable tags are monotonic `bNNNN` build numbers. Unknown formats
/// retain the old unequal-version behavior so custom channels can still update.
pub fn release_is_newer(current: Option<&str>, candidate: &str) -> bool {
    let Some(current) = current else {
        return true;
    };
    if current.trim().eq_ignore_ascii_case(candidate.trim()) {
        return false;
    }
    match (llama_build_number(current), llama_build_number(candidate)) {
        (Some(current_build), Some(candidate_build)) => candidate_build > current_build,
        _ => true,
    }
}

/// Represent a release asset from GitHub.
#[derive(Debug, serde::Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

#[derive(Debug, serde::Deserialize)]
struct GithubRelease {
    tag_name: String,
    assets: Vec<GithubAsset>,
}

/// Check for a newer version of llama-server.
/// Scans multiple recent releases in case the latest doesn't have Windows binaries.
/// Returns (tag, download_url, size) if an update is available, or None if up-to-date.
pub async fn check_for_update() -> anyhow::Result<Option<(String, String, u64)>> {
    let client = reqwest::Client::builder()
        .user_agent("InferenceBridge/0.1")
        .timeout(std::time::Duration::from_secs(15))
        .build()?;

    let releases: Vec<GithubRelease> = client.get(GITHUB_RELEASES_API).send().await?.json().await?;

    let current = current_version();
    let patterns = server_binary_patterns();

    // Scan releases for the first one with a matching Windows server binary
    for release in &releases {
        // Releases are newest-first. If the installed build is equal to or
        // ahead of this release, every remaining stable release is older too.
        if !release_is_newer(current.as_deref(), &release.tag_name) {
            tracing::info!(
                installed = ?current,
                available = %release.tag_name,
                "llama-server is up-to-date or ahead of the stable channel"
            );
            return Ok(None);
        }

        // Try to find a server binary in this release
        if let Some(asset) = patterns
            .iter()
            .find_map(|p| find_matching_asset(&release.assets, p))
        {
            tracing::info!(
                version = %release.tag_name,
                asset = %asset.name,
                size_mb = asset.size / (1024 * 1024),
                "Update available for llama-server"
            );
            return Ok(Some((
                release.tag_name.clone(),
                asset.browser_download_url.clone(),
                asset.size,
            )));
        }

        tracing::debug!(
            version = %release.tag_name,
            "Release has no Windows server binary, checking older releases..."
        );
    }

    let releases_checked: Vec<&str> = releases.iter().map(|r| r.tag_name.as_str()).collect();
    Err(anyhow::anyhow!(
        "No compatible llama-server binary found in {len} recent releases: {releases_checked:?}",
        len = releases.len()
    ))
}

fn find_matching_asset<'a>(assets: &'a [GithubAsset], pattern: &str) -> Option<&'a GithubAsset> {
    // The pattern uses glob-like wildcards, convert to simple matching
    let parts: Vec<&str> = pattern.split('*').collect();
    assets.iter().find(|a| {
        let name = a.name.to_lowercase();
        // Must contain .zip for Windows or .tar.gz for others
        let is_archive = name.ends_with(".zip") || name.ends_with(".tar.gz");
        if !is_archive {
            return false;
        }
        // Check all fixed parts of the pattern appear in order
        let mut pos = 0usize;
        for part in &parts {
            let part_lower = part.to_lowercase();
            if let Some(found) = name[pos..].find(&part_lower) {
                pos += found + part_lower.len();
            } else {
                return false;
            }
        }
        true
    })
}

fn safe_download_name(url: &str, tag: &str) -> String {
    let tail = url
        .rsplit('/')
        .next()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("llama-server.zip");
    let mut name = format!("{tag}-{tail}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();
    if name.len() > 180 {
        name.truncate(180);
    }
    name
}

fn copy_dir_all(src: &Path, dst: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let target = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else if ty.is_file() {
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

/// Download and extract llama-server to our managed directory.
/// Emits `model-load-progress` events with stage "downloading".
pub async fn download_llama_server(
    app: &tauri::AppHandle,
    url: &str,
    tag: &str,
    total_size: u64,
) -> anyhow::Result<PathBuf> {
    use futures_util::StreamExt;
    use reqwest::header::{CONTENT_LENGTH, RANGE};
    use std::io::{Seek, SeekFrom, Write};

    let dir = LlamaProcess::managed_binary_dir();
    std::fs::create_dir_all(&dir)?;

    let download_name = safe_download_name(url, tag);
    let temp_file = dir.join(format!("{download_name}.part"));
    let complete_file = dir.join(format!("{download_name}.download"));
    let stage_dir = dir.join(format!("{download_name}.extracting"));
    let _ = std::fs::remove_file(&complete_file);

    let client = reqwest::Client::builder()
        .user_agent("InferenceBridge/0.1")
        .redirect(reqwest::redirect::Policy::limited(10))
        .connect_timeout(std::time::Duration::from_secs(20))
        .timeout(std::time::Duration::from_secs(1800))
        .build()?;

    // Emit progress during download
    let emit_download = |downloaded: u64, total: u64| {
        let pct = if total > 0 {
            (downloaded as f32 / total as f32).min(1.0)
        } else {
            0.0
        };
        let mb_done = downloaded / (1024 * 1024);
        let mb_total = total / (1024 * 1024);
        let _ = app.emit(
            "model-load-progress",
            crate::state::LoadProgress {
                stage: "downloading".to_string(),
                message: format!("Downloading llama-server ({tag})... {mb_done}/{mb_total} MB"),
                progress: pct * 0.8, // 0-80% for download, 20% for extraction
                done: false,
                error: None,
            },
        );
    };

    let mut last_error: Option<anyhow::Error> = None;
    let mut final_total = total_size;

    for attempt in 1..=5u32 {
        let existing = std::fs::metadata(&temp_file).map(|m| m.len()).unwrap_or(0);
        let mut request = client.get(url);
        if existing > 0 {
            request = request.header(RANGE, format!("bytes={existing}-"));
        }

        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                last_error = Some(error.into());
                tokio::time::sleep(std::time::Duration::from_millis(600 * attempt as u64)).await;
                continue;
            }
        };

        let status = response.status();
        if existing > 0 && status == reqwest::StatusCode::OK {
            tracing::warn!(
                existing,
                "Download server ignored Range resume; restarting partial download"
            );
            let _ = std::fs::remove_file(&temp_file);
        } else if !(status.is_success() || status == reqwest::StatusCode::PARTIAL_CONTENT) {
            last_error = Some(anyhow::anyhow!(
                "Download failed with status: {} (url: {})",
                status,
                url
            ));
            tokio::time::sleep(std::time::Duration::from_millis(600 * attempt as u64)).await;
            continue;
        }

        let existing = std::fs::metadata(&temp_file).map(|m| m.len()).unwrap_or(0);
        let response_len = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        final_total = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            total_size.max(existing + response_len)
        } else {
            response.content_length().unwrap_or(total_size)
        };

        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_string();
        tracing::info!(
            attempt,
            resume_from = existing,
            final_url = %final_url,
            content_type = %content_type,
            "Download started"
        );

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(&temp_file)?;
        file.seek(SeekFrom::Start(existing))?;
        let mut downloaded = existing;
        emit_download(downloaded, final_total);

        let mut stream = response.bytes_stream();
        let mut attempt_failed = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(chunk) => {
                    file.write_all(&chunk)?;
                    downloaded += chunk.len() as u64;

                    if downloaded % (2 * 1024 * 1024) < chunk.len() as u64 {
                        emit_download(downloaded, final_total);
                    }
                }
                Err(error) => {
                    last_error = Some(error.into());
                    attempt_failed = true;
                    break;
                }
            }
        }
        file.sync_all()?;
        drop(file);

        if attempt_failed {
            tokio::time::sleep(std::time::Duration::from_millis(600 * attempt as u64)).await;
            continue;
        }

        if final_total > 0 && downloaded < final_total {
            last_error = Some(anyhow::anyhow!(
                "Download incomplete: {} of {} bytes",
                downloaded,
                final_total
            ));
            tokio::time::sleep(std::time::Duration::from_millis(600 * attempt as u64)).await;
            continue;
        }

        std::fs::rename(&temp_file, &complete_file)?;
        break;
    }

    if !complete_file.exists() {
        return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Download failed")));
    }

    emit_download(final_total, final_total);

    // Extract
    let _ = app.emit(
        "model-load-progress",
        crate::state::LoadProgress {
            stage: "downloading".to_string(),
            message: "Extracting llama-server...".to_string(),
            progress: 0.85,
            done: false,
            error: None,
        },
    );

    let is_cuda_runtime = url.to_lowercase().contains("cudart");
    let extract_dest = if is_cuda_runtime {
        wait_for_managed_binary_release(app, &dir).await?;
        dir.clone()
    } else {
        let _ = std::fs::remove_dir_all(&stage_dir);
        std::fs::create_dir_all(&stage_dir)?;
        stage_dir.clone()
    };

    let server_path = extract_archive(&complete_file, &extract_dest)?;

    if !is_cuda_runtime {
        wait_for_managed_binary_release(app, &dir).await?;
        copy_dir_all(&stage_dir, &dir)?;
        let _ = std::fs::remove_dir_all(&stage_dir);
    }

    // Clean up temp file
    let _ = std::fs::remove_file(&complete_file);

    // Write version file only for server binaries. CUDA runtime sidecar
    // packages must not overwrite the installed llama-server version.
    if !is_cuda_runtime {
        let version_path = dir.join(VERSION_FILE);
        std::fs::write(version_path, tag)?;
    }

    tracing::info!(
        path = %dir.join("llama-server.exe").display(),
        version = %tag,
        "llama-server downloaded and installed"
    );

    let _ = app.emit(
        "model-load-progress",
        crate::state::LoadProgress {
            stage: "downloading".to_string(),
            message: format!("Finalizing llama-server {tag} install..."),
            progress: 0.95,
            done: false,
            error: None,
        },
    );

    if is_cuda_runtime {
        Ok(server_path)
    } else {
        Ok(dir.join("llama-server.exe"))
    }
}

async fn wait_for_managed_binary_release(app: &tauri::AppHandle, dir: &Path) -> anyhow::Result<()> {
    let server_path = dir.join("llama-server.exe");
    if !server_path.exists() {
        return Ok(());
    }

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let mut emitted_wait = false;

    loop {
        match std::fs::OpenOptions::new().write(true).open(&server_path) {
            Ok(_) => return Ok(()),
            Err(error) if file_is_locked(&error) && std::time::Instant::now() < deadline => {
                if !emitted_wait {
                    emitted_wait = true;
                    tracing::warn!(
                        path = %server_path.display(),
                        "llama-server binary is still locked; waiting before install"
                    );
                    let _ = app.emit(
                        "model-load-progress",
                        crate::state::LoadProgress {
                            stage: "downloading".to_string(),
                            message:
                                "Waiting for running llama-server to stop before installing..."
                                    .to_string(),
                            progress: 0.9,
                            done: false,
                            error: None,
                        },
                    );
                }
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
            Err(error) if file_is_locked(&error) => {
                return Err(anyhow::anyhow!(
                    "{} is still in use after waiting. Stop the loaded model and retry the download.",
                    server_path.display()
                ));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn file_is_locked(error: &std::io::Error) -> bool {
    matches!(error.raw_os_error(), Some(32) | Some(33))
        || error.kind() == std::io::ErrorKind::PermissionDenied
}

#[cfg(test)]
mod tests {
    use super::{file_is_locked, release_is_newer};

    #[test]
    fn newer_llama_build_is_an_update() {
        assert!(release_is_newer(Some("b9968"), "b9983"));
    }

    #[test]
    fn older_llama_build_is_not_an_update() {
        assert!(!release_is_newer(Some("b9983"), "b9968"));
    }

    #[test]
    fn equal_llama_build_is_not_an_update() {
        assert!(!release_is_newer(Some("B9983"), "b9983"));
    }

    #[test]
    fn missing_runtime_can_install_latest_build() {
        assert!(release_is_newer(None, "b9983"));
    }

    #[test]
    fn windows_sharing_violation_is_treated_as_locked_file() {
        let error = std::io::Error::from_raw_os_error(32);

        assert!(file_is_locked(&error));
    }

    #[test]
    fn windows_lock_violation_is_treated_as_locked_file() {
        let error = std::io::Error::from_raw_os_error(33);

        assert!(file_is_locked(&error));
    }

    #[test]
    fn unrelated_io_errors_are_not_treated_as_locked_files() {
        let error = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");

        assert!(!file_is_locked(&error));
    }
}

/// Extract the llama-server binary from a downloaded archive.
pub fn extract_archive(archive_path: &Path, dest_dir: &Path) -> anyhow::Result<PathBuf> {
    // Validate the file looks like a zip before trying to extract
    let mut file = std::fs::File::open(archive_path)?;
    let mut magic = [0u8; 4];
    use std::io::Read;
    file.read_exact(&mut magic)?;
    drop(file);

    if magic[0..2] != [0x50, 0x4B] {
        // Not a zip file — log first bytes for debugging
        let file_size = std::fs::metadata(archive_path)?.len();
        // Read start of file to see if it's an HTML error page
        let preview = std::fs::read_to_string(archive_path)
            .unwrap_or_default()
            .chars()
            .take(200)
            .collect::<String>();
        return Err(anyhow::anyhow!(
            "Downloaded file is not a valid zip archive (size: {} bytes, starts with: {:?}). \
             Preview: {}",
            file_size,
            &magic,
            preview
        ));
    }

    extract_zip(archive_path, dest_dir)
}

/// Detect if a zip archive has a single top-level directory wrapping all entries.
/// Returns 1 if yes (skip first path component), 0 if entries are at root level.
fn detect_top_level_dir(archive: &mut zip::ZipArchive<std::fs::File>) -> usize {
    if archive.len() == 0 {
        return 0;
    }

    // Get the first path component of the first entry
    let first_entry = match archive.by_index_raw(0) {
        Ok(e) => e.name().to_string(),
        Err(_) => return 0,
    };

    let top_dir = match first_entry.split('/').next() {
        Some(d) if !d.is_empty() => d.to_string(),
        _ => return 0,
    };

    // Check that ALL entries start with this same top-level directory
    let prefix = format!("{top_dir}/");
    for i in 0..archive.len() {
        if let Ok(entry) = archive.by_index_raw(i) {
            let name = entry.name().to_string();
            // Allow the directory entry itself (e.g. "top_dir/") or entries under it
            if name != format!("{top_dir}") && !name.starts_with(&prefix) {
                tracing::info!(
                    entry = %name,
                    expected_prefix = %prefix,
                    "Archive has mixed top-level entries, not stripping prefix"
                );
                return 0;
            }
        }
    }

    tracing::info!(top_dir = %top_dir, "Archive has common top-level directory, stripping it");
    1
}

/// Extract a zip archive and find the llama-server binary.
fn extract_zip(archive_path: &Path, dest_dir: &Path) -> anyhow::Result<PathBuf> {
    let file = std::fs::File::open(archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    // Detect whether the archive has a common top-level directory to strip
    let skip_components = detect_top_level_dir(&mut archive);

    // Log first few entry names for debugging
    let sample_entries: Vec<String> = (0..archive.len().min(5))
        .filter_map(|i| archive.by_index_raw(i).ok().map(|e| e.name().to_string()))
        .collect();
    tracing::info!(
        entries = archive.len(),
        skip_components,
        sample = ?sample_entries,
        "Extracting zip archive"
    );

    let mut server_path: Option<PathBuf> = None;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;

        // Validate entry name to prevent path traversal
        let entry_name = entry.name().to_string();
        if entry_name.contains("..") {
            continue;
        }

        let rel_path: String = entry_name
            .split('/')
            .skip(skip_components)
            .collect::<Vec<_>>()
            .join("/");

        if rel_path.is_empty() {
            continue;
        }

        let outpath = dest_dir.join(&rel_path);

        if entry.is_dir() {
            std::fs::create_dir_all(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut outfile = std::fs::File::create(&outpath)?;
            std::io::copy(&mut entry, &mut outfile)?;

            // Track the llama-server binary
            if let Some(name) = outpath.file_name() {
                let n = name.to_string_lossy().to_lowercase();
                if n == "llama-server" || n == "llama-server.exe" {
                    server_path = Some(outpath.clone());
                }
            }
        }
    }

    server_path
        .ok_or_else(|| {
            // CUDA runtime packages (cudart-*) might only contain DLLs, not llama-server.exe.
            // Check if llama-server.exe already exists in the dest directory.
            let existing = dest_dir.join("llama-server.exe");
            if existing.exists() {
                tracing::info!(
                    "Downloaded archive didn't contain llama-server.exe, but it exists in {}",
                    dest_dir.display()
                );
                return anyhow::anyhow!("__use_existing__");
            }
            anyhow::anyhow!("llama-server binary not found in the downloaded archive")
        })
        .or_else(|e| {
            // Special case: CUDA runtime package extracted DLLs alongside existing llama-server
            if e.to_string() == "__use_existing__" {
                Ok(dest_dir.join("llama-server.exe"))
            } else {
                Err(e)
            }
        })
}
