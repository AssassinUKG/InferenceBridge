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
        // Skip if we already have this version
        if current.as_deref() == Some(&release.tag_name) {
            tracing::info!(
                version = %release.tag_name,
                "llama-server is up-to-date"
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

/// Download and extract llama-server to our managed directory.
/// Emits `model-load-progress` events with stage "downloading".
pub async fn download_llama_server(
    app: &tauri::AppHandle,
    url: &str,
    tag: &str,
    total_size: u64,
) -> anyhow::Result<PathBuf> {
    use futures_util::StreamExt;

    let dir = LlamaProcess::managed_binary_dir();
    std::fs::create_dir_all(&dir)?;

    let temp_file = dir.join("download.tmp");

    let client = reqwest::Client::builder()
        .user_agent("InferenceBridge/0.1")
        .redirect(reqwest::redirect::Policy::limited(10))
        .timeout(std::time::Duration::from_secs(600))
        .build()?;

    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Download failed with status: {} (url: {})",
            response.status(),
            url
        ));
    }

    // Log the actual content type and final URL to help debug redirect issues
    let final_url = response.url().to_string();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();
    tracing::info!(
        final_url = %final_url,
        content_type = %content_type,
        "Download started"
    );

    let total = response.content_length().unwrap_or(total_size);
    let mut stream = response.bytes_stream();
    let mut file = std::fs::File::create(&temp_file)?;
    let mut downloaded: u64 = 0;

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

    emit_download(0, total);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        use std::io::Write;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;

        // Emit every ~2MB to avoid flooding
        if downloaded % (2 * 1024 * 1024) < chunk.len() as u64 {
            emit_download(downloaded, total);
        }
    }
    drop(file);

    emit_download(total, total);

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

    let server_path = extract_archive(&temp_file, &dir)?;

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_file);

    // Write version file
    let version_path = dir.join(VERSION_FILE);
    std::fs::write(version_path, tag)?;

    tracing::info!(
        path = %server_path.display(),
        version = %tag,
        "llama-server downloaded and installed"
    );

    let _ = app.emit(
        "model-load-progress",
        crate::state::LoadProgress {
            stage: "downloading".to_string(),
            message: format!("llama-server {tag} installed"),
            progress: 0.95,
            done: false,
            error: None,
        },
    );

    Ok(server_path)
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
