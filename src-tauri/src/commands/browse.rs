//! Model browser commands backed by live Hugging Face GGUF search and downloads.

use std::path::{Component, PathBuf};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::Emitter;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use crate::state::SharedState;


// Shared model browser types used by the UI and HF search results.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubQuant {
    pub quant: String,
    pub size_gb: f32,
    pub url: String,
    pub filename: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubModel {
    pub id: String,
    pub name: String,
    pub family: String,
    pub params: String,
    pub description: String,
    pub tags: Vec<String>,
    pub supports_vision: bool,
    pub quants: Vec<HubQuant>,
}

// Hugging Face live search metadata returned by the public model API.

#[derive(Debug, serde::Deserialize)]
struct HfSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct HfApiModel {
    model_id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    disabled: bool,
    #[serde(default)]
    gated: Option<serde_json::Value>,
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

fn extract_quant(filename: &str) -> String {
    const KNOWN: &[&str] = &[
        "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_XS", "IQ2_XXS", "IQ2_XS", "Q8_0", "Q6_K", "Q5_K_M",
        "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_L", "Q3_K_S",
        "Q2_K", "F16", "BF16",
    ];
    let upper = filename.to_uppercase();
    for &k in KNOWN {
        if upper.contains(k) {
            return k.to_string();
        }
    }
    filename
        .trim_end_matches(".gguf")
        .rsplit('-')
        .next()
        .unwrap_or("GGUF")
        .to_uppercase()
}

fn format_downloads(downloads: u64) -> String {
    let value = downloads.to_string();
    let mut grouped = String::with_capacity(value.len() + value.len() / 3);

    for (index, ch) in value.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }

    grouped.chars().rev().collect()
}

fn is_hf_downloadable(model: &HfApiModel) -> bool {
    if model.private || model.disabled {
        return false;
    }

    match &model.gated {
        None => true,
        Some(serde_json::Value::Bool(false)) => true,
        Some(serde_json::Value::Null) => true,
        _ => false,
    }
}

fn is_hf_featured_candidate(model: &HfApiModel) -> bool {
    let id = model.model_id.to_lowercase();
    if id.contains("models-moved") || id.contains("embed") || id.contains("embedding") {
        return false;
    }

    let blocked_tags = [
        "sentence-transformers",
        "bert",
        "feature-extraction",
        "onnx",
        "openvino",
        "reranker",
    ];

    !model.tags.iter().any(|tag| {
        let lowered = tag.to_lowercase();
        blocked_tags.contains(&lowered.as_str())
    })
}

fn hf_supports_vision(model: &HfApiModel) -> bool {
    let pipeline = model
        .pipeline_tag
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();

    if matches!(
        pipeline.as_str(),
        "image-text-to-text" | "image-to-text" | "visual-question-answering"
    ) {
        return true;
    }

    model.tags.iter().any(|tag| {
        let lowered = tag.to_lowercase();
        matches!(
            lowered.as_str(),
            "vision" | "multimodal" | "image-text-to-text" | "image-to-text"
        )
    })
}

fn hf_api_to_hub(m: HfApiModel) -> Option<HubModel> {
    if !is_hf_downloadable(&m) {
        return None;
    }

    let gguf_files: Vec<&HfSibling> = m
        .siblings
        .iter()
        .filter(|s| s.rfilename.to_lowercase().ends_with(".gguf"))
        .collect();
    if gguf_files.is_empty() {
        return None;
    }

    let mut quants: Vec<HubQuant> = gguf_files
        .iter()
        .map(|s| HubQuant {
            quant: extract_quant(&s.rfilename),
            size_gb: s.size.map(|sz| sz as f32 / 1_073_741_824.0).unwrap_or(0.0),
            url: format!(
                "https://huggingface.co/{}/resolve/main/{}",
                m.model_id, s.rfilename
            ),
            filename: s.rfilename.clone(),
        })
        .collect();
    quants.sort_by(|left, right| left.size_gb.total_cmp(&right.size_gb));

    let mut parts = m.model_id.split('/');
    let owner = parts.next().unwrap_or("HuggingFace");
    let repo_name = parts.next().unwrap_or(&m.model_id);
    let name = repo_name.replace('-', " ").replace('_', " ");
    let supports_vision = hf_supports_vision(&m);

    let mut tags: Vec<String> = m
        .tags
        .into_iter()
        .filter(|t| !t.contains(':') && !t.starts_with("base_model") && t.len() < 24)
        .take(5)
        .collect();
    if supports_vision && !tags.iter().any(|tag| tag.eq_ignore_ascii_case("vision")) {
        tags.insert(0, "vision".to_string());
    }

    Some(HubModel {
        id: m.model_id.clone(),
        name,
        family: owner.to_string(),
        params: String::new(),
        description: format!("{} downloads | {}", format_downloads(m.downloads), m.model_id),
        tags,
        supports_vision,
        quants,
    })
}

async fn fetch_hub_models(
    query: Option<&str>,
    offset: u32,
    limit: u32,
    featured_only: bool,
) -> Result<Vec<HubModel>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("InferenceBridge/1.0")
        .build()
        .map_err(|e| e.to_string())?;

    let mut req = client.get("https://huggingface.co/api/models").query(&[
        ("filter", "gguf".to_string()),
        ("sort", "downloads".to_string()),
        ("direction", "-1".to_string()),
        ("limit", limit.to_string()),
        ("offset", offset.to_string()),
        ("full", "true".to_string()),
    ]);

    if let Some(query) = query.map(str::trim).filter(|value| !value.is_empty()) {
        req = req.query(&[("search", query.to_string())]);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("HuggingFace request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HuggingFace returned HTTP {}", resp.status()));
    }

    let models: Vec<HfApiModel> = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse HuggingFace response: {e}"))?;

    Ok(models
        .into_iter()
        .filter(|model| !featured_only || is_hf_featured_candidate(model))
        .filter_map(hf_api_to_hub)
        .collect())
}

/// Search HuggingFace for GGUF models. Returns up to 20 results sorted by downloads.
/// `offset` is the number of results to skip (for pagination).
#[tauri::command]
pub async fn search_hub_models(query: String, offset: u32) -> Result<Vec<HubModel>, String> {
    fetch_hub_models(Some(&query), offset, 20, false).await
}

// Download progress event payload emitted during model downloads.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub id: String,
    pub filename: String,
    pub dest_path: Option<String>,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f32,
    pub done: bool,
    pub status: String,
    pub error: Option<String>,
}

// Tauri commands for browsing, downloading, and deleting models.

fn sanitize_download_subpath(filename: &str) -> Result<PathBuf, String> {
    let mut sanitized = PathBuf::new();

    for component in std::path::Path::new(filename).components() {
        match component {
            Component::Normal(segment) => sanitized.push(segment),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("Invalid model path from Hugging Face: {filename}"));
            }
        }
    }

    if sanitized.as_os_str().is_empty() {
        return Err("Hugging Face returned an empty model path".to_string());
    }

    Ok(sanitized)
}

async fn upsert_download(
    state: &SharedState,
    progress: DownloadProgress,
    cancel_token: Option<CancellationToken>,
) {
    let mut s = state.write().await;
    if let Some(existing) = s.active_downloads.get_mut(&progress.id) {
        existing.progress = progress;
        if let Some(token) = cancel_token {
            existing.cancel_token = token;
        }
        return;
    }

    if let Some(token) = cancel_token {
        s.active_downloads.insert(
            progress.id.clone(),
            crate::state::ActiveDownload {
                progress,
                cancel_token: token,
            },
        );
    }
}

async fn emit_download_progress(
    app: &tauri::AppHandle,
    state: &SharedState,
    progress: DownloadProgress,
    cancel_token: Option<CancellationToken>,
) {
    upsert_download(state, progress.clone(), cancel_token).await;
    let _ = app.emit("model-download-progress", progress);
}

/// Open the containing folder for a path in the native file manager.
/// On Windows, selects the file itself in Explorer.
#[tauri::command]
pub async fn show_in_folder(path: String) -> Result<(), String> {
    let p = std::path::Path::new(&path);

    #[cfg(target_os = "windows")]
    {
        // /select highlights the specific file inside Explorer
        std::process::Command::new("explorer")
            .arg("/select,")
            .arg(p)
            .spawn()
            .map_err(|e| format!("Failed to open Explorer: {e}"))?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("-R")
            .arg(p)
            .spawn()
            .map_err(|e| format!("Failed to open Finder: {e}"))?;
    }
    #[cfg(target_os = "linux")]
    {
        let dir = p.parent().unwrap_or(p);
        std::process::Command::new("xdg-open")
            .arg(dir)
            .spawn()
            .map_err(|e| format!("Failed to open file manager: {e}"))?;
    }
    Ok(())
}

#[tauri::command]
pub async fn list_downloads(
    state: tauri::State<'_, SharedState>,
) -> Result<Vec<DownloadProgress>, String> {
    let s = state.read().await;
    let mut items: Vec<DownloadProgress> = s
        .active_downloads
        .values()
        .map(|entry| entry.progress.clone())
        .collect();
    items.sort_by(|left, right| {
        left.done
            .cmp(&right.done)
            .then_with(|| left.filename.cmp(&right.filename))
    });
    Ok(items)
}

#[tauri::command]
pub async fn cancel_download(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
    id: String,
) -> Result<(), String> {
    let progress = {
        let mut s = state.write().await;
        let entry = s
            .active_downloads
            .get_mut(&id)
            .ok_or_else(|| "Download not found.".to_string())?;
        if entry.progress.done {
            return Ok(());
        }
        entry.cancel_token.cancel();
        entry.progress.status = "Cancelling".to_string();
        entry.progress.clone()
    };

    let _ = app.emit("model-download-progress", progress);
    Ok(())
}

#[tauri::command]
pub async fn clear_completed_downloads(
    state: tauri::State<'_, SharedState>,
) -> Result<(), String> {
    let mut s = state.write().await;
    s.active_downloads.retain(|_, entry| !entry.progress.done);
    Ok(())
}

/// Stream-download a GGUF file into the first configured scan directory.
/// Emits `model-download-progress` events with live byte counts (~4/s).
#[tauri::command]
pub async fn download_hub_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
    url: String,
    filename: String,
    supports_vision: Option<bool>,
) -> Result<String, String> {
    let relative_path = sanitize_download_subpath(&filename)?;
    let download_id = url.clone();
    let dest_dir: std::path::PathBuf = {
        let s = state.read().await;
        match s.config.models.scan_dirs.first() {
            Some(d) => std::path::PathBuf::from(d),
            None => {
                return Err(
                    "No model directory configured. Add one in Settings > Model Directories."
                        .to_string(),
                )
            }
        }
    };

    tokio::fs::create_dir_all(&dest_dir)
        .await
        .map_err(|e| format!("Cannot create {}: {e}", dest_dir.display()))?;

    let dest_path = dest_dir.join(&relative_path);
    if let Some(parent) = dest_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Cannot create {}: {e}", parent.display()))?;
    }

    {
        let s = state.read().await;
        if let Some(existing) = s.active_downloads.get(&download_id) {
            if !existing.progress.done {
                return Err(format!("Download already in progress for {}", existing.progress.filename));
            }
        }
    }

    let cancel_token = CancellationToken::new();
    let dest_path_string = dest_path.to_string_lossy().to_string();
    emit_download_progress(
        &app,
        state.inner(),
        DownloadProgress {
            id: download_id.clone(),
            filename: filename.clone(),
            dest_path: Some(dest_path_string.clone()),
            downloaded_bytes: 0,
            total_bytes: 0,
            percent: 0.0,
            done: false,
            status: "Starting".to_string(),
            error: None,
        },
        Some(cancel_token.clone()),
    )
    .await;

    let result: Result<String, String> = async {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(7200)) // 2-hour ceiling for large models
            .user_agent("InferenceBridge/1.0")
            .build()
            .map_err(|e| e.to_string())?;

        let resp = client
            .get(&url)
            .header(reqwest::header::ACCEPT, "application/octet-stream")
            .send()
            .await
            .map_err(|e| format!("Download request failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Server returned HTTP {} for {}",
                resp.status(),
                url
            ));
        }

        let total_bytes = resp.content_length().unwrap_or(0);
        let mut file = tokio::fs::File::create(&dest_path)
            .await
            .map_err(|e| format!("Cannot create {}: {e}", dest_path.display()))?;

        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = 0;
        let mut last_emit = std::time::Instant::now();
        let mut cancelled = false;

        while let Some(chunk) = tokio::select! {
            _ = cancel_token.cancelled() => {
                cancelled = true;
                None
            }
            chunk = stream.next() => chunk
        } {
            let chunk = chunk.map_err(|e| format!("Download error: {e}"))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| format!("Write error: {e}"))?;
            downloaded += chunk.len() as u64;

            if last_emit.elapsed().as_millis() >= 250 {
                let percent = if total_bytes > 0 {
                    downloaded as f32 / total_bytes as f32
                } else {
                    0.0
                };
                emit_download_progress(
                    &app,
                    state.inner(),
                    DownloadProgress {
                        id: download_id.clone(),
                        filename: filename.clone(),
                        dest_path: Some(dest_path_string.clone()),
                        downloaded_bytes: downloaded,
                        total_bytes,
                        percent,
                        done: false,
                        status: "Downloading".to_string(),
                        error: None,
                    },
                    None,
                )
                .await;
                last_emit = std::time::Instant::now();
            }
        }

        if cancelled {
            drop(file);
            let _ = tokio::fs::remove_file(&dest_path).await;
            emit_download_progress(
                &app,
                state.inner(),
                DownloadProgress {
                    id: download_id.clone(),
                    filename: filename.clone(),
                    dest_path: Some(dest_path_string.clone()),
                    downloaded_bytes: downloaded,
                    total_bytes,
                    percent: if total_bytes > 0 {
                        downloaded as f32 / total_bytes as f32
                    } else {
                        0.0
                    },
                    done: true,
                    status: "Cancelled".to_string(),
                    error: None,
                },
                None,
            )
            .await;
            return Err("Download cancelled".to_string());
        }

        file.flush()
            .await
            .map_err(|e| format!("Flush error: {e}"))?;
        drop(file);

        if let (Some(value), Some(model_filename)) = (
            supports_vision,
            relative_path
                .file_name()
                .and_then(|value| value.to_str())
                .map(str::to_string),
        ) {
            if let Err(error) =
                crate::models::overrides::set_model_supports_vision_override(&model_filename, value)
            {
                tracing::warn!(
                    model = %model_filename,
                    supports_vision = value,
                    error = %error,
                    "Failed to persist Hugging Face vision capability override"
                );
            }
        }

        {
            let s = state.read().await;
            let dirs = s.config.models.scan_dirs.clone();
            drop(s);
            let scanned = tokio::task::spawn_blocking(move || crate::models::scanner::scan_all(&dirs))
                .await
                .unwrap_or_default();
            state.write().await.model_registry.update(scanned);
        }

        emit_download_progress(
            &app,
            state.inner(),
            DownloadProgress {
                id: download_id.clone(),
                filename: filename.clone(),
                dest_path: Some(dest_path_string.clone()),
                downloaded_bytes: downloaded,
                total_bytes,
                percent: 1.0,
                done: true,
                status: "Completed".to_string(),
                error: None,
            },
            None,
        )
        .await;

        Ok(dest_path.to_string_lossy().to_string())
    }
    .await;

    if let Err(error) = &result {
        if error != "Download cancelled" {
            let _ = tokio::fs::remove_file(&dest_path).await;
            emit_download_progress(
                &app,
                state.inner(),
                DownloadProgress {
                    id: download_id,
                    filename,
                    dest_path: Some(dest_path_string),
                    downloaded_bytes: 0,
                    total_bytes: 0,
                    percent: 0.0,
                    done: true,
                    status: "Failed".to_string(),
                    error: Some(error.clone()),
                },
                None,
            )
            .await;
        }
    }

    result
}

/// Delete a local .gguf file and refresh the model registry.
#[tauri::command]
pub async fn delete_model_file(
    state: tauri::State<'_, SharedState>,
    path: String,
) -> Result<(), String> {
    let p = std::path::Path::new(&path);

    match p.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("gguf") => {}
        _ => return Err("Only .gguf files can be deleted via this command.".to_string()),
    }

    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }

    tokio::fs::remove_file(p)
        .await
        .map_err(|e| format!("Delete failed for {}: {e}", p.display()))?;

    // Rescan so deleted model vanishes from the UI
    {
        let s = state.read().await;
        let dirs = s.config.models.scan_dirs.clone();
        drop(s);
        let scanned = tokio::task::spawn_blocking(move || crate::models::scanner::scan_all(&dirs))
            .await
            .unwrap_or_default();
        state.write().await.model_registry.update(scanned);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{hf_supports_vision, sanitize_download_subpath, HfApiModel};
    use std::path::PathBuf;

    #[test]
    fn allows_nested_repo_paths() {
        let path = sanitize_download_subpath("BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf").unwrap();
        assert_eq!(
            path,
            PathBuf::from("BF16").join("Qwen3.5-27B-BF16-00001-of-00002.gguf")
        );
    }

    #[test]
    fn rejects_parent_traversal() {
        assert!(sanitize_download_subpath("../escape.gguf").is_err());
    }

    #[test]
    fn detects_vision_from_hugging_face_pipeline_tag() {
        let model = HfApiModel {
            model_id: "Qwen/Qwen3.5-35B-A3B".to_string(),
            downloads: 0,
            pipeline_tag: Some("image-text-to-text".to_string()),
            tags: vec!["gguf".to_string()],
            private: false,
            disabled: false,
            gated: None,
            siblings: vec![],
        };

        assert!(hf_supports_vision(&model));
    }

    #[test]
    fn detects_vision_from_hugging_face_tags() {
        let model = HfApiModel {
            model_id: "HauhauCS/Qwen3.5-35B-A3B-Uncensored".to_string(),
            downloads: 0,
            pipeline_tag: None,
            tags: vec!["gguf".to_string(), "vision".to_string(), "multimodal".to_string()],
            private: false,
            disabled: false,
            gated: None,
            siblings: vec![],
        };

        assert!(hf_supports_vision(&model));
    }
}

