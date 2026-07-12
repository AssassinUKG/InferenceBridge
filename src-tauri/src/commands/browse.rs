//! Model browser commands backed by live Hugging Face GGUF search and downloads.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::Emitter;
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

use crate::{models::overrides::HfModelMetadata, state::SharedState};

// Shared model browser types used by the UI and HF search results.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubQuant {
    pub quant: String,
    pub size_bytes: Option<u64>,
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
    pub hf_url: String,
    pub readme: Option<String>,
    pub license: Option<String>,
    pub base_model: Option<String>,
    pub pipeline_tag: Option<String>,
    pub tags: Vec<String>,
    pub supports_vision: bool,
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: Option<String>,
    pub quants: Vec<HubQuant>,
}

// Hugging Face live search metadata returned by the public model API.

#[derive(Debug, serde::Deserialize)]
struct HfSibling {
    #[serde(default)]
    rfilename: String,
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<HfLfs>,
}

#[derive(Debug, serde::Deserialize)]
struct HfLfs {
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    size: Option<u64>,
}

#[derive(Debug)]
struct HfApiModel {
    model_id: String,
    downloads: u64,
    likes: u64,
    last_modified: Option<String>,
    pipeline_tag: Option<String>,
    tags: Vec<String>,
    private: bool,
    disabled: bool,
    gated: Option<serde_json::Value>,
    siblings: Vec<HfSibling>,
}

impl<'de> serde::Deserialize<'de> for HfApiModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let object = value.as_object().ok_or_else(|| {
            serde::de::Error::custom("HuggingFace model record was not an object")
        })?;
        let model_id = value_string(object.get("modelId"))
            .or_else(|| value_string(object.get("id")))
            .unwrap_or_default();

        Ok(Self {
            model_id,
            downloads: value_u64(object.get("downloads")).unwrap_or(0),
            likes: value_u64(object.get("likes")).unwrap_or(0),
            last_modified: value_string(object.get("lastModified")),
            pipeline_tag: value_string(object.get("pipeline_tag")),
            tags: value_string_vec(object.get("tags")),
            private: value_bool(object.get("private")),
            disabled: value_bool(object.get("disabled")),
            gated: object.get("gated").cloned(),
            siblings: value_siblings(object.get("siblings")),
        })
    }
}

fn value_u64(value: Option<&serde_json::Value>) -> Option<u64> {
    match value {
        Some(serde_json::Value::Number(number)) => number.as_u64(),
        Some(serde_json::Value::String(text)) => text.parse::<u64>().ok(),
        _ => None,
    }
}

fn value_string(value: Option<&serde_json::Value>) -> Option<String> {
    match value {
        Some(serde_json::Value::String(text)) => Some(text.clone()),
        Some(serde_json::Value::Number(number)) => Some(number.to_string()),
        Some(serde_json::Value::Bool(flag)) => Some(flag.to_string()),
        _ => None,
    }
}

fn value_string_vec(value: Option<&serde_json::Value>) -> Vec<String> {
    match value {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

fn value_bool(value: Option<&serde_json::Value>) -> bool {
    match value {
        Some(serde_json::Value::Bool(flag)) => *flag,
        Some(serde_json::Value::String(text)) => text.eq_ignore_ascii_case("true"),
        Some(serde_json::Value::Number(number)) => number.as_u64().is_some_and(|value| value != 0),
        _ => false,
    }
}

fn value_siblings(value: Option<&serde_json::Value>) -> Vec<HfSibling> {
    match value {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|item| serde_json::from_value::<HfSibling>(item.clone()).ok())
            .collect(),
        _ => Vec::new(),
    }
}

fn deserialize_optional_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(match value {
        Some(serde_json::Value::Number(number)) => number.as_u64(),
        Some(serde_json::Value::String(text)) => text.parse::<u64>().ok(),
        _ => None,
    })
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

fn file_size_bytes(file: &HfSibling) -> Option<u64> {
    file.size
        .or_else(|| file.lfs.as_ref().and_then(|lfs| lfs.size))
}

fn file_size_gb(file: &HfSibling) -> f32 {
    file_size_bytes(file)
        .map(|sz| sz as f32 / 1_073_741_824.0)
        .unwrap_or(0.0)
}

fn tag_value(tags: &[String], prefix: &str) -> Option<String> {
    tags.iter()
        .find_map(|tag| tag.strip_prefix(prefix).map(str::to_string))
}

fn is_hf_downloadable(model: &HfApiModel, authenticated: bool) -> bool {
    if model.disabled {
        return false;
    }
    if model.private && !authenticated {
        return false;
    }

    match &model.gated {
        None => true,
        Some(serde_json::Value::Bool(false)) => true,
        Some(serde_json::Value::Null) => true,
        _ => authenticated,
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

fn hf_api_to_hub(m: HfApiModel, authenticated: bool, readme: Option<String>) -> Option<HubModel> {
    if !is_hf_downloadable(&m, authenticated) {
        return None;
    }

    let gguf_files: Vec<&HfSibling> = m
        .siblings
        .iter()
        .filter(|s| {
            let filename = s.rfilename.to_lowercase();
            filename.ends_with(".gguf")
                && !filename.starts_with("mmproj")
                && !filename.contains("/mmproj")
        })
        .collect();
    if gguf_files.is_empty() {
        return None;
    }

    let mut quants: Vec<HubQuant> = gguf_files
        .iter()
        .map(|s| HubQuant {
            quant: extract_quant(&s.rfilename),
            size_bytes: file_size_bytes(s),
            size_gb: file_size_gb(s),
            url: format!(
                "https://huggingface.co/{}/resolve/main/{}",
                m.model_id, s.rfilename
            ),
            filename: s.rfilename.clone(),
        })
        .collect();
    quants.sort_by(|left, right| {
        left.size_bytes
            .unwrap_or(u64::MAX)
            .cmp(&right.size_bytes.unwrap_or(u64::MAX))
    });

    let mut parts = m.model_id.split('/');
    let owner = parts.next().unwrap_or("HuggingFace");
    let repo_name = parts.next().unwrap_or(&m.model_id);
    let name = repo_name.replace('-', " ").replace('_', " ");
    let supports_vision = hf_supports_vision(&m);
    let license = tag_value(&m.tags, "license:");
    let base_model = tag_value(&m.tags, "base_model:quantized:")
        .or_else(|| tag_value(&m.tags, "base_model:adapter:"))
        .or_else(|| tag_value(&m.tags, "base_model:"));

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
        description: format!(
            "{} downloads | {}",
            format_downloads(m.downloads),
            m.model_id
        ),
        hf_url: format!("https://huggingface.co/{}", m.model_id),
        readme,
        license,
        base_model,
        pipeline_tag: m.pipeline_tag,
        tags,
        supports_vision,
        downloads: m.downloads,
        likes: m.likes,
        last_modified: m.last_modified,
        quants,
    })
}

async fn search_hf_api_models(
    query: Option<&str>,
    offset: u32,
    limit: u32,
    sort: Option<&str>,
    hf_api_key: Option<&str>,
) -> Result<Vec<HfApiModel>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("InferenceBridge/1.0")
        .build()
        .map_err(|e| e.to_string())?;

    let sort = match sort.unwrap_or("downloads") {
        "lastModified" | "createdAt" | "likes" | "downloads" => sort.unwrap_or("downloads"),
        _ => "downloads",
    };

    let mut req = client.get("https://huggingface.co/api/models").query(&[
        ("filter", "gguf".to_string()),
        ("sort", sort.to_string()),
        ("direction", "-1".to_string()),
        ("limit", limit.to_string()),
        ("offset", offset.to_string()),
        ("full", "true".to_string()),
        ("blobs", "true".to_string()),
    ]);

    if let Some(query) = query.map(str::trim).filter(|value| !value.is_empty()) {
        req = req.query(&[("search", query.to_string())]);
    }
    if let Some(key) = hf_api_key.map(str::trim).filter(|value| !value.is_empty()) {
        req = req.bearer_auth(key);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("HuggingFace request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HuggingFace returned HTTP {}", resp.status()));
    }

    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read HuggingFace response: {e}"))?;
    serde_json::from_str::<Vec<HfApiModel>>(&body).map_err(|e| {
        let snippet: String = body.chars().take(240).collect();
        format!("Failed to parse HuggingFace response: {e}. Body starts with: {snippet}")
    })
}

async fn fetch_hf_model_details(
    repo_id: &str,
    hf_api_key: Option<&str>,
) -> Result<HfApiModel, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("InferenceBridge/1.0")
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!("https://huggingface.co/api/models/{}", repo_id.trim());
    let mut req = client.get(url).query(&[("blobs", "true")]);
    if let Some(key) = hf_api_key.map(str::trim).filter(|value| !value.is_empty()) {
        req = req.bearer_auth(key);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("HuggingFace detail request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HuggingFace returned HTTP {}", resp.status()));
    }

    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read HuggingFace model details: {e}"))?;
    serde_json::from_str::<HfApiModel>(&body).map_err(|e| {
        let snippet: String = body.chars().take(240).collect();
        format!("Failed to parse HuggingFace model details: {e}. Body starts with: {snippet}")
    })
}

async fn fetch_hf_readme(
    repo_id: &str,
    hf_api_key: Option<&str>,
) -> Result<Option<String>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("InferenceBridge/1.0")
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!(
        "https://huggingface.co/{}/raw/main/README.md",
        repo_id.trim()
    );
    let mut req = client.get(url);
    if let Some(key) = hf_api_key.map(str::trim).filter(|value| !value.is_empty()) {
        req = req.bearer_auth(key);
    }

    let resp = req
        .send()
        .await
        .map_err(|e| format!("HuggingFace README request failed: {e}"))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        return Err(format!(
            "HuggingFace README returned HTTP {}",
            resp.status()
        ));
    }

    let text = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read HuggingFace README: {e}"))?;
    let trimmed: String = text.chars().take(16_000).collect();
    Ok(Some(trimmed))
}

async fn fetch_hub_models(
    query: Option<&str>,
    offset: u32,
    limit: u32,
    featured_only: bool,
    sort: Option<&str>,
    hf_api_key: Option<&str>,
) -> Result<Vec<HubModel>, String> {
    let models = search_hf_api_models(query, offset, limit, sort, hf_api_key).await?;
    let authenticated = hf_api_key
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());

    let mut seen = HashSet::new();
    Ok(models
        .into_iter()
        .filter(|model| !featured_only || is_hf_featured_candidate(model))
        .filter_map(|model| hf_api_to_hub(model, authenticated, None))
        .filter(|model| seen.insert(model.id.to_lowercase()))
        .collect())
}

fn basename_lower(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
        .unwrap_or_else(|| path.to_lowercase())
}

fn find_repo_chat_template_path(model: &HfApiModel) -> Option<String> {
    model
        .siblings
        .iter()
        .find(|sibling| basename_lower(&sibling.rfilename) == "chat_template.jinja")
        .map(|sibling| sibling.rfilename.clone())
}

fn push_unique_query(queries: &mut Vec<String>, seen: &mut HashSet<String>, value: &str) {
    let trimmed = value.trim().trim_matches('-').to_string();
    if trimmed.len() >= 4 && seen.insert(trimmed.to_lowercase()) {
        queries.push(trimmed);
    }
}

fn build_hf_sync_queries(filename: &str) -> Vec<String> {
    let stem = Path::new(filename)
        .file_stem()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| filename.to_string());
    let normalized = stem.replace('_', "-");

    let mut queries = Vec::new();
    let mut seen = HashSet::new();
    push_unique_query(&mut queries, &mut seen, &stem);
    push_unique_query(&mut queries, &mut seen, &normalized);

    let mut parts: Vec<&str> = normalized
        .split('-')
        .filter(|segment| !segment.is_empty())
        .collect();
    while parts.len() >= 3 && queries.len() < 6 {
        parts.pop();
        let candidate = parts.join("-");
        push_unique_query(&mut queries, &mut seen, &candidate);
    }

    queries
}

async fn lookup_hf_metadata(
    filename: &str,
    hf_api_key: Option<&str>,
) -> Result<Option<HfModelMetadata>, String> {
    let target = filename.to_lowercase();
    for query in build_hf_sync_queries(filename) {
        let models =
            search_hf_api_models(Some(&query), 0, 20, Some("downloads"), hf_api_key).await?;
        for model in models {
            let authenticated = hf_api_key
                .map(str::trim)
                .is_some_and(|value| !value.is_empty());
            if !is_hf_downloadable(&model, authenticated) {
                continue;
            }

            let has_exact_sibling = model.siblings.iter().any(|sibling| {
                let sibling_name = sibling.rfilename.to_lowercase();
                sibling_name == target || basename_lower(&sibling_name) == target
            });

            if has_exact_sibling {
                let template_path = find_repo_chat_template_path(&model);
                return Ok(Some(HfModelMetadata {
                    repo_id: Some(model.model_id.clone()),
                    file: Some(filename.to_string()),
                    has_repo_template: template_path.is_some(),
                    template_path,
                    supports_vision: Some(hf_supports_vision(&model)),
                }));
            }
        }
    }

    Ok(None)
}

/// Search HuggingFace for GGUF models. Returns up to 20 results sorted by downloads.
/// `offset` is the number of results to skip (for pagination).
#[tauri::command]
pub async fn search_hub_models(
    state: tauri::State<'_, SharedState>,
    query: String,
    offset: u32,
    sort: Option<String>,
    tag: Option<String>,
) -> Result<Vec<HubModel>, String> {
    let hf_api_key = {
        let s = state.read().await;
        s.config.hub.hf_api_key.clone()
    };
    let trimmed_query = query.trim();
    let trimmed_tag = tag
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let effective_query = match (trimmed_query.is_empty(), trimmed_tag) {
        (true, Some(tag)) => tag.to_string(),
        (false, Some(tag)) if !trimmed_query.eq_ignore_ascii_case(tag) => {
            format!("{trimmed_query} {tag}")
        }
        _ => trimmed_query.to_string(),
    };
    let mut models = fetch_hub_models(
        Some(&effective_query),
        offset,
        60,
        false,
        sort.as_deref(),
        hf_api_key.as_deref(),
    )
    .await?;
    models.truncate(20);
    Ok(models)
}

/// Fetch one HuggingFace repo with blob metadata. The search endpoint often omits file sizes.
#[tauri::command]
pub async fn get_hub_model_details(
    state: tauri::State<'_, SharedState>,
    repo_id: String,
    include_readme: Option<bool>,
) -> Result<Option<HubModel>, String> {
    let hf_api_key = {
        let s = state.read().await;
        s.config.hub.hf_api_key.clone()
    };
    let authenticated = hf_api_key
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    let model = fetch_hf_model_details(&repo_id, hf_api_key.as_deref()).await?;
    let readme = if include_readme.unwrap_or(false) {
        fetch_hf_readme(&repo_id, hf_api_key.as_deref()).await?
    } else {
        None
    };
    Ok(hf_api_to_hub(model, authenticated, readme))
}

// Download progress event payload emitted during model downloads.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub id: String,
    pub filename: String,
    pub dest_path: Option<String>,
    pub partial_path: Option<String>,
    pub supports_vision: Option<bool>,
    pub repo_id: Option<String>,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f32,
    pub speed_bps: Option<u64>,
    pub eta_seconds: Option<u64>,
    pub resumable: bool,
    pub attempt: u32,
    pub done: bool,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSyncSummary {
    pub scanned_models: usize,
    pub matched_models: usize,
    pub updated_models: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubAccessStatus {
    pub configured: bool,
    pub reachable: bool,
    pub user: Option<String>,
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

fn path_is_inside_any_dir<P: AsRef<std::path::Path>>(path: &std::path::Path, dirs: &[P]) -> bool {
    dirs.iter().any(|dir| {
        std::fs::canonicalize(dir)
            .map(|scan_dir| path.starts_with(scan_dir))
            .unwrap_or(false)
    })
}

fn partial_path_for(dest_path: &Path) -> PathBuf {
    let mut partial = dest_path.as_os_str().to_os_string();
    partial.push(".part");
    PathBuf::from(partial)
}

fn progress_percent(downloaded: u64, total: u64) -> f32 {
    if total > 0 {
        (downloaded as f32 / total as f32).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn estimate_eta_seconds(downloaded: u64, total: u64, speed_bps: u64) -> Option<u64> {
    if total == 0 || speed_bps == 0 || downloaded >= total {
        return None;
    }
    Some(((total - downloaded) / speed_bps).max(1))
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
pub async fn open_external_url(url: String) -> Result<(), String> {
    let trimmed = url.trim();
    if !(trimmed.starts_with("https://huggingface.co/")
        || trimmed.starts_with("https://www.huggingface.co/"))
    {
        return Err("Only Hugging Face URLs can be opened from the model browser.".to_string());
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", trimmed])
            .spawn()
            .map_err(|e| format!("Failed to open URL: {e}"))?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(trimmed)
            .spawn()
            .map_err(|e| format!("Failed to open URL: {e}"))?;
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::process::Command::new("xdg-open")
            .arg(trimmed)
            .spawn()
            .map_err(|e| format!("Failed to open URL: {e}"))?;
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
    let (progress, paused_path) = {
        let mut s = state.write().await;
        let entry = s
            .active_downloads
            .get_mut(&id)
            .ok_or_else(|| "Download not found.".to_string())?;
        if entry.progress.done && !(entry.progress.status == "Failed" && entry.progress.resumable) {
            return Ok(());
        }
        if entry.progress.status == "Paused"
            || (entry.progress.status == "Failed" && entry.progress.resumable)
        {
            entry.progress.done = true;
            entry.progress.status = "Cancelled".to_string();
            (
                entry.progress.clone(),
                entry
                    .progress
                    .partial_path
                    .clone()
                    .or_else(|| entry.progress.dest_path.clone()),
            )
        } else {
            entry.cancel_token.cancel();
            entry.progress.status = "Cancelling".to_string();
            (entry.progress.clone(), None)
        }
    };

    if let Some(path) = paused_path {
        let _ = tokio::fs::remove_file(path).await;
    }
    let _ = app.emit("model-download-progress", progress);
    Ok(())
}

#[tauri::command]
pub async fn pause_download(
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
        if entry.progress.done || entry.progress.status == "Paused" {
            return Ok(());
        }
        entry.cancel_token.cancel();
        entry.progress.status = "Pausing".to_string();
        entry.progress.clone()
    };

    let _ = app.emit("model-download-progress", progress);
    Ok(())
}

#[tauri::command]
pub async fn clear_completed_downloads(state: tauri::State<'_, SharedState>) -> Result<(), String> {
    let mut s = state.write().await;
    s.active_downloads.retain(|_, entry| !entry.progress.done);
    Ok(())
}

#[tauri::command]
pub async fn get_hub_access_status(
    state: tauri::State<'_, SharedState>,
) -> Result<HubAccessStatus, String> {
    let hf_api_key = {
        let s = state.read().await;
        s.config.hub.hf_api_key.clone()
    };
    let Some(key) = hf_api_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
    else {
        return Ok(HubAccessStatus {
            configured: false,
            reachable: true,
            user: None,
            error: None,
        });
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("InferenceBridge/1.0")
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get("https://huggingface.co/api/whoami-v2")
        .bearer_auth(&key)
        .send()
        .await
        .map_err(|e| format!("Hugging Face status request failed: {e}"))?;
    if !resp.status().is_success() {
        return Ok(HubAccessStatus {
            configured: true,
            reachable: false,
            user: None,
            error: Some(format!("Hugging Face returned HTTP {}", resp.status())),
        });
    }
    let value: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse Hugging Face status: {e}"))?;
    let user = value
        .get("name")
        .or_else(|| value.get("fullname"))
        .and_then(|value| value.as_str())
        .map(str::to_string);
    Ok(HubAccessStatus {
        configured: true,
        reachable: true,
        user,
        error: None,
    })
}

#[tauri::command]
pub async fn sync_local_model_metadata(
    state: tauri::State<'_, SharedState>,
) -> Result<MetadataSyncSummary, String> {
    let (filenames, scan_dirs, hf_api_key) = {
        let s = state.read().await;
        (
            s.model_registry
                .list()
                .iter()
                .map(|model| model.filename.clone())
                .collect::<Vec<_>>(),
            s.config.models.scan_dirs.clone(),
            s.config.hub.hf_api_key.clone(),
        )
    };

    let mut matched_models = 0usize;
    let mut updated_models = 0usize;

    for filename in &filenames {
        match lookup_hf_metadata(filename, hf_api_key.as_deref()).await {
            Ok(Some(metadata)) => {
                matched_models += 1;
                match crate::models::overrides::set_model_hf_metadata_override(
                    filename,
                    metadata.clone(),
                ) {
                    Ok(()) => updated_models += 1,
                    Err(error) => tracing::warn!(
                        model = %filename,
                        repo = ?metadata.repo_id,
                        error = %error,
                        "Failed to persist synced Hugging Face metadata override"
                    ),
                }
            }
            Ok(None) => {}
            Err(error) => tracing::warn!(
                model = %filename,
                error = %error,
                "Failed to sync model metadata from Hugging Face"
            ),
        }
    }

    let rescanned =
        tokio::task::spawn_blocking(move || crate::models::scanner::scan_all(&scan_dirs))
            .await
            .unwrap_or_default();

    state.write().await.model_registry.update(rescanned);

    Ok(MetadataSyncSummary {
        scanned_models: filenames.len(),
        matched_models,
        updated_models,
    })
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
    repo_id: Option<String>,
) -> Result<String, String> {
    let relative_path = sanitize_download_subpath(&filename)?;
    let download_id = url.clone();
    let (dest_dir, hf_api_key): (std::path::PathBuf, Option<String>) = {
        let s = state.read().await;
        let dest_dir = match s.config.models.scan_dirs.first() {
            Some(d) => std::path::PathBuf::from(d),
            None => {
                return Err(
                    "No model directory configured. Add one in Settings > Model Directories."
                        .to_string(),
                )
            }
        };
        (dest_dir, s.config.hub.hf_api_key.clone())
    };

    tokio::fs::create_dir_all(&dest_dir)
        .await
        .map_err(|e| format!("Cannot create {}: {e}", dest_dir.display()))?;

    let dest_path = dest_dir.join(&relative_path);
    let partial_path = partial_path_for(&dest_path);
    if let Some(parent) = dest_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Cannot create {}: {e}", parent.display()))?;
    }

    {
        let s = state.read().await;
        if let Some(existing) = s.active_downloads.get(&download_id) {
            if !existing.progress.done && existing.progress.status != "Paused" {
                return Err(format!(
                    "Download already in progress for {}",
                    existing.progress.filename
                ));
            }
        }
    }

    let cancel_token = CancellationToken::new();
    let dest_path_string = dest_path.to_string_lossy().to_string();
    let partial_path_string = partial_path.to_string_lossy().to_string();
    let resume_from = {
        let s = state.read().await;
        let can_resume = s
            .active_downloads
            .get(&download_id)
            .is_some_and(|entry| entry.progress.status == "Paused");
        drop(s);
        let partial_len = tokio::fs::metadata(&partial_path)
            .await
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        if can_resume || partial_len > 0 {
            partial_len
        } else {
            0
        }
    };
    emit_download_progress(
        &app,
        state.inner(),
        DownloadProgress {
            id: download_id.clone(),
            filename: filename.clone(),
            dest_path: Some(dest_path_string.clone()),
            partial_path: Some(partial_path_string.clone()),
            supports_vision,
            repo_id: repo_id.clone(),
            downloaded_bytes: resume_from,
            total_bytes: 0,
            percent: 0.0,
            speed_bps: None,
            eta_seconds: None,
            resumable: resume_from > 0,
            attempt: 1,
            done: false,
            status: if resume_from > 0 {
                "Resuming".to_string()
            } else {
                "Starting".to_string()
            },
            error: None,
        },
        Some(cancel_token.clone()),
    )
    .await;

    let result: Result<String, String> = async {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(20))
            .timeout(std::time::Duration::from_secs(7200)) // 2-hour ceiling for large models
            .redirect(reqwest::redirect::Policy::limited(10))
            .user_agent("InferenceBridge/1.0")
            .build()
            .map_err(|e| e.to_string())?;

        let mut downloaded = resume_from;
        let mut total_bytes = 0u64;
        let mut cancelled = false;
        let mut completed = false;
        let mut last_error: Option<String> = None;

        for attempt in 1..=5u32 {
            let disk_resume = tokio::fs::metadata(&partial_path)
                .await
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            let mut request = client
                .get(&url)
                .header(reqwest::header::ACCEPT, "application/octet-stream");
            if disk_resume > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={disk_resume}-"));
            }
            if let Some(key) = hf_api_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                request = request.bearer_auth(key);
            }

            let resp = match request.send().await {
                Ok(resp) => resp,
                Err(error) => {
                    last_error = Some(format!("Download request failed: {error}"));
                    tokio::time::sleep(std::time::Duration::from_millis(750 * attempt as u64))
                        .await;
                    continue;
                }
            };

            if disk_resume > 0 && resp.status() == reqwest::StatusCode::OK {
                tracing::warn!(
                    url = %url,
                    resume_from = disk_resume,
                    "Hugging Face ignored Range resume; restarting partial model download"
                );
                let _ = tokio::fs::remove_file(&partial_path).await;
            } else if !(resp.status().is_success()
                || resp.status() == reqwest::StatusCode::PARTIAL_CONTENT)
            {
                last_error = Some(format!(
                    "Server returned HTTP {} for {}",
                    resp.status(),
                    url
                ));
                tokio::time::sleep(std::time::Duration::from_millis(750 * attempt as u64)).await;
                continue;
            }

            let resumed = disk_resume > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
            downloaded = if resumed { disk_resume } else { 0 };
            total_bytes = resp
                .content_length()
                .map(|length| length + downloaded)
                .unwrap_or(downloaded);
            let mut file = if resumed {
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&partial_path)
                    .await
                    .map_err(|e| format!("Cannot open {}: {e}", partial_path.display()))?
            } else {
                tokio::fs::File::create(&partial_path)
                    .await
                    .map_err(|e| format!("Cannot create {}: {e}", partial_path.display()))?
            };

            let mut stream = resp.bytes_stream();
            let mut last_emit = std::time::Instant::now();
            let mut last_speed_sample = std::time::Instant::now();
            let mut last_speed_bytes = downloaded;
            let mut speed_bps = 0u64;
            let mut attempt_failed = false;

            while let Some(chunk) = tokio::select! {
                _ = cancel_token.cancelled() => {
                    cancelled = true;
                    None
                }
                chunk = stream.next() => chunk
            } {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error) => {
                        last_error = Some(format!("Download error: {error}"));
                        attempt_failed = true;
                        break;
                    }
                };
                file.write_all(&chunk)
                    .await
                    .map_err(|e| format!("Write error: {e}"))?;
                downloaded += chunk.len() as u64;

                if last_speed_sample.elapsed().as_millis() >= 1000 {
                    let elapsed = last_speed_sample.elapsed().as_secs_f64().max(0.001);
                    speed_bps = ((downloaded - last_speed_bytes) as f64 / elapsed).max(0.0) as u64;
                    last_speed_bytes = downloaded;
                    last_speed_sample = std::time::Instant::now();
                }

                if last_emit.elapsed().as_millis() >= 250 {
                    emit_download_progress(
                        &app,
                        state.inner(),
                        DownloadProgress {
                            id: download_id.clone(),
                            filename: filename.clone(),
                            dest_path: Some(dest_path_string.clone()),
                            partial_path: Some(partial_path_string.clone()),
                            supports_vision,
                            repo_id: repo_id.clone(),
                            downloaded_bytes: downloaded,
                            total_bytes,
                            percent: progress_percent(downloaded, total_bytes),
                            speed_bps: Some(speed_bps),
                            eta_seconds: estimate_eta_seconds(downloaded, total_bytes, speed_bps),
                            resumable: true,
                            attempt,
                            done: false,
                            status: if attempt > 1 {
                                "Retrying".to_string()
                            } else {
                                "Downloading".to_string()
                            },
                            error: None,
                        },
                        None,
                    )
                    .await;
                    last_emit = std::time::Instant::now();
                }
            }

            file.flush()
                .await
                .map_err(|e| format!("Flush error: {e}"))?;
            drop(file);

            if cancelled {
                break;
            }
            if attempt_failed || (total_bytes > 0 && downloaded < total_bytes) {
                if !attempt_failed {
                    last_error = Some(format!(
                        "Download incomplete: {} of {} bytes",
                        downloaded, total_bytes
                    ));
                }
                emit_download_progress(
                    &app,
                    state.inner(),
                    DownloadProgress {
                        id: download_id.clone(),
                        filename: filename.clone(),
                        dest_path: Some(dest_path_string.clone()),
                        partial_path: Some(partial_path_string.clone()),
                        supports_vision,
                        repo_id: repo_id.clone(),
                        downloaded_bytes: downloaded,
                        total_bytes,
                        percent: progress_percent(downloaded, total_bytes),
                        speed_bps: Some(speed_bps),
                        eta_seconds: estimate_eta_seconds(downloaded, total_bytes, speed_bps),
                        resumable: true,
                        attempt,
                        done: false,
                        status: "Retrying".to_string(),
                        error: last_error.clone(),
                    },
                    None,
                )
                .await;
                tokio::time::sleep(std::time::Duration::from_millis(750 * attempt as u64)).await;
                continue;
            }
            completed = true;
            break;
        }

        if cancelled {
            let paused = {
                let s = state.read().await;
                s.active_downloads
                    .get(&download_id)
                    .is_some_and(|entry| entry.progress.status == "Pausing")
            };
            if !paused {
                let _ = tokio::fs::remove_file(&partial_path).await;
            }
            emit_download_progress(
                &app,
                state.inner(),
                DownloadProgress {
                    id: download_id.clone(),
                    filename: filename.clone(),
                    dest_path: Some(dest_path_string.clone()),
                    partial_path: Some(partial_path_string.clone()),
                    supports_vision,
                    repo_id: repo_id.clone(),
                    downloaded_bytes: downloaded,
                    total_bytes,
                    percent: progress_percent(downloaded, total_bytes),
                    speed_bps: None,
                    eta_seconds: None,
                    resumable: paused,
                    attempt: 1,
                    done: !paused,
                    status: if paused {
                        "Paused".to_string()
                    } else {
                        "Cancelled".to_string()
                    },
                    error: None,
                },
                None,
            )
            .await;
            return Err(if paused {
                "Download paused".to_string()
            } else {
                "Download cancelled".to_string()
            });
        }

        if !completed {
            return Err(last_error.unwrap_or_else(|| "Download failed after retries".to_string()));
        }

        if total_bytes > 0 && downloaded < total_bytes {
            return Err(last_error.unwrap_or_else(|| {
                format!(
                    "Download incomplete: {} of {} bytes",
                    downloaded, total_bytes
                )
            }));
        }

        tokio::fs::rename(&partial_path, &dest_path)
            .await
            .map_err(|e| format!("Cannot finalize {}: {e}", dest_path.display()))?;

        if let Some(model_filename) = relative_path
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_string)
        {
            let metadata = HfModelMetadata {
                repo_id: repo_id.clone(),
                file: Some(filename.clone()),
                template_path: None,
                has_repo_template: false,
                supports_vision,
            };
            if let Err(error) =
                crate::models::overrides::set_model_hf_metadata_override(&model_filename, metadata)
            {
                tracing::warn!(
                    model = %model_filename,
                    repo = ?repo_id,
                    error = %error,
                    "Failed to persist Hugging Face metadata override"
                );
            }
        }

        {
            let s = state.read().await;
            let dirs = s.config.models.scan_dirs.clone();
            drop(s);
            let scanned =
                tokio::task::spawn_blocking(move || crate::models::scanner::scan_all(&dirs))
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
                partial_path: None,
                supports_vision,
                repo_id: repo_id.clone(),
                downloaded_bytes: downloaded,
                total_bytes,
                percent: 1.0,
                speed_bps: None,
                eta_seconds: None,
                resumable: false,
                attempt: 1,
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
        if error != "Download cancelled" && error != "Download paused" {
            emit_download_progress(
                &app,
                state.inner(),
                DownloadProgress {
                    id: download_id,
                    filename,
                    dest_path: Some(dest_path_string),
                    partial_path: Some(partial_path_string),
                    supports_vision,
                    repo_id,
                    downloaded_bytes: tokio::fs::metadata(&partial_path)
                        .await
                        .map(|m| m.len())
                        .unwrap_or(0),
                    total_bytes: 0,
                    percent: 0.0,
                    speed_bps: None,
                    eta_seconds: None,
                    resumable: tokio::fs::metadata(&partial_path)
                        .await
                        .map(|m| m.len())
                        .unwrap_or(0)
                        > 0,
                    attempt: 5,
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
) -> Result<String, String> {
    let p = std::path::PathBuf::from(&path);

    match p.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("gguf") => {}
        _ => return Err("Only .gguf files can be deleted via this command.".to_string()),
    }

    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }

    let canonical_path = tokio::fs::canonicalize(&p)
        .await
        .map_err(|e| format!("Could not resolve {}: {e}", p.display()))?;

    let (scan_dirs, loaded_model) = {
        let s = state.read().await;
        (s.config.models.scan_dirs.clone(), s.loaded_model.clone())
    };

    if !path_is_inside_any_dir(&canonical_path, &scan_dirs) {
        return Err(format!(
            "Refusing to delete {}; it is not inside a configured model directory.",
            canonical_path.display()
        ));
    }

    let filename = canonical_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model")
        .to_string();
    let mut stopped_loaded_model = false;
    if loaded_model.as_deref() == Some(&filename) {
        let load_mutex = {
            let s = state.read().await;
            s.model_load_mutex.clone()
        };
        let _load_guard = load_mutex.lock().await;
        crate::commands::model::stop_model_for_binary_update(state.inner().clone()).await?;
        stopped_loaded_model = true;
    }

    tokio::fs::remove_file(&canonical_path)
        .await
        .map_err(|e| format!("Delete failed for {}: {e}", canonical_path.display()))?;

    // Rescan so deleted model vanishes from the UI
    {
        let scanned =
            tokio::task::spawn_blocking(move || crate::models::scanner::scan_all(&scan_dirs))
                .await
                .unwrap_or_default();
        state.write().await.model_registry.update(scanned);
    }

    if stopped_loaded_model {
        Ok(format!("Deleted {filename} after unloading it."))
    } else {
        Ok(format!("Deleted {filename}."))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_hf_sync_queries, find_repo_chat_template_path, hf_supports_vision,
        is_hf_downloadable, path_is_inside_any_dir, sanitize_download_subpath, HfApiModel,
        HfSibling,
    };
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
    fn public_mode_rejects_gated_models() {
        let model = HfApiModel {
            model_id: "owner/model".to_string(),
            downloads: 0,
            pipeline_tag: None,
            tags: vec![],
            private: false,
            disabled: false,
            gated: Some(serde_json::json!("manual")),
            siblings: vec![],
            likes: 0,
            last_modified: None,
        };

        assert!(!is_hf_downloadable(&model, false));
    }

    #[test]
    fn authenticated_mode_allows_gated_models() {
        let model = HfApiModel {
            model_id: "owner/model".to_string(),
            downloads: 0,
            pipeline_tag: None,
            tags: vec![],
            private: false,
            disabled: false,
            gated: Some(serde_json::json!("manual")),
            siblings: vec![],
            likes: 0,
            last_modified: None,
        };

        assert!(is_hf_downloadable(&model, true));
    }

    #[test]
    fn authenticated_mode_allows_private_models_returned_by_hf() {
        let model = HfApiModel {
            model_id: "owner/private".to_string(),
            downloads: 0,
            pipeline_tag: None,
            tags: vec![],
            private: true,
            disabled: false,
            gated: None,
            siblings: vec![],
            likes: 0,
            last_modified: None,
        };

        assert!(is_hf_downloadable(&model, true));
    }

    #[test]
    fn delete_guard_allows_models_inside_scan_dir() {
        let root = std::env::temp_dir().join(format!(
            "inference-bridge-delete-test-{}",
            std::process::id()
        ));
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let model = nested.join("model.gguf");
        std::fs::write(&model, b"test").unwrap();

        let canonical_model = std::fs::canonicalize(&model).unwrap();
        assert!(path_is_inside_any_dir(
            &canonical_model,
            &[root.to_string_lossy().to_string()]
        ));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn delete_guard_rejects_models_outside_scan_dir() {
        let root = std::env::temp_dir().join(format!(
            "inference-bridge-delete-test-{}-root",
            std::process::id()
        ));
        let outside = std::env::temp_dir().join(format!(
            "inference-bridge-delete-test-{}-outside",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let model = outside.join("model.gguf");
        std::fs::write(&model, b"test").unwrap();

        let canonical_model = std::fs::canonicalize(&model).unwrap();
        assert!(!path_is_inside_any_dir(
            &canonical_model,
            &[root.to_string_lossy().to_string()]
        ));

        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn reduces_quantized_filename_to_searchable_hf_queries() {
        let queries = build_hf_sync_queries("Qwen3.5-35B-A3B-Q4_K_M.gguf");
        assert!(queries.iter().any(|query| query == "Qwen3.5-35B-A3B"));
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
            likes: 0,
            last_modified: None,
        };

        assert!(hf_supports_vision(&model));
    }

    #[test]
    fn detects_vision_from_hugging_face_tags() {
        let model = HfApiModel {
            model_id: "HauhauCS/Qwen3.5-35B-A3B-Uncensored".to_string(),
            downloads: 0,
            pipeline_tag: None,
            tags: vec![
                "gguf".to_string(),
                "vision".to_string(),
                "multimodal".to_string(),
            ],
            private: false,
            disabled: false,
            gated: None,
            siblings: vec![],
            likes: 0,
            last_modified: None,
        };

        assert!(hf_supports_vision(&model));
    }

    #[test]
    fn detects_actual_repo_chat_template_sibling() {
        let model = HfApiModel {
            model_id: "owner/model".to_string(),
            downloads: 0,
            likes: 0,
            last_modified: None,
            pipeline_tag: None,
            tags: vec![],
            private: false,
            disabled: false,
            gated: None,
            siblings: vec![
                HfSibling {
                    rfilename: "Qwen3-8B-Q4_K_M.gguf".to_string(),
                    size: None,
                    lfs: None,
                },
                HfSibling {
                    rfilename: "nested/chat_template.jinja".to_string(),
                    size: None,
                    lfs: None,
                },
            ],
        };

        assert_eq!(
            find_repo_chat_template_path(&model).as_deref(),
            Some("nested/chat_template.jinja")
        );
    }

    #[test]
    fn leaves_repo_template_empty_when_sibling_is_absent() {
        let model = HfApiModel {
            model_id: "owner/model".to_string(),
            downloads: 0,
            likes: 0,
            last_modified: None,
            pipeline_tag: None,
            tags: vec![],
            private: false,
            disabled: false,
            gated: None,
            siblings: vec![
                HfSibling {
                    rfilename: "Qwen3-8B-Q4_K_M.gguf".to_string(),
                    size: None,
                    lfs: None,
                },
                HfSibling {
                    rfilename: "tokenizer_config.json".to_string(),
                    size: None,
                    lfs: None,
                },
            ],
        };

        assert!(find_repo_chat_template_path(&model).is_none());
    }
}
