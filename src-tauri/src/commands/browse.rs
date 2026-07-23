//! Model browser commands backed by live Hugging Face GGUF search and downloads.

use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions as StdOpenOptions;
use std::io::Write as StdWrite;
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
    pub author: Option<String>,
    pub params: String,
    pub description: String,
    pub hf_url: String,
    pub readme: Option<String>,
    pub license: Option<String>,
    pub base_model: Option<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    pub supports_vision: bool,
    pub downloads: u64,
    pub likes: u64,
    pub created_at: Option<String>,
    pub last_modified: Option<String>,
    pub gguf_total: Option<u64>,
    pub gguf_architecture: Option<String>,
    pub gguf_context_length: Option<u64>,
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

#[derive(Debug, serde::Deserialize)]
struct HfGgufMetadata {
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    total: Option<u64>,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    context_length: Option<u64>,
}

#[derive(Debug)]
struct HfApiModel {
    model_id: String,
    author: Option<String>,
    downloads: u64,
    likes: u64,
    created_at: Option<String>,
    last_modified: Option<String>,
    pipeline_tag: Option<String>,
    library_name: Option<String>,
    tags: Vec<String>,
    gguf: Option<HfGgufMetadata>,
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
            author: value_string(object.get("author")),
            downloads: value_u64(object.get("downloads")).unwrap_or(0),
            likes: value_u64(object.get("likes")).unwrap_or(0),
            created_at: value_string(object.get("createdAt")),
            last_modified: value_string(object.get("lastModified")),
            pipeline_tag: value_string(object.get("pipeline_tag")),
            library_name: value_string(object.get("library_name")),
            tags: value_string_vec(object.get("tags")),
            gguf: object
                .get("gguf")
                .and_then(|value| serde_json::from_value(value.clone()).ok()),
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

fn format_parameter_count(total: u64) -> String {
    let (scale, suffix) = if total >= 1_000_000_000 {
        (1_000_000_000.0, "B")
    } else if total >= 1_000_000 {
        (1_000_000.0, "M")
    } else {
        return total.to_string();
    };

    let formatted = format!("{:.1}", total as f64 / scale);
    format!("{}{}", formatted.trim_end_matches(".0"), suffix)
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

    let (gguf_total, gguf_architecture, gguf_context_length) = m
        .gguf
        .map(|gguf| (gguf.total, gguf.architecture, gguf.context_length))
        .unwrap_or((None, None, None));
    let params = gguf_total.map(format_parameter_count).unwrap_or_default();

    let mut seen_tags = HashSet::new();
    let mut tags = Vec::with_capacity(12);
    for tag in m.tags {
        let tag = tag.trim();
        if tag.is_empty() || tag.contains(':') || tag.starts_with("base_model") || tag.len() >= 24 {
            continue;
        }
        if seen_tags.insert(tag.to_lowercase()) {
            tags.push(tag.to_string());
        }
        if tags.len() == 12 {
            break;
        }
    }
    if supports_vision && !tags.iter().any(|tag| tag.eq_ignore_ascii_case("vision")) {
        tags.insert(0, "vision".to_string());
        tags.truncate(12);
    }

    Some(HubModel {
        id: m.model_id.clone(),
        name,
        family: owner.to_string(),
        author: m.author,
        params,
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
        library_name: m.library_name,
        tags,
        supports_vision,
        downloads: m.downloads,
        likes: m.likes,
        created_at: m.created_at,
        last_modified: m.last_modified,
        gguf_total,
        gguf_architecture,
        gguf_context_length,
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

const DOWNLOAD_MANIFEST_VERSION: u32 = 1;
const DOWNLOAD_MANIFEST_FILE: &str = "download-jobs.json";
const DOWNLOAD_STATUS_CLEANUP_PENDING: &str = "Cleanup pending";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DownloadManifest {
    version: u32,
    jobs: Vec<DownloadProgress>,
}

impl Default for DownloadManifest {
    fn default() -> Self {
        Self {
            version: DOWNLOAD_MANIFEST_VERSION,
            jobs: Vec::new(),
        }
    }
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

fn download_manifest_path() -> PathBuf {
    crate::config::app_support_dir().join(DOWNLOAD_MANIFEST_FILE)
}

fn manifest_sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut sibling = path.as_os_str().to_os_string();
    sibling.push(suffix);
    PathBuf::from(sibling)
}

fn trusted_hugging_face_download_url(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|parsed| {
        parsed.scheme() == "https"
            && matches!(
                parsed.host_str(),
                Some("huggingface.co" | "www.huggingface.co")
            )
    })
}

fn normalized_path_key(path: &Path) -> Option<String> {
    let normalized = if path.exists() {
        std::fs::canonicalize(path).ok()?
    } else {
        let parent = std::fs::canonicalize(path.parent()?).ok()?;
        parent.join(path.file_name()?)
    };
    let text = normalized.to_string_lossy().replace('/', "\\");
    Some(if cfg!(windows) {
        text.to_lowercase()
    } else {
        text
    })
}

fn download_destination_key(path: &Path) -> String {
    normalized_path_key(path).unwrap_or_else(|| {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(path)
        };
        let text = absolute.to_string_lossy().replace('/', "\\");
        if cfg!(windows) {
            text.to_lowercase()
        } else {
            text
        }
    })
}

fn path_parent_is_within_scan_dir(path: &Path, scan_dir: &Path) -> bool {
    let Ok(scan_root) = std::fs::canonicalize(scan_dir) else {
        return false;
    };
    let Some(mut ancestor) = path.parent() else {
        return false;
    };
    while !ancestor.exists() {
        let Some(parent) = ancestor.parent() else {
            return false;
        };
        ancestor = parent;
    }
    std::fs::canonicalize(ancestor)
        .is_ok_and(|canonical_parent| canonical_parent.starts_with(scan_root))
}

fn validated_persisted_destination(
    progress: &DownloadProgress,
    scan_dirs: &[PathBuf],
) -> Option<(PathBuf, PathBuf)> {
    if !trusted_hugging_face_download_url(&progress.id) {
        return None;
    }
    let relative = sanitize_download_subpath(&progress.filename).ok()?;
    let stored_dest = PathBuf::from(progress.dest_path.as_deref()?);
    let stored_key = normalized_path_key(&stored_dest)?;

    for scan_dir in scan_dirs {
        let candidate = scan_dir.join(&relative);
        if path_parent_is_within_scan_dir(&candidate, scan_dir)
            && normalized_path_key(&candidate).as_deref() == Some(stored_key.as_str())
        {
            return Some((candidate.clone(), partial_path_for(&candidate)));
        }
    }
    None
}

fn should_auto_resume_status(status: &str) -> bool {
    matches!(
        status,
        "Starting" | "Resuming" | "Downloading" | "Retrying" | "Interrupted"
    )
}

fn read_download_manifest_at(path: &Path) -> Result<DownloadManifest, String> {
    let read_one = |candidate: &Path| -> Result<(DownloadManifest, std::time::SystemTime), String> {
        let contents = std::fs::read_to_string(candidate)
            .map_err(|error| format!("Failed to read {}: {error}", candidate.display()))?;
        let manifest: DownloadManifest = serde_json::from_str(&contents)
            .map_err(|error| format!("Failed to parse {}: {error}", candidate.display()))?;
        if manifest.version != DOWNLOAD_MANIFEST_VERSION {
            return Err(format!(
                "Unsupported download manifest version {} in {}",
                manifest.version,
                candidate.display()
            ));
        }
        let modified = std::fs::metadata(candidate)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        Ok((manifest, modified))
    };

    let candidates = [
        (path.to_path_buf(), 2u8),
        (manifest_sibling(path, ".next"), 3u8),
        (manifest_sibling(path, ".bak"), 1u8),
    ];
    let mut valid = Vec::new();
    let mut errors = Vec::new();
    for (candidate, priority) in candidates {
        if !candidate.exists() {
            continue;
        }
        match read_one(&candidate) {
            Ok((manifest, modified)) => valid.push((modified, priority, manifest)),
            Err(error) => errors.push(error),
        }
    }
    valid
        .into_iter()
        .max_by_key(|(modified, priority, _)| (*modified, *priority))
        .map(|(_, _, manifest)| manifest)
        .ok_or_else(|| {
            if errors.is_empty() {
                format!("No download manifest found at {}", path.display())
            } else {
                errors.join("; ")
            }
        })
}

fn write_download_manifest_atomic_at(
    path: &Path,
    manifest: &DownloadManifest,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create {}: {error}", parent.display()))?;
    }

    let next = manifest_sibling(path, ".next");
    let backup = manifest_sibling(path, ".bak");
    let contents = serde_json::to_vec_pretty(manifest)
        .map_err(|error| format!("Failed to serialize download manifest: {error}"))?;
    let mut file = StdOpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&next)
        .map_err(|error| format!("Failed to open {}: {error}", next.display()))?;
    file.write_all(&contents)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("Failed to write {}: {error}", next.display()))?;
    drop(file);

    if backup.exists() {
        std::fs::remove_file(&backup)
            .map_err(|error| format!("Failed to remove {}: {error}", backup.display()))?;
    }
    if path.exists() {
        std::fs::rename(path, &backup).map_err(|error| {
            format!(
                "Failed to rotate {} to {}: {error}",
                path.display(),
                backup.display()
            )
        })?;
    }
    if let Err(error) = std::fs::rename(&next, path) {
        if backup.exists() {
            let _ = std::fs::rename(&backup, path);
        }
        return Err(format!(
            "Failed to install download manifest {}: {error}",
            path.display()
        ));
    }
    if backup.exists() {
        let _ = std::fs::remove_file(backup);
    }
    Ok(())
}

pub(crate) fn load_persisted_downloads(scan_dirs: &[PathBuf]) -> Vec<DownloadProgress> {
    let path = download_manifest_path();
    load_persisted_downloads_at(&path, scan_dirs)
}

fn load_persisted_downloads_at(path: &Path, scan_dirs: &[PathBuf]) -> Vec<DownloadProgress> {
    if !path.exists()
        && !manifest_sibling(path, ".next").exists()
        && !manifest_sibling(path, ".bak").exists()
    {
        return Vec::new();
    }
    let manifest = match read_download_manifest_at(path) {
        Ok(manifest) => manifest,
        Err(error) => {
            tracing::warn!(path = %path.display(), error = %error, "Could not restore download manifest");
            return Vec::new();
        }
    };
    let mut restored: HashMap<String, DownloadProgress> = HashMap::new();
    let mut owned_destinations: HashMap<String, String> = HashMap::new();

    for mut progress in manifest.jobs {
        let Some((dest_path, partial_path)) = validated_persisted_destination(&progress, scan_dirs)
        else {
            tracing::warn!(
                id = %progress.id,
                filename = %progress.filename,
                "Ignoring persisted download whose source or destination is no longer trusted"
            );
            continue;
        };

        let dest_len = std::fs::metadata(&dest_path)
            .map(|metadata| metadata.len())
            .ok();
        let partial_len = std::fs::metadata(&partial_path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        progress.dest_path = Some(dest_path.to_string_lossy().to_string());
        progress.speed_bps = None;
        progress.eta_seconds = None;

        if matches!(
            progress.status.as_str(),
            "Cancelling" | "Cancelled" | DOWNLOAD_STATUS_CLEANUP_PENDING
        ) {
            // Cancellation intent wins even if an older completed file already
            // exists at the destination. Only the recomputed trusted partial is
            // removed; a completed destination is never touched here.
            let cleanup_error = if partial_len > 0 {
                std::fs::remove_file(&partial_path).err()
            } else {
                None
            };
            if let Some(error) = cleanup_error {
                tracing::warn!(
                    path = %partial_path.display(),
                    error = %error,
                    "Could not finish deletion for a cancelled download"
                );
                progress.partial_path = Some(partial_path.to_string_lossy().to_string());
                progress.downloaded_bytes = partial_len;
                progress.percent = progress_percent(partial_len, progress.total_bytes);
                progress.resumable = false;
                progress.done = false;
                progress.status = DOWNLOAD_STATUS_CLEANUP_PENDING.to_string();
                progress.error = Some(format!(
                    "Could not discard partial {}: {error}. Cleanup will retry on next start.",
                    partial_path.display()
                ));
            } else {
                progress.dest_path = None;
                progress.partial_path = None;
                progress.downloaded_bytes = 0;
                progress.percent = 0.0;
                progress.resumable = false;
                progress.done = true;
                progress.status = "Cancelled".to_string();
                progress.error = None;
            }
        } else if let Some(length) = dest_len.filter(|length| {
            partial_len == 0
                && (progress.status == "Completed"
                    || (progress.total_bytes > 0 && progress.total_bytes == *length))
        }) {
            progress.partial_path = None;
            progress.downloaded_bytes = length;
            progress.total_bytes = progress.total_bytes.max(length);
            progress.percent = 1.0;
            progress.resumable = false;
            progress.done = true;
            progress.status = "Completed".to_string();
            progress.error = None;
        } else {
            progress.partial_path = Some(partial_path.to_string_lossy().to_string());
            progress.downloaded_bytes = partial_len;
            progress.total_bytes = progress.total_bytes.max(partial_len);
            progress.percent = progress_percent(partial_len, progress.total_bytes);
            progress.resumable = partial_len > 0;

            if dest_len.is_some() && partial_len == 0 {
                progress.done = true;
                progress.status = "Failed".to_string();
                progress.error = Some(
                    "A destination file exists but its size does not match the interrupted download."
                        .to_string(),
                );
            } else if should_auto_resume_status(&progress.status) {
                progress.done = false;
                progress.status = "Interrupted".to_string();
                progress.error = None;
            } else if progress.status == "Pausing" {
                progress.done = false;
                progress.status = "Paused".to_string();
                progress.error = None;
            } else if progress.status == "Paused" {
                progress.done = false;
                progress.error = None;
            } else if progress.status == "Completed" {
                progress.done = true;
                progress.status = "Failed".to_string();
                progress.error = Some("Completed file is missing from disk.".to_string());
            }
        }
        if !progress.done || progress.resumable {
            let destination_key = download_destination_key(&dest_path);
            if let Some(existing_id) = owned_destinations.get(&destination_key) {
                let conflict_error = format!(
                    "Multiple persisted sources target {}; reselect the intended Hub file.",
                    dest_path.display()
                );
                let current_cleanup = progress.status == DOWNLOAD_STATUS_CLEANUP_PENDING;
                if let Some(existing) = restored.get_mut(existing_id) {
                    let existing_cleanup = existing.status == DOWNLOAD_STATUS_CLEANUP_PENDING;
                    if existing_cleanup {
                        // Cancellation cleanup owns the path until deletion
                        // succeeds; never turn that intent back into Resume.
                    } else {
                        existing.done = true;
                        existing.status = "Failed".to_string();
                        existing.error = Some(conflict_error.clone());
                    }
                    if !current_cleanup || existing_cleanup {
                        progress.done = true;
                        progress.status = "Failed".to_string();
                        progress.error = Some(conflict_error);
                    }
                }
            } else {
                owned_destinations.insert(destination_key, progress.id.clone());
            }
        }
        restored.insert(progress.id.clone(), progress);
    }

    restored.into_values().collect()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParsedContentRange {
    Satisfied {
        start: u64,
        end: u64,
        total: Option<u64>,
    },
    Unsatisfied {
        total: u64,
    },
}

fn parse_content_range(value: &str) -> Option<ParsedContentRange> {
    let value = value.trim().strip_prefix("bytes ")?;
    let (range, total) = value.split_once('/')?;
    if range == "*" {
        return total
            .parse::<u64>()
            .ok()
            .map(|total| ParsedContentRange::Unsatisfied { total });
    }
    let (start, end) = range.split_once('-')?;
    let start = start.parse::<u64>().ok()?;
    let end = end.parse::<u64>().ok()?;
    if end < start {
        return None;
    }
    let total = if total == "*" {
        None
    } else {
        Some(total.parse::<u64>().ok()?)
    };
    Some(ParsedContentRange::Satisfied { start, end, total })
}

fn response_content_range(response: &reqwest::Response) -> Option<ParsedContentRange> {
    response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()
        .and_then(parse_content_range)
}

fn content_range_matches_resume(
    range: Option<ParsedContentRange>,
    expected_start: u64,
    content_length: Option<u64>,
) -> bool {
    matches!(
        range,
        Some(ParsedContentRange::Satisfied { start, end, total })
            if start == expected_start
                && total.map_or(true, |total| end < total)
                && content_length.map_or(true, |length| length == end - start + 1)
    )
}

async fn upsert_download(
    state: &SharedState,
    progress: DownloadProgress,
    cancel_token: Option<CancellationToken>,
) -> bool {
    let mut s = state.write().await;
    if let Some(existing) = s.active_downloads.get_mut(&progress.id) {
        let persist = existing.progress.status != progress.status
            || existing.progress.done != progress.done
            || existing.progress.error != progress.error
            || existing.progress.total_bytes != progress.total_bytes
            || existing.progress.dest_path != progress.dest_path
            || existing.progress.partial_path != progress.partial_path;
        existing.progress = progress;
        if let Some(token) = cancel_token {
            existing.cancel_token = token;
        }
        return persist;
    }

    if let Some(token) = cancel_token {
        s.active_downloads.insert(
            progress.id.clone(),
            crate::state::ActiveDownload {
                progress,
                cancel_token: token,
            },
        );
        return true;
    }
    false
}

async fn persist_downloads(state: &SharedState) {
    let persist_mutex = {
        let s = state.read().await;
        s.download_persist_mutex.clone()
    };
    let _persist_guard = persist_mutex.lock().await;
    let manifest = {
        let s = state.read().await;
        let mut jobs = s
            .active_downloads
            .values()
            .map(|entry| entry.progress.clone())
            .collect::<Vec<_>>();
        jobs.sort_by(|left, right| left.id.cmp(&right.id));
        DownloadManifest {
            version: DOWNLOAD_MANIFEST_VERSION,
            jobs,
        }
    };
    let path = download_manifest_path();
    if let Err(error) =
        tokio::task::spawn_blocking(move || write_download_manifest_atomic_at(&path, &manifest))
            .await
            .unwrap_or_else(|error| Err(format!("Download manifest writer failed: {error}")))
    {
        tracing::warn!(error = %error, "Failed to persist download manifest");
    }
}

async fn emit_download_progress(
    app: &tauri::AppHandle,
    state: &SharedState,
    progress: DownloadProgress,
    cancel_token: Option<CancellationToken>,
) {
    let persist = upsert_download(state, progress.clone(), cancel_token).await;
    if persist {
        persist_downloads(state).await;
    }
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
        let scan_dirs = s.config.models.scan_dirs.clone();
        let entry = s
            .active_downloads
            .get_mut(&id)
            .ok_or_else(|| "Download not found.".to_string())?;
        if entry.progress.done && !(entry.progress.status == "Failed" && entry.progress.resumable) {
            return Ok(());
        }
        if entry.progress.status == "Paused"
            || entry.progress.status == "Interrupted"
            || entry.progress.status == DOWNLOAD_STATUS_CLEANUP_PENDING
            || (entry.progress.status == "Failed" && entry.progress.resumable)
        {
            let previous = entry.progress.clone();
            let partial = validated_persisted_destination(&entry.progress, &scan_dirs)
                .map(|(_, partial)| partial)
                .ok_or_else(|| {
                    "Refusing to discard a partial outside configured model directories."
                        .to_string()
                })?;
            entry.cancel_token.cancel();
            entry.progress.done = false;
            entry.progress.status = "Cancelling".to_string();
            (entry.progress.clone(), Some((partial, previous)))
        } else {
            entry.cancel_token.cancel();
            entry.progress.status = "Cancelling".to_string();
            (entry.progress.clone(), None)
        }
    };

    let _ = app.emit("model-download-progress", progress.clone());
    persist_downloads(state.inner()).await;

    if let Some((path, previous)) = paused_path {
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            if let Err(error) = tokio::fs::remove_file(&path).await {
                tracing::warn!(path = %path.display(), error = %error, "Failed to discard cancelled partial download");
                let failed = {
                    let mut s = state.write().await;
                    let entry = s
                        .active_downloads
                        .get_mut(&id)
                        .ok_or_else(|| "Download not found.".to_string())?;
                    entry.progress = previous;
                    entry.progress.done = false;
                    entry.progress.status = DOWNLOAD_STATUS_CLEANUP_PENDING.to_string();
                    entry.progress.resumable = false;
                    entry.progress.error = Some(format!(
                        "Could not discard partial {}: {error}",
                        path.display()
                    ));
                    entry.progress.clone()
                };
                let _ = app.emit("model-download-progress", failed);
                persist_downloads(state.inner()).await;
                return Err(format!(
                    "Could not discard partial {}: {error}",
                    path.display()
                ));
            }
        }
        let cancelled = {
            let mut s = state.write().await;
            let entry = s
                .active_downloads
                .get_mut(&id)
                .ok_or_else(|| "Download not found.".to_string())?;
            entry.progress.dest_path = None;
            entry.progress.partial_path = None;
            entry.progress.downloaded_bytes = 0;
            entry.progress.percent = 0.0;
            entry.progress.speed_bps = None;
            entry.progress.eta_seconds = None;
            entry.progress.resumable = false;
            entry.progress.done = true;
            entry.progress.status = "Cancelled".to_string();
            entry.progress.error = None;
            entry.progress.clone()
        };
        persist_downloads(state.inner()).await;
        let _ = app.emit("model-download-progress", cancelled);
    }
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
        if entry.progress.done
            || entry.progress.status == "Paused"
            || entry.progress.status == DOWNLOAD_STATUS_CLEANUP_PENDING
        {
            return Ok(());
        }
        if entry.progress.status == "Interrupted" {
            entry.progress.status = "Paused".to_string();
            entry.progress.error = None;
            entry.progress.clone()
        } else {
            entry.cancel_token.cancel();
            entry.progress.status = "Pausing".to_string();
            entry.progress.clone()
        }
    };

    let _ = app.emit("model-download-progress", progress);
    persist_downloads(state.inner()).await;
    Ok(())
}

#[tauri::command]
pub async fn clear_completed_downloads(state: tauri::State<'_, SharedState>) -> Result<(), String> {
    {
        let mut s = state.write().await;
        s.active_downloads
            .retain(|_, entry| retain_download_after_clear(&entry.progress));
    }
    persist_downloads(state.inner()).await;
    Ok(())
}

fn retain_download_after_clear(progress: &DownloadProgress) -> bool {
    !progress.done
        || (progress.status == "Failed" && progress.resumable)
        || progress.status == DOWNLOAD_STATUS_CLEANUP_PENDING
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
    download_hub_model_inner(
        app,
        state.inner().clone(),
        url,
        filename,
        supports_vision,
        repo_id,
        false,
    )
    .await
}

async fn download_hub_model_inner(
    app: tauri::AppHandle,
    state: SharedState,
    url: String,
    filename: String,
    supports_vision: Option<bool>,
    repo_id: Option<String>,
    startup_resume: bool,
) -> Result<String, String> {
    if !trusted_hugging_face_download_url(&url) {
        return Err("Only HTTPS downloads from huggingface.co are supported.".to_string());
    }
    let relative_path = sanitize_download_subpath(&filename)?;
    let download_id = url.clone();
    let (scan_dirs, hf_api_key, existing_progress): (
        Vec<PathBuf>,
        Option<String>,
        Option<DownloadProgress>,
    ) = {
        let s = state.read().await;
        let scan_dirs = s
            .config
            .models
            .scan_dirs
            .iter()
            .map(PathBuf::from)
            .collect::<Vec<_>>();
        if scan_dirs.is_empty() {
            return Err(
                "No model directory configured. Add one in Settings > Model Directories."
                    .to_string(),
            );
        }
        (
            scan_dirs,
            s.config.hub.hf_api_key.clone(),
            s.active_downloads
                .get(&download_id)
                .map(|entry| entry.progress.clone()),
        )
    };

    let dest_path = existing_progress
        .as_ref()
        .and_then(|progress| validated_persisted_destination(progress, &scan_dirs))
        .map(|(dest, _)| dest)
        .unwrap_or_else(|| scan_dirs[0].join(&relative_path));
    let partial_path = partial_path_for(&dest_path);
    let destination_key = download_destination_key(&dest_path);
    let owning_scan_dir = scan_dirs
        .iter()
        .find(|scan_dir| {
            let candidate = scan_dir.join(&relative_path);
            download_destination_key(&candidate) == destination_key
                && path_parent_is_within_scan_dir(&candidate, scan_dir)
        })
        .ok_or_else(|| {
            "Refusing to download through a path outside configured model directories.".to_string()
        })?;
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create {}: {e}", parent.display()))?;
    }
    if !path_parent_is_within_scan_dir(&dest_path, owning_scan_dir) {
        return Err(
            "Refusing to download through a junction or symlink outside the model directory."
                .to_string(),
        );
    }
    let cancel_token = CancellationToken::new();
    let dest_path_string = dest_path.to_string_lossy().to_string();
    let partial_path_string = partial_path.to_string_lossy().to_string();
    let resume_from = std::fs::metadata(&partial_path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let prior_total = existing_progress
        .as_ref()
        .map(|progress| progress.total_bytes)
        .unwrap_or(0)
        .max(resume_from);
    let initial_progress = DownloadProgress {
        id: download_id.clone(),
        filename: filename.clone(),
        dest_path: Some(dest_path_string.clone()),
        partial_path: Some(partial_path_string.clone()),
        supports_vision,
        repo_id: repo_id.clone(),
        downloaded_bytes: resume_from,
        total_bytes: prior_total,
        percent: progress_percent(resume_from, prior_total),
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
    };

    // Claim both the source id and destination under one lock. This prevents a
    // startup auto-resume and a user double-click from creating two writers for
    // the same partial file.
    {
        let mut s = state.write().await;
        if startup_resume
            && !s
                .active_downloads
                .get(&download_id)
                .is_some_and(|entry| entry.progress.status == "Interrupted")
        {
            return Err("Interrupted download state changed before auto-resume.".to_string());
        }
        if let Some(existing) = s.active_downloads.get(&download_id) {
            if !existing.progress.done
                && existing.progress.status != "Paused"
                && existing.progress.status != "Interrupted"
            {
                return Err(format!(
                    "Download already in progress for {}",
                    existing.progress.filename
                ));
            }
        }
        if let Some(conflict) = s.active_downloads.values().find(|entry| {
            entry.progress.id != download_id
                && (!entry.progress.done
                    || (entry.progress.resumable
                        && entry
                            .progress
                            .partial_path
                            .as_deref()
                            .is_some_and(|path| Path::new(path).exists())))
                && entry.progress.dest_path.as_deref().is_some_and(|path| {
                    download_destination_key(Path::new(path)) == destination_key
                })
        }) {
            return Err(format!(
                "Another download ({}) already owns destination {}",
                conflict.progress.filename,
                dest_path.display()
            ));
        }
        s.active_downloads.insert(
            download_id.clone(),
            crate::state::ActiveDownload {
                progress: initial_progress.clone(),
                cancel_token: cancel_token.clone(),
            },
        );
    }
    persist_downloads(&state).await;
    let _ = app.emit("model-download-progress", initial_progress);

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
        let mut reset_partial_once = false;

        for attempt in 1..=5u32 {
            let disk_resume = tokio::fs::metadata(&partial_path)
                .await
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            let mut request = client
                .get(&url)
                .header(reqwest::header::ACCEPT, "application/octet-stream")
                .header(reqwest::header::ACCEPT_ENCODING, "identity");
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

            let status = resp.status();
            let content_range = response_content_range(&resp);

            if disk_resume > 0 && status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                if matches!(
                    content_range,
                    Some(ParsedContentRange::Unsatisfied { total }) if total == disk_resume
                ) {
                    downloaded = disk_resume;
                    total_bytes = disk_resume;
                    completed = true;
                    break;
                }
                let reason = format!(
                    "Server rejected resume offset {disk_resume} with HTTP 416 ({content_range:?})"
                );
                if reset_partial_once {
                    last_error = Some(reason);
                    break;
                }
                tokio::fs::remove_file(&partial_path).await.map_err(|error| {
                    format!(
                        "Cannot reset incompatible partial {}: {error}",
                        partial_path.display()
                    )
                })?;
                reset_partial_once = true;
                last_error = Some(reason);
                continue;
            }

            if status == reqwest::StatusCode::PARTIAL_CONTENT {
                let valid_range = content_range_matches_resume(
                    content_range,
                    disk_resume,
                    resp.content_length(),
                );
                if !valid_range {
                    let reason = format!(
                        "Server returned an invalid Content-Range for resume offset {disk_resume}: {content_range:?}"
                    );
                    if disk_resume == 0 || reset_partial_once {
                        last_error = Some(reason);
                        break;
                    }
                    tokio::fs::remove_file(&partial_path).await.map_err(|error| {
                        format!(
                            "Cannot reset incompatible partial {}: {error}",
                            partial_path.display()
                        )
                    })?;
                    reset_partial_once = true;
                    last_error = Some(reason);
                    continue;
                }
            }

            if disk_resume > 0 && status == reqwest::StatusCode::OK {
                tracing::warn!(
                    url = %url,
                    resume_from = disk_resume,
                    "Hugging Face ignored Range resume; restarting partial model download"
                );
                tokio::fs::remove_file(&partial_path).await.map_err(|error| {
                    format!(
                        "Cannot restart partial {} after Range was ignored: {error}",
                        partial_path.display()
                    )
                })?;
                reset_partial_once = true;
            } else if !status.is_success() {
                last_error = Some(format!(
                    "Server returned HTTP {} for {}",
                    status,
                    url
                ));
                tokio::time::sleep(std::time::Duration::from_millis(750 * attempt as u64)).await;
                continue;
            }

            let resumed = disk_resume > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
            downloaded = if resumed { disk_resume } else { 0 };
            total_bytes = match content_range {
                Some(ParsedContentRange::Satisfied {
                    total: Some(total),
                    ..
                }) => total,
                _ => resp
                    .content_length()
                    .map(|length| length + downloaded)
                    .unwrap_or(downloaded),
            };
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
                        &state,
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
            if attempt_failed || (total_bytes > 0 && downloaded != total_bytes) {
                if !attempt_failed {
                    last_error = Some(format!(
                        "Download size mismatch: {} of {} bytes",
                        downloaded, total_bytes
                    ));
                }
                emit_download_progress(
                    &app,
                    &state,
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
                if tokio::fs::try_exists(&partial_path).await.unwrap_or(false) {
                    if let Err(error) = tokio::fs::remove_file(&partial_path).await {
                        tracing::warn!(
                            path = %partial_path.display(),
                            error = %error,
                            "Failed to finish cancelling active model download"
                        );
                        emit_download_progress(
                            &app,
                            &state,
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
                                resumable: false,
                                attempt: 1,
                                done: false,
                                status: DOWNLOAD_STATUS_CLEANUP_PENDING.to_string(),
                                error: Some(format!(
                                    "Could not discard partial {}: {error}. Cleanup will retry on next start.",
                                    partial_path.display()
                                )),
                            },
                            None,
                        )
                        .await;
                        return Err("Download cancelled".to_string());
                    }
                }
            }
            emit_download_progress(
                &app,
                &state,
                DownloadProgress {
                    id: download_id.clone(),
                    filename: filename.clone(),
                    dest_path: paused.then(|| dest_path_string.clone()),
                    partial_path: paused.then(|| partial_path_string.clone()),
                    supports_vision,
                    repo_id: repo_id.clone(),
                    downloaded_bytes: if paused { downloaded } else { 0 },
                    total_bytes,
                    percent: if paused {
                        progress_percent(downloaded, total_bytes)
                    } else {
                        0.0
                    },
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

        if total_bytes > 0 && downloaded != total_bytes {
            return Err(last_error.unwrap_or_else(|| {
                format!(
                    "Download size mismatch: {} of {} bytes",
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
            &state,
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
            let downloaded_bytes = tokio::fs::metadata(&partial_path)
                .await
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            let total_bytes = {
                let s = state.read().await;
                s.active_downloads
                    .get(&download_id)
                    .map(|entry| entry.progress.total_bytes)
                    .unwrap_or(0)
                    .max(downloaded_bytes)
            };
            emit_download_progress(
                &app,
                &state,
                DownloadProgress {
                    id: download_id,
                    filename,
                    dest_path: Some(dest_path_string),
                    partial_path: Some(partial_path_string),
                    supports_vision,
                    repo_id,
                    downloaded_bytes,
                    total_bytes,
                    percent: progress_percent(downloaded_bytes, total_bytes),
                    speed_bps: None,
                    eta_seconds: None,
                    resumable: downloaded_bytes > 0,
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

/// Resume transfers that were running when the previous GUI process stopped.
/// Explicitly paused and failed transfers remain visible for manual action.
pub(crate) fn resume_interrupted_downloads(app: tauri::AppHandle, state: SharedState) {
    tauri::async_runtime::spawn(async move {
        // Commit reconciled byte counts and terminal cancellation cleanup before
        // any resumed transfer mutates the registry again.
        persist_downloads(&state).await;
        let jobs = {
            let s = state.read().await;
            s.active_downloads
                .values()
                .filter(|entry| entry.progress.status == "Interrupted")
                .map(|entry| entry.progress.clone())
                .collect::<Vec<_>>()
        };

        if !jobs.is_empty() {
            tracing::info!(count = jobs.len(), "Resuming interrupted model downloads");
        }
        for job in jobs {
            let app = app.clone();
            let state = state.clone();
            tauri::async_runtime::spawn(async move {
                let id = job.id.clone();
                let filename = job.filename.clone();
                if let Err(error) = download_hub_model_inner(
                    app,
                    state,
                    job.id,
                    job.filename,
                    job.supports_vision,
                    job.repo_id,
                    true,
                )
                .await
                {
                    tracing::warn!(
                        id = %id,
                        filename = %filename,
                        error = %error,
                        "Interrupted model download did not resume"
                    );
                }
            });
        }
    });
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
        build_hf_sync_queries, find_repo_chat_template_path, format_parameter_count, hf_api_to_hub,
        hf_supports_vision, is_hf_downloadable, load_persisted_downloads_at, manifest_sibling,
        parse_content_range, partial_path_for, path_is_inside_any_dir, retain_download_after_clear,
        sanitize_download_subpath, write_download_manifest_atomic_at, DownloadManifest,
        DownloadProgress, HfApiModel, HfSibling, ParsedContentRange, DOWNLOAD_MANIFEST_VERSION,
    };
    use std::path::{Path, PathBuf};

    fn temp_dir(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("inference-bridge-{name}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn download_fixture(dest: &Path, status: &str) -> DownloadProgress {
        DownloadProgress {
            id: "https://huggingface.co/owner/repo/resolve/main/model.gguf".to_string(),
            filename: "model.gguf".to_string(),
            dest_path: Some(dest.to_string_lossy().to_string()),
            partial_path: Some(partial_path_for(dest).to_string_lossy().to_string()),
            supports_vision: Some(false),
            repo_id: Some("owner/repo".to_string()),
            downloaded_bytes: 0,
            total_bytes: 1_000,
            percent: 0.0,
            speed_bps: None,
            eta_seconds: None,
            resumable: false,
            attempt: 1,
            done: false,
            status: status.to_string(),
            error: None,
        }
    }

    fn write_manifest(path: &Path, jobs: Vec<DownloadProgress>) {
        write_download_manifest_atomic_at(
            path,
            &DownloadManifest {
                version: DOWNLOAD_MANIFEST_VERSION,
                jobs,
            },
        )
        .unwrap();
    }

    #[test]
    fn interrupted_download_restores_from_actual_partial_length() {
        let root = temp_dir("download-restore");
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let dest = models.join("model.gguf");
        let partial = partial_path_for(&dest);
        std::fs::File::create(&partial)
            .unwrap()
            .set_len(321)
            .unwrap();
        let manifest = root.join("download-jobs.json");
        write_manifest(&manifest, vec![download_fixture(&dest, "Downloading")]);

        let restored = load_persisted_downloads_at(&manifest, &[models]);
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].status, "Interrupted");
        assert!(!restored[0].done);
        assert!(restored[0].resumable);
        assert_eq!(restored[0].downloaded_bytes, 321);
        assert_eq!(restored[0].total_bytes, 1_000);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn explicit_pause_stays_paused_after_restore() {
        let root = temp_dir("download-paused");
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let dest = models.join("model.gguf");
        std::fs::File::create(partial_path_for(&dest))
            .unwrap()
            .set_len(100)
            .unwrap();
        let manifest = root.join("download-jobs.json");
        write_manifest(&manifest, vec![download_fixture(&dest, "Paused")]);

        let restored = load_persisted_downloads_at(&manifest, &[models]);
        assert_eq!(restored[0].status, "Paused");
        assert!(!restored[0].done);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn cancelled_restore_deletes_only_partial_and_never_relabels_existing_dest_completed() {
        let root = temp_dir("download-cancelled");
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let dest = models.join("model.gguf");
        std::fs::write(&dest, b"existing model").unwrap();
        let partial = partial_path_for(&dest);
        std::fs::write(&partial, b"partial").unwrap();
        let manifest = root.join("download-jobs.json");
        write_manifest(&manifest, vec![download_fixture(&dest, "Cancelling")]);

        let restored = load_persisted_downloads_at(&manifest, &[models]);
        assert_eq!(restored[0].status, "Cancelled");
        assert!(restored[0].done);
        assert!(!restored[0].resumable);
        assert!(restored[0].dest_path.is_none());
        assert!(restored[0].partial_path.is_none());
        assert!(dest.exists(), "completed destination must be preserved");
        assert!(!partial.exists(), "trusted partial should be discarded");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn restore_rejects_destination_outside_configured_scan_dirs() {
        let root = temp_dir("download-path-guard");
        let models = root.join("models");
        let outside = root.join("outside");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let dest = outside.join("model.gguf");
        std::fs::write(partial_path_for(&dest), b"partial").unwrap();
        let manifest = root.join("download-jobs.json");
        write_manifest(&manifest, vec![download_fixture(&dest, "Downloading")]);

        assert!(load_persisted_downloads_at(&manifest, &[models]).is_empty());
        assert!(
            partial_path_for(&dest).exists(),
            "untrusted file is untouched"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn restore_rejects_nested_link_that_escapes_scan_directory() {
        let root = temp_dir("download-link-guard");
        let models = root.join("models");
        let outside = root.join("outside");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let link = models.join("linked");
        #[cfg(target_os = "windows")]
        let linked = std::os::windows::fs::symlink_dir(&outside, &link).is_ok();
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&outside, &link).is_ok();
        if !linked {
            let _ = std::fs::remove_dir_all(root);
            return;
        }

        let dest = link.join("model.gguf");
        std::fs::write(partial_path_for(&dest), b"partial").unwrap();
        let manifest = root.join("download-jobs.json");
        let mut job = download_fixture(&dest, "Downloading");
        job.filename = "linked/model.gguf".to_string();
        write_manifest(&manifest, vec![job]);

        assert!(load_persisted_downloads_at(&manifest, &[models]).is_empty());
        assert!(
            partial_path_for(&dest).exists(),
            "escaped file is untouched"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn restore_recovers_valid_next_manifest_after_first_write_crash() {
        let root = temp_dir("download-next");
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let dest = models.join("model.gguf");
        std::fs::write(partial_path_for(&dest), b"partial").unwrap();
        let manifest = root.join("download-jobs.json");
        let next = manifest_sibling(&manifest, ".next");
        let contents = serde_json::to_vec_pretty(&DownloadManifest {
            version: DOWNLOAD_MANIFEST_VERSION,
            jobs: vec![download_fixture(&dest, "Downloading")],
        })
        .unwrap();
        std::fs::write(next, contents).unwrap();

        let restored = load_persisted_downloads_at(&manifest, &[models]);
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].status, "Interrupted");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn restore_disables_conflicting_owners_of_one_partial() {
        let root = temp_dir("download-conflict");
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let dest = models.join("model.gguf");
        std::fs::write(partial_path_for(&dest), b"partial").unwrap();
        let manifest = root.join("download-jobs.json");
        let first = download_fixture(&dest, "Downloading");
        let mut second = download_fixture(&dest, "Paused");
        second.id = "https://huggingface.co/other/repo/resolve/main/model.gguf".to_string();
        second.repo_id = Some("other/repo".to_string());
        write_manifest(&manifest, vec![first, second]);

        let restored = load_persisted_downloads_at(&manifest, &[models]);
        assert_eq!(restored.len(), 2);
        assert!(restored
            .iter()
            .all(|job| job.status == "Failed" && job.done));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn parses_and_validates_http_content_ranges() {
        assert_eq!(
            parse_content_range("bytes 321-999/1000"),
            Some(ParsedContentRange::Satisfied {
                start: 321,
                end: 999,
                total: Some(1000)
            })
        );
        assert_eq!(
            parse_content_range("bytes */1000"),
            Some(ParsedContentRange::Unsatisfied { total: 1000 })
        );
        assert!(parse_content_range("bytes 999-321/1000").is_none());
        assert!(!super::content_range_matches_resume(
            parse_content_range("bytes 0-99/50"),
            0,
            Some(100)
        ));
        assert!(!super::content_range_matches_resume(
            parse_content_range("bytes 320-999/1000"),
            321,
            Some(680)
        ));
        assert!(super::content_range_matches_resume(
            parse_content_range("bytes 321-999/1000"),
            321,
            Some(679)
        ));
    }

    #[test]
    fn clear_done_retains_failed_resumable_metadata() {
        let root = temp_dir("download-clear");
        let mut failed = download_fixture(&root.join("model.gguf"), "Failed");
        failed.done = true;
        failed.resumable = true;
        assert!(retain_download_after_clear(&failed));
        failed.resumable = false;
        assert!(!retain_download_after_clear(&failed));
        failed.status = "Cleanup pending".to_string();
        assert!(retain_download_after_clear(&failed));
        let _ = std::fs::remove_dir_all(root);
    }

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
    fn maps_huihui_hugging_face_metadata_into_hub_model() {
        let api_model: HfApiModel = serde_json::from_value(serde_json::json!({
            "id": "huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF",
            "modelId": "huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF",
            "author": "huihui-ai",
            "createdAt": "2026-05-19T05:52:43.000Z",
            "lastModified": "2026-05-19T06:04:10.000Z",
            "library_name": "transformers",
            "pipeline_tag": "image-text-to-text",
            "downloads": 95945,
            "likes": 76,
            "private": false,
            "disabled": false,
            "gated": false,
            "gguf": {
                "total": 27320697856_u64,
                "architecture": "qwen35",
                "context_length": 262144
            },
            "tags": [
                "transformers",
                "gguf",
                "abliterated",
                "uncensored",
                "GGUF",
                "MTP",
                "image-text-to-text",
                "base_model:Qwen/Qwen3.6-27B",
                "license:apache-2.0",
                "endpoints_compatible",
                "region:us",
                "conversational",
                "qwen35",
                "vision",
                "tool-use",
                "reasoning",
                "extra"
            ],
            "siblings": [
                {
                    "rfilename": "Huihui-Qwen3.6-27B-abliterated-Q4_K_M.gguf",
                    "size": 17740000000_u64
                },
                {
                    "rfilename": "mmproj-model-f16.gguf",
                    "size": 900000000_u64
                }
            ]
        }))
        .expect("fixture should deserialize");

        let hub = hf_api_to_hub(api_model, false, Some("# Model card".to_string()))
            .expect("public GGUF fixture should map");

        assert_eq!(hub.author.as_deref(), Some("huihui-ai"));
        assert_eq!(hub.created_at.as_deref(), Some("2026-05-19T05:52:43.000Z"));
        assert_eq!(hub.library_name.as_deref(), Some("transformers"));
        assert_eq!(hub.gguf_total, Some(27_320_697_856));
        assert_eq!(hub.gguf_architecture.as_deref(), Some("qwen35"));
        assert_eq!(hub.gguf_context_length, Some(262_144));
        assert_eq!(hub.params, "27.3B");
        assert!(hub.supports_vision);
        assert_eq!(hub.quants.len(), 1, "mmproj files are not model quants");
        assert_eq!(hub.tags.len(), 12);
        assert_eq!(
            hub.tags
                .iter()
                .filter(|tag| tag.eq_ignore_ascii_case("gguf"))
                .count(),
            1,
            "case variants of GGUF should be deduplicated"
        );
        assert!(hub.tags.iter().any(|tag| tag == "reasoning"));
        assert!(!hub.tags.iter().any(|tag| tag.contains(':')));
    }

    #[test]
    fn compacts_gguf_parameter_counts_in_billions_or_millions() {
        assert_eq!(format_parameter_count(27_320_697_856), "27.3B");
        assert_eq!(format_parameter_count(7_500_000_000), "7.5B");
        assert_eq!(format_parameter_count(355_000_000), "355M");
    }

    #[test]
    fn public_mode_rejects_gated_models() {
        let model = HfApiModel {
            model_id: "owner/model".to_string(),
            author: None,
            downloads: 0,
            created_at: None,
            pipeline_tag: None,
            library_name: None,
            tags: vec![],
            gguf: None,
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
            author: None,
            downloads: 0,
            created_at: None,
            pipeline_tag: None,
            library_name: None,
            tags: vec![],
            gguf: None,
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
            author: None,
            downloads: 0,
            created_at: None,
            pipeline_tag: None,
            library_name: None,
            tags: vec![],
            gguf: None,
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
            author: None,
            downloads: 0,
            created_at: None,
            pipeline_tag: Some("image-text-to-text".to_string()),
            library_name: None,
            tags: vec!["gguf".to_string()],
            gguf: None,
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
            author: None,
            downloads: 0,
            created_at: None,
            pipeline_tag: None,
            library_name: None,
            tags: vec![
                "gguf".to_string(),
                "vision".to_string(),
                "multimodal".to_string(),
            ],
            gguf: None,
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
            author: None,
            downloads: 0,
            likes: 0,
            created_at: None,
            last_modified: None,
            pipeline_tag: None,
            library_name: None,
            tags: vec![],
            gguf: None,
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
            author: None,
            downloads: 0,
            likes: 0,
            created_at: None,
            last_modified: None,
            pipeline_tag: None,
            library_name: None,
            tags: vec![],
            gguf: None,
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
