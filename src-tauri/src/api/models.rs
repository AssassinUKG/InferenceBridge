use crate::state::{LoadProgress, ModelLoadState, ModelStats, SharedState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::Serialize;
use tokio::task;

#[derive(serde::Deserialize)]
pub struct LoadModelRequest {
    pub model: String,
    /// Optional context size override. Uses model profile default when omitted.
    pub context_size: Option<u32>,
}

#[derive(serde::Deserialize)]
pub struct ModelStatsRequest {
    pub model: Option<String>,
}

#[derive(serde::Serialize)]
pub struct LoadModelResponse {
    pub status: String,
    pub progress: Option<LoadProgress>,
    pub model_info: Option<ModelStats>,
}

pub async fn load_model(
    State(state): State<SharedState>,
    Json(req): Json<LoadModelRequest>,
) -> Json<LoadModelResponse> {
    {
        let mut s = state.write().await;
        s.model_load_state = ModelLoadState::Loading;
        s.model_load_progress = Some(LoadProgress {
            stage: "starting".to_string(),
            message: format!("Loading model {}...", req.model),
            progress: 0.0,
            done: false,
            error: None,
        });
    }

    let state_clone = state.clone();
    let model_name = req.model.clone();
    let context_size = req.context_size;
    task::spawn(async move {
        use crate::commands::model::backend_load_model;
        use crate::context::tracker::reset_slots_warning;
        let result =
            backend_load_model(state_clone.clone(), model_name.clone(), context_size).await;
        let mut s = state_clone.write().await;
        match result {
            Ok(_msg) => {
                s.model_load_state = ModelLoadState::Loaded;
                s.model_load_progress = Some(LoadProgress {
                    stage: "ready".to_string(),
                    message: format!("Model {} loaded.", model_name),
                    progress: 1.0,
                    done: true,
                    error: None,
                });
                reset_slots_warning();
            }
            Err(e) => {
                s.model_load_state = ModelLoadState::Error(e.clone());
                s.model_load_progress = Some(LoadProgress {
                    stage: "error".to_string(),
                    message: format!("Failed to load model {}: {}", model_name, e),
                    progress: 0.0,
                    done: true,
                    error: Some(e),
                });
            }
        }
    });

    let s = state.read().await;
    Json(LoadModelResponse {
        status: "loading".to_string(),
        progress: s.model_load_progress.clone(),
        model_info: s.model_stats.clone(),
    })
}

pub async fn unload_model(State(state): State<SharedState>) -> Json<serde_json::Value> {
    match crate::commands::model::backend_unload_model(state).await {
        Ok(message) => Json(serde_json::json!({ "status": "unloaded", "message": message })),
        Err(error) => Json(serde_json::json!({ "status": "error", "error": error })),
    }
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Serialize, Clone)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub active: bool,
}

#[derive(Serialize, Clone)]
pub struct ModelDetailResponse {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub active: bool,
    pub path: String,
    pub size_bytes: u64,
    pub size_gb: f64,
    pub family: String,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub supports_vision: bool,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub quant: Option<String>,
    pub tool_call_format: String,
    pub think_tag_style: String,
}

#[derive(Serialize)]
pub struct ModelStatsResponse {
    pub requested_model: Option<String>,
    pub active_model: Option<String>,
    pub matches_active_model: bool,
    pub state: ModelLoadState,
    pub progress: Option<LoadProgress>,
    pub stats: Option<ModelStats>,
    pub model: Option<ModelDetailResponse>,
}

pub async fn list_models(State(state): State<SharedState>) -> Json<ModelsResponse> {
    let snapshot = get_or_scan_models(&state).await;
    let loaded = snapshot.loaded_model.as_deref();

    let data: Vec<ModelObject> = snapshot
        .models
        .iter()
        .map(|model| model_object_from_scanned(model, loaded))
        .collect();

    Json(ModelsResponse {
        object: "list".to_string(),
        data,
    })
}

pub async fn get_model(
    State(state): State<SharedState>,
    Path(model_name): Path<String>,
) -> Result<Json<ModelDetailResponse>, StatusCode> {
    let snapshot = get_or_scan_models(&state).await;
    let loaded = snapshot.loaded_model.as_deref();
    let Some(model) = find_model_in_snapshot(&snapshot.models, &model_name) else {
        return Err(StatusCode::NOT_FOUND);
    };

    Ok(Json(model_detail_from_scanned(model, loaded)))
}

pub async fn model_stats(
    State(state): State<SharedState>,
    Json(req): Json<ModelStatsRequest>,
) -> Result<Json<ModelStatsResponse>, StatusCode> {
    model_stats_inner(state, req.model).await
}

pub async fn current_model_stats(
    State(state): State<SharedState>,
) -> Result<Json<ModelStatsResponse>, StatusCode> {
    model_stats_inner(state, None).await
}

async fn model_stats_inner(
    state: SharedState,
    requested_model: Option<String>,
) -> Result<Json<ModelStatsResponse>, StatusCode> {
    let snapshot = get_or_scan_models(&state).await;
    let active_model = snapshot.loaded_model.clone();
    let requested_model = requested_model
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| active_model.clone());

    let model = requested_model
        .as_ref()
        .and_then(|name| find_model_in_snapshot(&snapshot.models, name))
        .cloned();

    if requested_model.is_some() && model.is_none() {
        return Err(StatusCode::NOT_FOUND);
    }

    let (state_value, progress, stats, matches_active_model) = {
        let s = state.read().await;
        let matches_active = requested_model
            .as_deref()
            .and_then(|name| {
                model
                    .as_ref()
                    .map(|model| model_matches_runtime(name, &model.filename, &s))
            })
            .unwrap_or(false);

        if requested_model.is_none() || matches_active {
            (
                s.model_load_state.clone(),
                s.model_load_progress.clone(),
                s.model_stats.clone(),
                matches_active,
            )
        } else {
            (ModelLoadState::Idle, None, None, false)
        }
    };

    Ok(Json(ModelStatsResponse {
        requested_model: requested_model.clone(),
        active_model: active_model.clone(),
        matches_active_model,
        state: state_value,
        progress,
        stats,
        model: model.map(|model| model_detail_from_scanned(&model, active_model.as_deref())),
    }))
}

#[derive(Clone)]
struct ModelRegistrySnapshot {
    models: Vec<crate::models::scanner::ScannedModel>,
    loaded_model: Option<String>,
}

async fn get_or_scan_models(state: &SharedState) -> ModelRegistrySnapshot {
    {
        let s = state.read().await;
        if !s.model_registry.list().is_empty() {
            return ModelRegistrySnapshot {
                models: s.model_registry.list().to_vec(),
                loaded_model: s.loaded_model.clone(),
            };
        }
    }

    let scan_dirs = {
        let s = state.read().await;
        s.config.models.scan_dirs.clone()
    };
    let scanned = task::spawn_blocking(move || crate::models::scanner::scan_all(&scan_dirs))
        .await
        .unwrap_or_default();
    let count = scanned.len();

    let mut s = state.write().await;
    if s.model_registry.list().is_empty() {
        s.model_registry.update(scanned.clone());
        tracing::info!(count, "API model listing triggered registry auto-scan");
    }

    ModelRegistrySnapshot {
        models: s.model_registry.list().to_vec(),
        loaded_model: s.loaded_model.clone(),
    }
}

fn find_model_in_snapshot<'a>(
    models: &'a [crate::models::scanner::ScannedModel],
    name: &str,
) -> Option<&'a crate::models::scanner::ScannedModel> {
    let lower = name.to_lowercase();

    if let Some(model) = models.iter().find(|model| {
        let filename = model.filename.to_lowercase();
        filename == lower || filename.trim_end_matches(".gguf") == lower
    }) {
        return Some(model);
    }

    models
        .iter()
        .find(|model| model.filename.to_lowercase().contains(&lower))
}

fn model_matches_runtime(requested: &str, canonical: &str, state: &crate::state::AppState) -> bool {
    let requested_matches_loaded = state
        .loaded_model
        .as_deref()
        .map(|loaded| names_match(loaded, requested) || names_match(loaded, canonical))
        .unwrap_or(false);

    if requested_matches_loaded {
        return true;
    }

    let progress_mentions_model = state.model_load_progress.as_ref().map(|progress| {
        let message = progress.message.to_lowercase();
        let requested = requested.to_lowercase();
        let canonical = canonical.to_lowercase();
        message.contains(&requested) || message.contains(&canonical)
    });

    if progress_mentions_model.unwrap_or(false) {
        return true;
    }

    state
        .model_stats
        .as_ref()
        .map(|stats| names_match(&stats.model, requested) || names_match(&stats.model, canonical))
        .unwrap_or(false)
}

fn names_match(left: &str, right: &str) -> bool {
    let left = left.to_lowercase();
    let right = right.to_lowercase();
    left == right
        || left.trim_end_matches(".gguf") == right
        || left == right.trim_end_matches(".gguf")
}

fn model_object_from_scanned(
    model: &crate::models::scanner::ScannedModel,
    loaded_model: Option<&str>,
) -> ModelObject {
    ModelObject {
        id: model.filename.clone(),
        object: "model".to_string(),
        owned_by: "local".to_string(),
        active: loaded_model
            .map(|loaded| names_match(loaded, &model.filename))
            .unwrap_or(false),
    }
}

fn model_detail_from_scanned(
    model: &crate::models::scanner::ScannedModel,
    loaded_model: Option<&str>,
) -> ModelDetailResponse {
    use crate::models::profiles::ThinkTagStyle;

    // All current ToolCallFormats (NativeApi, HermesXml, QwenXml) represent tool-capable
    // models. NativeApi is the standard tool_calls field — it IS tool support, not a lack
    // of it. supports_parallel_tools is an extra capability flag on top.
    // (If a future ToolCallFormat::NoTools variant is added, exclude it here.)
    let supports_tools = true;
    let _ = &model.profile.supports_parallel_tools; // kept for future per-model flags
    let supports_reasoning = !matches!(model.profile.think_tag_style, ThinkTagStyle::None);

    ModelDetailResponse {
        id: model.filename.clone(),
        object: "model".to_string(),
        owned_by: "local".to_string(),
        active: loaded_model
            .map(|loaded| names_match(loaded, &model.filename))
            .unwrap_or(false),
        path: model.path.to_string_lossy().to_string(),
        size_bytes: model.size_bytes,
        size_gb: model.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        family: model.profile.family.to_string(),
        supports_tools,
        supports_reasoning,
        supports_vision: model_supports_vision(&model.filename),
        context_window: model.profile.default_context_window,
        max_output_tokens: model.profile.default_max_output_tokens,
        quant: extract_quant(&model.filename),
        tool_call_format: format!("{:?}", model.profile.tool_call_format),
        think_tag_style: format!("{:?}", model.profile.think_tag_style),
    }
}

fn extract_quant(filename: &str) -> Option<String> {
    let upper = filename.to_uppercase();
    let re = regex::Regex::new(
        r"[_.-]((?:I?Q\d+_[A-Z0-9]+(?:_[A-Z]+)?)|F(?:16|32)|BF16)(?:[_.-]|\.GGUF$)",
    )
    .ok()?;
    re.captures(&upper).map(|c| c[1].to_string())
}

fn model_supports_vision(filename: &str) -> bool {
    let name = filename.to_lowercase();
    name.contains("vision")
        || name.contains("llava")
        || name.contains("multimodal")
        || name.contains("qwen2.5-vl")
        || name.contains("-vl")
        || name.contains("_vl")
}
