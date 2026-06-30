use crate::api::errors::{ApiErrorResponse, ApiResult};
use crate::state::{LoadProgress, ModelLoadState, ModelStats, SharedState};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
};
use serde::Serialize;
use std::collections::HashMap;
use tokio::task;

// ── Load / Unload ──────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct LoadModelRequest {
    pub model: String,
    #[serde(
        default,
        alias = "context_size",
        alias = "contextLength",
        alias = "context_length",
        alias = "contextlength",
        alias = "ctx_size",
        alias = "n_ctx",
        alias = "maxContextLength",
        alias = "num_ctx",
        alias = "numCtx"
    )]
    pub context_size: Option<u32>,
    #[serde(default, alias = "hfRepo", alias = "hf_repo", alias = "repo_id")]
    pub hf_repo: Option<String>,
    #[serde(default, alias = "hfFile", alias = "hf_file", alias = "file")]
    pub hf_file: Option<String>,
    #[serde(default, alias = "fit", alias = "fit_mode")]
    pub fit_mode: Option<String>,
    #[serde(
        default,
        alias = "cacheRam",
        alias = "cache_ram",
        alias = "cache_ram_mb"
    )]
    pub cache_ram_mb: Option<u32>,
    #[serde(default)]
    pub ctxcp: Option<u32>,
    #[serde(default, alias = "jinja", alias = "use_jinja")]
    pub use_jinja: Option<bool>,
    #[serde(default, alias = "reasoning", alias = "reasoning_mode")]
    pub reasoning_mode: Option<String>,
    #[serde(default, alias = "reasoningPreserve", alias = "reasoning_preserve")]
    pub reasoning_preserve: Option<bool>,
    #[serde(default, alias = "templateMode", alias = "template_mode")]
    pub template_mode: Option<String>,
    #[serde(default, alias = "templateName", alias = "template_name")]
    pub template_name: Option<String>,
    #[serde(default, alias = "customTemplatePath", alias = "custom_template_path")]
    pub custom_template_path: Option<String>,
    #[serde(
        default,
        alias = "chatTemplateKwargs",
        alias = "chat_template_kwargs",
        alias = "chat_template_kwargs_json"
    )]
    pub chat_template_kwargs_json: Option<String>,
    #[serde(
        default,
        alias = "draftModelPath",
        alias = "draft_model",
        alias = "draftModel",
        alias = "spec_draft_model",
        alias = "specDraftModel"
    )]
    pub draft_model_path: Option<String>,
    #[serde(default, alias = "specType", alias = "spec_type")]
    pub spec_type: Option<String>,
    #[serde(
        default,
        alias = "specDraftNMax",
        alias = "spec_draft_n_max",
        alias = "spec_draft_tokens",
        alias = "draftNMax"
    )]
    pub spec_draft_n_max: Option<u32>,
    #[serde(
        default,
        alias = "draftMaxTokens",
        alias = "draft_max",
        alias = "draftMax"
    )]
    pub draft_max_tokens: Option<u32>,
    #[serde(
        default,
        alias = "draftMinTokens",
        alias = "draft_min",
        alias = "draftMin"
    )]
    pub draft_min_tokens: Option<u32>,
    #[serde(default, alias = "draftPMin", alias = "draft_p_min")]
    pub draft_p_min: Option<f32>,
    #[serde(default, alias = "extraArgs", alias = "extra_args")]
    pub extra_args: Option<Vec<String>>,
    #[serde(default)]
    pub echo_load_config: bool,
    #[serde(default, alias = "forceReload", alias = "force_reload")]
    pub force_reload: bool,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl LoadModelRequest {
    fn requested_context_size(&self) -> Option<u32> {
        self.context_size
            .filter(|value| *value > 0)
            .or_else(|| {
                self.extra
                    .get("load_config")
                    .or_else(|| self.extra.get("loadConfig"))
                    .and_then(crate::api::completions::extract_context_size_from_value)
            })
            .or_else(|| crate::api::completions::extract_context_size_from_hash_map(&self.extra))
    }
}

/// LM Studio-compatible load response.
#[derive(serde::Serialize)]
pub struct LoadModelResponse {
    /// Always `"llm"` for GGUF models.
    #[serde(rename = "type")]
    pub model_type: String,
    /// Identifier of the loaded instance (matches the model filename).
    pub instance_id: String,
    /// Time taken to load the model in seconds.
    pub load_time_seconds: f64,
    /// `"loaded"` on success.
    pub status: String,
    /// Echoed load config (only when `echo_load_config: true` in request).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_config: Option<LoadConfig>,
    // InferenceBridge extensions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<LoadProgress>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_info: Option<ModelStats>,
}

#[derive(serde::Serialize)]
pub struct LoadConfig {
    pub context_length: u32,
}

pub async fn load_model(
    State(state): State<SharedState>,
    Json(req): Json<LoadModelRequest>,
) -> ApiResult<Json<LoadModelResponse>> {
    let model_name = req.model.clone();
    let context_size = req.requested_context_size();
    let echo_load_config = req.echo_load_config;

    tracing::info!(
        model = %model_name,
        requested_context_size = context_size,
        "Native model load requested"
    );

    let load_start = std::time::Instant::now();
    crate::commands::model::backend_load_model_with_overrides(
        state.clone(),
        model_name.clone(),
        context_size,
        crate::commands::model::RuntimeLoadOverrides {
            hf_repo: req.hf_repo.clone(),
            hf_file: req.hf_file.clone(),
            fit_mode: req.fit_mode.clone(),
            cache_ram_mb: req.cache_ram_mb,
            ctxcp: req.ctxcp,
            use_jinja: req.use_jinja,
            reasoning_mode: req.reasoning_mode.clone(),
            reasoning_preserve: req.reasoning_preserve,
            template_mode: req.template_mode.clone(),
            template_name: req.template_name.clone(),
            custom_template_path: req.custom_template_path.clone(),
            chat_template_kwargs_json: req.chat_template_kwargs_json.clone(),
            draft_model_path: req.draft_model_path.clone(),
            spec_type: req.spec_type.clone(),
            spec_draft_n_max: req.spec_draft_n_max,
            draft_max_tokens: req.draft_max_tokens,
            draft_min_tokens: req.draft_min_tokens,
            draft_p_min: req.draft_p_min,
            extra_args: req.extra_args.clone(),
            attach_mmproj: None,
            force_reload: req.force_reload,
        },
    )
    .await
    .map_err(|error| {
        ApiErrorResponse::service_unavailable(format!(
            "Failed to load model '{model_name}': {error}"
        ))
    })?;
    let load_time_seconds = load_start.elapsed().as_secs_f64();

    let s = state.read().await;
    let loaded_name = s.loaded_model.clone().unwrap_or_else(|| model_name.clone());
    let actual_ctx = s
        .model_stats
        .as_ref()
        .map(|st| st.context_size)
        .or_else(|| {
            s.last_launch_preview
                .as_ref()
                .and_then(|preview| preview.context_size)
        })
        .or(context_size)
        .unwrap_or(0);

    Ok(Json(LoadModelResponse {
        model_type: "llm".to_string(),
        instance_id: loaded_name,
        load_time_seconds,
        status: "loaded".to_string(),
        load_config: echo_load_config.then(|| LoadConfig {
            context_length: actual_ctx,
        }),
        progress: s.model_load_progress.clone(),
        model_info: s.model_stats.clone(),
    }))
}

#[cfg(test)]
mod tests {
    use super::LoadModelRequest;

    #[test]
    fn deserializes_context_length_alias_for_model_load() {
        let request: LoadModelRequest = serde_json::from_str(
            r#"{
                "model": "Qwen3.5-9B-Q4_K_S.gguf",
                "contextLength": 32768
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.requested_context_size(), Some(32768));
    }

    #[test]
    fn deserializes_force_reload_alias_for_model_load() {
        let request: LoadModelRequest = serde_json::from_str(
            r#"{
                "model": "Qwen3.6-35B.gguf",
                "context_size": 16384,
                "force_reload": true
            }"#,
        )
        .expect("load request should parse force_reload");

        assert!(request.force_reload);
        assert_eq!(request.requested_context_size(), Some(16384));
    }

    #[test]
    fn deserializes_helixclaw_snake_case_context_length() {
        // Exact payload HelixClaw sends: {"context_length":32768,"model":"qwen/qwen3.5-4b"}
        let request: LoadModelRequest = serde_json::from_str(
            r#"{
                "model": "qwen/qwen3.5-4b",
                "context_length": 32768
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(
            request.requested_context_size(),
            Some(32768),
            "context_length should be picked up as context_size"
        );
        // Also verify it's not swallowed by flatten extra
        assert!(
            !request.extra.contains_key("context_length"),
            "context_length should NOT end up in extra (flatten)"
        );
    }

    #[test]
    fn deserializes_draft_mtp_load_options() {
        let request: LoadModelRequest = serde_json::from_str(
            r#"{
                "model": "gemma-4-26B-A4B-it-QAT-Q4_0.gguf",
                "context_size": 49152,
                "draft_model_path": "C:\\models\\gemma-draft.gguf",
                "spec_type": "draft-mtp",
                "spec_draft_n_max": 3
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(
            request.draft_model_path.as_deref(),
            Some("C:\\models\\gemma-draft.gguf")
        );
        assert_eq!(request.spec_type.as_deref(), Some("draft-mtp"));
        assert_eq!(request.spec_draft_n_max, Some(3));
    }

    #[test]
    fn path_style_model_name_find_resolves() {
        // Test that "qwen/qwen3.5-4b" matches "Qwen3.5-4B-Q4_K_M.gguf"
        // This uses the completions helper
        assert!(
            crate::api::completions::loaded_model_matches_request_pub(
                "Qwen3.5-4B-Q4_K_M.gguf",
                "qwen/qwen3.5-4b"
            ),
            "path-style name should match loaded model"
        );
    }

    #[test]
    fn deserializes_nested_context_length_alias_for_model_load() {
        let request: LoadModelRequest = serde_json::from_str(
            r#"{
                "model": "Qwen3.5-9B-Q4_K_S.gguf",
                "load_config": {
                    "context_size": 32768
                }
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.requested_context_size(), Some(32768));
    }
}

/// LM Studio-compatible unload.  Accepts `{"instance_id": "..."}` or
/// `{"model": "..."}` — both are treated as the model identifier.
#[derive(serde::Deserialize, Default)]
pub struct UnloadModelRequest {
    #[serde(default)]
    pub instance_id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

pub async fn unload_model(
    State(state): State<SharedState>,
    body: Option<Json<UnloadModelRequest>>,
) -> Json<serde_json::Value> {
    let instance_id = body.and_then(|b| b.instance_id.clone().or_else(|| b.model.clone()));

    // Fall back to the currently loaded model if no body was sent.
    let instance_id = match instance_id {
        Some(id) => Some(id),
        None => {
            let s = state.read().await;
            s.loaded_model.clone()
        }
    };

    match crate::commands::model::backend_unload_model(state).await {
        Ok(_) => Json(serde_json::json!({
            "instance_id": instance_id,
            "status": "unloaded"
        })),
        Err(error) => Json(serde_json::json!({ "status": "error", "error": error })),
    }
}

// ── OpenAI-compatible /v1/models ────────────────────────────────────────

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

/// OpenAI-spec model object.  Required fields: id, object, created, owned_by.
#[derive(Serialize, Clone)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    // LM Studio / InferenceBridge extensions — always included so HelixClaw
    // and similar clients can discover context window & load state.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    pub reasoning: ReasoningCapability,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_runtime_ready: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mmproj_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_file: Option<String>,
    pub provider_type: String,
    pub provider_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_base_url: Option<String>,
    pub provider_managed: bool,
}

#[derive(Serialize, Clone)]
pub struct ModelDetailResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
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
    pub max_context_length: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub quant: Option<String>,
    pub tool_call_format: String,
    pub think_tag_style: String,
    pub reasoning: ReasoningCapability,
    pub template_mode: Option<String>,
    pub template_source: Option<String>,
    pub vision_runtime_ready: bool,
    pub mmproj_status: String,
    pub vision_block_reason: Option<String>,
    pub hf_repo: Option<String>,
    pub hf_file: Option<String>,
    pub hf_has_repo_template: bool,
}

#[derive(Serialize, Clone)]
pub struct ReasoningCapability {
    pub supported: bool,
    pub separates_content: bool,
    pub effort_values: Vec<String>,
    pub supports_reasoning_tokens: bool,
    pub default_effort: Option<String>,
}

// ── Model stats ─────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct ModelStatsRequest {
    pub model: Option<String>,
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

// ── Handlers ────────────────────────────────────────────────────────────

pub async fn list_models(State(state): State<SharedState>) -> Json<ModelsResponse> {
    if let Some(response) = list_active_upstream_models(&state).await {
        return Json(response);
    }

    let snapshot = get_or_scan_models(&state).await;
    let loaded = snapshot.loaded_model.as_deref();
    let loaded_ctx = snapshot.loaded_context_size;

    let data: Vec<ModelObject> = snapshot
        .models
        .iter()
        .map(|model| {
            model_object_from_scanned(model, loaded, loaded_ctx, snapshot.launch_preview.as_ref())
        })
        .collect();

    Json(ModelsResponse {
        object: "list".to_string(),
        data,
    })
}

#[derive(serde::Deserialize)]
struct UpstreamModelsResponse {
    #[serde(default)]
    data: Vec<UpstreamModelObject>,
}

#[derive(serde::Deserialize)]
struct UpstreamModelObject {
    id: String,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    owned_by: Option<String>,
    #[serde(default)]
    max_context_length: Option<u32>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    context_window: Option<u32>,
}

async fn list_active_upstream_models(state: &SharedState) -> Option<ModelsResponse> {
    let (active, lm_studio, sglang) = {
        let s = state.read().await;
        (
            s.config.providers.active.clone(),
            s.config.providers.lm_studio.clone(),
            s.config.providers.sglang.clone(),
        )
    };

    let (provider_name, provider_type, base_url, api_key) = match active.as_str() {
        "lm_studio" if lm_studio.enabled => (
            "LM Studio",
            "lm_studio",
            lm_studio.base_url,
            lm_studio.api_key,
        ),
        "sglang" if sglang.enabled => ("SGLang", "sglang", sglang.base_url, sglang.api_key),
        _ => return None,
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;
    let mut request = client.get(format!(
        "{}/models",
        crate::providers::normalize_openai_base_url(&base_url)
    ));
    if let Some(api_key) = api_key.filter(|value| !value.trim().is_empty()) {
        request = request.bearer_auth(api_key);
    }
    let response = request.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let upstream: UpstreamModelsResponse = response.json().await.ok()?;
    let now = now_unix_secs();
    let data = upstream
        .data
        .into_iter()
        .map(|model| ModelObject {
            id: model.id,
            object: "model".to_string(),
            created: model.created.unwrap_or(now),
            owned_by: model.owned_by.unwrap_or_else(|| provider_type.to_string()),
            active: true,
            max_context_length: model
                .max_context_length
                .or(model.context_length)
                .or(model.context_window),
            state: Some("external".to_string()),
            reasoning: ReasoningCapability {
                supported: false,
                separates_content: false,
                effort_values: Vec::new(),
                supports_reasoning_tokens: false,
                default_effort: None,
            },
            template_mode: None,
            template_source: Some(provider_type.to_string()),
            vision_runtime_ready: None,
            mmproj_status: None,
            hf_repo: None,
            hf_file: None,
            provider_type: provider_type.to_string(),
            provider_name: provider_name.to_string(),
            provider_base_url: Some(crate::providers::normalize_openai_base_url(&base_url)),
            provider_managed: false,
        })
        .collect();

    Some(ModelsResponse {
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
    let loaded_ctx = snapshot.loaded_context_size;
    let Some(model) = find_model_in_snapshot(&snapshot.models, &model_name) else {
        return Err(StatusCode::NOT_FOUND);
    };

    Ok(Json(model_detail_from_scanned(
        model,
        loaded,
        loaded_ctx,
        snapshot.launch_preview.as_ref(),
    )))
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
    let requested_model = requested_model
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    if requested_model.is_none() {
        let s = state.read().await;
        let active_model = s.loaded_model.clone();
        let matches_active_model = active_model.is_some();
        let (state_value, progress, stats, _) =
            effective_model_stats_state(&s, matches_active_model);
        return Ok(Json(ModelStatsResponse {
            requested_model: active_model.clone(),
            active_model,
            matches_active_model,
            state: state_value,
            progress,
            stats,
            model: None,
        }));
    }

    let snapshot = get_or_scan_models(&state).await;
    let active_model = snapshot.loaded_model.clone();
    let loaded_ctx = snapshot.loaded_context_size;

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
            effective_model_stats_state(&s, matches_active)
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
        model: model.map(|model| {
            model_detail_from_scanned(
                &model,
                active_model.as_deref(),
                loaded_ctx,
                snapshot.launch_preview.as_ref(),
            )
        }),
    }))
}

// ── Snapshot / scanning ─────────────────────────────────────────────────

fn effective_model_stats_state(
    state: &crate::state::AppState,
    matches_active_model: bool,
) -> (
    ModelLoadState,
    Option<LoadProgress>,
    Option<ModelStats>,
    bool,
) {
    let running = matches!(
        state.process.state(),
        crate::engine::process::ProcessState::Running
    );
    let has_loaded_model = state.loaded_model.is_some() || state.model_stats.is_some();
    if running && has_loaded_model && matches_active_model {
        return (
            ModelLoadState::Loaded,
            state
                .model_load_progress
                .as_ref()
                .filter(|progress| progress.done)
                .cloned(),
            state.model_stats.clone(),
            matches_active_model,
        );
    }

    (
        state.model_load_state.clone(),
        state.model_load_progress.clone(),
        state.model_stats.clone(),
        matches_active_model,
    )
}

#[derive(Clone)]
struct ModelRegistrySnapshot {
    models: Vec<crate::models::scanner::ScannedModel>,
    loaded_model: Option<String>,
    loaded_context_size: Option<u32>,
    launch_preview: Option<crate::engine::process::LaunchPreview>,
}

async fn get_or_scan_models(state: &SharedState) -> ModelRegistrySnapshot {
    {
        let s = state.read().await;
        if !s.model_registry.list().is_empty() {
            return ModelRegistrySnapshot {
                models: s.model_registry.list().to_vec(),
                loaded_model: s.loaded_model.clone(),
                loaded_context_size: s
                    .model_stats
                    .as_ref()
                    .map(|st| st.context_size)
                    .filter(|v| *v > 0),
                launch_preview: s.last_launch_preview.clone(),
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
        loaded_context_size: s
            .model_stats
            .as_ref()
            .map(|st| st.context_size)
            .filter(|v| *v > 0),
        launch_preview: s.last_launch_preview.clone(),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

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
    _loaded_ctx: Option<u32>,
    launch_preview: Option<&crate::engine::process::LaunchPreview>,
) -> ModelObject {
    let is_active = loaded_model
        .map(|loaded| names_match(loaded, &model.filename))
        .unwrap_or(false);
    let (vision_runtime_ready, mmproj_status, _vision_block_reason) =
        vision_runtime_info(model, is_active, launch_preview);

    ModelObject {
        id: model.filename.clone(),
        object: "model".to_string(),
        created: now_unix_secs(),
        owned_by: "inference-bridge".to_string(),
        active: is_active,
        max_context_length: model.profile.max_context_window,
        state: Some(if is_active { "loaded" } else { "not-loaded" }.to_string()),
        reasoning: reasoning_capability(&model.profile),
        template_mode: launch_preview
            .filter(|_| is_active)
            .map(|preview| preview.template_mode.clone()),
        template_source: launch_preview
            .filter(|_| is_active)
            .and_then(|preview| preview.template_source.clone()),
        vision_runtime_ready: Some(vision_runtime_ready),
        mmproj_status: Some(mmproj_status),
        hf_repo: model
            .hf_metadata
            .as_ref()
            .and_then(|metadata| metadata.repo_id.clone()),
        hf_file: model
            .hf_metadata
            .as_ref()
            .and_then(|metadata| metadata.file.clone()),
        provider_type: "managed_llamacpp".to_string(),
        provider_name: "Managed llama.cpp".to_string(),
        provider_base_url: None,
        provider_managed: true,
    }
}

fn model_detail_from_scanned(
    model: &crate::models::scanner::ScannedModel,
    loaded_model: Option<&str>,
    loaded_ctx: Option<u32>,
    launch_preview: Option<&crate::engine::process::LaunchPreview>,
) -> ModelDetailResponse {
    use crate::models::profiles::ThinkTagStyle;

    let supports_tools = true;
    let _ = &model.profile.supports_parallel_tools;
    let supports_reasoning = !matches!(model.profile.think_tag_style, ThinkTagStyle::None);
    let is_active = loaded_model
        .map(|loaded| names_match(loaded, &model.filename))
        .unwrap_or(false);
    let (vision_runtime_ready, mmproj_status, vision_block_reason) =
        vision_runtime_info(model, is_active, launch_preview);

    ModelDetailResponse {
        id: model.filename.clone(),
        object: "model".to_string(),
        created: now_unix_secs(),
        owned_by: "inference-bridge".to_string(),
        active: is_active,
        path: model.path.to_string_lossy().to_string(),
        size_bytes: model.size_bytes,
        size_gb: model.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        family: model.profile.family.to_string(),
        supports_tools,
        supports_reasoning,
        supports_vision: model.profile.supports_vision,
        context_window: if is_active {
            loaded_ctx
        } else {
            model.profile.default_context_window
        },
        max_context_length: model.profile.max_context_window,
        max_output_tokens: model.profile.default_max_output_tokens,
        quant: extract_quant(&model.filename),
        tool_call_format: format!("{:?}", model.profile.tool_call_format),
        think_tag_style: format!("{:?}", model.profile.think_tag_style),
        reasoning: reasoning_capability(&model.profile),
        template_mode: launch_preview
            .filter(|_| is_active)
            .map(|preview| preview.template_mode.clone()),
        template_source: launch_preview
            .filter(|_| is_active)
            .and_then(|preview| preview.template_source.clone()),
        vision_runtime_ready,
        mmproj_status,
        vision_block_reason,
        hf_repo: model
            .hf_metadata
            .as_ref()
            .and_then(|metadata| metadata.repo_id.clone()),
        hf_file: model
            .hf_metadata
            .as_ref()
            .and_then(|metadata| metadata.file.clone()),
        hf_has_repo_template: model
            .hf_metadata
            .as_ref()
            .map(|metadata| metadata.has_repo_template)
            .unwrap_or(false),
    }
}

fn vision_runtime_info(
    model: &crate::models::scanner::ScannedModel,
    is_active: bool,
    launch_preview: Option<&crate::engine::process::LaunchPreview>,
) -> (bool, String, Option<String>) {
    if !model.profile.supports_vision {
        return (
            false,
            "not-capable".to_string(),
            Some("Model metadata does not advertise vision support".to_string()),
        );
    }

    if !is_active {
        return (
            false,
            "vision-capable".to_string(),
            Some("Model supports vision but is not the active runtime".to_string()),
        );
    }

    if launch_preview
        .and_then(|preview| preview.mmproj_path.as_ref())
        .is_some()
    {
        return (true, "vision-ready".to_string(), None);
    }

    (
        false,
        "mmproj-missing".to_string(),
        Some("Runtime is missing a matching mmproj sidecar".to_string()),
    )
}

fn reasoning_capability(profile: &crate::models::profiles::ModelProfile) -> ReasoningCapability {
    let supported = !matches!(
        profile.think_tag_style,
        crate::models::profiles::ThinkTagStyle::None
    );

    ReasoningCapability {
        supported,
        separates_content: supported,
        effort_values: if supported {
            vec![
                "none".to_string(),
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
                "xhigh".to_string(),
            ]
        } else {
            Vec::new()
        },
        supports_reasoning_tokens: supported,
        default_effort: supported.then(|| "medium".to_string()),
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
