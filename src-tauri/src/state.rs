use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use tauri::Emitter;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::commands::browse::DownloadProgress;
use crate::config::AppConfig;
use crate::context::tracker::ContextStatus;
use crate::engine::process::{LaunchPreview, LlamaProcess};
use crate::engine::scheduler::RequestScheduler;
use crate::models::overrides::ModelProfileOverride;
use crate::models::profiles::ModelProfile;
use crate::models::registry::ModelRegistry;
use crate::session::db::SessionDb;

#[derive(Clone, serde::Serialize)]
pub enum ModelLoadState {
    Idle,
    Loading,
    Swapping,
    Unloading,
    Loaded,
    Error(String),
}

#[derive(Debug, Clone, serde::Serialize)]
pub enum ApiServerState {
    Idle,
    Starting,
    Running,
    Error,
}

#[derive(Clone, serde::Serialize)]
pub struct ModelStats {
    pub model: String,
    pub context_size: u32,
    pub tokens_per_sec: f32,
    pub memory_mb: u32,
}

#[derive(Clone, serde::Serialize)]
pub struct LoadProgress {
    pub stage: String,
    pub message: String,
    pub progress: f32,
    pub done: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GenerationRequest {
    pub id: String,
    pub source: String,
    pub session_id: Option<String>,
    pub model: String,
    pub started_at: String,
    pub status: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RuntimePerformanceMetrics {
    pub source: String,
    pub model: String,
    pub request_id: String,
    pub started_at: String,
    pub finished_at: String,
    pub elapsed_ms: u64,
    pub time_to_first_token_ms: Option<u64>,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub prompt_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub end_to_end_tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveStreamEvent {
    pub timestamp: String,
    pub kind: String,
    pub text: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveStreamSnapshot {
    pub request_id: String,
    pub source: String,
    pub model: String,
    pub started_at: String,
    pub status: String,
    pub raw_output: String,
    pub visible_output: String,
    pub reasoning_output: String,
    pub events: Vec<LiveStreamEvent>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EffectiveProfileInfo {
    pub requested_model: Option<String>,
    pub resolved_model: Option<String>,
    pub profile: ModelProfile,
    pub override_entry: Option<ModelProfileOverride>,
}

#[derive(Clone)]
pub struct ActiveDownload {
    pub progress: DownloadProgress,
    pub cancel_token: CancellationToken,
}

/// Cumulative API metrics for the /v1/metrics endpoint.
#[derive(Debug, Clone, serde::Serialize, Default)]
pub struct CumulativeMetrics {
    pub total_requests: u64,
    pub total_errors: u64,
    pub total_cancellations: u64,
    pub total_model_loads: u64,
    pub total_model_unloads: u64,
    pub backend_restart_count: u64,
}

pub struct AppState {
    pub config: AppConfig,
    pub process: LlamaProcess,
    pub model_registry: ModelRegistry,
    pub session_db: Mutex<SessionDb>,
    pub loaded_model: Option<String>,
    pub loading_generation: u64,
    pub previous_model: Option<String>,
    pub generation_cancel: CancellationToken,
    pub model_load_cancel: CancellationToken,
    pub active_generation: Option<GenerationRequest>,
    pub last_prompt: Option<String>,
    pub last_parse_trace: Option<String>,
    pub last_launch_preview: Option<LaunchPreview>,
    pub last_known_good_config: Option<LaunchPreview>,
    pub last_context_status: Option<ContextStatus>,
    pub last_generation_metrics: Option<RuntimePerformanceMetrics>,
    pub live_stream: Option<LiveStreamSnapshot>,
    pub last_startup_duration_ms: Option<u64>,
    pub model_load_state: ModelLoadState,
    pub model_load_progress: Option<LoadProgress>,
    pub model_stats: Option<ModelStats>,
    pub api_server_state: ApiServerState,
    pub api_server_error: Option<String>,
    pub app_handle: Option<tauri::AppHandle>,
    pub active_downloads: HashMap<String, ActiveDownload>,
    pub request_scheduler: Arc<RequestScheduler>,
    pub model_load_mutex: Arc<AsyncMutex<()>>,
    pub cumulative_metrics: CumulativeMetrics,
}

pub type SharedState = Arc<RwLock<AppState>>;

impl AppState {
    pub fn new(config: AppConfig) -> anyhow::Result<Self> {
        let session_db = SessionDb::open()?;
        let scheduler_limit = config.process.parallel_slots;
        Ok(Self {
            config,
            process: LlamaProcess::new(),
            model_registry: ModelRegistry::new(),
            session_db: Mutex::new(session_db),
            loaded_model: None,
            loading_generation: 0,
            previous_model: None,
            generation_cancel: CancellationToken::new(),
            model_load_cancel: CancellationToken::new(),
            active_generation: None,
            last_prompt: None,
            last_parse_trace: None,
            last_launch_preview: None,
            last_known_good_config: None,
            last_context_status: None,
            last_generation_metrics: None,
            live_stream: None,
            last_startup_duration_ms: None,
            model_load_state: ModelLoadState::Idle,
            model_load_progress: None,
            model_stats: None,
            api_server_state: ApiServerState::Idle,
            api_server_error: None,
            app_handle: None,
            active_downloads: HashMap::new(),
            request_scheduler: Arc::new(RequestScheduler::new(scheduler_limit)),
            model_load_mutex: Arc::new(AsyncMutex::new(())),
            cumulative_metrics: CumulativeMetrics::default(),
        })
    }
}

pub struct GenerationHandle {
    pub cancel: CancellationToken,
    pub request_id: String,
}

pub async fn begin_api_generation(state: &SharedState, model: String) -> GenerationHandle {
    let mut s = state.write().await;
    s.generation_cancel.cancel();
    s.generation_cancel = CancellationToken::new();
    s.cumulative_metrics.total_requests += 1;
    let request_id = uuid::Uuid::new_v4().to_string();
    let started_at = chrono::Utc::now().to_rfc3339();
    s.active_generation = Some(GenerationRequest {
        id: request_id.clone(),
        source: "api".to_string(),
        session_id: None,
        model,
        started_at: started_at.clone(),
        status: "running".to_string(),
    });
    let model = s
        .active_generation
        .as_ref()
        .map(|generation| generation.model.clone())
        .unwrap_or_default();
    s.live_stream = Some(LiveStreamSnapshot {
        request_id: request_id.clone(),
        source: "api".to_string(),
        model,
        started_at,
        status: "running".to_string(),
        raw_output: String::new(),
        visible_output: String::new(),
        reasoning_output: String::new(),
        events: Vec::new(),
    });
    if let (Some(handle), Some(snapshot)) = (s.app_handle.clone(), s.live_stream.clone()) {
        let _ = handle.emit("llm-stream-start", snapshot);
    }
    GenerationHandle {
        cancel: s.generation_cancel.clone(),
        request_id,
    }
}

pub async fn finish_api_generation(state: &SharedState, status: &str) {
    let mut s = state.write().await;
    if let Some(active) = s.active_generation.as_mut() {
        active.status = status.to_string();
    }
    if let Some(stream) = s.live_stream.as_mut() {
        stream.status = status.to_string();
    }
    if let (Some(handle), Some(snapshot)) = (s.app_handle.clone(), s.live_stream.clone()) {
        let _ = handle.emit("llm-stream-done", snapshot);
    }
    s.active_generation = None;
}

pub async fn begin_live_generation(
    state: &SharedState,
    source: &str,
    session_id: Option<String>,
    model: String,
) -> GenerationHandle {
    let mut s = state.write().await;
    s.generation_cancel.cancel();
    s.generation_cancel = CancellationToken::new();
    s.cumulative_metrics.total_requests += 1;
    let request_id = uuid::Uuid::new_v4().to_string();
    let started_at = chrono::Utc::now().to_rfc3339();
    s.active_generation = Some(GenerationRequest {
        id: request_id.clone(),
        source: source.to_string(),
        session_id,
        model: model.clone(),
        started_at: started_at.clone(),
        status: "running".to_string(),
    });
    s.live_stream = Some(LiveStreamSnapshot {
        request_id: request_id.clone(),
        source: source.to_string(),
        model,
        started_at,
        status: "running".to_string(),
        raw_output: String::new(),
        visible_output: String::new(),
        reasoning_output: String::new(),
        events: Vec::new(),
    });
    if let (Some(handle), Some(snapshot)) = (s.app_handle.clone(), s.live_stream.clone()) {
        let _ = handle.emit("llm-stream-start", snapshot);
    }
    GenerationHandle {
        cancel: s.generation_cancel.clone(),
        request_id,
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveStreamDelta {
    pub request_id: String,
    pub timestamp: String,
    pub kind: String,
    pub text: String,
}

pub async fn append_live_stream_delta(state: &SharedState, kind: &str, text: &str) {
    if text.is_empty() {
        return;
    }

    let (handle, delta) = {
        let mut s = state.write().await;
        let Some(stream) = s.live_stream.as_mut() else {
            return;
        };
        let timestamp = chrono::Utc::now().to_rfc3339();
        match kind {
            "content" => {
                stream.visible_output.push_str(text);
            }
            "reasoning" => {
                stream.reasoning_output.push_str(text);
            }
            "tool_call" | "raw" | "error" => {
                stream.raw_output.push_str(text);
            }
            _ => {
                stream.raw_output.push_str(text);
            }
        }
        let event = LiveStreamEvent {
            timestamp: timestamp.clone(),
            kind: kind.to_string(),
            text: text.to_string(),
        };
        stream.events.push(event);
        if stream.events.len() > 500 {
            let excess = stream.events.len() - 500;
            stream.events.drain(0..excess);
        }
        let delta = LiveStreamDelta {
            request_id: stream.request_id.clone(),
            timestamp,
            kind: kind.to_string(),
            text: text.to_string(),
        };
        (s.app_handle.clone(), delta)
    };

    if let Some(handle) = handle {
        let _ = handle.emit("llm-stream-delta", delta);
    }
}

pub fn summarize_reasoning_tokens(
    total_completion_tokens: Option<u32>,
    visible_text: &str,
    reasoning_text: &str,
) -> u32 {
    if reasoning_text.trim().is_empty() {
        return 0;
    }

    if let Some(total_completion_tokens) = total_completion_tokens {
        let visible_chars = visible_text.chars().count() as u32;
        let reasoning_chars = reasoning_text.chars().count() as u32;
        let total_chars = visible_chars + reasoning_chars;
        if total_chars > 0 {
            let estimated = ((total_completion_tokens as f64)
                * (reasoning_chars as f64 / total_chars as f64))
                .round() as u32;
            return estimated.min(total_completion_tokens);
        }
    }

    crate::normalize::think_strip::estimate_token_count(reasoning_text)
}
