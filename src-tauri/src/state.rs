use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::config::AppConfig;
use crate::context::tracker::ContextStatus;
use crate::engine::scheduler::RequestScheduler;
use crate::engine::process::{LaunchPreview, LlamaProcess};
use crate::models::overrides::ModelProfileOverride;
use crate::models::profiles::ModelProfile;
use crate::models::registry::ModelRegistry;
use crate::session::db::SessionDb;
use crate::commands::browse::DownloadProgress;

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
    pub started_at: String,
    pub finished_at: String,
    pub elapsed_ms: u64,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub prompt_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub end_to_end_tokens_per_second: Option<f64>,
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

pub struct AppState {
    pub config: AppConfig,
    pub process: LlamaProcess,
    pub model_registry: ModelRegistry,
    pub session_db: Mutex<SessionDb>,
    pub loaded_model: Option<String>,
    pub loading_generation: u64,
    pub previous_model: Option<String>,
    pub generation_cancel: CancellationToken,
    pub active_generation: Option<GenerationRequest>,
    pub last_prompt: Option<String>,
    pub last_parse_trace: Option<String>,
    pub last_launch_preview: Option<LaunchPreview>,
    pub last_known_good_config: Option<LaunchPreview>,
    pub last_context_status: Option<ContextStatus>,
    pub last_generation_metrics: Option<RuntimePerformanceMetrics>,
    pub last_startup_duration_ms: Option<u64>,
    pub model_load_state: ModelLoadState,
    pub model_load_progress: Option<LoadProgress>,
    pub model_stats: Option<ModelStats>,
    pub api_server_state: ApiServerState,
    pub api_server_error: Option<String>,
    pub app_handle: Option<tauri::AppHandle>,
    pub active_downloads: HashMap<String, ActiveDownload>,
    pub request_scheduler: Arc<RequestScheduler>,
    /// Serialises concurrent model-load requests.  Only one load runs at a time;
    /// the others wait and then coalesce (skip the load if the right model is
    /// already running by the time they acquire the lock).
    pub model_load_mutex: Arc<AsyncMutex<()>>,
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
            active_generation: None,
            last_prompt: None,
            last_parse_trace: None,
            last_launch_preview: None,
            last_known_good_config: None,
            last_context_status: None,
            last_generation_metrics: None,
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
        })
    }
}

pub async fn begin_api_generation(
    state: &SharedState,
    model: String,
) -> tokio_util::sync::CancellationToken {
    let mut s = state.write().await;
    s.generation_cancel.cancel();
    s.generation_cancel = tokio_util::sync::CancellationToken::new();
    s.active_generation = Some(GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        source: "api".to_string(),
        session_id: None,
        model,
        started_at: chrono::Utc::now().to_rfc3339(),
        status: "running".to_string(),
    });
    s.generation_cancel.clone()
}

pub async fn finish_api_generation(state: &SharedState, status: &str) {
    let mut s = state.write().await;
    if let Some(active) = s.active_generation.as_mut() {
        active.status = status.to_string();
    }
    s.active_generation = None;
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
            let estimated =
                ((total_completion_tokens as f64) * (reasoning_chars as f64 / total_chars as f64))
                    .round() as u32;
            return estimated.min(total_completion_tokens);
        }
    }

    crate::normalize::think_strip::estimate_token_count(reasoning_text)
}
