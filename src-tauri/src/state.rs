use std::sync::Arc;
use std::sync::Mutex;

use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::config::AppConfig;
use crate::context::tracker::ContextStatus;
use crate::engine::process::{LaunchPreview, LlamaProcess};
use crate::models::overrides::ModelProfileOverride;
use crate::models::profiles::ModelProfile;
use crate::models::registry::ModelRegistry;
use crate::session::db::SessionDb;

#[derive(Clone, serde::Serialize)]
pub enum ModelLoadState {
    Idle,
    Loading,
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
pub struct EffectiveProfileInfo {
    pub requested_model: Option<String>,
    pub resolved_model: Option<String>,
    pub profile: ModelProfile,
    pub override_entry: Option<ModelProfileOverride>,
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
    pub last_startup_duration_ms: Option<u64>,
    pub model_load_state: ModelLoadState,
    pub model_load_progress: Option<LoadProgress>,
    pub model_stats: Option<ModelStats>,
    pub api_server_state: ApiServerState,
    pub api_server_error: Option<String>,
    pub app_handle: Option<tauri::AppHandle>,
}

pub type SharedState = Arc<RwLock<AppState>>;

impl AppState {
    pub fn new(config: AppConfig) -> anyhow::Result<Self> {
        let session_db = SessionDb::open()?;
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
            last_startup_duration_ms: None,
            model_load_state: ModelLoadState::Idle,
            model_load_progress: None,
            model_stats: None,
            api_server_state: ApiServerState::Idle,
            api_server_error: None,
            app_handle: None,
        })
    }
}
