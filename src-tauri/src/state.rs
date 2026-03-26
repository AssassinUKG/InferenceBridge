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
    // Add more fields as needed
}

#[derive(Clone, serde::Serialize)]
pub struct LoadProgress {
    pub stage: String,
    pub message: String,
    pub progress: f32, // 0.0 to 1.0
    pub done: bool,
    pub error: Option<String>,
}
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::RwLock;

use crate::config::AppConfig;
use crate::engine::process::LlamaProcess;
use crate::models::registry::ModelRegistry;
use crate::session::db::SessionDb;

/// Shared application state, accessible from both Tauri commands and the Axum API server.
pub struct AppState {
    pub config: AppConfig,
    pub process: LlamaProcess,
    pub model_registry: ModelRegistry,
    pub session_db: Mutex<SessionDb>,
    /// Currently loaded model ID (filename), if any.
    pub loaded_model: Option<String>,
    /// Monotonic generation counter — prevents stale load_model calls from
    /// overwriting the result of a newer swap.
    pub loading_generation: u64,
    /// Previously loaded model name — enables quick swap-back.
    pub previous_model: Option<String>,
    /// Cancellation flag for in-flight streaming generation.
    /// Set to true by stop_generation, reset to false at the start of each send_message.
    pub generation_stop: Arc<AtomicBool>,
    pub model_load_state: ModelLoadState,
    pub model_load_progress: Option<LoadProgress>,
    pub model_stats: Option<ModelStats>,
    pub api_server_state: ApiServerState,
    pub api_server_error: Option<String>,
    /// Tauri app handle — set in GUI mode so API/backend paths can emit
    /// frontend events (model-load-progress, api-server-state-changed, etc.).
    /// Always `None` in headless/CLI mode.
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
            generation_stop: Arc::new(AtomicBool::new(false)),
            model_load_state: ModelLoadState::Idle,
            model_load_progress: None,
            model_stats: None,
            api_server_state: ApiServerState::Idle,
            api_server_error: None,
            app_handle: None,
        })
    }
}
