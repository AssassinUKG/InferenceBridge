use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use tauri::Emitter;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::commands::browse::DownloadProgress;
use crate::config::AppConfig;
use crate::context::tracker::ContextStatus;
use crate::engine::process::{LaunchPreview, LlamaProcess, SamplingDefaults};
use crate::engine::scheduler::RequestScheduler;
use crate::image_generation::ImageGenerationProgress;
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
    pub generation_cancels: HashMap<String, CancellationToken>,
    pub model_load_cancel: CancellationToken,
    pub active_generation: Option<GenerationRequest>,
    pub last_prompt: Option<String>,
    pub last_parse_trace: Option<String>,
    pub last_launch_preview: Option<LaunchPreview>,
    pub last_known_good_config: Option<LaunchPreview>,
    pub last_context_status: Option<ContextStatus>,
    pub last_generation_metrics: Option<RuntimePerformanceMetrics>,
    pub live_stream: Option<LiveStreamSnapshot>,
    pub live_streams: Vec<LiveStreamSnapshot>,
    pub last_startup_duration_ms: Option<u64>,
    pub model_load_state: ModelLoadState,
    pub model_load_progress: Option<LoadProgress>,
    pub model_stats: Option<ModelStats>,
    pub api_server_state: ApiServerState,
    pub api_server_error: Option<String>,
    pub api_server_host: Option<String>,
    pub api_server_port: Option<u16>,
    pub app_handle: Option<tauri::AppHandle>,
    pub active_downloads: HashMap<String, ActiveDownload>,
    pub download_persist_mutex: Arc<AsyncMutex<()>>,
    pub request_scheduler: Arc<RequestScheduler>,
    pub model_load_mutex: Arc<AsyncMutex<()>>,
    pub image_generation_progress: Option<ImageGenerationProgress>,
    pub image_generation_cancel: CancellationToken,
    pub image_generation_mutex: Arc<AsyncMutex<()>>,
    pub cumulative_metrics: CumulativeMetrics,
}

pub type SharedState = Arc<RwLock<AppState>>;

impl AppState {
    pub fn new(config: AppConfig) -> anyhow::Result<Self> {
        let session_db = SessionDb::open()?;
        let scheduler_limit = config.process.parallel_slots;
        let active_downloads =
            crate::commands::browse::load_persisted_downloads(&config.models.scan_dirs)
                .into_iter()
                .map(|progress| {
                    (
                        progress.id.clone(),
                        ActiveDownload {
                            progress,
                            cancel_token: CancellationToken::new(),
                        },
                    )
                })
                .collect();
        let mut process = LlamaProcess::new();
        if !config.process.llama_server_path.trim().is_empty() {
            process.set_server_path(config.process.llama_server_path.clone().into());
        }
        if !config.process.llama_diffusion_cli_path.trim().is_empty() {
            process.set_diffusion_cli_path(config.process.llama_diffusion_cli_path.clone().into());
        }
        Ok(Self {
            config,
            process,
            model_registry: ModelRegistry::new(),
            session_db: Mutex::new(session_db),
            loaded_model: None,
            loading_generation: 0,
            previous_model: None,
            generation_cancel: CancellationToken::new(),
            generation_cancels: HashMap::new(),
            model_load_cancel: CancellationToken::new(),
            active_generation: None,
            last_prompt: None,
            last_parse_trace: None,
            last_launch_preview: None,
            last_known_good_config: None,
            last_context_status: None,
            last_generation_metrics: None,
            live_stream: None,
            live_streams: Vec::new(),
            last_startup_duration_ms: None,
            model_load_state: ModelLoadState::Idle,
            model_load_progress: None,
            model_stats: None,
            api_server_state: ApiServerState::Idle,
            api_server_error: None,
            api_server_host: None,
            api_server_port: None,
            app_handle: None,
            active_downloads,
            download_persist_mutex: Arc::new(AsyncMutex::new(())),
            request_scheduler: Arc::new(RequestScheduler::new(scheduler_limit)),
            model_load_mutex: Arc::new(AsyncMutex::new(())),
            image_generation_progress: None,
            image_generation_cancel: CancellationToken::new(),
            image_generation_mutex: Arc::new(AsyncMutex::new(())),
            cumulative_metrics: CumulativeMetrics::default(),
        })
    }

    /// Resolve a model profile through the scanned registry so request-time
    /// rendering/parsing uses the GGUF architecture rather than filename-only
    /// heuristics.
    pub fn effective_profile_for_model(&self, model_name: &str) -> ModelProfile {
        self.model_registry.effective_profile_for_name(model_name)
    }

    pub fn active_sampling_defaults(&self) -> SamplingDefaults {
        self.last_launch_preview
            .as_ref()
            .map(|preview| preview.sampling_defaults)
            .unwrap_or_default()
    }
}

pub struct GenerationHandle {
    pub cancel: CancellationToken,
    pub request_id: String,
}

pub struct GenerationDropGuard {
    state: SharedState,
    request_id: String,
    cancel: CancellationToken,
    completed: Arc<AtomicBool>,
}

impl GenerationDropGuard {
    pub fn new(state: SharedState, request_id: String, cancel: CancellationToken) -> Self {
        Self {
            state,
            request_id,
            cancel,
            completed: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn mark_completed(&self) {
        self.completed.store(true, Ordering::SeqCst);
    }
}

impl Drop for GenerationDropGuard {
    fn drop(&mut self) {
        if self.completed.swap(true, Ordering::SeqCst) {
            return;
        }
        self.cancel.cancel();
        let state = self.state.clone();
        let request_id = self.request_id.clone();
        tokio::spawn(async move {
            finish_api_generation_for_request(&state, &request_id, "disconnected").await;
        });
    }
}

pub async fn begin_api_generation(state: &SharedState, model: String) -> GenerationHandle {
    let mut s = state.write().await;
    let cancel = CancellationToken::new();
    s.generation_cancel = cancel.clone();
    s.cumulative_metrics.total_requests += 1;
    let request_id = uuid::Uuid::new_v4().to_string();
    s.generation_cancels
        .insert(request_id.clone(), cancel.clone());
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
    let snapshot = LiveStreamSnapshot {
        request_id: request_id.clone(),
        source: "api".to_string(),
        model,
        started_at,
        status: "running".to_string(),
        raw_output: String::new(),
        visible_output: String::new(),
        reasoning_output: String::new(),
        events: Vec::new(),
    };
    push_live_stream_locked(&mut s, snapshot.clone());
    if let Some(handle) = s.app_handle.clone() {
        let _ = handle.emit("llm-stream-start", snapshot);
    }
    GenerationHandle { cancel, request_id }
}

pub async fn finish_api_generation(state: &SharedState, status: &str) {
    let request_id = {
        let s = state.read().await;
        s.live_stream
            .as_ref()
            .map(|stream| stream.request_id.clone())
            .or_else(|| {
                s.active_generation
                    .as_ref()
                    .map(|generation| generation.id.clone())
            })
    };
    if let Some(request_id) = request_id {
        finish_api_generation_for_request(state, &request_id, status).await;
    }
}

pub async fn finish_api_generation_for_request(
    state: &SharedState,
    request_id: &str,
    status: &str,
) {
    finish_generation_for_request(state, request_id, status).await;
}

pub async fn finish_generation_for_request(state: &SharedState, request_id: &str, status: &str) {
    let mut s = state.write().await;
    s.generation_cancels.remove(request_id);
    let effective_status = s
        .live_streams
        .iter()
        .find(|stream| stream.request_id == request_id)
        .map(|stream| {
            if stream.status == "cancelled" {
                "cancelled"
            } else {
                status
            }
        })
        .unwrap_or(status)
        .to_string();
    if let Some(active) = s
        .active_generation
        .as_mut()
        .filter(|active| active.id == request_id)
    {
        active.status = effective_status.clone();
    }
    let snapshot = update_live_stream_locked(&mut s, request_id, |stream| {
        stream.status = effective_status;
    });
    if let (Some(handle), Some(snapshot)) = (s.app_handle.clone(), snapshot) {
        let _ = handle.emit("llm-stream-done", snapshot);
    }
    if s.active_generation
        .as_ref()
        .map(|active| active.id == request_id)
        .unwrap_or(false)
    {
        s.active_generation = None;
    }
}

pub async fn begin_live_generation(
    state: &SharedState,
    source: &str,
    session_id: Option<String>,
    model: String,
) -> GenerationHandle {
    let mut s = state.write().await;
    let cancel = CancellationToken::new();
    s.generation_cancel = cancel.clone();
    s.cumulative_metrics.total_requests += 1;
    let request_id = uuid::Uuid::new_v4().to_string();
    s.generation_cancels
        .insert(request_id.clone(), cancel.clone());
    let started_at = chrono::Utc::now().to_rfc3339();
    s.active_generation = Some(GenerationRequest {
        id: request_id.clone(),
        source: source.to_string(),
        session_id,
        model: model.clone(),
        started_at: started_at.clone(),
        status: "running".to_string(),
    });
    let snapshot = LiveStreamSnapshot {
        request_id: request_id.clone(),
        source: source.to_string(),
        model,
        started_at,
        status: "running".to_string(),
        raw_output: String::new(),
        visible_output: String::new(),
        reasoning_output: String::new(),
        events: Vec::new(),
    };
    push_live_stream_locked(&mut s, snapshot.clone());
    if let Some(handle) = s.app_handle.clone() {
        let _ = handle.emit("llm-stream-start", snapshot);
    }
    GenerationHandle { cancel, request_id }
}

pub async fn cancel_all_generations(state: &SharedState) -> usize {
    let (handle, snapshots, count) = {
        let mut s = state.write().await;
        let request_ids = s.generation_cancels.keys().cloned().collect::<Vec<_>>();
        let count = request_ids.len();
        for cancel in s.generation_cancels.values() {
            cancel.cancel();
        }
        s.generation_cancel.cancel();
        s.generation_cancels.clear();
        if count > 0 {
            s.cumulative_metrics.total_cancellations += count as u64;
        }
        if s.active_generation
            .as_ref()
            .is_some_and(|active| request_ids.contains(&active.id))
        {
            s.active_generation = None;
        }

        let mut snapshots = Vec::with_capacity(count);
        for request_id in &request_ids {
            if let Some(snapshot) = update_live_stream_locked(&mut s, request_id, |stream| {
                stream.status = "cancelled".to_string();
            }) {
                snapshots.push(snapshot);
            }
        }
        (s.app_handle.clone(), snapshots, count)
    };

    if let Some(handle) = handle {
        for snapshot in snapshots {
            let _ = handle.emit("llm-stream-done", snapshot);
        }
    }
    count
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveStreamDelta {
    pub request_id: String,
    pub timestamp: String,
    pub kind: String,
    pub text: String,
}

pub async fn append_live_stream_delta(state: &SharedState, kind: &str, text: &str) {
    let request_id = {
        let s = state.read().await;
        s.live_stream
            .as_ref()
            .map(|stream| stream.request_id.clone())
            .or_else(|| {
                s.active_generation
                    .as_ref()
                    .map(|generation| generation.id.clone())
            })
    };
    if let Some(request_id) = request_id {
        append_live_stream_delta_for_request(state, &request_id, kind, text).await;
    }
}

pub async fn append_live_stream_delta_for_request(
    state: &SharedState,
    request_id: &str,
    kind: &str,
    text: &str,
) {
    if text.is_empty() {
        return;
    }

    let (handle, delta) = {
        let mut s = state.write().await;
        let timestamp = chrono::Utc::now().to_rfc3339();
        let Some(delta) = update_live_stream_locked(&mut s, request_id, |stream| {
            match kind {
                "content" => {
                    stream.visible_output.push_str(text);
                }
                "reasoning" => {
                    stream.reasoning_output.push_str(text);
                }
                "raw" | "error" => {
                    stream.raw_output.push_str(text);
                }
                "content_buffered" => {}
                "input" => {}
                "tool_call" => {}
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
        }) else {
            return;
        };
        let delta = LiveStreamDelta {
            request_id: delta.request_id,
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

fn push_live_stream_locked(s: &mut AppState, snapshot: LiveStreamSnapshot) {
    if let Some(existing) = s
        .live_streams
        .iter_mut()
        .find(|stream| stream.request_id == snapshot.request_id)
    {
        *existing = snapshot.clone();
    } else {
        s.live_streams.push(snapshot.clone());
        if s.live_streams.len() > 30 {
            let excess = s.live_streams.len() - 30;
            s.live_streams.drain(0..excess);
        }
    }
    s.live_stream = Some(snapshot);
}

fn update_live_stream_locked<F>(
    s: &mut AppState,
    request_id: &str,
    update: F,
) -> Option<LiveStreamSnapshot>
where
    F: FnOnce(&mut LiveStreamSnapshot),
{
    let index = s
        .live_streams
        .iter()
        .position(|stream| stream.request_id == request_id)?;
    update(&mut s.live_streams[index]);
    let snapshot = s.live_streams[index].clone();
    if s.live_stream
        .as_ref()
        .map(|stream| stream.request_id == request_id)
        .unwrap_or(false)
    {
        s.live_stream = Some(snapshot.clone());
    }
    Some(snapshot)
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

#[cfg(test)]
mod tests {
    use super::{
        append_live_stream_delta_for_request, begin_api_generation, begin_live_generation,
        cancel_all_generations, finish_generation_for_request, AppState, GenerationDropGuard,
    };
    use crate::config::AppConfig;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn starting_second_api_generation_does_not_cancel_first() {
        let state = Arc::new(RwLock::new(
            AppState::new(AppConfig::default()).expect("state should initialize"),
        ));

        let first = begin_api_generation(&state, "model-a".to_string()).await;
        let second = begin_api_generation(&state, "model-b".to_string()).await;

        assert!(!first.cancel.is_cancelled());
        assert!(!second.cancel.is_cancelled());

        let cancelled = cancel_all_generations(&state).await;

        assert_eq!(cancelled, 2);
        assert!(first.cancel.is_cancelled());
        assert!(second.cancel.is_cancelled());

        let s = state.read().await;
        assert!(s.generation_cancels.is_empty());
        assert!(s.active_generation.is_none());
        assert!(s
            .live_streams
            .iter()
            .all(|stream| stream.status == "cancelled"));
    }

    #[tokio::test]
    async fn buffered_content_events_do_not_duplicate_raw_log_output() {
        let state = Arc::new(RwLock::new(
            AppState::new(AppConfig::default()).expect("state should initialize"),
        ));

        let generation = begin_api_generation(&state, "model-a".to_string()).await;
        append_live_stream_delta_for_request(&state, &generation.request_id, "raw", "Hello").await;
        append_live_stream_delta_for_request(
            &state,
            &generation.request_id,
            "content_buffered",
            "Hello",
        )
        .await;

        let s = state.read().await;
        let stream = s.live_stream.as_ref().expect("stream should exist");
        assert_eq!(stream.raw_output, "Hello");
        assert_eq!(stream.events.len(), 2);
        assert_eq!(stream.events[1].kind, "content_buffered");
    }

    #[tokio::test]
    async fn request_input_is_recorded_without_polluting_model_output() {
        let state = Arc::new(RwLock::new(
            AppState::new(AppConfig::default()).expect("state should initialize"),
        ));

        let generation = begin_live_generation(
            &state,
            "gui",
            Some("session-input".to_string()),
            "model-a".to_string(),
        )
        .await;
        append_live_stream_delta_for_request(
            &state,
            &generation.request_id,
            "input",
            "hello from the user",
        )
        .await;

        let s = state.read().await;
        let stream = s.live_stream.as_ref().expect("stream should exist");
        assert!(stream.raw_output.is_empty());
        assert!(stream.visible_output.is_empty());
        assert_eq!(stream.events.len(), 1);
        assert_eq!(stream.events[0].kind, "input");
        assert_eq!(stream.events[0].text, "hello from the user");
    }

    #[tokio::test]
    async fn request_scoped_updates_do_not_cross_interleaved_generations() {
        let state = Arc::new(RwLock::new(
            AppState::new(AppConfig::default()).expect("state should initialize"),
        ));

        let gui = begin_live_generation(
            &state,
            "gui",
            Some("session-a".to_string()),
            "model-a".to_string(),
        )
        .await;
        let api = begin_api_generation(&state, "model-b".to_string()).await;

        append_live_stream_delta_for_request(&state, &gui.request_id, "raw", "gui raw").await;
        append_live_stream_delta_for_request(&state, &gui.request_id, "content", "gui text").await;
        append_live_stream_delta_for_request(&state, &api.request_id, "raw", "api raw").await;

        let s = state.read().await;
        let gui_stream = s
            .live_streams
            .iter()
            .find(|stream| stream.request_id == gui.request_id)
            .expect("GUI stream should remain in history");
        let api_stream = s
            .live_streams
            .iter()
            .find(|stream| stream.request_id == api.request_id)
            .expect("API stream should remain in history");
        assert_eq!(gui_stream.raw_output, "gui raw");
        assert_eq!(gui_stream.visible_output, "gui text");
        assert_eq!(api_stream.raw_output, "api raw");
        assert!(api_stream.visible_output.is_empty());
        assert_eq!(
            s.live_stream.as_ref().map(|stream| &stream.request_id),
            Some(&api.request_id)
        );
    }

    #[tokio::test]
    async fn request_scoped_finish_updates_history_and_cleans_only_its_cancel() {
        let state = Arc::new(RwLock::new(
            AppState::new(AppConfig::default()).expect("state should initialize"),
        ));

        let gui = begin_live_generation(
            &state,
            "gui",
            Some("session-a".to_string()),
            "model-a".to_string(),
        )
        .await;
        let api = begin_api_generation(&state, "model-b".to_string()).await;

        finish_generation_for_request(&state, &gui.request_id, "completed").await;

        {
            let s = state.read().await;
            let gui_stream = s
                .live_streams
                .iter()
                .find(|stream| stream.request_id == gui.request_id)
                .expect("GUI stream should remain in history");
            let api_stream = s
                .live_streams
                .iter()
                .find(|stream| stream.request_id == api.request_id)
                .expect("API stream should remain in history");
            assert_eq!(gui_stream.status, "completed");
            assert_eq!(api_stream.status, "running");
            assert!(!s.generation_cancels.contains_key(&gui.request_id));
            assert!(s.generation_cancels.contains_key(&api.request_id));
            assert_eq!(
                s.live_stream.as_ref().map(|stream| &stream.request_id),
                Some(&api.request_id)
            );
            assert_eq!(
                s.active_generation.as_ref().map(|active| &active.id),
                Some(&api.request_id)
            );
        }

        finish_generation_for_request(&state, &api.request_id, "completed").await;

        let s = state.read().await;
        assert!(s.generation_cancels.is_empty());
        assert!(s.active_generation.is_none());
        assert_eq!(
            s.live_stream.as_ref().map(|stream| stream.status.as_str()),
            Some("completed")
        );
        assert!(s
            .live_streams
            .iter()
            .all(|stream| stream.status == "completed"));
    }

    #[tokio::test]
    async fn cancellation_is_terminal_when_worker_finishes_after_cancel() {
        let state = Arc::new(RwLock::new(
            AppState::new(AppConfig::default()).expect("state should initialize"),
        ));

        let generation = begin_live_generation(
            &state,
            "gui",
            Some("session-a".to_string()),
            "model-a".to_string(),
        )
        .await;

        assert_eq!(cancel_all_generations(&state).await, 1);
        finish_generation_for_request(&state, &generation.request_id, "completed").await;

        let s = state.read().await;
        let stream = s
            .live_streams
            .iter()
            .find(|stream| stream.request_id == generation.request_id)
            .expect("cancelled stream should remain in history");
        assert_eq!(stream.status, "cancelled");
        assert!(s.generation_cancels.is_empty());
    }

    #[tokio::test]
    async fn terminal_guard_drop_preserves_completed_and_error_statuses() {
        for terminal_status in ["completed", "error"] {
            let state = Arc::new(RwLock::new(
                AppState::new(AppConfig::default()).expect("state should initialize"),
            ));
            let generation = begin_api_generation(&state, "model-a".to_string()).await;
            let request_id = generation.request_id.clone();
            let cancellation = generation.cancel.clone();
            let guard =
                GenerationDropGuard::new(state.clone(), request_id.clone(), generation.cancel);

            // SSE producers must perform these two operations before yielding a
            // terminal frame because clients commonly drop immediately on it.
            guard.mark_completed();
            finish_generation_for_request(&state, &request_id, terminal_status).await;
            drop(guard);
            tokio::task::yield_now().await;

            assert!(!cancellation.is_cancelled());
            let state = state.read().await;
            let stream = state
                .live_streams
                .iter()
                .find(|stream| stream.request_id == request_id)
                .expect("terminal stream should remain in history");
            assert_eq!(stream.status, terminal_status);
        }
    }
}
