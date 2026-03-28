use axum::extract::State;
use axum::response::Json;
use crate::state::SharedState;

#[derive(serde::Serialize)]
pub struct MetricsResponse {
    pub model: Option<String>,
    pub model_load_state: String,
    pub context_size: Option<u32>,
    pub last_load_duration_ms: Option<u64>,
    pub last_inference: Option<crate::state::RuntimePerformanceMetrics>,
    pub cumulative: crate::state::CumulativeMetrics,
    pub process_state: String,
    pub uptime_secs: u64,
}

pub async fn get_metrics(
    State(state): State<SharedState>,
) -> Json<MetricsResponse> {
    let s = state.read().await;

    let process_state = format!("{:?}", s.process.state());
    let context_size = s.model_stats.as_ref().map(|st| st.context_size).filter(|v| *v > 0);
    let model_load_state = match &s.model_load_state {
        crate::state::ModelLoadState::Idle => "idle".to_string(),
        crate::state::ModelLoadState::Loading => "loading".to_string(),
        crate::state::ModelLoadState::Swapping => "swapping".to_string(),
        crate::state::ModelLoadState::Unloading => "unloading".to_string(),
        crate::state::ModelLoadState::Loaded => "loaded".to_string(),
        crate::state::ModelLoadState::Error(e) => format!("error: {e}"),
    };

    Json(MetricsResponse {
        model: s.loaded_model.clone(),
        model_load_state,
        context_size,
        last_load_duration_ms: s.last_startup_duration_ms,
        last_inference: s.last_generation_metrics.clone(),
        cumulative: s.cumulative_metrics.clone(),
        process_state,
        uptime_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    })
}

#[derive(serde::Serialize)]
pub struct CancelResponse {
    pub cancelled: bool,
    pub message: String,
}

pub async fn cancel_inference(
    State(state): State<SharedState>,
) -> Json<CancelResponse> {
    let mut s = state.write().await;

    if s.active_generation.is_some() {
        s.generation_cancel.cancel();
        s.cumulative_metrics.total_cancellations += 1;
        if let Some(gen) = s.active_generation.as_mut() {
            gen.status = "cancelled".to_string();
        }
        tracing::info!("Active inference cancelled via API");
        Json(CancelResponse {
            cancelled: true,
            message: "Active inference request cancelled".to_string(),
        })
    } else {
        Json(CancelResponse {
            cancelled: false,
            message: "No active inference request to cancel".to_string(),
        })
    }
}
