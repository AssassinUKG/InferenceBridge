//! GET /health — Health check with KV cache metrics (inspired by Fox/vLLM).

use axum::extract::State;
use axum::response::Json;
use serde::Serialize;

use crate::context::tracker;
use crate::engine::client::LlamaClient;
use crate::state::SharedState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub model: Option<String>,
    pub kv_cache: Option<KvCacheHealth>,
}

#[derive(Serialize)]
pub struct KvCacheHealth {
    pub total_tokens: u32,
    pub used_tokens: u32,
    pub fill_ratio: f32,
}

pub async fn health_check(State(state): State<SharedState>) -> Json<HealthResponse> {
    let s = state.read().await;
    if s.loaded_model.is_none() {
        return Json(HealthResponse {
            status: "no_model",
            model: None,
            kv_cache: None,
        });
    }

    let model = s.loaded_model.clone();
    let client = LlamaClient::new(s.process.port());
    drop(s);

    let healthy = client.health().await.unwrap_or(false);
    let ctx = tracker::poll_context_status(&client).await;

    Json(HealthResponse {
        status: if healthy { "ok" } else { "unhealthy" },
        model,
        kv_cache: Some(KvCacheHealth {
            total_tokens: ctx.total_tokens,
            used_tokens: ctx.used_tokens,
            fill_ratio: ctx.fill_ratio,
        }),
    })
}
