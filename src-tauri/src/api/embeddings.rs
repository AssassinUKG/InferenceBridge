use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};

use crate::api::errors::ApiErrorResponse;
use crate::engine::client::LlamaClient;
use crate::engine::process::ProcessState;
use crate::state::SharedState;

/// OpenAI-compatible embeddings endpoint.
///
/// Phase A intentionally proxies to the currently loaded managed llama-server.
/// TODO(phase-b): add a dedicated embedding llama-server instance so chat and
/// embeddings can run concurrently with separate models.
pub async fn embeddings(
    State(state): State<SharedState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiErrorResponse> {
    if let Some(upstream) = crate::api::upstream::active_openai_provider(&state).await {
        return crate::api::upstream::proxy_json_to_openai_provider(
            state.clone(),
            upstream,
            "/embeddings",
            body,
        )
        .await;
    }

    let llama_port = {
        let s = state.read().await;
        if s.loaded_model.is_none() || s.process.state() != ProcessState::Running {
            return Err(ApiErrorResponse::no_model());
        }
        s.process.port()
    };

    let client = LlamaClient::new(llama_port);
    match client.embeddings(&body).await {
        Ok(value) => Ok(Json(value).into_response()),
        Err(error) => Err(ApiErrorResponse::not_supported(format!(
            "Embeddings are not available for the currently loaded runtime. Load an embedding GGUF and launch llama.cpp with embeddings support when you want this endpoint; no separate embedding model is started automatically. Underlying: {error}"
        ))),
    }
}
