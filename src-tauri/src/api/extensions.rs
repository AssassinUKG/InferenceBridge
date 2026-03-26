//! Extension endpoints beyond the OpenAI-compatible API.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::Deserialize;

use crate::api::errors::ApiErrorResponse;
use crate::context::tracker;
use crate::engine::client::LlamaClient;
use crate::state::SharedState;

// ─── Context ──────────────────────────────────────────────────────────────────

pub async fn context_status(State(state): State<SharedState>) -> Json<tracker::ContextStatus> {
    let s = state.read().await;
    if s.loaded_model.is_none() {
        return Json(tracker::ContextStatus::empty());
    }
    let client = LlamaClient::new(s.process.port());
    Json(tracker::poll_context_status(&client).await)
}

// ─── Session list ─────────────────────────────────────────────────────────────

pub async fn list_sessions(
    State(state): State<SharedState>,
) -> Result<Json<Vec<crate::session::db::SessionInfo>>, ApiErrorResponse> {
    let s = state.read().await;
    let db = s
        .session_db
        .lock()
        .map_err(|_| ApiErrorResponse::service_unavailable("Session DB lock poisoned"))?;
    db.list_sessions()
        .map(Json)
        .map_err(|e| ApiErrorResponse::service_unavailable(format!("Failed to list sessions: {e}")))
}

// ─── Session create ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    /// Human-readable session name.
    pub name: Option<String>,
    /// Model ID to associate with this session (optional).
    pub model_id: Option<String>,
}

pub async fn create_session(
    State(state): State<SharedState>,
    Json(req): Json<CreateSessionRequest>,
) -> Result<Json<crate::session::db::SessionInfo>, ApiErrorResponse> {
    let name = req.name.unwrap_or_else(|| "New Session".to_string());
    let s = state.read().await;
    let db = s
        .session_db
        .lock()
        .map_err(|_| ApiErrorResponse::service_unavailable("Session DB lock poisoned"))?;

    let id = db
        .create_session(&name, req.model_id.as_deref())
        .map_err(|e| ApiErrorResponse::service_unavailable(format!("Failed to create session: {e}")))?;

    // Re-fetch the created session to get timestamps.
    let sessions = db
        .list_sessions()
        .map_err(|e| ApiErrorResponse::service_unavailable(format!("Failed to fetch session: {e}")))?;

    sessions
        .into_iter()
        .find(|s| s.id == id)
        .map(Json)
        .ok_or_else(|| ApiErrorResponse::service_unavailable("Session created but not found"))
}

// ─── Session delete ───────────────────────────────────────────────────────────

pub async fn delete_session(
    State(state): State<SharedState>,
    Path(session_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiErrorResponse> {
    let s = state.read().await;
    let db = s
        .session_db
        .lock()
        .map_err(|_| ApiErrorResponse::service_unavailable("Session DB lock poisoned"))?;

    db.delete_session(&session_id)
        .map_err(|e| {
            ApiErrorResponse::service_unavailable(format!("Failed to delete session: {e}"))
        })?;

    Ok(Json(serde_json::json!({ "deleted": true, "id": session_id })))
}

// ─── Session messages ─────────────────────────────────────────────────────────

pub async fn get_session_messages(
    State(state): State<SharedState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<crate::session::db::MessageInfo>>, ApiErrorResponse> {
    let s = state.read().await;
    let db = s
        .session_db
        .lock()
        .map_err(|_| ApiErrorResponse::service_unavailable("Session DB lock poisoned"))?;

    db.get_messages(&session_id)
        .map(Json)
        .map_err(|e| {
            if e.to_string().contains("no rows") {
                ApiErrorResponse(
                    StatusCode::NOT_FOUND,
                    axum::Json(crate::api::errors::ApiError {
                        error: crate::api::errors::ApiErrorBody {
                            message: format!("Session '{session_id}' not found"),
                            error_type: "invalid_request_error".to_string(),
                            code: None,
                            param: Some("session_id".to_string()),
                        },
                    }),
                )
            } else {
                ApiErrorResponse::service_unavailable(format!("Failed to fetch messages: {e}"))
            }
        })
}
