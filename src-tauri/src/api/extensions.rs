use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde::Deserialize;

use crate::api::errors::{ApiError, ApiErrorBody, ApiErrorResponse};
use crate::context::tracker;
use crate::engine::client::LlamaClient;
use crate::state::{EffectiveProfileInfo, SharedState};

pub async fn context_status(State(state): State<SharedState>) -> Json<tracker::ContextStatus> {
    let (loaded, port, stored) = {
        let s = state.read().await;
        (
            s.loaded_model.is_some(),
            s.process.port(),
            s.last_context_status.clone(),
        )
    };

    if !loaded {
        return Json(stored.unwrap_or_else(tracker::ContextStatus::empty));
    }

    let client = LlamaClient::new(port);
    let polled = tracker::poll_context_status(&client).await;

    if let Some(stored) = stored {
        Json(polled.with_breakdown(
            stored.pinned_tokens,
            stored.rolling_tokens,
            stored.compressed_tokens,
            stored.last_compaction_action,
        ))
    } else {
        Json(polled)
    }
}

pub async fn runtime_status(
    State(state): State<SharedState>,
) -> Result<Json<crate::commands::model::ProcessStatusInfo>, ApiErrorResponse> {
    crate::commands::model::collect_process_status(state)
        .await
        .map(Json)
        .map_err(ApiErrorResponse::service_unavailable)
}

#[derive(Debug, Deserialize)]
pub struct DebugProfileQuery {
    pub model: Option<String>,
}

pub async fn debug_profile(
    State(state): State<SharedState>,
    Query(query): Query<DebugProfileQuery>,
) -> Result<Json<EffectiveProfileInfo>, ApiErrorResponse> {
    let s = state.read().await;
    crate::commands::model::get_effective_profile_for_shared(&s, query.model.as_deref())
        .map(Json)
        .map_err(ApiErrorResponse::bad_request)
}

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

#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    pub name: Option<String>,
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

    let sessions = db
        .list_sessions()
        .map_err(|e| ApiErrorResponse::service_unavailable(format!("Failed to fetch session: {e}")))?;

    sessions
        .into_iter()
        .find(|session| session.id == id)
        .map(Json)
        .ok_or_else(|| ApiErrorResponse::service_unavailable("Session created but not found"))
}

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
        .map_err(|e| ApiErrorResponse::service_unavailable(format!("Failed to delete session: {e}")))?;

    Ok(Json(serde_json::json!({ "deleted": true, "id": session_id })))
}

pub async fn get_session_messages(
    State(state): State<SharedState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<crate::session::db::MessageInfo>>, ApiErrorResponse> {
    let s = state.read().await;
    let db = s
        .session_db
        .lock()
        .map_err(|_| ApiErrorResponse::service_unavailable("Session DB lock poisoned"))?;

    db.get_messages(&session_id).map(Json).map_err(|e| {
        if e.to_string().contains("no rows") {
            ApiErrorResponse(
                StatusCode::NOT_FOUND,
                axum::Json(ApiError {
                    error: ApiErrorBody {
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
