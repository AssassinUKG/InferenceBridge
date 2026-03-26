//! Tauri commands for debug inspector.

use axum::body::to_bytes;
use axum::extract::Path;
use axum::extract::State as AxumState;
use axum::response::IntoResponse;
use percent_encoding::percent_decode_str;
use serde::{Deserialize, Serialize};

use crate::state::SharedState;

#[tauri::command]
pub async fn get_logs(limit: Option<usize>) -> Result<Vec<crate::logging::LogEntry>, String> {
    Ok(crate::logging::list(limit.unwrap_or(500)))
}

#[tauri::command]
pub async fn clear_logs() -> Result<(), String> {
    crate::logging::clear();
    Ok(())
}

#[derive(Deserialize)]
pub struct DebugApiRequest {
    pub method: String,
    pub path: String,
    pub body: Option<String>,
}

#[derive(Serialize)]
pub struct DebugApiResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
    pub transport: String,
}

#[tauri::command]
pub async fn debug_api_request(
    state: tauri::State<'_, SharedState>,
    request: DebugApiRequest,
) -> Result<DebugApiResponse, String> {
    let shared = state.inner().clone();
    let method = request.method.to_uppercase();
    let path = request.path.trim().to_string();

    let response = if method == "GET" && path == "/v1/health" {
        crate::api::health::health_check(AxumState(shared))
            .await
            .into_response()
    } else if method == "GET" && path == "/v1/models" {
        crate::api::models::list_models(AxumState(shared))
            .await
            .into_response()
    } else if method == "GET" && path == "/v1/models/stats" {
        match crate::api::models::current_model_stats(AxumState(shared)).await {
            Ok(json) => json.into_response(),
            Err(status) => status.into_response(),
        }
    } else if method == "POST" && path == "/v1/models/stats" {
        let body = request.body.unwrap_or_else(|| "{}".to_string());
        let parsed = serde_json::from_str::<crate::api::models::ModelStatsRequest>(&body)
            .map_err(|e| format!("Invalid JSON for /v1/models/stats: {e}"))?;
        match crate::api::models::model_stats(AxumState(shared), axum::Json(parsed)).await {
            Ok(json) => json.into_response(),
            Err(status) => status.into_response(),
        }
    } else if method == "GET" && path == "/v1/context/status" {
        crate::api::extensions::context_status(AxumState(shared))
            .await
            .into_response()
    } else if method == "GET" && path == "/v1/sessions" {
        match crate::api::extensions::list_sessions(AxumState(shared)).await {
            Ok(json) => json.into_response(),
            Err(status) => status.into_response(),
        }
    } else if method == "POST" && path == "/v1/models/load" {
        let body = request.body.unwrap_or_else(|| "{}".to_string());
        let parsed = serde_json::from_str::<crate::api::models::LoadModelRequest>(&body)
            .map_err(|e| format!("Invalid JSON for /v1/models/load: {e}"))?;
        crate::api::models::load_model(AxumState(shared), axum::Json(parsed))
            .await
            .into_response()
    } else if method == "POST" && path == "/v1/models/unload" {
        crate::api::models::unload_model(AxumState(shared))
            .await
            .into_response()
    } else if method == "POST" && path == "/v1/chat/completions" {
        let body = request.body.unwrap_or_else(|| "{}".to_string());
        let mut parsed =
            serde_json::from_str::<crate::api::completions::ChatCompletionRequest>(&body)
                .map_err(|e| format!("Invalid JSON for /v1/chat/completions: {e}"))?;
        if parsed.stream {
            parsed.stream = false;
        }
        match crate::api::completions::chat_completions(AxumState(shared), axum::Json(parsed)).await
        {
            Ok(response) => response,
            Err(status) => status.into_response(),
        }
    } else if method == "GET" {
        if let Some(model_name) = path.strip_prefix("/v1/models/") {
            if !model_name.is_empty() && !model_name.contains('/') {
                let decoded_name = percent_decode_str(model_name)
                    .decode_utf8_lossy()
                    .into_owned();
                match crate::api::models::get_model(AxumState(shared), Path(decoded_name)).await {
                    Ok(json) => json.into_response(),
                    Err(status) => status.into_response(),
                }
            } else {
                return Ok(DebugApiResponse {
                    status: 404,
                    headers: vec![(
                        "content-type".to_string(),
                        "text/plain; charset=utf-8".to_string(),
                    )],
                    body: format!("No internal debug route for {method} {path}"),
                    transport: "direct".to_string(),
                });
            }
        } else {
            return Ok(DebugApiResponse {
                status: 404,
                headers: vec![(
                    "content-type".to_string(),
                    "text/plain; charset=utf-8".to_string(),
                )],
                body: format!("No internal debug route for {method} {path}"),
                transport: "direct".to_string(),
            });
        }
    } else {
        return Ok(DebugApiResponse {
            status: 404,
            headers: vec![(
                "content-type".to_string(),
                "text/plain; charset=utf-8".to_string(),
            )],
            body: format!("No internal debug route for {method} {path}"),
            transport: "direct".to_string(),
        });
    };

    let status = response.status().as_u16();
    let headers = response
        .headers()
        .iter()
        .map(
            |(key, value): (&axum::http::header::HeaderName, &axum::http::HeaderValue)| {
                (
                    key.as_str().to_string(),
                    value.to_str().unwrap_or_default().to_string(),
                )
            },
        )
        .collect::<Vec<_>>();
    let bytes = to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .map_err(|e| format!("Failed to read direct response body: {e}"))?;
    let body = String::from_utf8_lossy(&bytes).to_string();

    Ok(DebugApiResponse {
        status,
        headers,
        body,
        transport: "direct".to_string(),
    })
}

#[tauri::command]
pub async fn get_raw_prompt() -> Result<String, String> {
    // TODO: store last prompt sent to llama-server
    Ok("No prompt captured yet".to_string())
}

#[tauri::command]
pub async fn get_parse_trace() -> Result<String, String> {
    // TODO: store last normalization pipeline trace
    Ok("No parse trace captured yet".to_string())
}
