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
    } else if method == "GET" && path == "/v1/runtime/status" {
        match crate::api::extensions::runtime_status(AxumState(shared)).await {
            Ok(json) => json.into_response(),
            Err(status) => status.into_response(),
        }
    } else if method == "GET" && path.starts_with("/v1/debug/profile") {
        let query = path
            .split_once('?')
            .map(|(_, query)| query)
            .unwrap_or_default();
        let model = query
            .split('&')
            .find_map(|pair| {
                let (key, value) = pair.split_once('=')?;
                if key == "model" {
                    Some(
                        percent_decode_str(value)
                            .decode_utf8_lossy()
                            .into_owned(),
                    )
                } else {
                    None
                }
            });
        match crate::api::extensions::debug_profile(
            AxumState(shared),
            axum::extract::Query(crate::api::extensions::DebugProfileQuery { model }),
        )
        .await
        {
            Ok(json) => json.into_response(),
            Err(status) => status.into_response(),
        }
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
                not_found_response(&method, &path)
            }
        } else {
            not_found_response(&method, &path)
        }
    } else {
        not_found_response(&method, &path)
    };

    let status = response.status().as_u16();
    let headers = response
        .headers()
        .iter()
        .map(|(key, value)| {
            (
                key.as_str().to_string(),
                value.to_str().unwrap_or_default().to_string(),
            )
        })
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

fn not_found_response(method: &str, path: &str) -> axum::response::Response {
    (
        axum::http::StatusCode::NOT_FOUND,
        [("content-type", "text/plain; charset=utf-8")],
        format!("No internal debug route for {method} {path}"),
    )
        .into_response()
}

#[tauri::command]
pub async fn get_raw_prompt(state: tauri::State<'_, SharedState>) -> Result<String, String> {
    let s = state.read().await;
    Ok(s.last_prompt
        .clone()
        .unwrap_or_else(|| "No prompt captured yet".to_string()))
}

#[tauri::command]
pub async fn get_parse_trace(state: tauri::State<'_, SharedState>) -> Result<String, String> {
    let s = state.read().await;
    Ok(s.last_parse_trace
        .clone()
        .unwrap_or_else(|| "No parse trace captured yet".to_string()))
}

#[tauri::command]
pub async fn get_launch_preview(state: tauri::State<'_, SharedState>) -> Result<String, String> {
    let s = state.read().await;
    let preview = s
        .last_launch_preview
        .clone()
        .ok_or_else(|| "No launch preview captured yet".to_string())?;
    serde_json::to_string_pretty(&preview).map_err(|e| e.to_string())
}
