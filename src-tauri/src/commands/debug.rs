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
    } else if method == "GET" && path == "/v1/runtime/doctor" {
        crate::api::extensions::runtime_doctor(AxumState(shared))
            .await
            .into_response()
    } else if method == "GET" && path.starts_with("/v1/debug/profile") {
        let query = path
            .split_once('?')
            .map(|(_, query)| query)
            .unwrap_or_default();
        let model = query.split('&').find_map(|pair| {
            let (key, value) = pair.split_once('=')?;
            if key == "model" {
                Some(percent_decode_str(value).decode_utf8_lossy().into_owned())
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
        crate::api::models::unload_model(AxumState(shared), None)
            .await
            .into_response()
    } else if method == "POST" && path == "/v1/chat/completions" {
        let body = request.body.unwrap_or_else(|| "{}".to_string());
        let mut parsed = serde_json::from_str::<serde_json::Value>(&body)
            .map_err(|e| format!("Invalid JSON for /v1/chat/completions: {e}"))?;
        if let Some(object) = parsed.as_object_mut() {
            object.insert("stream".to_string(), serde_json::Value::Bool(false));
        }
        match crate::api::completions::chat_completions(
            AxumState(shared),
            axum::http::HeaderMap::new(),
            axum::Json(parsed),
        )
        .await
        {
            Ok(response) => response,
            Err(status) => status.into_response(),
        }
    } else if method == "POST" && path == "/v1/responses" {
        let body = request.body.unwrap_or_else(|| "{}".to_string());
        let mut parsed = serde_json::from_str::<serde_json::Value>(&body)
            .map_err(|e| format!("Invalid JSON for /v1/responses: {e}"))?;
        if let Some(object) = parsed.as_object_mut() {
            object.insert("stream".to_string(), serde_json::Value::Bool(false));
        }
        match crate::api::responses::responses(
            AxumState(shared),
            axum::http::HeaderMap::new(),
            axum::Json(parsed),
        )
        .await
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

#[tauri::command]
pub async fn get_runtime_doctor(
    state: tauri::State<'_, SharedState>,
) -> Result<crate::providers::RuntimeDoctorReport, String> {
    Ok(crate::providers::collect_runtime_doctor(state.inner().clone()).await)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemplateDryRunRequest {
    pub model_name: Option<String>,
    pub use_jinja: bool,
    pub template_mode: String,
    pub template_name: Option<String>,
    pub custom_template_path: Option<String>,
    pub chat_template_kwargs_json: Option<String>,
    pub reasoning_mode: String,
    pub parallel_slots: u32,
}

#[derive(Debug, Serialize)]
pub struct TemplateDryRunReport {
    pub model_name: String,
    pub family: String,
    pub renderer: String,
    pub tool_format: String,
    pub prompt: String,
    pub checks: Vec<String>,
    pub warnings: Vec<String>,
}

#[tauri::command]
pub async fn template_dry_run(
    state: tauri::State<'_, SharedState>,
    request: TemplateDryRunRequest,
) -> Result<TemplateDryRunReport, String> {
    let model_name = match request
        .model_name
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(name) => name.to_string(),
        None => {
            let s = state.read().await;
            s.loaded_model
                .clone()
                .or_else(|| {
                    s.last_launch_preview.as_ref().and_then(|preview| {
                        std::path::Path::new(&preview.model_path)
                            .file_name()
                            .map(|name| name.to_string_lossy().to_string())
                    })
                })
                .unwrap_or_else(|| "Qwen3.6-27B-GGUF".to_string())
        }
    };

    let profile = crate::models::profiles::ModelProfile::detect(&model_name);
    let messages = vec![
        crate::templates::engine::ChatMessage {
            role: "system".to_string(),
            content: "You are InferenceBridge's template dry run. Use tools only when needed."
                .to_string(),
        },
        crate::templates::engine::ChatMessage {
            role: "user".to_string(),
            content:
                "Use the get_weather tool for London in celsius. Return exactly one tool call."
                    .to_string(),
        },
    ];
    let prompt = crate::templates::engine::render_prompt_with_tools(&messages, &profile, true);

    let mut checks = Vec::new();
    let mut warnings = Vec::new();
    let lower = model_name.to_ascii_lowercase();
    let is_qwen = lower.contains("qwen");
    let is_gemma = lower.contains("gemma");
    let template_mode = request.template_mode.trim().to_ascii_lowercase();

    if request.use_jinja {
        checks.push("Jinja rendering is enabled for llama.cpp launch.".to_string());
    } else if is_qwen || is_gemma {
        warnings.push(format!(
            "{model_name} looks template-sensitive but Use Jinja is disabled."
        ));
    }

    match template_mode.as_str() {
        "repo" => checks.push("Template mode is repo.".to_string()),
        "builtin" => checks.push(format!(
            "Template mode is builtin{}.",
            request
                .template_name
                .as_deref()
                .map(|name| format!(" ({name})"))
                .unwrap_or_default()
        )),
        "custom" => {
            if let Some(path) = request
                .custom_template_path
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                if std::path::Path::new(path).exists() {
                    checks.push(format!("Custom template exists: {path}."));
                } else {
                    warnings.push(format!("Custom template path does not exist: {path}."));
                }
            } else {
                warnings.push(
                    "Template mode is custom but no custom template path is set.".to_string(),
                );
            }
        }
        "" => {
            warnings.push("Template mode is blank; launch will fall back to defaults.".to_string())
        }
        other => warnings.push(format!(
            "Template mode {other:?} is not one of repo/custom/builtin."
        )),
    }

    if is_qwen {
        let reasoning = request.reasoning_mode.trim();
        if reasoning.is_empty() {
            warnings.push("Qwen launch has no explicit reasoning mode.".to_string());
        } else {
            checks.push(format!("Qwen reasoning mode is {reasoning}."));
        }

        if request.parallel_slots > 1 {
            warnings.push(format!(
                "Qwen tool reliability is usually best with one slot; current setting is {}.",
                request.parallel_slots
            ));
        } else {
            checks.push("Parallel slots set to 1 for Qwen tool reliability.".to_string());
        }

        let kwargs = request
            .chat_template_kwargs_json
            .as_deref()
            .map(str::trim)
            .unwrap_or("");
        if kwargs.is_empty() {
            warnings.push(
                "Qwen launch has no chat_template_kwargs_json; thinking/tool behavior may depend on template defaults."
                    .to_string(),
            );
        } else if serde_json::from_str::<serde_json::Value>(kwargs).is_ok() {
            checks.push("Template kwargs JSON parses successfully.".to_string());
        } else {
            warnings.push("Template kwargs JSON is not valid JSON.".to_string());
        }
    }

    if prompt.contains("<|im_start|>system") || prompt.contains("<|turn>system") {
        checks.push("Rendered prompt contains a system turn marker.".to_string());
    } else {
        warnings
            .push("Rendered prompt did not include a recognizable system turn marker.".to_string());
    }
    if prompt.contains("<|im_start|>assistant") || prompt.contains("<|turn>model") {
        checks.push("Rendered prompt contains an assistant generation marker.".to_string());
    } else {
        warnings.push(
            "Rendered prompt did not include a recognizable assistant generation marker."
                .to_string(),
        );
    }

    Ok(TemplateDryRunReport {
        model_name,
        family: format!("{:?}", profile.family),
        renderer: format!("{:?}", profile.renderer_type),
        tool_format: format!("{:?}", profile.tool_call_format),
        prompt,
        checks,
        warnings,
    })
}
