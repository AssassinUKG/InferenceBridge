//! POST /v1/chat/completions — OpenAI-compatible chat completions endpoint.
//!
//! Supports automatic model swapping: if the `model` field in the request differs
//! from the currently loaded model, the server will unload the current model and
//! load the requested one before serving the request.

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::api::errors::ApiErrorResponse;
use crate::engine::client::{CompletionRequest, LlamaClient};
use crate::models::profiles::ModelProfile;
use crate::state::SharedState;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ApiMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub stop: Option<StopParam>,
    pub tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: Option<String>,
}

/// OpenAI `stop` can be a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopParam {
    Single(String),
    Multiple(Vec<String>),
}

impl StopParam {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopParam::Single(s) => vec![s],
            StopParam::Multiple(v) => v,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiErrorResponse> {
    // Determine which model is needed
    let requested_model = req.model.as_deref().unwrap_or("");

    // Auto-swap: if a model is specified and it's not the currently loaded one, swap
    let model_name = {
        let needs_swap = {
            let s = state.read().await;
            if requested_model.is_empty() {
                // No model specified — use whatever is loaded
                if s.loaded_model.is_none() {
                    return Err(ApiErrorResponse::no_model());
                }
                false
            } else {
                match &s.loaded_model {
                    Some(loaded) => {
                        // Case-insensitive substring match (same as registry.find_by_name)
                        !loaded
                            .to_lowercase()
                            .contains(&requested_model.to_lowercase())
                    }
                    None => true, // Nothing loaded, need to load
                }
            }
        };

        if needs_swap {
            swap_model_for_api(&state, requested_model)
                .await
                .map_err(|e| {
                    tracing::error!(error = %e, model = %requested_model, "Model swap failed");
                    ApiErrorResponse::service_unavailable(format!(
                        "Could not load model '{requested_model}': {e}"
                    ))
                })?;
        }

        let s = state.read().await;
        s.loaded_model
            .clone()
            .ok_or_else(ApiErrorResponse::no_model)?
    };

    let s = state.read().await;
    let profile = ModelProfile::detect(&model_name);

    // Server-level sampling defaults (set via --temperature etc. in headless mode
    // or via inference-bridge.toml [server] section). Request params > profile defaults
    // > server defaults.
    let srv_temp = s.config.server.default_temperature;
    let srv_top_p = s.config.server.default_top_p;
    let srv_top_k = s.config.server.default_top_k;
    let srv_max_tokens = s.config.server.default_max_tokens;

    // Convert API messages to ChatMessages
    let messages: Vec<crate::templates::engine::ChatMessage> = req
        .messages
        .iter()
        .map(|m| crate::templates::engine::ChatMessage {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();

    let prompt = crate::templates::engine::render_prompt(&messages, &profile);

    // Merge user-provided stop sequences with profile defaults
    let mut stop = profile.stop_markers.clone();
    if let Some(user_stop) = req.stop {
        stop.extend(user_stop.into_vec());
    }

    let request = CompletionRequest {
        prompt,
        n_predict: req
            .max_tokens
            .or(profile.default_max_output_tokens)
            .or(srv_max_tokens)
            .map(|t| t as i32),
        temperature: req.temperature.or(profile.default_temperature).or(srv_temp),
        top_p: req.top_p.or(profile.default_top_p).or(srv_top_p),
        top_k: req.top_k.or(profile.default_top_k).or(srv_top_k),
        min_p: profile.default_min_p,
        presence_penalty: req.presence_penalty.or(profile.default_presence_penalty),
        frequency_penalty: req.frequency_penalty,
        repeat_penalty: req.repetition_penalty,
        seed: req.seed,
        stream: req.stream,
        stop,
        special: true,
        image_data: vec![],
    };

    let client = LlamaClient::new(s.process.port());
    drop(s);

    if req.stream {
        return stream_chat_completion(client, request, model_name).await;
    }

    // Non-streaming path
    let response = client.complete(&request).await.map_err(|e| {
        tracing::error!(error = %e, "Completion failed");
        ApiErrorResponse::inference_failed(&e.to_string())
    })?;

    // Normalize output
    let stripped = crate::normalize::think_strip::strip_think_tags(&response.content);
    let (tool_calls, text) = crate::normalize::tool_extract::extract_tool_calls(&stripped);

    let content = if text.is_empty() { None } else { Some(text) };
    let api_tool_calls: Vec<serde_json::Value> = tool_calls
        .iter()
        .map(|tc| {
            serde_json::json!({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default()
                }
            })
        })
        .collect();

    let finish_reason = if !tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: model_name.clone(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content,
                tool_calls: api_tool_calls,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: response.tokens_evaluated.unwrap_or(0),
            completion_tokens: response.tokens_predicted.unwrap_or(0),
            total_tokens: response.tokens_evaluated.unwrap_or(0)
                + response.tokens_predicted.unwrap_or(0),
        },
    })
    .into_response())
}

/// SSE streaming for /v1/chat/completions (OpenAI-compatible format).
async fn stream_chat_completion(
    client: LlamaClient,
    request: CompletionRequest,
    model_name: String,
) -> Result<Response, ApiErrorResponse> {
    let response = client.complete_stream(&request).await.map_err(|e| {
        tracing::error!(error = %e, "Stream completion failed");
        ApiErrorResponse::inference_failed(&e.to_string())
    })?;

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let stream = async_stream::stream! {
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "SSE chunk error");
                    break;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        yield Ok::<Event, std::convert::Infallible>(
                            Event::default().data("[DONE]")
                        );
                        return;
                    }

                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        let content = json["content"].as_str().unwrap_or("");
                        let is_stop = json["stop"].as_bool().unwrap_or(false);

                        let finish_reason = if is_stop {
                            Some("stop".to_string())
                        } else {
                            None
                        };

                        let chunk_json = serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": content
                                },
                                "finish_reason": finish_reason
                            }]
                        });

                        yield Ok(Event::default().data(chunk_json.to_string()));

                        if is_stop {
                            yield Ok(Event::default().data("[DONE]"));
                            return;
                        }
                    }
                }
            }
        }

        // Stream ended — send final DONE
        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// Swap to a different model for API requests.
/// Delegates to `backend_load_model` which handles health-checking and
/// emits `model-load-progress` GUI events when an app_handle is present.
async fn swap_model_for_api(state: &SharedState, model_name: &str) -> Result<(), String> {
    tracing::info!(model = %model_name, "API: auto-swapping model");
    crate::commands::model::backend_load_model(state.clone(), model_name.to_string(), None)
        .await
        .map(|_| ())
}

// ─── POST /v1/completions  (OpenAI text completion — non-chat) ────────────────

#[derive(Debug, serde::Deserialize)]
pub struct TextCompletionRequest {
    pub model: Option<String>,
    /// Prompt string (plain text, not chat-formatted).
    pub prompt: Option<String>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub seed: Option<i64>,
    pub stop: Option<StopParam>,
}

#[derive(Debug, serde::Serialize)]
pub struct TextCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
}

#[derive(Debug, serde::Serialize)]
pub struct TextChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

pub async fn text_completions(
    axum::extract::State(state): axum::extract::State<SharedState>,
    axum::Json(req): axum::Json<TextCompletionRequest>,
) -> Result<axum::response::Response, ApiErrorResponse> {
    let requested_model = req.model.as_deref().unwrap_or("");

    // Auto-swap model if needed.
    {
        let needs_swap = {
            let s = state.read().await;
            if requested_model.is_empty() {
                if s.loaded_model.is_none() {
                    return Err(ApiErrorResponse::no_model());
                }
                false
            } else {
                match &s.loaded_model {
                    Some(loaded) => !loaded
                        .to_lowercase()
                        .contains(&requested_model.to_lowercase()),
                    None => true,
                }
            }
        };
        if needs_swap {
            swap_model_for_api(&state, requested_model)
                .await
                .map_err(|e| {
                    tracing::error!(error = %e, "Text completion model swap failed");
                    ApiErrorResponse::service_unavailable(format!(
                        "Could not load model '{requested_model}': {e}"
                    ))
                })?;
        }
    }

    let s = state.read().await;
    let model_name = s
        .loaded_model
        .clone()
        .ok_or_else(ApiErrorResponse::no_model)?;
    let profile = ModelProfile::detect(&model_name);

    let srv_temp = s.config.server.default_temperature;
    let srv_top_p = s.config.server.default_top_p;
    let srv_top_k = s.config.server.default_top_k;
    let srv_max_tokens = s.config.server.default_max_tokens;

    let prompt = req.prompt.unwrap_or_default();
    if prompt.is_empty() {
        return Err(ApiErrorResponse::bad_request(
            "The `prompt` field is required and must not be empty.",
        ));
    }

    let mut stop = profile.stop_markers.clone();
    if let Some(user_stop) = req.stop {
        stop.extend(user_stop.into_vec());
    }

    let completion_req = CompletionRequest {
        prompt,
        n_predict: req
            .max_tokens
            .or(srv_max_tokens)
            .or(profile.default_max_output_tokens)
            .map(|t| t as i32),
        temperature: req.temperature.or(srv_temp).or(profile.default_temperature),
        top_p: req.top_p.or(srv_top_p).or(profile.default_top_p),
        top_k: req.top_k.or(srv_top_k).or(profile.default_top_k),
        min_p: profile.default_min_p,
        presence_penalty: profile.default_presence_penalty,
        frequency_penalty: None,
        repeat_penalty: None,
        seed: req.seed,
        stream: false, // text completions: non-streaming for now
        stop,
        special: true,
        image_data: vec![],
    };

    let client = LlamaClient::new(s.process.port());
    drop(s);

    let response = client.complete(&completion_req).await.map_err(|e| {
        tracing::error!(error = %e, "Text completion failed");
        ApiErrorResponse::inference_failed(&e.to_string())
    })?;

    let text = crate::normalize::think_strip::strip_think_tags(&response.content);

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(axum::Json(TextCompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created,
        model: model_name,
        choices: vec![TextChoice {
            index: 0,
            text,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: response.tokens_evaluated.unwrap_or(0),
            completion_tokens: response.tokens_predicted.unwrap_or(0),
            total_tokens: response.tokens_evaluated.unwrap_or(0)
                + response.tokens_predicted.unwrap_or(0),
        },
    })
    .into_response())
}
