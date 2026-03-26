use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use crate::api::errors::ApiErrorResponse;
use crate::engine::client::{CompletionRequest, LlamaClient};
use crate::engine::streaming::{self, StreamEvent};
use crate::models::profiles::ModelProfile;
use crate::state::{GenerationRequest, SharedState};

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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopParam {
    Single(String),
    Multiple(Vec<String>),
}

impl StopParam {
    fn into_vec(self) -> Vec<String> {
        match self {
            StopParam::Single(value) => vec![value],
            StopParam::Multiple(values) => values,
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

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn build_parse_trace(raw: &str, stripped: &str) -> String {
    let (tool_calls, visible_text) = crate::normalize::tool_extract::extract_tool_calls(stripped);
    serde_json::to_string_pretty(&serde_json::json!({
        "raw_response": raw,
        "stripped_response": stripped,
        "visible_text": visible_text,
        "tool_calls": tool_calls,
    }))
    .unwrap_or_else(|_| "Failed to serialize parse trace".to_string())
}

async fn begin_api_generation(state: &SharedState, model: String) -> tokio_util::sync::CancellationToken {
    let mut s = state.write().await;
    s.generation_cancel.cancel();
    s.generation_cancel = tokio_util::sync::CancellationToken::new();
    s.active_generation = Some(GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        source: "api".to_string(),
        session_id: None,
        model,
        started_at: chrono::Utc::now().to_rfc3339(),
        status: "running".to_string(),
    });
    s.generation_cancel.clone()
}

async fn finish_api_generation(state: &SharedState, status: &str) {
    let mut s = state.write().await;
    if let Some(active) = s.active_generation.as_mut() {
        active.status = status.to_string();
    }
    s.active_generation = None;
}

async fn swap_model_for_api(state: &SharedState, model_name: &str) -> Result<(), String> {
    crate::commands::model::backend_load_model(state.clone(), model_name.to_string(), None)
        .await
        .map(|_| ())
}

async fn resolve_loaded_model(
    state: &SharedState,
    requested_model: &str,
) -> Result<String, ApiErrorResponse> {
    let needs_swap = {
        let s = state.read().await;
        if requested_model.is_empty() {
            if s.loaded_model.is_none() {
                return Err(ApiErrorResponse::no_model());
            }
            false
        } else {
            match &s.loaded_model {
                Some(loaded) => !loaded.to_lowercase().contains(&requested_model.to_lowercase()),
                None => true,
            }
        }
    };

    if needs_swap {
        swap_model_for_api(state, requested_model).await.map_err(|e| {
            ApiErrorResponse::service_unavailable(format!(
                "Could not load model '{requested_model}': {e}"
            ))
        })?;
    }

    let s = state.read().await;
    s.loaded_model.clone().ok_or_else(ApiErrorResponse::no_model)
}

fn build_chat_request(
    profile: &ModelProfile,
    req: ChatCompletionRequest,
    server_defaults: (&Option<f32>, &Option<f32>, &Option<i32>, &Option<u32>),
) -> CompletionRequest {
    let messages: Vec<crate::templates::engine::ChatMessage> = req
        .messages
        .iter()
        .map(|message| crate::templates::engine::ChatMessage {
            role: message.role.clone(),
            content: message.content.clone().unwrap_or_default(),
        })
        .collect();

    let mut stop = profile.stop_markers.clone();
    if let Some(user_stop) = req.stop {
        stop.extend(user_stop.into_vec());
    }

    CompletionRequest {
        prompt: crate::templates::engine::render_prompt(&messages, profile),
        n_predict: req
            .max_tokens
            .or(profile.default_max_output_tokens)
            .or(*server_defaults.3)
            .map(|value| value as i32),
        temperature: req
            .temperature
            .or(profile.default_temperature)
            .or(*server_defaults.0),
        top_p: req.top_p.or(profile.default_top_p).or(*server_defaults.1),
        top_k: req.top_k.or(profile.default_top_k).or(*server_defaults.2),
        min_p: profile.default_min_p,
        presence_penalty: req.presence_penalty.or(profile.default_presence_penalty),
        frequency_penalty: req.frequency_penalty,
        repeat_penalty: req.repetition_penalty,
        seed: req.seed,
        stream: req.stream,
        stop,
        special: true,
        image_data: vec![],
    }
}

pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiErrorResponse> {
    let requested_model = req.model.clone().unwrap_or_default();
    let model_name = resolve_loaded_model(&state, &requested_model).await?;
    let profile = crate::models::overrides::detect_effective_profile(&model_name);

    let server_defaults = {
        let s = state.read().await;
        (
            s.config.server.default_temperature,
            s.config.server.default_top_p,
            s.config.server.default_top_k,
            s.config.server.default_max_tokens,
            s.process.port(),
        )
    };

    let request = build_chat_request(
        &profile,
        req,
        (
            &server_defaults.0,
            &server_defaults.1,
            &server_defaults.2,
            &server_defaults.3,
        ),
    );

    {
        let mut s = state.write().await;
        s.last_prompt = Some(request.prompt.clone());
    }

    let client = LlamaClient::new(server_defaults.4);

    if request.stream {
        return stream_chat_completion(state, client, request, model_name).await;
    }

    let response = client.complete(&request).await.map_err(|e| {
        tracing::error!(error = %e, "Completion failed");
        ApiErrorResponse::inference_failed(&e.to_string())
    })?;

    let stripped = crate::normalize::think_strip::strip_think_tags(&response.content);
    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(build_parse_trace(&response.content, &stripped));
    }
    let (tool_calls, text) = crate::normalize::tool_extract::extract_tool_calls(&stripped);
    let content = if text.is_empty() { None } else { Some(text) };
    let api_tool_calls: Vec<serde_json::Value> = tool_calls
        .iter()
        .map(|tool_call| {
            serde_json::json!({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": serde_json::to_string(&tool_call.arguments).unwrap_or_default()
                }
            })
        })
        .collect();

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: now_unix_secs(),
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content,
                tool_calls: api_tool_calls.clone(),
            },
            finish_reason: if api_tool_calls.is_empty() {
                "stop".to_string()
            } else {
                "tool_calls".to_string()
            },
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

async fn stream_chat_completion(
    state: SharedState,
    client: LlamaClient,
    request: CompletionRequest,
    model_name: String,
) -> Result<Response, ApiErrorResponse> {
    let response = client.complete_stream(&request).await.map_err(|e| {
        tracing::error!(error = %e, "Stream completion failed");
        ApiErrorResponse::inference_failed(&e.to_string())
    })?;

    let cancel = begin_api_generation(&state, model_name.clone()).await;
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let _ = streaming::consume_sse_stream(response, tx, cancel).await;
    });

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = now_unix_secs();
    let state_for_stream = state.clone();

    let stream = async_stream::stream! {
        let mut raw_full_text = String::new();

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(token) => {
                    let chunk_json = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": token },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    yield Ok::<Event, std::convert::Infallible>(Event::default().data(chunk_json.to_string()));
                }
                StreamEvent::ReasoningDelta(reasoning) => {
                    raw_full_text.push_str("<think>");
                    raw_full_text.push_str(&reasoning);
                    raw_full_text.push_str("</think>");
                }
                StreamEvent::Done {
                    full_text,
                    tokens_predicted,
                    tokens_evaluated,
                    ..
                } => {
                    let stripped = crate::normalize::think_strip::strip_think_tags(&full_text);
                    let mut s = state_for_stream.write().await;
                    s.last_parse_trace = Some(build_parse_trace(&full_text, &stripped));
                    drop(s);

                    let final_chunk = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": tokens_evaluated,
                            "completion_tokens": tokens_predicted,
                            "total_tokens": tokens_evaluated + tokens_predicted,
                        }
                    });
                    yield Ok(Event::default().data(final_chunk.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    finish_api_generation(&state_for_stream, "completed").await;
                    return;
                }
                StreamEvent::Error(error) => {
                    finish_api_generation(&state_for_stream, "error").await;
                    let error_chunk = serde_json::json!({
                        "error": {
                            "message": error,
                            "type": "server_error"
                        }
                    });
                    yield Ok(Event::default().data(error_chunk.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }

        if !raw_full_text.is_empty() {
            let stripped = crate::normalize::think_strip::strip_think_tags(&raw_full_text);
            let mut s = state_for_stream.write().await;
            s.last_parse_trace = Some(build_parse_trace(&raw_full_text, &stripped));
        }
        finish_api_generation(&state_for_stream, "completed").await;
        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[derive(Debug, Deserialize)]
pub struct TextCompletionRequest {
    pub model: Option<String>,
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

#[derive(Debug, Serialize)]
pub struct TextCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<TextChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct TextChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

pub async fn text_completions(
    State(state): State<SharedState>,
    Json(req): Json<TextCompletionRequest>,
) -> Result<Response, ApiErrorResponse> {
    let requested_model = req.model.clone().unwrap_or_default();
    let model_name = resolve_loaded_model(&state, &requested_model).await?;
    let profile = crate::models::overrides::detect_effective_profile(&model_name);

    let prompt = req.prompt.unwrap_or_default();
    if prompt.is_empty() {
        return Err(ApiErrorResponse::bad_request(
            "The `prompt` field is required and must not be empty.",
        ));
    }

    let (srv_temp, srv_top_p, srv_top_k, srv_max_tokens, port) = {
        let s = state.read().await;
        (
            s.config.server.default_temperature,
            s.config.server.default_top_p,
            s.config.server.default_top_k,
            s.config.server.default_max_tokens,
            s.process.port(),
        )
    };

    let mut stop = profile.stop_markers.clone();
    if let Some(user_stop) = req.stop {
        stop.extend(user_stop.into_vec());
    }

    let completion_req = CompletionRequest {
        prompt: prompt.clone(),
        n_predict: req
            .max_tokens
            .or(srv_max_tokens)
            .or(profile.default_max_output_tokens)
            .map(|value| value as i32),
        temperature: req.temperature.or(srv_temp).or(profile.default_temperature),
        top_p: req.top_p.or(srv_top_p).or(profile.default_top_p),
        top_k: req.top_k.or(srv_top_k).or(profile.default_top_k),
        min_p: profile.default_min_p,
        presence_penalty: profile.default_presence_penalty,
        frequency_penalty: None,
        repeat_penalty: None,
        seed: req.seed,
        stream: false,
        stop,
        special: true,
        image_data: vec![],
    };

    {
        let mut s = state.write().await;
        s.last_prompt = Some(prompt);
    }

    let client = LlamaClient::new(port);
    let response = client.complete(&completion_req).await.map_err(|e| {
        tracing::error!(error = %e, "Text completion failed");
        ApiErrorResponse::inference_failed(&e.to_string())
    })?;

    let text = crate::normalize::think_strip::strip_think_tags(&response.content);
    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(build_parse_trace(&response.content, &text));
    }

    Ok(Json(TextCompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: now_unix_secs(),
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
