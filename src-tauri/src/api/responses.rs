use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::api::completions::{
    ApiMessage, ChatCompletionRequest, StopParam, TopParam, build_chat_request, build_parse_trace,
    end_to_end_tokens_per_second, ensure_runtime_vision_ready,
    extract_context_size_from_hash_map, resolve_loaded_model,
};
use crate::api::errors::ApiErrorResponse;
use crate::engine::client::LlamaClient;
use crate::engine::streaming::{self, StreamEvent};
use crate::normalize::think_strip::{
    estimate_token_count, extract_reasoning_content_with_style, strip_think_tags_with_style,
};
use crate::state::{SharedState, begin_api_generation, finish_api_generation, summarize_reasoning_tokens};

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Messages(Vec<ApiMessage>),
}

#[derive(Debug, Deserialize, Clone)]
pub struct ResponseReasoningConfig {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    pub input: ResponseInput,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, alias = "maxOutputTokens")]
    pub max_output_tokens: Option<u32>,
    #[serde(default, alias = "temp")]
    pub temperature: Option<f32>,
    #[serde(default, alias = "topP")]
    pub top_p: Option<f32>,
    #[serde(default, alias = "topK")]
    pub top_k: Option<i32>,
    #[serde(default, alias = "minP")]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub top: Option<TopParam>,
    #[serde(default, alias = "presencePenalty")]
    pub presence_penalty: Option<f32>,
    #[serde(default, alias = "frequencyPenalty")]
    pub frequency_penalty: Option<f32>,
    #[serde(
        default,
        alias = "repetitionPenalty",
        alias = "repeatPenalty",
        alias = "repeat_penalty"
    )]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(
        default,
        alias = "contextLength",
        alias = "context_length",
        alias = "contextlength",
        alias = "context_size",
        alias = "ctx_size",
        alias = "n_ctx",
        alias = "maxContextLength"
    )]
    pub context_size: Option<u32>,
    #[serde(default)]
    pub stop: Option<StopParam>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub reasoning: Option<ResponseReasoningConfig>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ResponsesRequest {
    fn requested_context_size(&self) -> Option<u32> {
        self.context_size
            .filter(|value| *value > 0)
            .or_else(|| extract_context_size_from_hash_map(&self.extra))
    }
}

#[derive(Debug, Serialize)]
pub struct ResponsesUsageInputDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ResponsesUsageOutputDetails {
    pub reasoning_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub input_tokens_details: ResponsesUsageInputDetails,
    pub output_tokens_details: ResponsesUsageOutputDetails,
}

#[derive(Debug, Serialize)]
pub struct ResponsesOutputText {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    pub annotations: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesOutputReasoning {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct ResponsesOutputMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub status: &'static str,
    pub role: &'static str,
    pub content: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub status: &'static str,
    pub model: String,
    pub output: Vec<ResponsesOutputMessage>,
    pub usage: ResponsesUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn into_chat_request(request: ResponsesRequest) -> ChatCompletionRequest {
    let requested_context_size = request.requested_context_size();
    let messages = match request.input {
        ResponseInput::Text(text) => vec![ApiMessage {
            role: "user".to_string(),
            content: Some(crate::api::completions::ApiMessageContent::Text(text)),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            refusal: None,
        }],
        ResponseInput::Messages(messages) => messages,
    };

    ChatCompletionRequest {
        model: request.model,
        messages,
        stream: request.stream,
        max_tokens: request.max_output_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        min_p: request.min_p,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        repetition_penalty: request.repetition_penalty,
        seed: request.seed,
        stop: request.stop,
        tools: request.tools,
        context_size: requested_context_size,
        top: request.top,
        reasoning: request.reasoning.map(|reasoning| crate::api::completions::ReasoningRequest {
            effort: reasoning.effort,
            max_tokens: reasoning.max_tokens,
        }),
        reasoning_effort: request.reasoning_effort,
        reasoning_tokens: request.reasoning_tokens,
        stream_options: None,
        extra: request.extra,
    }
}

fn build_response_output(visible_text: String, reasoning_text: String) -> Vec<serde_json::Value> {
    let mut content = Vec::new();
    if !reasoning_text.is_empty() {
        content.push(serde_json::to_value(ResponsesOutputReasoning {
            kind: "reasoning",
            text: reasoning_text,
        }).unwrap_or_default());
    }
    if !visible_text.is_empty() {
        content.push(serde_json::to_value(ResponsesOutputText {
            kind: "output_text",
            text: visible_text,
            annotations: Vec::new(),
        }).unwrap_or_default());
    }
    content
}

fn build_usage(
    prompt_tokens: u32,
    completion_tokens: u32,
    reasoning_tokens: u32,
) -> ResponsesUsage {
    ResponsesUsage {
        input_tokens: prompt_tokens,
        output_tokens: completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        input_tokens_details: ResponsesUsageInputDetails { cached_tokens: 0 },
        output_tokens_details: ResponsesUsageOutputDetails { reasoning_tokens },
    }
}

pub async fn responses(
    State(state): State<SharedState>,
    Json(req): Json<ResponsesRequest>,
) -> Result<Response, ApiErrorResponse> {
    let previous_response_id = req.previous_response_id.clone();
    let chat_request = into_chat_request(req);
    let requested_context_size = chat_request.requested_context_size();
    let requested_model = chat_request.model.clone().unwrap_or_default();
    let model_name =
        resolve_loaded_model(&state, &requested_model, requested_context_size).await?;
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
        chat_request,
        (
            &server_defaults.0,
            &server_defaults.1,
            &server_defaults.2,
            &server_defaults.3,
        ),
    )
    .await?;

    ensure_runtime_vision_ready(&state, &model_name, &profile, !request.image_data.is_empty())
        .await?;

    {
        let mut s = state.write().await;
        s.last_prompt = Some(request.prompt.clone());
    }

    let scheduler = {
        let s = state.read().await;
        s.request_scheduler.clone()
    };
    let _permit = scheduler.acquire().await;

    let client = LlamaClient::new(server_defaults.4);
    let generation_started_at = chrono::Utc::now().to_rfc3339();
    let generation_started = std::time::Instant::now();

    if request.stream {
        let response = client.complete_stream(&request).await.map_err(|e| {
            tracing::error!(error = %e, "Responses stream completion failed");
            ApiErrorResponse::inference_failed(&e.to_string())
        })?;

        let cancel = begin_api_generation(&state, model_name.clone()).await;
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            let _ = streaming::consume_sse_stream(response, tx, cancel).await;
        });

        let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let created_at = now_unix_secs();
        let state_for_stream = state.clone();

        let stream = async_stream::stream! {
            let mut full_text = String::new();
            let opening = serde_json::json!({
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "in_progress",
                    "model": model_name,
                    "output": []
                }
            });
            yield Ok::<Event, std::convert::Infallible>(Event::default().data(opening.to_string()));

            while let Some(event) = rx.recv().await {
                match event {
                    StreamEvent::Token(token) => {
                        full_text.push_str(&token);
                        let chunk = serde_json::json!({
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "item_id": message_id,
                            "delta": token
                        });
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                    StreamEvent::ReasoningDelta(reasoning) => {
                        let chunk = serde_json::json!({
                            "type": "response.reasoning.delta",
                            "response_id": response_id,
                            "item_id": message_id,
                            "delta": reasoning
                        });
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                    StreamEvent::Done {
                        full_text: text,
                        tokens_predicted,
                        tokens_evaluated,
                        decode_tokens_per_second,
                        prompt_tokens_per_second,
                    } => {
                        let visible = strip_think_tags_with_style(&text, profile.think_tag_style);
                        let reasoning = extract_reasoning_content_with_style(&text, profile.think_tag_style);
                        let reasoning_tokens =
                            summarize_reasoning_tokens(Some(tokens_predicted), &visible, &reasoning);

                        let mut s = state_for_stream.write().await;
                        s.last_parse_trace = Some(build_parse_trace(&profile, &text, &visible));
                        s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
                            source: "responses-api".to_string(),
                            model: model_name.clone(),
                            started_at: generation_started_at.clone(),
                            finished_at: chrono::Utc::now().to_rfc3339(),
                            elapsed_ms: generation_started.elapsed().as_millis() as u64,
                            prompt_tokens: Some(tokens_evaluated),
                            completion_tokens: Some(tokens_predicted),
                            total_tokens: Some(tokens_evaluated + tokens_predicted),
                            prompt_tokens_per_second,
                            decode_tokens_per_second: Some(decode_tokens_per_second),
                            end_to_end_tokens_per_second: end_to_end_tokens_per_second(
                                Some(tokens_predicted),
                                generation_started.elapsed().as_millis() as u64,
                            ),
                        });
                        drop(s);

                        let completed = serde_json::json!({
                            "type": "response.completed",
                            "response": {
                                "id": response_id,
                                "object": "response",
                                "created_at": created_at,
                                "status": "completed",
                                "model": model_name,
                                "output": [{
                                    "id": message_id,
                                    "type": "message",
                                    "status": "completed",
                                    "role": "assistant",
                                    "content": build_response_output(visible, reasoning)
                                }],
                                "usage": build_usage(tokens_evaluated, tokens_predicted, reasoning_tokens),
                                "previous_response_id": previous_response_id
                            }
                        });
                        yield Ok(Event::default().data(completed.to_string()));
                        yield Ok(Event::default().data("[DONE]"));
                        finish_api_generation(&state_for_stream, "completed").await;
                        return;
                    }
                    StreamEvent::Error(error) => {
                        finish_api_generation(&state_for_stream, "error").await;
                        let error_chunk = serde_json::json!({
                            "type": "response.error",
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
        };

        return Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response());
    }

    let response = client.complete(&request).await.map_err(|e| {
        tracing::error!(error = %e, "Responses completion failed");
        ApiErrorResponse::inference_failed(&e.to_string())
    })?;

    let visible_text = strip_think_tags_with_style(&response.content, profile.think_tag_style);
    let reasoning_text = extract_reasoning_content_with_style(&response.content, profile.think_tag_style);
    let reasoning_tokens = summarize_reasoning_tokens(
        response.tokens_predicted.or_else(|| Some(estimate_token_count(&response.content))),
        &visible_text,
        &reasoning_text,
    );

    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(build_parse_trace(&profile, &response.content, &visible_text));
        s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
            source: "responses-api".to_string(),
            model: model_name.clone(),
            started_at: generation_started_at,
            finished_at: chrono::Utc::now().to_rfc3339(),
            elapsed_ms: generation_started.elapsed().as_millis() as u64,
            prompt_tokens: response.tokens_evaluated,
            completion_tokens: response.tokens_predicted,
            total_tokens: match (response.tokens_evaluated, response.tokens_predicted) {
                (Some(prompt), Some(completion)) => Some(prompt + completion),
                _ => None,
            },
            prompt_tokens_per_second: response.timings.as_ref().and_then(|timings| timings.prompt_per_second),
            decode_tokens_per_second: response.timings.as_ref().and_then(|timings| timings.predicted_per_second),
            end_to_end_tokens_per_second: end_to_end_tokens_per_second(
                response.tokens_predicted,
                generation_started.elapsed().as_millis() as u64,
            ),
        });
    }

    let output = vec![ResponsesOutputMessage {
        id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        kind: "message",
        status: "completed",
        role: "assistant",
        content: build_response_output(visible_text, reasoning_text),
    }];

    Ok(Json(ResponsesResponse {
        id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
        object: "response",
        created_at: now_unix_secs(),
        status: "completed",
        model: model_name,
        output,
        usage: build_usage(
            response.tokens_evaluated.unwrap_or(0),
            response.tokens_predicted.unwrap_or(0),
            reasoning_tokens,
        ),
        previous_response_id,
    }).into_response())
}
