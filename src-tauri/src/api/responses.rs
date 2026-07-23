use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::api::completions::{
    api_messages_have_images, build_chat_request, build_native_chat_body,
    compact_native_messages_to_fit, complete_native_chat_with_live_capture,
    complete_with_live_capture, end_to_end_tokens_per_second, ensure_runtime_vision_ready,
    extract_context_size_from_hash_map, extract_runtime_load_overrides, native_fixed_prompt_tokens,
    native_replay_request, resolve_loaded_model, uses_native_chat_api, ApiMessage,
    ChatCompletionRequest, StopParam, TopParam,
};
use crate::api::errors::ApiErrorResponse;
use crate::engine::client::{CompletionResponse, LlamaClient, Timings};
use crate::engine::scheduler::RequestPermit;
use crate::engine::streaming::{self, StreamEvent};
use crate::normalize::think_strip::{
    estimate_token_count, extract_reasoning_content_with_style, strip_think_tags_with_style,
};
use crate::state::{
    append_live_stream_delta_for_request, begin_api_generation, finish_api_generation_for_request,
    summarize_reasoning_tokens, GenerationDropGuard, SharedState,
};

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<serde_json::Value>),
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
    #[serde(default)]
    pub text: Option<serde_json::Value>,
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
    /// Ollama-format options object — e.g. {"options": {"num_ctx": 32768}}.
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ResponsesRequest {
    fn requested_context_size(&self) -> Option<u32> {
        self.context_size
            .filter(|value| *value > 0)
            .or_else(|| {
                self.options
                    .as_ref()
                    .and_then(|v| crate::api::completions::extract_context_size_from_value(v))
            })
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
    pub output: Vec<serde_json::Value>,
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

fn response_output_value_to_text(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text,
        serde_json::Value::Array(parts) => parts
            .into_iter()
            .map(|part| {
                part.get("text")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| part.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

fn response_input_items_to_chat_messages(
    items: Vec<serde_json::Value>,
) -> Result<Vec<ApiMessage>, String> {
    let mut messages: Vec<ApiMessage> = Vec::new();

    for mut item in items {
        let item_type = item
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("message")
            .to_string();
        match item_type.as_str() {
            "function_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| "Responses function_call item requires call_id".to_string())?;
                let name = item
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| "Responses function_call item requires name".to_string())?;
                let arguments = match item.get("arguments") {
                    Some(serde_json::Value::String(arguments)) => arguments.clone(),
                    Some(arguments) => serde_json::to_string(arguments)
                        .map_err(|error| format!("invalid function_call arguments: {error}"))?,
                    None => "{}".to_string(),
                };
                let tool_call = serde_json::json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                });

                if let Some(previous) = messages.last_mut().filter(|message| {
                    message.role == "assistant"
                        && message.content.is_none()
                        && message.tool_calls.is_some()
                }) {
                    previous
                        .tool_calls
                        .as_mut()
                        .expect("checked above")
                        .push(tool_call);
                } else {
                    messages.push(ApiMessage {
                        role: "assistant".to_string(),
                        content: None,
                        name: None,
                        tool_call_id: None,
                        tool_calls: Some(vec![tool_call]),
                        refusal: None,
                    });
                }
            }
            "function_call_output" => {
                let call_id = item
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| {
                        "Responses function_call_output item requires call_id".to_string()
                    })?
                    .to_string();
                let output = item
                    .get_mut("output")
                    .map(serde_json::Value::take)
                    .map(response_output_value_to_text)
                    .unwrap_or_default();
                messages.push(ApiMessage {
                    role: "tool".to_string(),
                    content: Some(crate::api::completions::ApiMessageContent::Text(output)),
                    name: None,
                    tool_call_id: Some(call_id),
                    tool_calls: None,
                    refusal: None,
                });
            }
            "message" => {
                // Responses output messages use output_text while chat messages use
                // text. Accept either so callers can feed a prior response back in.
                if let Some(parts) = item
                    .get_mut("content")
                    .and_then(|value| value.as_array_mut())
                {
                    for part in parts {
                        if part.get("type").and_then(serde_json::Value::as_str)
                            == Some("output_text")
                        {
                            part["type"] = serde_json::Value::String("text".to_string());
                        }
                    }
                }
                messages.push(
                    serde_json::from_value(item)
                        .map_err(|error| format!("invalid Responses message item: {error}"))?,
                );
            }
            "input_text" => {
                let text = item
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                messages.push(ApiMessage {
                    role: "user".to_string(),
                    content: Some(crate::api::completions::ApiMessageContent::Text(text)),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                    refusal: None,
                });
            }
            unsupported => {
                return Err(format!(
                    "unsupported Responses input item type '{unsupported}'"
                ));
            }
        }
    }

    Ok(messages)
}

fn responses_tools_to_chat_tools(
    tools: Option<Vec<serde_json::Value>>,
) -> Result<Option<Vec<serde_json::Value>>, String> {
    let Some(tools) = tools else {
        return Ok(None);
    };
    let mut translated = Vec::with_capacity(tools.len());
    for tool in tools {
        let tool_type = tool
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("function");
        if tool_type != "function" {
            return Err(format!(
                "unsupported local Responses tool type '{tool_type}'; only function tools are supported"
            ));
        }
        if tool
            .get("function")
            .and_then(serde_json::Value::as_object)
            .is_some()
        {
            translated.push(tool);
            continue;
        }
        let name = tool
            .get("name")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "Responses function tool requires name".to_string())?;
        let mut function = serde_json::Map::new();
        function.insert(
            "name".to_string(),
            serde_json::Value::String(name.to_string()),
        );
        if let Some(description) = tool.get("description").cloned() {
            function.insert("description".to_string(), description);
        }
        function.insert(
            "parameters".to_string(),
            tool.get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} })),
        );
        if let Some(strict) = tool.get("strict").cloned() {
            function.insert("strict".to_string(), strict);
        }
        translated.push(serde_json::json!({
            "type": "function",
            "function": serde_json::Value::Object(function),
        }));
    }
    Ok(Some(translated))
}

fn response_tool_choice_to_chat(choice: &mut serde_json::Value) {
    let Some(object) = choice.as_object_mut() else {
        return;
    };
    if object.get("type").and_then(serde_json::Value::as_str) != Some("function")
        || object.contains_key("function")
    {
        return;
    }
    if let Some(name) = object.remove("name") {
        object.insert("function".to_string(), serde_json::json!({ "name": name }));
    }
}

fn into_chat_request(mut request: ResponsesRequest) -> Result<ChatCompletionRequest, String> {
    let requested_context_size = request.requested_context_size();
    let response_format = response_text_to_chat_response_format(request.text.as_ref());
    let messages = match request.input {
        ResponseInput::Text(text) => vec![ApiMessage {
            role: "user".to_string(),
            content: Some(crate::api::completions::ApiMessageContent::Text(text)),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            refusal: None,
        }],
        ResponseInput::Items(items) => response_input_items_to_chat_messages(items)?,
    };
    let tools = responses_tools_to_chat_tools(request.tools)?;
    for key in ["tool_choice", "toolChoice"] {
        if let Some(choice) = request.extra.get_mut(key) {
            response_tool_choice_to_chat(choice);
        }
    }

    Ok(ChatCompletionRequest {
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
        tools,
        response_format,
        context_size: requested_context_size,
        top: request.top,
        reasoning: request
            .reasoning
            .map(|reasoning| crate::api::completions::ReasoningRequest {
                effort: reasoning.effort,
                max_tokens: reasoning.max_tokens,
            }),
        reasoning_effort: request.reasoning_effort,
        reasoning_tokens: request.reasoning_tokens,
        stream_options: None,
        options: request.options,
        extra: request.extra,
    })
}

fn response_text_to_chat_response_format(
    text: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    let format = text?.get("format")?;
    match format.get("type").and_then(|value| value.as_str()) {
        Some("json_schema") => Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": format
                    .get("name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("response"),
                "strict": format.get("strict").and_then(|value| value.as_bool()).unwrap_or(true),
                "schema": format.get("schema").cloned().unwrap_or_else(|| serde_json::json!({ "type": "object" }))
            }
        })),
        Some("json_object") => Some(serde_json::json!({ "type": "json_object" })),
        _ => None,
    }
}

fn build_response_output(visible_text: String, reasoning_text: String) -> Vec<serde_json::Value> {
    let mut content = Vec::new();
    if !reasoning_text.is_empty() {
        content.push(
            serde_json::to_value(ResponsesOutputReasoning {
                kind: "reasoning",
                text: reasoning_text,
            })
            .unwrap_or_default(),
        );
    }
    if !visible_text.is_empty() {
        content.push(
            serde_json::to_value(ResponsesOutputText {
                kind: "output_text",
                text: visible_text,
                annotations: Vec::new(),
            })
            .unwrap_or_default(),
        );
    }
    content
}

fn build_response_items(
    profile: &crate::models::profiles::ModelProfile,
    raw_text: &str,
) -> (
    Vec<serde_json::Value>,
    Vec<serde_json::Value>,
    String,
    String,
) {
    let reasoning = extract_reasoning_content_with_style(raw_text, profile.think_tag_style);
    let without_reasoning = strip_think_tags_with_style(raw_text, profile.think_tag_style);
    let (tool_calls, visible) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(&without_reasoning, profile);
    let mut output = Vec::new();
    if !visible.trim().is_empty() || !reasoning.trim().is_empty() {
        output.push(serde_json::json!({
            "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": build_response_output(visible.clone(), reasoning.clone()),
        }));
    }
    let function_calls = tool_calls
        .into_iter()
        .map(|tool_call| {
            serde_json::json!({
                "id": format!("fc_{}", uuid::Uuid::new_v4().simple()),
                "type": "function_call",
                "status": "completed",
                "call_id": tool_call.id,
                "name": tool_call.name,
                "arguments": serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string()),
            })
        })
        .collect::<Vec<_>>();
    output.extend(function_calls.iter().cloned());
    (output, function_calls, visible, reasoning)
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
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiErrorResponse> {
    if let Some(upstream) = crate::api::upstream::active_openai_provider(&state).await {
        return crate::api::upstream::proxy_json_to_openai_provider(
            state.clone(),
            upstream,
            "/responses",
            body,
        )
        .await;
    }

    let original_request_body = body.clone();
    let client_request_id =
        crate::replay::preferred_client_correlation_id(&headers, &original_request_body);
    let req: ResponsesRequest = serde_json::from_value(body)
        .map_err(|error| ApiErrorResponse::bad_request(error.to_string()))?;
    let previous_response_id = req.previous_response_id.clone();
    let chat_request = into_chat_request(req).map_err(ApiErrorResponse::bad_request)?;
    let buffer_tool_content = chat_request
        .tools
        .as_ref()
        .map(|tools| !tools.is_empty())
        .unwrap_or(false);
    let requested_context_size = chat_request.requested_context_size();
    let requested_model = chat_request.model.clone().unwrap_or_default();
    let requested_overrides =
        extract_runtime_load_overrides(chat_request.options.as_ref(), &chat_request.extra);
    let model_name = resolve_loaded_model(
        &state,
        &requested_model,
        requested_context_size,
        requested_overrides,
    )
    .await?;
    let profile = {
        let s = state.read().await;
        s.effective_profile_for_model(&model_name)
    };

    let (server_defaults, launch_defaults, llama_port, context_limit, scheduler) = {
        let s = state.read().await;
        (
            (
                s.config.server.default_temperature,
                s.config.server.default_top_p,
                s.config.server.default_top_k,
                s.config.server.default_max_tokens,
            ),
            s.active_sampling_defaults(),
            s.process.port(),
            s.model_stats
                .as_ref()
                .map(|stats| stats.context_size)
                .or_else(|| {
                    s.last_launch_preview
                        .as_ref()
                        .and_then(|preview| preview.context_size)
                })
                .or(requested_context_size),
            s.request_scheduler.clone(),
        )
    };
    let client = LlamaClient::new(llama_port);
    let permit = scheduler.acquire().await;

    let use_native_chat = uses_native_chat_api(&profile);
    let has_images = api_messages_have_images(&chat_request.messages);
    let native_compaction = if use_native_chat {
        Some(compact_native_messages_to_fit(
            &chat_request.messages,
            context_limit,
            chat_request
                .max_tokens
                .or(server_defaults.3)
                .or(profile.default_max_output_tokens),
            native_fixed_prompt_tokens(&chat_request),
        )?)
    } else {
        None
    };
    let native_chat_body = use_native_chat.then(|| {
        let mut body = build_native_chat_body(
            &original_request_body,
            &chat_request,
            &model_name,
            &profile,
            server_defaults,
            &launch_defaults,
        );
        if let Some((messages, _)) = native_compaction.as_ref() {
            body["messages"] = crate::api::completions::api_messages_to_native_value(messages);
        }
        if chat_request.stream {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }
        body
    });

    let (request, _compaction) = if let Some(body) = native_chat_body.as_ref() {
        (native_replay_request(body), None)
    } else {
        build_chat_request(
            &profile,
            chat_request,
            (
                &server_defaults.0,
                &server_defaults.1,
                &server_defaults.2,
                &server_defaults.3,
            ),
            &launch_defaults,
            context_limit,
            Some(&client),
        )
        .await?
    };

    ensure_runtime_vision_ready(
        &state,
        &model_name,
        &profile,
        has_images || !request.image_data.is_empty(),
    )
    .await?;

    {
        let mut s = state.write().await;
        s.last_prompt = Some(match native_chat_body.as_ref() {
            Some(body) => serde_json::to_string_pretty(body).unwrap_or_else(|_| body.to_string()),
            None => request.prompt.clone(),
        });
    }

    let generation_started_at = chrono::Utc::now().to_rfc3339();
    let generation_started = std::time::Instant::now();

    if request.stream {
        let response = if let Some(body) = native_chat_body.as_ref() {
            client.chat_completion_response(body).await
        } else {
            client.complete_stream(&request).await
        }
        .map_err(|e| {
            tracing::error!(error = %e, "Responses stream completion failed");
            ApiErrorResponse::inference_failed(&e.to_string())
        })?;
        if !response.status().is_success() {
            let status = response.status();
            let response_body = response.text().await.unwrap_or_default();
            return Err(ApiErrorResponse::inference_failed(&format!(
                "llama-server returned {status}: {response_body}"
            )));
        }

        let gen = begin_api_generation(&state, model_name.clone()).await;
        let request_id = gen.request_id.clone();
        let stream_cancel = gen.cancel.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            let result = if use_native_chat {
                streaming::consume_chat_sse_stream(response, tx, gen.cancel).await
            } else {
                streaming::consume_sse_stream(response, tx, gen.cancel).await
            };
            let _ = result;
        });

        let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let created_at = now_unix_secs();
        let state_for_stream = state.clone();

        let stream = async_stream::stream! {
            let _permit: RequestPermit = permit;
            let guard = GenerationDropGuard::new(
                state_for_stream.clone(),
                request_id.clone(),
                stream_cancel,
            );
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
                    StreamEvent::RawDelta(raw) => {
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "raw", &raw).await;
                    }
                    StreamEvent::Token(token) => {
                        full_text.push_str(&token);
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "content", &token).await;
                        let chunk = serde_json::json!({
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "item_id": message_id,
                            "delta": token
                        });
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                    StreamEvent::ReasoningDelta(reasoning) => {
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "reasoning", &reasoning).await;
                        let chunk = serde_json::json!({
                            "type": "response.reasoning.delta",
                            "response_id": response_id,
                            "item_id": message_id,
                            "delta": reasoning
                        });
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                    StreamEvent::ToolCallDelta(tool_call) => {
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "tool_call", &tool_call).await;
                    }
                    StreamEvent::Done {
                        full_text: text,
                        tokens_predicted,
                        tokens_evaluated,
                        decode_tokens_per_second,
                        prompt_tokens_per_second,
                        stopped_limit,
                        stop_type,
                    } => {
                        let (response_output, function_calls, visible, reasoning) =
                            build_response_items(&profile, &text);
                        let reasoning_tokens =
                            summarize_reasoning_tokens(Some(tokens_predicted), &visible, &reasoning);
                        let replay_visible = visible.clone();

                        let elapsed_ms = generation_started.elapsed().as_millis() as u64;
                        let end_to_end_tps = end_to_end_tokens_per_second(
                            Some(tokens_predicted),
                            elapsed_ms,
                        );
                        let mut s = state_for_stream.write().await;
                        s.last_parse_trace = Some(crate::normalize::parse_trace::build_parse_trace(
                            &profile,
                            &text,
                            &replay_visible,
                            None,
                        ));
                        s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
                            source: "responses-api".to_string(),
                            model: model_name.clone(),
                            request_id: request_id.clone(),
                            started_at: generation_started_at.clone(),
                            finished_at: chrono::Utc::now().to_rfc3339(),
                            elapsed_ms,
                            time_to_first_token_ms: None,
                            prompt_tokens: Some(tokens_evaluated),
                            completion_tokens: Some(tokens_predicted),
                            total_tokens: Some(tokens_evaluated + tokens_predicted),
                            prompt_tokens_per_second,
                            decode_tokens_per_second: Some(decode_tokens_per_second),
                            end_to_end_tokens_per_second: end_to_end_tps,
                        });
                        drop(s);
                        let stream_response = CompletionResponse {
                            content: text.clone(),
                            stop: true,
                            stopped_limit,
                            stop_type,
                            tokens_predicted: Some(tokens_predicted),
                            tokens_evaluated: Some(tokens_evaluated),
                            timings: Some(Timings {
                                predicted_per_second: Some(decode_tokens_per_second),
                                prompt_per_second: prompt_tokens_per_second,
                            }),
                        };
                        let canonical = crate::replay::build_canonical_response(
                            &profile,
                            &request_id,
                            client_request_id.clone(),
                            "responses-api-stream",
                            &model_name,
                            &text,
                            &replay_visible,
                            function_calls.clone(),
                            &stream_response,
                            elapsed_ms,
                            end_to_end_tps,
                            llama_port,
                            context_limit,
                        );
                        crate::replay::append_api_replay_record(
                            "/v1/responses",
                            original_request_body.clone(),
                            &request,
                            canonical,
                        )
                        .await;

                        let function_call_offset = response_output.len().saturating_sub(function_calls.len());
                        for (call_index, function_call) in function_calls.iter().enumerate() {
                            let output_index = function_call_offset + call_index;
                            yield Ok(Event::default().data(serde_json::json!({
                                "type": "response.output_item.added",
                                "response_id": response_id,
                                "output_index": output_index,
                                "item": function_call,
                            }).to_string()));
                            yield Ok(Event::default().data(serde_json::json!({
                                "type": "response.function_call_arguments.done",
                                "response_id": response_id,
                                "item_id": function_call.get("id").cloned().unwrap_or(serde_json::Value::Null),
                                "output_index": output_index,
                                "arguments": function_call.get("arguments").cloned().unwrap_or_else(|| serde_json::json!("{}")),
                            }).to_string()));
                            yield Ok(Event::default().data(serde_json::json!({
                                "type": "response.output_item.done",
                                "response_id": response_id,
                                "output_index": output_index,
                                "item": function_call,
                            }).to_string()));
                        }

                        let completed = serde_json::json!({
                            "type": "response.completed",
                            "response": {
                                "id": response_id,
                                "object": "response",
                                "created_at": created_at,
                                "status": "completed",
                                "model": model_name,
                                "output": response_output,
                                "usage": build_usage(tokens_evaluated, tokens_predicted, reasoning_tokens),
                                "previous_response_id": previous_response_id
                            }
                        });
                        guard.mark_completed();
                        finish_api_generation_for_request(
                            &state_for_stream,
                            &request_id,
                            "completed",
                        )
                        .await;
                        yield Ok(Event::default().data(completed.to_string()));
                        yield Ok(Event::default().data("[DONE]"));
                        return;
                    }
                    StreamEvent::Error(error) => {
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "error", &error).await;
                        guard.mark_completed();
                        finish_api_generation_for_request(&state_for_stream, &request_id, "error").await;
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

        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let gen = begin_api_generation(&state, model_name.clone()).await;
    let request_id = gen.request_id.clone();
    let response_result = if let Some(body) = native_chat_body.as_ref() {
        complete_native_chat_with_live_capture(
            &state,
            &client,
            body,
            &model_name,
            &request_id,
            "responses-api-native-chat",
            &generation_started_at,
            generation_started,
            gen.cancel,
            buffer_tool_content,
        )
        .await
    } else {
        complete_with_live_capture(
            &state,
            &client,
            &request,
            &model_name,
            &request_id,
            "responses-api",
            &generation_started_at,
            generation_started,
            gen.cancel,
            buffer_tool_content,
        )
        .await
    };
    let response = match response_result {
        Ok(response) => response,
        Err(e) => {
            finish_api_generation_for_request(&state, &request_id, "error").await;
            tracing::error!(error = %e, "Responses completion failed");
            return Err(ApiErrorResponse::inference_failed(&e.to_string()));
        }
    };

    let (output, function_calls, visible_text, reasoning_text) =
        build_response_items(&profile, &response.content);
    let reasoning_tokens = summarize_reasoning_tokens(
        response
            .tokens_predicted
            .or_else(|| Some(estimate_token_count(&response.content))),
        &visible_text,
        &reasoning_text,
    );

    let elapsed_ms = generation_started.elapsed().as_millis() as u64;
    let end_to_end_tps = end_to_end_tokens_per_second(response.tokens_predicted, elapsed_ms);
    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(crate::normalize::parse_trace::build_parse_trace(
            &profile,
            &response.content,
            &visible_text,
            None,
        ));
        s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
            source: "responses-api".to_string(),
            model: model_name.clone(),
            request_id: request_id.clone(),
            started_at: generation_started_at,
            finished_at: chrono::Utc::now().to_rfc3339(),
            elapsed_ms,
            time_to_first_token_ms: None,
            prompt_tokens: response.tokens_evaluated,
            completion_tokens: response.tokens_predicted,
            total_tokens: match (response.tokens_evaluated, response.tokens_predicted) {
                (Some(prompt), Some(completion)) => Some(prompt + completion),
                _ => None,
            },
            prompt_tokens_per_second: response
                .timings
                .as_ref()
                .and_then(|timings| timings.prompt_per_second),
            decode_tokens_per_second: response
                .timings
                .as_ref()
                .and_then(|timings| timings.predicted_per_second),
            end_to_end_tokens_per_second: end_to_end_tps,
        });
    }

    let replay_visible_text = visible_text.clone();
    let canonical = crate::replay::build_canonical_response(
        &profile,
        &request_id,
        client_request_id,
        "responses-api",
        &model_name,
        &response.content,
        &replay_visible_text,
        function_calls,
        &response,
        elapsed_ms,
        end_to_end_tps,
        llama_port,
        context_limit,
    );
    crate::replay::append_api_replay_record(
        "/v1/responses",
        original_request_body,
        &request,
        canonical,
    )
    .await;
    finish_api_generation_for_request(&state, &request_id, "completed").await;

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
    })
    .into_response())
}

#[cfg(test)]
mod tests {
    use super::{build_response_items, into_chat_request, ResponseInput, ResponsesRequest};
    use crate::api::completions::build_native_chat_body;
    use crate::engine::process::SamplingDefaults;
    use crate::models::profiles::ModelProfile;
    use std::collections::HashMap;

    fn base_request(text: Option<serde_json::Value>) -> ResponsesRequest {
        ResponsesRequest {
            model: Some("local".to_string()),
            input: ResponseInput::Text("Return weather JSON.".to_string()),
            stream: false,
            max_output_tokens: Some(64),
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            top: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            tools: None,
            text,
            context_size: None,
            stop: None,
            seed: None,
            reasoning: None,
            reasoning_effort: None,
            reasoning_tokens: None,
            previous_response_id: None,
            options: None,
            extra: HashMap::new(),
        }
    }

    #[test]
    fn maps_responses_text_format_to_chat_response_format() {
        let request = base_request(Some(serde_json::json!({
            "format": {
                "type": "json_schema",
                "name": "weather",
                "strict": true,
                "schema": {
                    "type": "object",
                    "required": ["forecast"],
                    "properties": { "forecast": { "type": "string" } }
                }
            }
        })));

        let chat = into_chat_request(request).expect("Responses request should translate");
        let response_format = chat.response_format.expect("response format");
        assert_eq!(response_format["type"], "json_schema");
        assert_eq!(response_format["json_schema"]["name"], "weather");
        assert_eq!(
            response_format["json_schema"]["schema"]["required"][0],
            "forecast"
        );
    }

    #[test]
    fn responses_standard_tool_shapes_round_trip_to_native_chat() {
        let request: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "Tess-4-27B-Q4_K_M.gguf",
            "input": [
                {"type": "function_call", "id": "fc_7", "call_id": "call_7", "name": "lookup", "arguments": "{\"query\":\"tess\"}", "status": "completed"},
                {"type": "function_call", "id": "fc_8", "call_id": "call_8", "name": "clock", "arguments": {"zone": "Europe/London"}, "status": "completed"},
                {"type": "function_call_output", "call_id": "call_7", "output": "found"},
                {"type": "function_call_output", "call_id": "call_8", "output": [{"type": "input_text", "text": "12:00"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Summarize both results"}]}
            ],
            "tools": [{
                "type": "function",
                "name": "lookup",
                "description": "Look something up",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                },
                "strict": true
            }],
            "tool_choice": {"type": "function", "name": "lookup"},
            "parallel_tool_calls": false
        }))
        .expect("Responses request should deserialize");
        let chat = into_chat_request(request).expect("standard Responses tools should translate");
        assert_eq!(chat.messages.len(), 4);
        assert_eq!(chat.messages[0].role, "assistant");
        assert_eq!(chat.messages[0].tool_calls.as_ref().map(Vec::len), Some(2));
        assert_eq!(chat.messages[1].role, "tool");
        assert_eq!(chat.messages[1].tool_call_id.as_deref(), Some("call_7"));
        assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("call_8"));
        assert_eq!(chat.messages[3].role, "user");
        assert_eq!(
            chat.tools.as_ref().unwrap()[0]["function"]["name"],
            "lookup"
        );
        assert_eq!(chat.extra["tool_choice"]["function"]["name"], "lookup");
        let profile = ModelProfile::detect("Tess-4-27B-Q4_K_M.gguf");
        let body = build_native_chat_body(
            &serde_json::json!({}),
            &chat,
            "Tess-4-27B-Q4_K_M.gguf",
            &profile,
            (None, None, None, None),
            &SamplingDefaults::default(),
        );

        assert_eq!(body["messages"].as_array().map(Vec::len), Some(4));
        assert_eq!(body["messages"][0]["tool_calls"][0]["id"], "call_7");
        assert_eq!(body["messages"][0]["tool_calls"][1]["id"], "call_8");
        assert_eq!(body["messages"][1]["tool_call_id"], "call_7");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_8");
        assert_eq!(body["tools"][0]["function"]["name"], "lookup");
        assert_eq!(body["tool_choice"]["function"]["name"], "lookup");
        assert_eq!(body["parallel_tool_calls"], false);
    }

    #[test]
    fn responses_maps_two_native_tool_calls_to_function_call_items() {
        let profile = ModelProfile::detect("Tess-4-27B-Q4_K_M.gguf");
        let raw = concat!(
            "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"London\"}}</tool_call>",
            "<tool_call>{\"name\":\"clock\",\"arguments\":{\"zone\":\"Europe/London\"}}</tool_call>"
        );

        let (output, calls, visible, reasoning) = build_response_items(&profile, raw);

        assert_eq!(calls.len(), 2);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["name"], "weather");
        assert_eq!(output[1]["name"], "clock");
        assert!(visible.trim().is_empty());
        assert!(reasoning.trim().is_empty());
        assert!(output
            .iter()
            .all(|item| !item.to_string().contains("<tool_call>")));
    }
}
