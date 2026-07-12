use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use crate::api::completions::{
    api_tool_call_value, build_chat_request, completion_failure_diagnostics,
    ensure_runtime_vision_ready, extract_runtime_load_overrides, ApiContentPart, ApiImageUrl,
    ApiMessage, ApiMessageContent, ChatCompletionRequest, StopParam,
};
use crate::api::errors::ApiErrorResponse;
use crate::engine::client::{CompletionResponse, LlamaClient, Timings};
use crate::engine::scheduler::RequestPermit;
use crate::engine::streaming::{self, StreamEvent};
use crate::state::{
    append_live_stream_delta_for_request, begin_api_generation, finish_api_generation_for_request,
    GenerationDropGuard, SharedState,
};

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct AnthropicMessageResponse {
    id: String,
    #[serde(rename = "type")]
    object_type: &'static str,
    role: &'static str,
    model: String,
    content: Vec<serde_json::Value>,
    stop_reason: String,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

pub async fn messages(
    State(state): State<SharedState>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiErrorResponse> {
    let anthropic: AnthropicMessagesRequest = serde_json::from_value(body.clone())
        .map_err(|error| ApiErrorResponse::bad_request(error.to_string()))?;
    if anthropic.max_tokens.unwrap_or(0) == 0 {
        return Err(ApiErrorResponse::bad_request(
            "`max_tokens` is required for /v1/messages and must be greater than 0.",
        ));
    }

    let original_stream = anthropic.stream;
    let chat_request = anthropic_to_chat_request(anthropic)?;
    run_anthropic_chat(state, body, chat_request, original_stream).await
}

fn anthropic_to_chat_request(
    request: AnthropicMessagesRequest,
) -> Result<ChatCompletionRequest, ApiErrorResponse> {
    let mut messages = Vec::new();
    if let Some(system) = request.system.as_ref().and_then(anthropic_system_to_text) {
        messages.push(ApiMessage {
            role: "system".to_string(),
            content: Some(ApiMessageContent::Text(system)),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            refusal: None,
        });
    }

    for message in request.messages {
        messages.extend(anthropic_message_to_api_messages(message)?);
    }

    Ok(ChatCompletionRequest {
        model: request.model,
        messages,
        stream: request.stream,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        min_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        repetition_penalty: None,
        seed: None,
        stop: request.stop_sequences.map(StopParam::Multiple),
        tools: request.tools.map(anthropic_tools_to_openai_tools),
        response_format: None,
        context_size: None,
        top: None,
        reasoning: None,
        reasoning_effort: None,
        reasoning_tokens: None,
        stream_options: None,
        options: None,
        extra: std::collections::HashMap::new(),
    })
}

fn anthropic_system_to_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => non_empty_string(text),
        serde_json::Value::Array(blocks) => {
            let text = blocks
                .iter()
                .filter_map(|block| {
                    (block.get("type").and_then(|value| value.as_str()) == Some("text"))
                        .then(|| block.get("text").and_then(|value| value.as_str()))
                        .flatten()
                })
                .collect::<Vec<_>>()
                .join("\n");
            non_empty_string(&text)
        }
        _ => None,
    }
}

fn non_empty_string(text: &str) -> Option<String> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn anthropic_message_to_api_messages(
    message: AnthropicMessage,
) -> Result<Vec<ApiMessage>, ApiErrorResponse> {
    let role = message.role.trim().to_ascii_lowercase();
    let mut out = Vec::new();

    if role == "assistant" {
        let (content, tool_calls) = assistant_content_to_text_and_tool_calls(&message.content)?;
        out.push(ApiMessage {
            role,
            content: content.map(ApiMessageContent::Text),
            name: None,
            tool_call_id: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            refusal: None,
        });
        return Ok(out);
    }

    if role == "user" {
        let (content, tool_results) = user_content_to_parts_and_tool_results(&message.content)?;
        if let Some(content) = content {
            out.push(ApiMessage {
                role: "user".to_string(),
                content: Some(content),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                refusal: None,
            });
        }
        out.extend(tool_results);
        if out.is_empty() {
            out.push(ApiMessage {
                role: "user".to_string(),
                content: Some(ApiMessageContent::Text(String::new())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                refusal: None,
            });
        }
        return Ok(out);
    }

    Err(ApiErrorResponse::bad_request(format!(
        "Unsupported Anthropic message role `{}`.",
        message.role
    )))
}

fn assistant_content_to_text_and_tool_calls(
    content: &serde_json::Value,
) -> Result<(Option<String>, Vec<serde_json::Value>), ApiErrorResponse> {
    match content {
        serde_json::Value::String(text) => Ok((non_empty_string(text), Vec::new())),
        serde_json::Value::Array(blocks) => {
            let mut text = Vec::new();
            let mut tool_calls = Vec::new();
            for block in blocks {
                match block.get("type").and_then(|value| value.as_str()) {
                    Some("text") => {
                        if let Some(part) = block.get("text").and_then(|value| value.as_str()) {
                            text.push(part.to_string());
                        }
                    }
                    Some("tool_use") => {
                        let name = block
                            .get("name")
                            .and_then(|value| value.as_str())
                            .ok_or_else(|| {
                                ApiErrorResponse::bad_request(
                                    "Anthropic tool_use blocks require `name`.",
                                )
                            })?;
                        let input = block
                            .get("input")
                            .cloned()
                            .unwrap_or_else(|| serde_json::json!({}));
                        let id = block
                            .get("id")
                            .and_then(|value| value.as_str())
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4()));
                        tool_calls.push(serde_json::json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string())
                            }
                        }));
                    }
                    _ => {}
                }
            }
            Ok((non_empty_string(&text.join("\n")), tool_calls))
        }
        _ => Ok((None, Vec::new())),
    }
}

fn user_content_to_parts_and_tool_results(
    content: &serde_json::Value,
) -> Result<(Option<ApiMessageContent>, Vec<ApiMessage>), ApiErrorResponse> {
    match content {
        serde_json::Value::String(text) => {
            Ok((Some(ApiMessageContent::Text(text.clone())), Vec::new()))
        }
        serde_json::Value::Array(blocks) => {
            let mut parts = Vec::new();
            let mut tool_results = Vec::new();
            for block in blocks {
                match block.get("type").and_then(|value| value.as_str()) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|value| value.as_str()) {
                            parts.push(ApiContentPart::Text {
                                text: text.to_string(),
                            });
                        }
                    }
                    Some("image") => {
                        if let Some(url) = anthropic_image_block_to_data_url(block) {
                            parts.push(ApiContentPart::ImageUrl {
                                image_url: ApiImageUrl::Object { url },
                            });
                        }
                    }
                    Some("tool_result") => {
                        let tool_call_id = block
                            .get("tool_use_id")
                            .and_then(|value| value.as_str())
                            .map(ToOwned::to_owned);
                        tool_results.push(ApiMessage {
                            role: "tool".to_string(),
                            content: Some(ApiMessageContent::Text(
                                anthropic_tool_result_content_to_text(block.get("content")),
                            )),
                            name: None,
                            tool_call_id,
                            tool_calls: None,
                            refusal: None,
                        });
                    }
                    _ => {}
                }
            }
            let content = if parts.is_empty() {
                None
            } else {
                Some(ApiMessageContent::Parts(parts))
            };
            Ok((content, tool_results))
        }
        _ => Ok((None, Vec::new())),
    }
}

fn anthropic_image_block_to_data_url(block: &serde_json::Value) -> Option<String> {
    let source = block.get("source")?;
    if source.get("type").and_then(|value| value.as_str()) == Some("url") {
        return source
            .get("url")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned);
    }

    let media_type = source
        .get("media_type")
        .and_then(|value| value.as_str())
        .unwrap_or("image/png");
    let data = source.get("data").and_then(|value| value.as_str())?;
    Some(format!("data:{media_type};base64,{data}"))
}

fn anthropic_tool_result_content_to_text(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(text)) => text.clone(),
        Some(serde_json::Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|value| value.as_str()) == Some("text") {
                    block.get("text").and_then(|value| value.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
        None => String::new(),
    }
}

fn anthropic_tools_to_openai_tools(tools: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    tools
        .into_iter()
        .filter_map(|tool| {
            let name = tool.get("name")?.as_str()?;
            Some(serde_json::json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description").and_then(|value| value.as_str()).unwrap_or(""),
                    "parameters": tool.get("input_schema").cloned().unwrap_or_else(|| serde_json::json!({ "type": "object" }))
                }
            }))
        })
        .collect()
}

async fn run_anthropic_chat(
    state: SharedState,
    original_request_body: serde_json::Value,
    req: ChatCompletionRequest,
    original_stream: bool,
) -> Result<Response, ApiErrorResponse> {
    let requested_context_size = req.requested_context_size();
    let requested_model = req.model.clone().unwrap_or_default();
    let requested_tools = req.tools.clone();
    let requested_overrides = extract_runtime_load_overrides(req.options.as_ref(), &req.extra);
    let model_name = crate::api::completions::resolve_loaded_model(
        &state,
        &requested_model,
        requested_context_size,
        requested_overrides,
    )
    .await?;
    let profile = crate::models::overrides::detect_effective_profile(&model_name);

    let (server_defaults, scheduler, llama_port, context_limit, tool_argument_repair_enabled) = {
        let s = state.read().await;
        (
            (
                s.config.server.default_temperature,
                s.config.server.default_top_p,
                s.config.server.default_top_k,
                s.config.server.default_max_tokens,
            ),
            s.request_scheduler.clone(),
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
            s.config.server.tool_argument_repair_enabled,
        )
    };
    let client = LlamaClient::new(llama_port);
    let permit = scheduler.acquire().await;

    let (request, _compaction) = build_chat_request(
        &profile,
        req,
        (
            &server_defaults.0,
            &server_defaults.1,
            &server_defaults.2,
            &server_defaults.3,
        ),
        context_limit,
        Some(&client),
    )
    .await?;

    ensure_runtime_vision_ready(
        &state,
        &model_name,
        &profile,
        !request.image_data.is_empty(),
    )
    .await?;
    {
        let mut s = state.write().await;
        s.last_prompt = Some(request.prompt.clone());
    }

    let generation_started_at = chrono::Utc::now().to_rfc3339();
    let generation_started = std::time::Instant::now();
    let gen = begin_api_generation(&state, model_name.clone()).await;
    let request_id = gen.request_id.clone();

    if original_stream {
        return stream_anthropic_message(
            state,
            client,
            request,
            model_name,
            profile,
            generation_started_at,
            generation_started,
            request_id,
            gen.cancel,
            requested_tools,
            permit,
            tool_argument_repair_enabled,
        )
        .await;
    }

    let response = match client.complete(&request).await {
        Ok(response) => response,
        Err(error) => {
            let diagnostics = completion_failure_diagnostics(&state, &model_name, &error).await;
            finish_api_generation_for_request(&state, &request_id, "error").await;
            return Err(ApiErrorResponse::inference_failed(&diagnostics));
        }
    };

    let anthropic = anthropic_response_from_completion(
        Some(&client),
        &profile,
        &model_name,
        &response,
        requested_tools.as_ref(),
        tool_argument_repair_enabled,
    )
    .await;
    append_live_stream_delta_for_request(&state, &request_id, "raw", &response.content).await;
    finish_api_generation_for_request(&state, &request_id, "completed").await;
    let _ = original_request_body;
    let _ = generation_started_at;
    let _ = generation_started;
    let _permit = permit;
    Ok(Json(anthropic).into_response())
}

async fn anthropic_response_from_completion(
    client: Option<&LlamaClient>,
    profile: &crate::models::profiles::ModelProfile,
    model_name: &str,
    response: &CompletionResponse,
    requested_tools: Option<&Vec<serde_json::Value>>,
    repair_enabled: bool,
) -> AnthropicMessageResponse {
    let reasoning_text = crate::normalize::think_strip::extract_reasoning_content_with_style(
        &response.content,
        profile.think_tag_style,
    );
    let stripped = crate::normalize::think_strip::strip_think_tags_with_style(
        &response.content,
        profile.think_tag_style,
    );
    let (tool_calls, text) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(&stripped, profile);
    let mut content = Vec::new();
    if !reasoning_text.trim().is_empty() {
        content.push(serde_json::json!({
            "type": "thinking",
            "thinking": reasoning_text
        }));
    }
    if !text.trim().is_empty() {
        content.push(serde_json::json!({
            "type": "text",
            "text": text
        }));
    }
    for (index, tool_call) in tool_calls.iter().enumerate() {
        let openai_tool_call = api_tool_call_value(
            client,
            profile,
            tool_call,
            requested_tools,
            Some(index),
            repair_enabled,
        )
        .await;
        content.push(openai_tool_call_to_anthropic_tool_use(&openai_tool_call));
    }
    if content.is_empty() {
        content.push(serde_json::json!({ "type": "text", "text": "" }));
    }

    AnthropicMessageResponse {
        id: format!("msg_{}", uuid::Uuid::new_v4()),
        object_type: "message",
        role: "assistant",
        model: model_name.to_string(),
        stop_reason: anthropic_stop_reason(response, content_has_tool_use(&content)),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: response.tokens_evaluated.unwrap_or(0),
            output_tokens: response.tokens_predicted.unwrap_or(0),
        },
        content,
    }
}

fn content_has_tool_use(content: &[serde_json::Value]) -> bool {
    content
        .iter()
        .any(|block| block.get("type").and_then(|value| value.as_str()) == Some("tool_use"))
}

fn anthropic_stop_reason(response: &CompletionResponse, has_tool_use: bool) -> String {
    if has_tool_use {
        return "tool_use".to_string();
    }
    let stopped_by_limit = response.stopped_limit.unwrap_or(false)
        || response
            .stop_type
            .as_deref()
            .map(|value| value.to_ascii_lowercase().contains("limit"))
            .unwrap_or(false);
    if stopped_by_limit {
        "max_tokens".to_string()
    } else {
        "end_turn".to_string()
    }
}

fn openai_tool_call_to_anthropic_tool_use(tool_call: &serde_json::Value) -> serde_json::Value {
    let function = tool_call.get("function").unwrap_or(tool_call);
    let name = function
        .get("name")
        .and_then(|value| value.as_str())
        .unwrap_or("tool");
    let input = function
        .get("arguments")
        .and_then(|value| value.as_str())
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
        .or_else(|| function.get("arguments").cloned())
        .unwrap_or_else(|| serde_json::json!({}));
    serde_json::json!({
        "type": "tool_use",
        "id": tool_call
            .get("id")
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("toolu_{}", uuid::Uuid::new_v4())),
        "name": name,
        "input": input
    })
}

#[allow(clippy::too_many_arguments)]
async fn stream_anthropic_message(
    state: SharedState,
    client: LlamaClient,
    request: crate::engine::client::CompletionRequest,
    model_name: String,
    profile: crate::models::profiles::ModelProfile,
    generation_started_at: String,
    generation_started: std::time::Instant,
    request_id: String,
    cancel: tokio_util::sync::CancellationToken,
    requested_tools: Option<Vec<serde_json::Value>>,
    permit: RequestPermit,
    tool_argument_repair_enabled: bool,
) -> Result<Response, ApiErrorResponse> {
    let response = match client.complete_stream(&request).await {
        Ok(response) => response,
        Err(error) => {
            let diagnostics = completion_failure_diagnostics(&state, &model_name, &error).await;
            finish_api_generation_for_request(&state, &request_id, "error").await;
            return Err(ApiErrorResponse::inference_failed(&diagnostics));
        }
    };

    let stream_cancel = cancel.clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let _ = streaming::consume_sse_stream(response, tx, cancel).await;
    });

    let buffer_tool_content = requested_tools
        .as_ref()
        .map(|tools| !tools.is_empty())
        .unwrap_or(false);
    let message_id = format!("msg_{}", uuid::Uuid::new_v4());
    let state_for_stream = state.clone();

    let stream = async_stream::stream! {
        let _permit = permit;
        let guard = GenerationDropGuard::new(
            state_for_stream.clone(),
            request_id.clone(),
            stream_cancel,
        );
        let mut raw_full_text = String::new();
        let mut block_open = false;
        let mut first_token_at: Option<std::time::Instant> = None;
        let mut visible_tokens: u32 = 0;

        let message_start = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [],
                "stop_reason": serde_json::Value::Null,
                "stop_sequence": serde_json::Value::Null,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        });
        yield Ok::<Event, std::convert::Infallible>(anthropic_sse("message_start", message_start));

        if !buffer_tool_content {
            let block_start = serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            });
            block_open = true;
            yield Ok(anthropic_sse("content_block_start", block_start));
        }

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::RawDelta(raw) => {
                    raw_full_text.push_str(&raw);
                    append_live_stream_delta_for_request(&state_for_stream, &request_id, "raw", &raw).await;
                }
                StreamEvent::Token(token) => {
                    if first_token_at.is_none() {
                        first_token_at = Some(std::time::Instant::now());
                    }
                    visible_tokens = visible_tokens.saturating_add(1);
                    append_live_stream_delta_for_request(
                        &state_for_stream,
                        &request_id,
                        if buffer_tool_content { "content_buffered" } else { "content" },
                        &token,
                    ).await;
                    if !buffer_tool_content {
                        let delta = serde_json::json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": { "type": "text_delta", "text": token }
                        });
                        yield Ok(anthropic_sse("content_block_delta", delta));
                    }
                }
                StreamEvent::ReasoningDelta(reasoning) => {
                    raw_full_text.push_str(&reasoning);
                    append_live_stream_delta_for_request(&state_for_stream, &request_id, "reasoning", &reasoning).await;
                }
                StreamEvent::Done {
                    full_text,
                    tokens_predicted,
                    tokens_evaluated,
                    decode_tokens_per_second,
                    prompt_tokens_per_second,
                    stopped_limit,
                    stop_type,
                } => {
                    if block_open {
                        yield Ok(anthropic_sse("content_block_stop", serde_json::json!({
                            "type": "content_block_stop",
                            "index": 0
                        })));
                    }
                    let response = CompletionResponse {
                        content: full_text.clone(),
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
                    let anthropic = anthropic_response_from_completion(
                        Some(&client),
                        &profile,
                        &model_name,
                        &response,
                        requested_tools.as_ref(),
                        tool_argument_repair_enabled,
                    ).await;

                    if buffer_tool_content {
                        for (index, block) in anthropic.content.iter().enumerate() {
                            yield Ok(anthropic_sse(
                                "content_block_start",
                                anthropic_content_block_start_event(index, block),
                            ));
                            if let Some(delta) = anthropic_content_block_delta_event(index, block) {
                                yield Ok(anthropic_sse("content_block_delta", delta));
                            }
                            yield Ok(anthropic_sse("content_block_stop", serde_json::json!({
                                "type": "content_block_stop",
                                "index": index
                            })));
                        }
                    }

                    let elapsed_ms = generation_started.elapsed().as_millis() as u64;
                    {
                        let mut s = state_for_stream.write().await;
                        s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
                            source: "api".to_string(),
                            model: model_name.clone(),
                            request_id: request_id.clone(),
                            started_at: generation_started_at.clone(),
                            finished_at: chrono::Utc::now().to_rfc3339(),
                            elapsed_ms,
                            time_to_first_token_ms: first_token_at.map(|t| t.duration_since(generation_started).as_millis() as u64),
                            prompt_tokens: Some(tokens_evaluated),
                            completion_tokens: Some(tokens_predicted),
                            total_tokens: Some(tokens_evaluated + tokens_predicted),
                            prompt_tokens_per_second,
                            decode_tokens_per_second: Some(decode_tokens_per_second),
                            end_to_end_tokens_per_second: crate::api::completions::end_to_end_tokens_per_second(
                                Some(tokens_predicted),
                                elapsed_ms,
                            ),
                        });
                    }

                    yield Ok(anthropic_sse("message_delta", serde_json::json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": anthropic.stop_reason,
                            "stop_sequence": serde_json::Value::Null
                        },
                        "usage": { "output_tokens": tokens_predicted }
                    })));
                    yield Ok(anthropic_sse("message_stop", serde_json::json!({
                        "type": "message_stop"
                    })));
                    finish_api_generation_for_request(&state_for_stream, &request_id, "completed").await;
                    let _ = &guard;
                }
                StreamEvent::Error(error) => {
                    yield Ok(anthropic_sse("error", serde_json::json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": error }
                    })));
                    finish_api_generation_for_request(&state_for_stream, &request_id, "error").await;
                }
            }
        }
    };

    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn anthropic_sse(event: &str, data: serde_json::Value) -> Event {
    Event::default().event(event).data(data.to_string())
}

fn anthropic_content_block_start_event(
    index: usize,
    block: &serde_json::Value,
) -> serde_json::Value {
    match block.get("type").and_then(|value| value.as_str()) {
        Some("tool_use") => serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {
                "type": "tool_use",
                "id": block
                    .get("id")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!(format!("toolu_{}", uuid::Uuid::new_v4()))),
                "name": block.get("name").cloned().unwrap_or_else(|| serde_json::json!("tool")),
                "input": {}
            }
        }),
        Some("thinking") => serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": { "type": "thinking", "thinking": "" }
        }),
        _ => serde_json::json!({
            "type": "content_block_start",
            "index": index,
            "content_block": { "type": "text", "text": "" }
        }),
    }
}

fn anthropic_content_block_delta_event(
    index: usize,
    block: &serde_json::Value,
) -> Option<serde_json::Value> {
    match block.get("type").and_then(|value| value.as_str()) {
        Some("tool_use") => {
            let input = block
                .get("input")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            Some(serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string())
                }
            }))
        }
        Some("thinking") => block
            .get("thinking")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(|thinking| {
                serde_json::json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "thinking_delta", "thinking": thinking }
                })
            }),
        _ => block
            .get("text")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(|text| {
                serde_json::json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": { "type": "text_delta", "text": text }
                })
            }),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        anthropic_content_block_delta_event, anthropic_content_block_start_event,
        anthropic_image_block_to_data_url, anthropic_to_chat_request, AnthropicMessagesRequest,
    };

    #[test]
    fn translates_anthropic_tools_to_openai_tools() {
        let request: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "local",
            "max_tokens": 32,
            "system": "You are terse.",
            "messages": [{ "role": "user", "content": "weather?" }],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "required": ["city"],
                    "properties": { "city": { "type": "string" } }
                }
            }]
        }))
        .expect("valid anthropic request");

        let chat = match anthropic_to_chat_request(request) {
            Ok(chat) => chat,
            Err(_) => panic!("translation should succeed"),
        };
        assert_eq!(chat.messages.len(), 2);
        let tools = chat.tools.expect("tools should translate");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["parameters"]["required"][0], "city");
    }

    #[test]
    fn streams_tool_use_input_as_json_delta() {
        let block = serde_json::json!({
            "type": "tool_use",
            "id": "toolu_123",
            "name": "lookup",
            "input": { "query": "qwen vision" }
        });

        let start = anthropic_content_block_start_event(0, &block);
        assert_eq!(start["content_block"]["type"], "tool_use");
        assert_eq!(start["content_block"]["input"], serde_json::json!({}));

        let delta = anthropic_content_block_delta_event(0, &block).expect("tool delta");
        assert_eq!(delta["delta"]["type"], "input_json_delta");
        assert_eq!(delta["delta"]["partial_json"], r#"{"query":"qwen vision"}"#);
    }

    #[test]
    fn accepts_anthropic_url_image_sources() {
        let block = serde_json::json!({
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.test/image.png"
            }
        });

        assert_eq!(
            anthropic_image_block_to_data_url(&block).as_deref(),
            Some("https://example.test/image.png")
        );
    }
}
