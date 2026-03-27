use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::api::errors::ApiErrorResponse;
use crate::engine::client::{CompletionRequest, ImageData, LlamaClient};
use crate::engine::streaming::{self, StreamEvent};
use crate::models::profiles::ModelProfile;
use crate::normalize::images::normalize_image_payload as normalize_image_payload_shared;
use crate::normalize::think_strip::{
    extract_reasoning_content_with_style, strip_think_tags_with_style,
};
use crate::state::{
    SharedState, begin_api_generation, finish_api_generation, summarize_reasoning_tokens,
};

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ApiMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, alias = "max_completion_tokens", alias = "maxTokens")]
    pub max_tokens: Option<u32>,
    #[serde(default, alias = "temp")]
    pub temperature: Option<f32>,
    #[serde(default, alias = "topP")]
    pub top_p: Option<f32>,
    #[serde(default, alias = "topK")]
    pub top_k: Option<i32>,
    #[serde(default, alias = "minP")]
    pub min_p: Option<f32>,
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
    pub seed: Option<i64>,
    pub stop: Option<StopParam>,
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(
        default,
        alias = "contextLength",
        alias = "context_length",
        alias = "contextlength",
        alias = "context_size",
        alias = "ctx_size",
        alias = "n_ctx",
        alias = "maxContextLength",
        // Ollama format (num_ctx inside top-level or inside options object)
        alias = "num_ctx",
        alias = "numCtx"
    )]
    pub context_size: Option<u32>,
    #[serde(default)]
    pub top: Option<TopParam>,
    #[serde(default)]
    pub reasoning: Option<ReasoningRequest>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Ollama-format options object — e.g. {"num_ctx": 32768}.
    /// Context size is extracted from here via requested_context_size() if not set
    /// at the top level.
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ReasoningRequest {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum TopParam {
    Integer(i64),
    Float(f64),
}

impl TopParam {
    fn as_top_p(&self) -> Option<f32> {
        match self {
            TopParam::Float(value) if (0.0..=1.0).contains(value) => Some(*value as f32),
            TopParam::Integer(value) if (0..=1).contains(value) => Some(*value as f32),
            _ => None,
        }
    }

    fn as_top_k(&self) -> Option<i32> {
        match self {
            TopParam::Integer(value) if *value > 1 => i32::try_from(*value).ok(),
            TopParam::Float(value) if *value > 1.0 && value.fract() == 0.0 => {
                i32::try_from(*value as i64).ok()
            }
            _ => None,
        }
    }
}

impl ChatCompletionRequest {
    pub fn requested_context_size(&self) -> Option<u32> {
        self.context_size
            .filter(|value| *value > 0)
            // Check Ollama-style options object: {"options": {"num_ctx": 32768}}
            .or_else(|| {
                self.options
                    .as_ref()
                    .and_then(|v| extract_context_size_from_value(v))
            })
            .or_else(|| extract_context_size_from_hash_map(&self.extra))
    }

    fn requested_top_p(&self) -> Option<f32> {
        self.top_p
            .or_else(|| self.top.as_ref().and_then(TopParam::as_top_p))
    }

    fn requested_top_k(&self) -> Option<i32> {
        self.top_k
            .or_else(|| self.top.as_ref().and_then(TopParam::as_top_k))
    }
}

const CONTEXT_SIZE_KEYS: &[&str] = &[
    "contextLength",
    "context_length",
    "contextlength",
    "context_size",
    "ctx_size",
    "ctxSize",
    "n_ctx",
    "nCtx",
    // Ollama format (also used by HelixClaw when sending to Ollama-compatible endpoints)
    "num_ctx",
    "numCtx",
    "maxContextLength",
    "max_context_length",
    "contextWindow",
    "context_window",
];

fn parse_context_size_string(text: &str) -> Option<u32> {
    let normalized = text.trim().to_ascii_lowercase().replace('_', "");
    if normalized.is_empty() {
        return None;
    }

    if let Ok(value) = normalized.parse::<u32>() {
        return (value > 0).then_some(value);
    }

    let stripped = normalized
        .trim_end_matches("tokens")
        .trim_end_matches("token")
        .trim();

    if let Some(number) = stripped.strip_suffix('k') {
        return number
            .trim()
            .parse::<f64>()
            .ok()
            .map(|value| (value * 1024.0).round() as u32)
            .filter(|value| *value > 0);
    }

    if let Some(number) = stripped.strip_suffix('m') {
        return number
            .trim()
            .parse::<f64>()
            .ok()
            .map(|value| (value * 1024.0 * 1024.0).round() as u32)
            .filter(|value| *value > 0);
    }

    None
}

pub(crate) fn extract_context_size_from_value(value: &serde_json::Value) -> Option<u32> {
    match value {
        serde_json::Value::Number(number) => number
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0),
        serde_json::Value::String(text) => parse_context_size_string(text),
        serde_json::Value::Object(map) => extract_context_size_from_json_map(map),
        serde_json::Value::Array(values) => values
            .iter()
            .find_map(extract_context_size_from_value),
        _ => None,
    }
}

pub(crate) fn extract_context_size_from_json_map(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Option<u32> {
    for key in CONTEXT_SIZE_KEYS {
        if let Some(value) = map.get(*key).and_then(extract_context_size_from_value) {
            return Some(value);
        }
    }

    for value in map.values() {
        if let Some(value) = extract_context_size_from_value(value) {
            return Some(value);
        }
    }

    None
}

pub(crate) fn extract_context_size_from_hash_map(
    map: &HashMap<String, serde_json::Value>,
) -> Option<u32> {
    for key in CONTEXT_SIZE_KEYS {
        if let Some(value) = map.get(*key).and_then(extract_context_size_from_value) {
            return Some(value);
        }
    }

    for value in map.values() {
        if let Some(value) = extract_context_size_from_value(value) {
            return Some(value);
        }
    }

    None
}

#[derive(Debug, Deserialize, Clone)]
pub struct ApiMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<ApiMessageContent>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub refusal: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ApiMessageContent {
    Text(String),
    Parts(Vec<ApiContentPart>),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApiContentPart {
    Text { text: String },
    InputText { text: String },
    ImageUrl { image_url: ApiImageUrl },
    InputImage {
        #[serde(default)]
        image_url: Option<ApiImageUrl>,
        #[serde(default)]
        image_base64: Option<String>,
    },
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ApiImageUrl {
    String(String),
    Object { url: String },
}

impl ApiImageUrl {
    fn into_url(self) -> String {
        match self {
            ApiImageUrl::String(url) => url,
            ApiImageUrl::Object { url } => url,
        }
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(crate) fn end_to_end_tokens_per_second(
    completion_tokens: Option<u32>,
    elapsed_ms: u64,
) -> Option<f64> {
    if elapsed_ms == 0 {
        return None;
    }

    completion_tokens.map(|tokens| tokens as f64 / (elapsed_ms as f64 / 1000.0))
}

pub(crate) fn build_parse_trace(profile: &ModelProfile, raw: &str, stripped: &str) -> String {
    let (tool_calls, visible_text) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(stripped, profile);
    let reasoning_text = extract_reasoning_content_with_style(raw, profile.think_tag_style);
    serde_json::to_string_pretty(&serde_json::json!({
        "parser_type": format!("{:?}", profile.parser_type),
        "tool_call_format": format!("{:?}", profile.tool_call_format),
        "think_tag_style": format!("{:?}", profile.think_tag_style),
        "raw_response": raw,
        "reasoning_text": reasoning_text,
        "stripped_response": stripped,
        "visible_text": visible_text,
        "tool_calls": tool_calls,
    }))
    .unwrap_or_else(|_| "Failed to serialize parse trace".to_string())
}

fn loaded_model_matches_request(loaded: &str, requested: &str) -> bool {
    let loaded = loaded.to_ascii_lowercase();
    let requested = requested.trim().to_ascii_lowercase();

    loaded == requested
        || loaded.trim_end_matches(".gguf") == requested
        || loaded == requested.trim_end_matches(".gguf")
        || (!requested.is_empty() && loaded.contains(&requested))
}

async fn swap_model_for_api(
    state: &SharedState,
    model_name: &str,
    context_size: Option<u32>,
) -> Result<(), String> {
    crate::commands::model::backend_load_model(state.clone(), model_name.to_string(), context_size)
        .await
        .map(|_| ())
}

pub(crate) async fn resolve_loaded_model(
    state: &SharedState,
    requested_model: &str,
    requested_context_size: Option<u32>,
) -> Result<String, ApiErrorResponse> {
    let (target_model, needs_swap) = {
        let s = state.read().await;
        let target_model = if requested_model.trim().is_empty() {
            s.loaded_model.clone().ok_or_else(ApiErrorResponse::no_model)?
        } else {
            requested_model.trim().to_string()
        };

        // Only reload if the model NAME mismatches. Never reload just because
        // the request includes a different context size — that would cause a
        // full multi-minute reload on every call (LM Studio never does this).
        let needs_swap = match &s.loaded_model {
            Some(loaded) => !loaded_model_matches_request(loaded, &target_model),
            None => true,
        };

        (target_model, needs_swap)
    };

    if needs_swap {
        swap_model_for_api(state, &target_model, requested_context_size).await.map_err(|e| {
            ApiErrorResponse::service_unavailable(format!(
                "Could not load model '{target_model}': {e}"
            ))
        })?;
    }

    let s = state.read().await;
    s.loaded_model.clone().ok_or_else(ApiErrorResponse::no_model)
}

async fn normalize_image_payload(value: &str) -> Result<String, ApiErrorResponse> {
    normalize_image_payload_shared(value)
        .await
        .map_err(ApiErrorResponse::bad_request)
}

pub(crate) async fn normalize_api_messages(
    messages: &[ApiMessage],
) -> Result<(Vec<crate::templates::engine::ChatMessage>, Vec<ImageData>), ApiErrorResponse> {
    let mut normalized = Vec::with_capacity(messages.len());
    let mut image_data = Vec::new();
    let mut next_image_id = 1u32;

    for message in messages {
        let mut content_parts = Vec::new();

        match message.content.clone() {
            None => {}
            Some(ApiMessageContent::Text(text)) => {
                if !text.is_empty() {
                    content_parts.push(text);
                }
            }
            Some(ApiMessageContent::Parts(parts)) => {
                for part in parts {
                    match part {
                        ApiContentPart::Text { text } | ApiContentPart::InputText { text } => {
                            if !text.is_empty() {
                                content_parts.push(text);
                            }
                        }
                        ApiContentPart::ImageUrl { image_url } => {
                            let image_id = next_image_id;
                            next_image_id += 1;
                            content_parts.push(format!("[img-{image_id}]"));
                            image_data.push(ImageData {
                                data: normalize_image_payload(&image_url.into_url()).await?,
                                id: image_id,
                            });
                        }
                        ApiContentPart::InputImage {
                            image_url,
                            image_base64,
                        } => {
                            let raw = if let Some(image_base64) = image_base64 {
                                normalize_image_payload(&image_base64).await?
                            } else if let Some(image_url) = image_url {
                                normalize_image_payload(&image_url.into_url()).await?
                            } else {
                                return Err(ApiErrorResponse::bad_request(
                                    "input_image parts require image_url or image_base64.",
                                ));
                            };
                            let image_id = next_image_id;
                            next_image_id += 1;
                            content_parts.push(format!("[img-{image_id}]"));
                            image_data.push(ImageData { data: raw, id: image_id });
                        }
                    }
                }
            }
        }

        if let Some(name) = &message.name {
            content_parts.insert(0, format!("[name:{name}]"));
        }

        if let Some(tool_call_id) = &message.tool_call_id {
            content_parts.insert(0, format!("[tool_call_id:{tool_call_id}]"));
        }

        if let Some(refusal) = &message.refusal {
            if !refusal.is_empty() {
                content_parts.push(format!("[refusal]\n{refusal}"));
            }
        }

        if let Some(tool_calls) = &message.tool_calls {
            if !tool_calls.is_empty() {
                let serialized = serde_json::to_string_pretty(tool_calls)
                    .unwrap_or_else(|_| "[]".to_string());
                content_parts.push(format!("[tool_calls]\n{serialized}"));
            }
        }

        normalized.push(crate::templates::engine::ChatMessage {
            role: message.role.clone(),
            content: content_parts.join("\n"),
        });
    }

    Ok((normalized, image_data))
}

fn prepend_tool_schema_message(
    messages: &mut Vec<crate::templates::engine::ChatMessage>,
    tools: Option<&Vec<serde_json::Value>>,
) {
    let Some(tools) = tools.filter(|tools| !tools.is_empty()) else {
        return;
    };

    let serialized = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_string());
    messages.insert(
        0,
        crate::templates::engine::ChatMessage {
            role: "system".to_string(),
            content: format!(
                "Available tools (OpenAI-style schema):\n{serialized}\n\nIf a tool is needed, reply using the tool-calling format appropriate for your model family."
            ),
        },
    );
}

pub(crate) async fn build_chat_request(
    profile: &ModelProfile,
    req: ChatCompletionRequest,
    server_defaults: (&Option<f32>, &Option<f32>, &Option<i32>, &Option<u32>),
) -> Result<CompletionRequest, ApiErrorResponse> {
    let requested_top_p = req.requested_top_p();
    let requested_top_k = req.requested_top_k();
    let (mut messages, image_data) = normalize_api_messages(&req.messages).await?;
    prepend_tool_schema_message(&mut messages, req.tools.as_ref());
    prepend_reasoning_guidance_message(&mut messages, profile, req.reasoning.as_ref(), req.reasoning_effort.as_deref(), req.reasoning_tokens);

    let mut stop = profile.stop_markers.clone();
    if let Some(user_stop) = req.stop {
        stop.extend(user_stop.into_vec());
    }

    Ok(CompletionRequest {
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
        top_p: requested_top_p.or(profile.default_top_p).or(*server_defaults.1),
        top_k: requested_top_k.or(profile.default_top_k).or(*server_defaults.2),
        min_p: req.min_p.or(profile.default_min_p),
        presence_penalty: req.presence_penalty.or(profile.default_presence_penalty),
        frequency_penalty: req.frequency_penalty,
        repeat_penalty: req.repetition_penalty,
        seed: req.seed,
        stream: req.stream,
        stop,
        special: true,
        image_data,
    })
}

fn launch_preview_matches_model(model_name: &str, preview_model_path: &str) -> bool {
    let requested = model_name.trim().to_ascii_lowercase();
    let preview_name = std::path::Path::new(preview_model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(preview_model_path)
        .trim()
        .to_ascii_lowercase();

    preview_name == requested
        || preview_name.trim_end_matches(".gguf") == requested
        || preview_name == requested.trim_end_matches(".gguf")
        || (!requested.is_empty() && preview_name.contains(&requested))
}

pub(crate) async fn ensure_runtime_vision_ready(
    state: &SharedState,
    model_name: &str,
    profile: &ModelProfile,
    has_images: bool,
) -> Result<(), ApiErrorResponse> {
    if !has_images {
        return Ok(());
    }

    if !profile.supports_vision {
        return Err(ApiErrorResponse::bad_request(format!(
            "The loaded model '{model_name}' does not advertise vision support. Load a vision-capable model first."
        )));
    }

    let launch_preview = {
        let s = state.read().await;
        s.last_launch_preview.clone()
    };

    let matching_preview = launch_preview
        .as_ref()
        .filter(|preview| launch_preview_matches_model(model_name, &preview.model_path));

    if matching_preview
        .and_then(|preview| preview.mmproj_path.as_ref())
        .is_some()
    {
        return Ok(());
    }

    Err(ApiErrorResponse::bad_request(format!(
        "The loaded model '{model_name}' was started without a matching mmproj sidecar, so image input is not actually available. Reload a vision-ready model first."
    )))
}

fn prepend_reasoning_guidance_message(
    messages: &mut Vec<crate::templates::engine::ChatMessage>,
    profile: &ModelProfile,
    reasoning: Option<&ReasoningRequest>,
    reasoning_effort: Option<&str>,
    reasoning_tokens: Option<u32>,
) {
    let supports_reasoning = !matches!(profile.think_tag_style, crate::models::profiles::ThinkTagStyle::None);
    let requested_effort = reasoning
        .and_then(|cfg| cfg.effort.as_deref())
        .or(reasoning_effort)
        .map(|value| value.trim().to_ascii_lowercase());
    let requested_reasoning_tokens = reasoning.and_then(|cfg| cfg.max_tokens).or(reasoning_tokens);

    if !supports_reasoning && requested_effort.is_none() && requested_reasoning_tokens.is_none() {
        return;
    }

    let mut guidance = Vec::new();
    match requested_effort.as_deref() {
        Some("none") => guidance.push("Respond directly without emitting <think> blocks.".to_string()),
        Some("low") => guidance.push("Keep reasoning brief and concise.".to_string()),
        Some("high") => guidance.push("Use thorough step-by-step reasoning before the final answer.".to_string()),
        Some("xhigh") => guidance.push("Use very detailed reasoning before the final answer.".to_string()),
        Some("medium") => {}
        Some(other) => guidance.push(format!("Reasoning effort requested: {other}.")),
        None => {}
    }

    if let Some(limit) = requested_reasoning_tokens {
        guidance.push(format!("Keep reasoning under roughly {limit} tokens."));
    }

    if guidance.is_empty() {
        return;
    }

    messages.insert(
        0,
        crate::templates::engine::ChatMessage {
            role: "system".to_string(),
            content: guidance.join(" "),
        },
    );
}

fn build_usage(
    prompt_tokens: u32,
    completion_tokens: u32,
    reasoning_tokens: u32,
    include_details: bool,
) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens_details: include_details.then(|| PromptTokensDetails { cached_tokens: 0 }),
        completion_tokens_details: include_details
            .then(|| CompletionTokensDetails { reasoning_tokens }),
    }
}

pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiErrorResponse> {
    let include_usage_details = req
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(true);
    let requested_context_size = req.requested_context_size();
    let requested_model = req.model.clone().unwrap_or_default();
    let model_name = resolve_loaded_model(&state, &requested_model, requested_context_size).await?;
    let profile = crate::models::overrides::detect_effective_profile(&model_name);

    let (server_defaults, scheduler, llama_port) = {
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
    )
    .await?;

    ensure_runtime_vision_ready(&state, &model_name, &profile, !request.image_data.is_empty())
        .await?;
    {
        let mut s = state.write().await;
        s.last_prompt = Some(request.prompt.clone());
    }
    let _permit = scheduler.acquire().await;

    let client = LlamaClient::new(llama_port);
    let generation_started_at = chrono::Utc::now().to_rfc3339();
    let generation_started = std::time::Instant::now();

    if request.stream {
        return stream_chat_completion(
            state,
            client,
            request,
            model_name,
            profile,
            generation_started_at,
            generation_started,
            include_usage_details,
        )
        .await;
    }

    let response = client.complete(&request).await.map_err(|e| {
        tracing::error!(error = %e, "Completion failed");
        ApiErrorResponse::inference_failed(&e.to_string())
    })?;

    let reasoning_text =
        extract_reasoning_content_with_style(&response.content, profile.think_tag_style);
    let stripped = strip_think_tags_with_style(&response.content, profile.think_tag_style);
    let reasoning_tokens =
        summarize_reasoning_tokens(response.tokens_predicted, &stripped, &reasoning_text);
    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(build_parse_trace(&profile, &response.content, &stripped));
        s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
            source: "api".to_string(),
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
            prompt_tokens_per_second: response
                .timings
                .as_ref()
                .and_then(|timings| timings.prompt_per_second),
            decode_tokens_per_second: response
                .timings
                .as_ref()
                .and_then(|timings| timings.predicted_per_second),
            end_to_end_tokens_per_second: end_to_end_tokens_per_second(
                response.tokens_predicted,
                generation_started.elapsed().as_millis() as u64,
            ),
        });
    }
    let (tool_calls, text) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(&stripped, &profile);
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
                reasoning: (!reasoning_text.is_empty()).then_some(reasoning_text),
                tool_calls: api_tool_calls.clone(),
            },
            finish_reason: if api_tool_calls.is_empty() {
                "stop".to_string()
            } else {
                "tool_calls".to_string()
            },
        }],
        usage: build_usage(
            response.tokens_evaluated.unwrap_or(0),
            response.tokens_predicted.unwrap_or(0),
            reasoning_tokens,
            include_usage_details,
        ),
    })
    .into_response())
}

async fn stream_chat_completion(
    state: SharedState,
    client: LlamaClient,
    request: CompletionRequest,
    model_name: String,
    profile: ModelProfile,
    generation_started_at: String,
    generation_started: std::time::Instant,
    include_usage_details: bool,
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

        let opening_chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant" },
                "finish_reason": serde_json::Value::Null
            }]
        });
        yield Ok::<Event, std::convert::Infallible>(Event::default().data(opening_chunk.to_string()));

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
                    let chunk_json = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": { "reasoning": reasoning },
                            "finish_reason": serde_json::Value::Null
                        }]
                    });
                    yield Ok::<Event, std::convert::Infallible>(Event::default().data(chunk_json.to_string()));
                }
                StreamEvent::Done {
                    full_text,
                    tokens_predicted,
                    tokens_evaluated,
                    decode_tokens_per_second,
                    prompt_tokens_per_second,
                } => {
                    let reasoning_text =
                        extract_reasoning_content_with_style(&full_text, profile.think_tag_style);
                    let stripped = strip_think_tags_with_style(&full_text, profile.think_tag_style);
                    let reasoning_tokens = summarize_reasoning_tokens(
                        Some(tokens_predicted),
                        &stripped,
                        &reasoning_text,
                    );
                    let mut s = state_for_stream.write().await;
                    s.last_parse_trace = Some(build_parse_trace(&profile, &full_text, &stripped));
                    s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
                        source: "api".to_string(),
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
                        "usage": build_usage(
                            tokens_evaluated,
                            tokens_predicted,
                            reasoning_tokens,
                            include_usage_details,
                        )
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
            let stripped = strip_think_tags_with_style(&raw_full_text, profile.think_tag_style);
            let mut s = state_for_stream.write().await;
            s.last_parse_trace = Some(build_parse_trace(&profile, &raw_full_text, &stripped));
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
    #[serde(default, alias = "maxTokens")]
    pub max_tokens: Option<u32>,
    #[serde(default, alias = "temp")]
    pub temperature: Option<f32>,
    #[serde(default, alias = "topP")]
    pub top_p: Option<f32>,
    #[serde(default, alias = "topK")]
    pub top_k: Option<i32>,
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
    pub top: Option<TopParam>,
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
    let requested_context_size = req.context_size.filter(|value| *value > 0);
    let requested_model = req.model.clone().unwrap_or_default();
    let model_name =
        resolve_loaded_model(&state, &requested_model, requested_context_size).await?;
    let profile = crate::models::overrides::detect_effective_profile(&model_name);

    let prompt = req.prompt.unwrap_or_default();
    if prompt.is_empty() {
        return Err(ApiErrorResponse::bad_request(
            "The `prompt` field is required and must not be empty.",
        ));
    }

    let (srv_temp, srv_top_p, srv_top_k, srv_max_tokens, port, scheduler) = {
        let s = state.read().await;
        (
            s.config.server.default_temperature,
            s.config.server.default_top_p,
            s.config.server.default_top_k,
            s.config.server.default_max_tokens,
            s.process.port(),
            s.request_scheduler.clone(),
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
        top_p: req
            .top_p
            .or_else(|| req.top.as_ref().and_then(TopParam::as_top_p))
            .or(srv_top_p)
            .or(profile.default_top_p),
        top_k: req
            .top_k
            .or_else(|| req.top.as_ref().and_then(TopParam::as_top_k))
            .or(srv_top_k)
            .or(profile.default_top_k),
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
    let _permit = scheduler.acquire().await;

    let client = LlamaClient::new(port);
    let response = client.complete(&completion_req).await.map_err(|e| {
        tracing::error!(error = %e, "Text completion failed");
        ApiErrorResponse::inference_failed(&e.to_string())
    })?;

    let reasoning_text =
        extract_reasoning_content_with_style(&response.content, profile.think_tag_style);
    let text = strip_think_tags_with_style(&response.content, profile.think_tag_style);
    let reasoning_tokens =
        summarize_reasoning_tokens(response.tokens_predicted, &text, &reasoning_text);
    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(build_parse_trace(&profile, &response.content, &text));
    }

    Ok(Json(TextCompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: now_unix_secs(),
        model: model_name,
        choices: vec![TextChoice {
            index: 0,
            text: text.clone(),
            finish_reason: "stop".to_string(),
        }],
        usage: build_usage(
            response.tokens_evaluated.unwrap_or(0),
            response.tokens_predicted.unwrap_or(0),
            reasoning_tokens,
            true,
        ),
    })
    .into_response())
}

#[cfg(test)]
mod tests {
    use super::{normalize_api_messages, ApiMessage, ChatCompletionRequest, loaded_model_matches_request};

    #[test]
    fn deserializes_string_message_content() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.messages.len(), 1);
    }

    #[tokio::test]
    async fn deserializes_array_message_content() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": "hello" },
                            { "type": "text", "text": "world" }
                        ]
                    }
                ]
            }"#,
        )
        .expect("request should deserialize");

        let (messages, image_data) = match normalize_api_messages(&request.messages).await {
            Ok(value) => value,
            Err(_) => panic!("messages should normalize"),
        };
        assert_eq!(messages[0].content, "hello\nworld");
        assert!(image_data.is_empty());
    }

    #[tokio::test]
    async fn normalizes_data_url_images() {
        let messages = vec![ApiMessage {
            role: "user".to_string(),
            content: Some(serde_json::from_str(
                r#"[
                    { "type": "text", "text": "what is in this image?" },
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } }
                ]"#,
            )
            .expect("content should deserialize")),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            refusal: None,
        }];

        let (normalized, image_data) = match normalize_api_messages(&messages).await {
            Ok(value) => value,
            Err(_) => panic!("messages should normalize"),
        };
        assert!(normalized[0].content.contains("[img-1]"));
        assert_eq!(image_data.len(), 1);
        assert_eq!(image_data[0].data, "AAAA");
    }

    #[test]
    fn supports_max_completion_tokens_alias() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "max_completion_tokens": 321,
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.max_tokens, Some(321));
    }

    #[test]
    fn deserializes_reasoning_effort_and_tokens() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "reasoning_effort": "low",
                "reasoning_tokens": 128,
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(request.reasoning_tokens, Some(128));
    }

    #[test]
    fn deserializes_nested_reasoning_config() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "reasoning": { "effort": "high", "max_tokens": 256 },
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(
            request.reasoning.as_ref().and_then(|value| value.effort.as_deref()),
            Some("high")
        );
        assert_eq!(request.reasoning.as_ref().and_then(|value| value.max_tokens), Some(256));
    }

    #[test]
    fn deserializes_context_length_and_vendor_sampling_aliases() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "contextLength": 32768,
                "topK": 40,
                "minP": 0.07,
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.requested_context_size(), Some(32768));
        assert_eq!(request.requested_top_k(), Some(40));
        assert_eq!(request.min_p, Some(0.07));
    }

    #[test]
    fn deserializes_nested_context_length_aliases() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [
                    { "role": "user", "content": "hello" }
                ],
                "loadConfig": {
                    "context_length": 32768
                }
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.requested_context_size(), Some(32768));
    }

    #[test]
    fn finds_context_size_in_arbitrary_nested_objects() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [
                    { "role": "user", "content": "hello" }
                ],
                "session": {
                    "modelPreferences": {
                        "runtime": {
                            "ctxSize": 32768
                        }
                    }
                }
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.requested_context_size(), Some(32768));
    }

    #[test]
    fn parses_context_size_strings_with_k_suffix() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [
                    { "role": "user", "content": "hello" }
                ],
                "loadConfig": {
                    "contextWindow": "32k"
                }
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.requested_context_size(), Some(32768));
    }

    #[test]
    fn infers_top_p_or_top_k_from_generic_top_field() {
        let top_k_request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "top": 40,
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");
        assert_eq!(top_k_request.requested_top_k(), Some(40));
        assert_eq!(top_k_request.requested_top_p(), None);

        let top_p_request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "top": 0.92,
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");
        assert_eq!(top_p_request.requested_top_p(), Some(0.92));
        assert_eq!(top_p_request.requested_top_k(), None);
    }

    #[test]
    fn matches_loaded_models_without_exact_filename_suffix() {
        assert!(loaded_model_matches_request(
            "Qwen3.5-9B-Q4_K_S.gguf",
            "Qwen3.5-9B-Q4_K_S"
        ));
        assert!(loaded_model_matches_request(
            "Qwen3.5-9B-Q4_K_S.gguf",
            "Qwen3.5-9B"
        ));
    }

    #[tokio::test]
    async fn preserves_tool_metadata_in_message_history() {
        let messages = vec![ApiMessage {
            role: "assistant".to_string(),
            content: None,
            name: Some("planner".to_string()),
            tool_call_id: Some("call_123".to_string()),
            tool_calls: Some(vec![serde_json::json!({
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "arguments": "{\"query\":\"qwen\"}"
                }
            })]),
            refusal: None,
        }];

        let (normalized, _) = match normalize_api_messages(&messages).await {
            Ok(value) => value,
            Err(_) => panic!("messages should normalize"),
        };

        assert!(normalized[0].content.contains("[name:planner]"));
        assert!(normalized[0].content.contains("[tool_call_id:call_123]"));
        assert!(normalized[0].content.contains("[tool_calls]"));
    }
}
