use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::api::errors::ApiErrorResponse;
use crate::commands::model::RuntimeLoadOverrides;
use crate::engine::client::{
    CompletionRequest, CompletionResponse, ImageData, LlamaClient, Timings,
};
use crate::engine::process::{LaunchPreview, ProcessState};
use crate::engine::streaming::{self, StreamEvent};
use crate::models::profiles::{ModelProfile, ToolCallFormat};
use crate::normalize::images::normalize_image_payload as normalize_image_payload_shared;
use crate::normalize::think_strip::{
    extract_reasoning_content_with_style, strip_think_tags_with_style,
};
use crate::state::{
    append_live_stream_delta_for_request, begin_api_generation, finish_api_generation_for_request,
    summarize_reasoning_tokens, SharedState,
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

    fn requested_thinking_disabled(&self) -> bool {
        ["enable_thinking", "enableThinking", "thinking"]
            .iter()
            .filter_map(|key| self.extra.get(*key))
            .any(|value| matches!(value, serde_json::Value::Bool(false)))
            || self
                .reasoning
                .as_ref()
                .and_then(|cfg| cfg.effort.as_deref())
                .or(self.reasoning_effort.as_deref())
                .map(|effort| effort.eq_ignore_ascii_case("none"))
                .unwrap_or(false)
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

const HF_REPO_KEYS: &[&str] = &["hf_repo", "hfRepo", "repo_id", "repoId", "repository"];
const HF_FILE_KEYS: &[&str] = &["hf_file", "hfFile", "file", "filename", "quant"];
const FIT_MODE_KEYS: &[&str] = &["fit_mode", "fitMode", "fit"];
const CACHE_RAM_KEYS: &[&str] = &["cache_ram_mb", "cacheRamMb", "cache_ram", "cacheRam"];
const CTXCP_KEYS: &[&str] = &["ctxcp", "ctx_cp"];
const JINJA_KEYS: &[&str] = &["use_jinja", "useJinja", "jinja"];
const REASONING_MODE_KEYS: &[&str] = &["reasoning_mode", "reasoningMode"];
const TEMPLATE_MODE_KEYS: &[&str] = &["template_mode", "templateMode"];
const TEMPLATE_NAME_KEYS: &[&str] = &["template_name", "templateName", "chat_template"];
const CUSTOM_TEMPLATE_PATH_KEYS: &[&str] = &[
    "custom_template_path",
    "customTemplatePath",
    "chat_template_file",
    "chatTemplateFile",
];
const TEMPLATE_KWARGS_KEYS: &[&str] = &[
    "chat_template_kwargs_json",
    "chatTemplateKwargsJson",
    "chat_template_kwargs",
    "chatTemplateKwargs",
];
const EXTRA_ARGS_KEYS: &[&str] = &["extra_args", "extraArgs"];

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

fn find_named_value_in_value(
    value: &serde_json::Value,
    keys: &[&str],
) -> Option<serde_json::Value> {
    match value {
        serde_json::Value::Object(map) => {
            for key in keys {
                if let Some(value) = map.get(*key) {
                    return Some(value.clone());
                }
            }
            for value in map.values() {
                if let Some(found) = find_named_value_in_value(value, keys) {
                    return Some(found);
                }
            }
            None
        }
        serde_json::Value::Array(values) => values
            .iter()
            .find_map(|value| find_named_value_in_value(value, keys)),
        _ => None,
    }
}

fn find_named_value_in_hash_map(
    map: &HashMap<String, serde_json::Value>,
    keys: &[&str],
) -> Option<serde_json::Value> {
    for key in keys {
        if let Some(value) = map.get(*key) {
            return Some(value.clone());
        }
    }
    for value in map.values() {
        if let Some(found) = find_named_value_in_value(value, keys) {
            return Some(found);
        }
    }
    None
}

fn parse_string_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => {
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        _ => None,
    }
}

fn parse_u32_value(value: &serde_json::Value) -> Option<u32> {
    extract_context_size_from_value(value)
}

fn parse_bool_value(value: &serde_json::Value) -> Option<bool> {
    match value {
        serde_json::Value::Bool(value) => Some(*value),
        serde_json::Value::String(value) => match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Some(true),
            "false" | "0" | "no" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn parse_string_array(value: &serde_json::Value) -> Option<Vec<String>> {
    match value {
        serde_json::Value::Array(items) => {
            let parsed = items
                .iter()
                .filter_map(parse_string_value)
                .collect::<Vec<_>>();
            (!parsed.is_empty()).then_some(parsed)
        }
        serde_json::Value::String(text) => {
            let parsed = text
                .split(|ch| ch == ',' || ch == '\n')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            (!parsed.is_empty()).then_some(parsed)
        }
        _ => None,
    }
}

pub(crate) fn extract_runtime_load_overrides(
    options: Option<&serde_json::Value>,
    extra: &HashMap<String, serde_json::Value>,
) -> RuntimeLoadOverrides {
    let lookup = |keys: &[&str]| {
        options
            .and_then(|value| find_named_value_in_value(value, keys))
            .or_else(|| find_named_value_in_hash_map(extra, keys))
    };

    RuntimeLoadOverrides {
        hf_repo: lookup(HF_REPO_KEYS).as_ref().and_then(parse_string_value),
        hf_file: lookup(HF_FILE_KEYS).as_ref().and_then(parse_string_value),
        fit_mode: lookup(FIT_MODE_KEYS).as_ref().and_then(parse_string_value),
        cache_ram_mb: lookup(CACHE_RAM_KEYS).as_ref().and_then(parse_u32_value),
        ctxcp: lookup(CTXCP_KEYS).as_ref().and_then(parse_u32_value),
        use_jinja: lookup(JINJA_KEYS).as_ref().and_then(parse_bool_value),
        reasoning_mode: lookup(REASONING_MODE_KEYS)
            .as_ref()
            .and_then(parse_string_value),
        template_mode: lookup(TEMPLATE_MODE_KEYS)
            .as_ref()
            .and_then(parse_string_value),
        template_name: lookup(TEMPLATE_NAME_KEYS)
            .as_ref()
            .and_then(parse_string_value),
        custom_template_path: lookup(CUSTOM_TEMPLATE_PATH_KEYS)
            .as_ref()
            .and_then(parse_string_value),
        chat_template_kwargs_json: lookup(TEMPLATE_KWARGS_KEYS).map(|value| {
            if let Some(text) = parse_string_value(&value) {
                text
            } else {
                value.to_string()
            }
        }),
        extra_args: lookup(EXTRA_ARGS_KEYS)
            .as_ref()
            .and_then(parse_string_array),
    }
}

fn has_runtime_load_overrides(overrides: &RuntimeLoadOverrides) -> bool {
    overrides.hf_repo.is_some()
        || overrides.hf_file.is_some()
        || overrides.fit_mode.is_some()
        || overrides.cache_ram_mb.is_some()
        || overrides.ctxcp.is_some()
        || overrides.use_jinja.is_some()
        || overrides.reasoning_mode.is_some()
        || overrides.template_mode.is_some()
        || overrides.template_name.is_some()
        || overrides.custom_template_path.is_some()
        || overrides.chat_template_kwargs_json.is_some()
        || overrides
            .extra_args
            .as_ref()
            .map(|value| !value.is_empty())
            .unwrap_or(false)
}

pub(crate) fn extract_context_size_from_value(value: &serde_json::Value) -> Option<u32> {
    match value {
        serde_json::Value::Number(number) => number
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0),
        serde_json::Value::String(text) => parse_context_size_string(text),
        serde_json::Value::Object(map) => extract_context_size_from_json_map(map),
        serde_json::Value::Array(values) => values.iter().find_map(extract_context_size_from_value),
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
    Text {
        text: String,
    },
    InputText {
        text: String,
    },
    ImageUrl {
        image_url: ApiImageUrl,
    },
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

/// Public test helper — exposes model name matching logic.
#[cfg(test)]
pub fn loaded_model_matches_request_pub(loaded: &str, requested: &str) -> bool {
    loaded_model_matches_request(loaded, requested)
}

fn loaded_model_matches_request(loaded: &str, requested: &str) -> bool {
    let loaded = normalize_model_match_name(loaded);
    let requested = normalize_model_match_name(requested);

    // Handle path-style names like "qwen/qwen3.5-4b" → extract "qwen3.5-4b"
    let search = if requested.contains('/') {
        requested.rsplit('/').next().unwrap_or(&requested)
    } else {
        &requested
    };

    loaded == requested
        || loaded.trim_end_matches(".gguf") == requested
        || loaded == requested.trim_end_matches(".gguf")
        || loaded.trim_end_matches(".gguf") == search
        || (!search.is_empty() && loaded.contains(search))
        || model_tokens_match(&loaded, search)
}

fn normalize_model_match_name(name: &str) -> String {
    let name = name.trim().replace('\\', "/").to_ascii_lowercase();
    let name = name.rsplit('/').next().unwrap_or(&name);
    name.trim_end_matches(".gguf").to_string()
}

fn model_match_tokens(name: &str) -> Vec<String> {
    const NOISE: &[&str] = &[
        "gguf",
        "think",
        "thinking",
        "reasoning",
        "chat",
        "instruct",
        "uncensored",
    ];

    name.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '.'))
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .filter(|token| !NOISE.contains(token))
        .filter(|token| !token.starts_with('q') || !token.contains('_'))
        .map(ToOwned::to_owned)
        .collect()
}

fn model_tokens_match(loaded: &str, requested: &str) -> bool {
    let requested_tokens = model_match_tokens(requested);
    if requested_tokens.is_empty() {
        return false;
    }

    let loaded_tokens = model_match_tokens(loaded);
    requested_tokens
        .iter()
        .all(|requested| loaded_tokens.iter().any(|loaded| loaded == requested))
}

fn optional_override_matches<T: PartialEq>(requested: Option<T>, current: Option<T>) -> bool {
    requested.map_or(true, |requested| current == Some(requested))
}

fn api_overrides_match_preview(overrides: &RuntimeLoadOverrides, preview: &LaunchPreview) -> bool {
    optional_override_matches(overrides.hf_repo.as_deref(), preview.hf_repo.as_deref())
        && optional_override_matches(overrides.hf_file.as_deref(), preview.hf_file.as_deref())
        && optional_override_matches(overrides.fit_mode.as_deref(), preview.fit_mode.as_deref())
        && optional_override_matches(overrides.cache_ram_mb, preview.cache_ram_mb)
        && optional_override_matches(overrides.ctxcp, preview.ctxcp)
        && optional_override_matches(overrides.use_jinja, Some(preview.use_jinja))
        && optional_override_matches(
            overrides.reasoning_mode.as_deref(),
            preview.reasoning_mode.as_deref(),
        )
        && optional_override_matches(
            overrides.template_mode.as_deref(),
            Some(preview.template_mode.as_str()),
        )
        && optional_override_matches(
            overrides.template_name.as_deref(),
            preview.template_name.as_deref(),
        )
        && optional_override_matches(
            overrides.custom_template_path.as_deref(),
            preview.template_path.as_deref(),
        )
        && optional_override_matches(
            overrides.chat_template_kwargs_json.as_deref(),
            preview.chat_template_kwargs_json.as_deref(),
        )
        && overrides.extra_args.as_ref().map_or(true, |requested| {
            preview
                .args
                .iter()
                .rev()
                .take(requested.len())
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                == *requested
        })
}

fn context_request_matches_preview(
    requested_context_size: Option<u32>,
    preview: Option<&LaunchPreview>,
) -> bool {
    requested_context_size.map_or(true, |requested| {
        preview
            .and_then(|preview| preview.context_size)
            .map_or(false, |current| current >= requested)
    })
}

fn context_size_for_reload(
    requested_context_size: Option<u32>,
    preview: Option<&LaunchPreview>,
    loaded_context_size: Option<u32>,
) -> Option<u32> {
    requested_context_size
        .or_else(|| preview.and_then(|preview| preview.context_size))
        .or(loaded_context_size)
}

async fn publish_live_generation_metrics(
    state: &SharedState,
    source: &str,
    model: &str,
    request_id: &str,
    started_at: &str,
    generation_started: std::time::Instant,
    first_token_at: Option<std::time::Instant>,
    completion_tokens: u32,
) {
    let elapsed_ms = generation_started.elapsed().as_millis() as u64;
    let mut s = state.write().await;
    if let Some(active) = s.active_generation.as_mut() {
        active.status = format!("streaming {} token(s)", completion_tokens);
    }
    s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
        source: source.to_string(),
        model: model.to_string(),
        request_id: request_id.to_string(),
        started_at: started_at.to_string(),
        finished_at: chrono::Utc::now().to_rfc3339(),
        elapsed_ms,
        time_to_first_token_ms: first_token_at
            .map(|t| t.duration_since(generation_started).as_millis() as u64),
        prompt_tokens: None,
        completion_tokens: Some(completion_tokens),
        total_tokens: Some(completion_tokens),
        prompt_tokens_per_second: None,
        decode_tokens_per_second: end_to_end_tokens_per_second(Some(completion_tokens), elapsed_ms),
        end_to_end_tokens_per_second: end_to_end_tokens_per_second(
            Some(completion_tokens),
            elapsed_ms,
        ),
    });
}

async fn completion_failure_diagnostics(
    state: &SharedState,
    model_name: &str,
    error: &anyhow::Error,
) -> String {
    let (port, loaded_model, loaded_ctx, launch_ctx, active_generation, stderr_lines) = {
        let s = state.read().await;
        (
            s.process.port(),
            s.loaded_model.clone(),
            s.model_stats.as_ref().map(|stats| stats.context_size),
            s.last_launch_preview
                .as_ref()
                .and_then(|preview| preview.context_size),
            s.active_generation
                .as_ref()
                .map(|generation| generation.status.clone()),
            s.process.last_stderr().await,
        )
    };

    let client = LlamaClient::new(port);
    let health = match client.health().await {
        Ok(ok) => ok.to_string(),
        Err(err) => format!("error: {err}"),
    };
    let slots = match client.get_slots().await {
        Ok(slots) => {
            let first = slots.first().map(|slot| {
                let decoded = slot.next_token.as_ref().map(|token| token.n_decoded);
                format!(
                    "id={} n_ctx={} is_processing={} n_past={} decoded={:?}",
                    slot.id, slot.n_ctx, slot.is_processing, slot.n_past, decoded
                )
            });
            format!(
                "ok count={} first={}",
                slots.len(),
                first.unwrap_or_else(|| "none".to_string())
            )
        }
        Err(err) => format!("error: {err}"),
    };
    let props = match client.get_props().await {
        Ok(props) => {
            let n_ctx = props
                .default_generation_settings
                .as_ref()
                .and_then(|settings| settings.n_ctx);
            format!(
                "ok n_ctx={:?} total_slots={:?} build_info={:?}",
                n_ctx, props.total_slots, props.build_info
            )
        }
        Err(err) => format!("error: {err}"),
    };
    let stderr_tail = stderr_lines
        .iter()
        .rev()
        .take(8)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" | ");

    format!(
        "request_error={error}; model={model_name}; loaded_model={:?}; port={port}; loaded_ctx={:?}; launch_ctx={:?}; active_generation={:?}; health={health}; slots={slots}; props={props}; stderr_tail={}",
        loaded_model, loaded_ctx, launch_ctx, active_generation, stderr_tail
    )
}

async fn swap_model_for_api(
    state: &SharedState,
    model_name: &str,
    context_size: Option<u32>,
    overrides: RuntimeLoadOverrides,
) -> Result<(), String> {
    crate::commands::model::backend_load_model_with_overrides(
        state.clone(),
        model_name.to_string(),
        context_size,
        overrides,
    )
    .await
    .map(|_| ())
}

pub(crate) async fn resolve_loaded_model(
    state: &SharedState,
    requested_model: &str,
    requested_context_size: Option<u32>,
    requested_overrides: RuntimeLoadOverrides,
) -> Result<String, ApiErrorResponse> {
    let has_overrides = has_runtime_load_overrides(&requested_overrides);
    let (target_model, needs_swap, context_size) = {
        let s = state.read().await;
        let runtime_running = matches!(s.process.state(), ProcessState::Running);
        let current_context_size = s.model_stats.as_ref().map(|stats| stats.context_size);
        let target_model = if requested_model.trim().is_empty() {
            s.loaded_model
                .clone()
                .ok_or_else(ApiErrorResponse::no_model)?
        } else {
            s.model_registry
                .find_by_name(requested_model.trim())
                .map(|model| model.filename.clone())
                .unwrap_or_else(|| requested_model.trim().to_string())
        };

        let needs_swap = match &s.loaded_model {
            Some(loaded) => {
                let model_matches = loaded_model_matches_request(loaded, &target_model)
                    || loaded_model_matches_request(loaded, requested_model);
                let context_matches = context_request_matches_preview(
                    requested_context_size,
                    s.last_launch_preview.as_ref(),
                );
                let overrides_match = !has_overrides
                    || s.last_launch_preview
                        .as_ref()
                        .map(|preview| api_overrides_match_preview(&requested_overrides, preview))
                        .unwrap_or(false);

                !runtime_running || !model_matches || !context_matches || !overrides_match
            }
            None => true,
        };

        let context_size = context_size_for_reload(
            requested_context_size,
            s.last_launch_preview.as_ref(),
            current_context_size,
        );

        (target_model, needs_swap, context_size)
    };

    if needs_swap {
        swap_model_for_api(state, &target_model, context_size, requested_overrides)
            .await
            .map_err(|e| {
                ApiErrorResponse::service_unavailable(format!(
                    "Could not load model '{target_model}': {e}"
                ))
            })?;
    }

    let s = state.read().await;
    s.loaded_model
        .clone()
        .ok_or_else(ApiErrorResponse::no_model)
}

async fn normalize_image_payload(value: &str) -> Result<String, ApiErrorResponse> {
    normalize_image_payload_shared(value)
        .await
        .map_err(ApiErrorResponse::bad_request)
}

pub(crate) async fn normalize_api_messages(
    messages: &[ApiMessage],
    profile: &ModelProfile,
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
                            image_data.push(ImageData {
                                data: raw,
                                id: image_id,
                            });
                        }
                    }
                }
            }
        }

        if let Some(name) = &message.name {
            content_parts.insert(0, format!("[name:{name}]"));
        }

        if message.role == "tool" {
            content_parts = vec![render_tool_response_history(
                message.name.as_deref(),
                message.tool_call_id.as_deref(),
                &content_parts.join("\n"),
                profile,
            )];
        } else if let Some(tool_call_id) = &message.tool_call_id {
            content_parts.insert(0, format!("Tool call id: {tool_call_id}"));
        }

        if let Some(refusal) = &message.refusal {
            if !refusal.is_empty() {
                content_parts.push(format!("[refusal]\n{refusal}"));
            }
        }

        if let Some(tool_calls) = &message.tool_calls {
            if !tool_calls.is_empty() {
                content_parts.push(render_tool_calls_history(tool_calls, profile));
            }
        }

        normalized.push(crate::templates::engine::ChatMessage {
            role: message.role.clone(),
            content: content_parts.join("\n"),
        });
    }

    Ok((normalized, image_data))
}

fn render_tool_calls_history(tool_calls: &[serde_json::Value], profile: &ModelProfile) -> String {
    let rendered = tool_calls
        .iter()
        .filter_map(|tool_call| render_tool_call_history(tool_call, profile.tool_call_format))
        .collect::<Vec<_>>();

    if rendered.is_empty() {
        serde_json::to_string(tool_calls).unwrap_or_else(|_| "[]".to_string())
    } else {
        rendered.join("\n")
    }
}

fn render_tool_call_history(
    tool_call: &serde_json::Value,
    format: ToolCallFormat,
) -> Option<String> {
    let function = tool_call.get("function").unwrap_or(tool_call);
    let name = function
        .get("name")
        .or_else(|| tool_call.get("name"))
        .and_then(|value| value.as_str())?
        .trim();
    if name.is_empty() {
        return None;
    }

    let arguments = function
        .get("arguments")
        .or_else(|| tool_call.get("arguments"))
        .map(normalize_tool_arguments)
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

    match format {
        ToolCallFormat::QwenXml => Some(render_qwen_tool_call(name, &arguments)),
        ToolCallFormat::Gemma4Native => Some(render_gemma4_tool_call(name, &arguments)),
        ToolCallFormat::HermesXml | ToolCallFormat::NativeApi => {
            let value = serde_json::json!({
                "name": name,
                "arguments": arguments,
            });
            Some(format!("<tool_call>{}</tool_call>", value))
        }
    }
}

fn normalize_tool_arguments(value: &serde_json::Value) -> serde_json::Value {
    if let Some(text) = value.as_str() {
        serde_json::from_str(text).unwrap_or_else(|_| serde_json::Value::String(text.to_string()))
    } else {
        value.clone()
    }
}

fn render_qwen_tool_call(name: &str, arguments: &serde_json::Value) -> String {
    let mut out = format!("<tool_call>\n<function={}>\n", escape_qwen_xml_attr(name));
    match arguments {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                out.push_str(&format!(
                    "<parameter={}>{}</parameter>\n",
                    escape_qwen_xml_attr(key),
                    escape_xml_text(&tool_argument_to_text(value))
                ));
            }
        }
        other => {
            out.push_str(&format!(
                "<parameter=arguments>{}</parameter>\n",
                escape_xml_text(&tool_argument_to_text(other))
            ));
        }
    }
    out.push_str("</function>\n</tool_call>");
    out
}

fn render_gemma4_tool_call(name: &str, arguments: &serde_json::Value) -> String {
    let mut out = format!("<|tool_call>call:{}", escape_qwen_xml_attr(name));
    out.push('{');
    if let serde_json::Value::Object(map) = arguments {
        let mut first = true;
        for (key, value) in map {
            if !first {
                out.push(',');
            }
            first = false;
            out.push_str(key);
            out.push(':');
            out.push_str(&gemma4_argument_to_text(value));
        }
    }
    out.push_str("}<tool_call|>");
    out
}

fn gemma4_argument_to_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => format!("<|\"|>{}<|\"|>", text),
        other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
    }
}

fn tool_argument_to_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
    }
}

fn render_tool_response_history(
    name: Option<&str>,
    tool_call_id: Option<&str>,
    content: &str,
    profile: &ModelProfile,
) -> String {
    let mut body = String::new();
    if let Some(name) = name.filter(|value| !value.trim().is_empty()) {
        body.push_str("Tool ");
        body.push_str(name.trim());
        body.push_str(" result");
        if let Some(id) = tool_call_id.filter(|value| !value.trim().is_empty()) {
            body.push_str(" (");
            body.push_str(id.trim());
            body.push(')');
        }
        body.push_str(":\n");
    }
    body.push_str(content);

    match profile.tool_call_format {
        ToolCallFormat::QwenXml => format!("<tool_response>\n{}\n</tool_response>", body.trim()),
        ToolCallFormat::Gemma4Native => format!("<|tool_response>{}<tool_response|>", body.trim()),
        ToolCallFormat::HermesXml | ToolCallFormat::NativeApi => body.trim().to_string(),
    }
}

fn escape_qwen_xml_attr(text: &str) -> String {
    text.trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-')
        .collect::<String>()
}

fn escape_xml_text(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn prepend_tool_schema_message(
    messages: &mut Vec<crate::templates::engine::ChatMessage>,
    profile: &ModelProfile,
    tools: Option<&Vec<serde_json::Value>>,
) {
    let Some(tools) = tools.filter(|tools| !tools.is_empty()) else {
        return;
    };

    let serialized = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".to_string());
    let no_think_guidance = if profile.disable_thinking_for_tools {
        "\n\nWhen deciding whether to call a tool, do not emit <think> blocks. Either produce a tool call directly or answer normally."
    } else {
        ""
    };
    let format_guidance = match profile.tool_call_format {
        ToolCallFormat::QwenXml => {
            "If a tool is needed, reply with exactly one Qwen XML tool call and no prose:\n<tool_call>\n<function=TOOL_NAME>\n<parameter=PARAM_NAME>VALUE</parameter>\n</function>\n</tool_call>"
        }
        ToolCallFormat::HermesXml => {
            "If a tool is needed, reply with exactly one Hermes XML tool call and no prose:\n<tool_call>{\"name\":\"TOOL_NAME\",\"arguments\":{\"PARAM_NAME\":\"VALUE\"}}</tool_call>"
        }
        ToolCallFormat::Gemma4Native => {
            "If a tool is needed, reply with exactly one Gemma 4 native tool call and no prose:\n<|tool_call>call:TOOL_NAME{PARAM_NAME:<|\"|>VALUE<|\"|>}<tool_call|>"
        }
        ToolCallFormat::NativeApi => {
            "If a tool is needed, reply using the tool-calling format appropriate for your model family."
        }
    };
    let schema_guidance =
        "Tool arguments MUST match the JSON schema exactly. Use bare JSON numbers for integer/number fields and bare true/false for boolean fields; do not quote them as strings.";
    messages.insert(
        0,
        crate::templates::engine::ChatMessage {
            role: "system".to_string(),
            content: format!(
                "Available tools (OpenAI-style schema):\n{serialized}\n\n{format_guidance}\n\n{schema_guidance}{no_think_guidance}"
            ),
        },
    );
}

fn tool_schema_type_matches(schema_type: &serde_json::Value, expected: &str) -> bool {
    match schema_type {
        serde_json::Value::String(value) => value == expected,
        serde_json::Value::Array(values) => values
            .iter()
            .any(|value| matches!(value, serde_json::Value::String(t) if t == expected)),
        _ => false,
    }
}

fn coerce_tool_arg_value(value: &mut serde_json::Value, schema: &serde_json::Value) {
    let schema_type = schema.get("type").unwrap_or(&serde_json::Value::Null);
    match value {
        serde_json::Value::String(text) if tool_schema_type_matches(schema_type, "integer") => {
            if let Ok(parsed) = text.trim().parse::<i64>() {
                *value = serde_json::Value::Number(parsed.into());
            }
        }
        serde_json::Value::String(text) if tool_schema_type_matches(schema_type, "number") => {
            if let Ok(parsed) = text.trim().parse::<f64>() {
                if let Some(number) = serde_json::Number::from_f64(parsed) {
                    *value = serde_json::Value::Number(number);
                }
            }
        }
        serde_json::Value::String(text) if tool_schema_type_matches(schema_type, "boolean") => {
            match text.trim().to_ascii_lowercase().as_str() {
                "true" => *value = serde_json::Value::Bool(true),
                "false" => *value = serde_json::Value::Bool(false),
                _ => {}
            }
        }
        serde_json::Value::String(text) if tool_schema_type_matches(schema_type, "array") => {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text.trim()) {
                if parsed.is_array() {
                    *value = parsed;
                }
            }
        }
        serde_json::Value::String(text) if tool_schema_type_matches(schema_type, "object") => {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text.trim()) {
                if parsed.is_object() {
                    *value = parsed;
                }
            }
        }
        _ => {}
    }
}

fn coerce_tool_arguments_for_schema(
    tool_name: &str,
    arguments: &serde_json::Value,
    tools: Option<&Vec<serde_json::Value>>,
) -> serde_json::Value {
    let mut repaired = arguments.clone();
    let Some(properties) = tools.and_then(|tools| {
        tools.iter().find_map(|tool| {
            let function = tool.get("function").unwrap_or(tool);
            let name = function.get("name")?.as_str()?;
            if name != tool_name {
                return None;
            }
            function
                .get("parameters")
                .and_then(|params| params.get("properties"))
                .and_then(|props| props.as_object())
        })
    }) else {
        return repaired;
    };

    if let Some(args) = repaired.as_object_mut() {
        for (arg_name, arg_value) in args.iter_mut() {
            if let Some(schema) = properties.get(arg_name) {
                coerce_tool_arg_value(arg_value, schema);
            }
        }
    }

    repaired
}

fn tool_parameters_for_name<'a>(
    tool_name: &str,
    tools: Option<&'a Vec<serde_json::Value>>,
) -> Option<&'a serde_json::Value> {
    tools.and_then(|tools| {
        tools.iter().find_map(|tool| {
            let function = tool.get("function").unwrap_or(tool);
            let name = function.get("name")?.as_str()?;
            (name == tool_name)
                .then(|| function.get("parameters"))
                .flatten()
        })
    })
}

fn validate_tool_arguments(
    tool_name: &str,
    arguments: &serde_json::Value,
    tools: Option<&Vec<serde_json::Value>>,
) -> Vec<String> {
    let Some(parameters) = tool_parameters_for_name(tool_name, tools) else {
        return Vec::new();
    };
    let Some(args) = arguments.as_object() else {
        return vec!["arguments must be a JSON object".to_string()];
    };

    let mut errors = Vec::new();
    if let Some(required) = parameters
        .get("required")
        .and_then(|value| value.as_array())
    {
        for name in required.iter().filter_map(|value| value.as_str()) {
            if !args.contains_key(name) {
                errors.push(format!("missing required field `{name}`"));
            }
        }
    }

    if let Some(properties) = parameters
        .get("properties")
        .and_then(|value| value.as_object())
    {
        for (name, value) in args {
            let Some(schema) = properties.get(name) else {
                continue;
            };
            let schema_type = schema.get("type").unwrap_or(&serde_json::Value::Null);
            let type_ok = (tool_schema_type_matches(schema_type, "integer") && value.is_i64())
                || (tool_schema_type_matches(schema_type, "number") && value.is_number())
                || (tool_schema_type_matches(schema_type, "boolean") && value.is_boolean())
                || (tool_schema_type_matches(schema_type, "string") && value.is_string())
                || (tool_schema_type_matches(schema_type, "array") && value.is_array())
                || (tool_schema_type_matches(schema_type, "object") && value.is_object())
                || matches!(schema_type, serde_json::Value::Null);
            if !type_ok {
                errors.push(format!(
                    "field `{name}` has wrong type for schema `{}`",
                    serde_json::to_string(schema_type).unwrap_or_default()
                ));
            }
            if let Some(enum_values) = schema.get("enum").and_then(|value| value.as_array()) {
                if !enum_values.iter().any(|allowed| allowed == value) {
                    errors.push(format!(
                        "field `{name}` must be one of {}",
                        serde_json::to_string(enum_values).unwrap_or_default()
                    ));
                }
            }
        }
    }

    errors
}

fn extract_first_json_object(text: &str) -> Option<serde_json::Value> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text.trim()) {
        if value.is_object() {
            return Some(value);
        }
    }

    let start = text.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in text[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    return serde_json::from_str::<serde_json::Value>(&text[start..end]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

fn build_tool_argument_repair_request(
    profile: &ModelProfile,
    tool_name: &str,
    schema: &serde_json::Value,
    arguments: &serde_json::Value,
    errors: &[String],
) -> CompletionRequest {
    let prompt = crate::templates::engine::render_prompt(
        &[
            crate::templates::engine::ChatMessage {
                role: "system".to_string(),
                content: "Repair tool-call JSON arguments. Return exactly one valid JSON object and nothing else. Do not call tools, explain, add Markdown, or include hidden reasoning.".to_string(),
            },
            crate::templates::engine::ChatMessage {
                role: "user".to_string(),
                content: format!(
                    "Tool name: {tool_name}\nSchema:\n{}\nCurrent arguments:\n{}\nValidation errors:\n{}\n\nReturn the corrected arguments JSON object only.",
                    serde_json::to_string_pretty(schema).unwrap_or_default(),
                    serde_json::to_string_pretty(arguments).unwrap_or_default(),
                    errors.join("\n")
                ),
            },
        ],
        profile,
    );

    CompletionRequest {
        prompt,
        n_predict: Some(512),
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(-1),
        min_p: None,
        presence_penalty: None,
        frequency_penalty: None,
        repeat_penalty: None,
        seed: None,
        stream: false,
        stop: vec![],
        special: true,
        image_data: vec![],
        grammar: None,
    }
}

async fn repair_tool_arguments_once(
    client: &LlamaClient,
    profile: &ModelProfile,
    tool_name: &str,
    arguments: &serde_json::Value,
    tools: Option<&Vec<serde_json::Value>>,
    errors: &[String],
) -> Option<serde_json::Value> {
    let schema = tool_parameters_for_name(tool_name, tools)?;
    let repair_request =
        build_tool_argument_repair_request(profile, tool_name, schema, arguments, errors);

    let response = client.complete(&repair_request).await.ok()?;
    extract_first_json_object(&response.content)
}

async fn repaired_tool_arguments(
    client: Option<&LlamaClient>,
    profile: &ModelProfile,
    tool_call: &crate::normalize::tool_extract::ToolCall,
    tools: Option<&Vec<serde_json::Value>>,
) -> serde_json::Value {
    let coerced = coerce_tool_arguments_for_schema(&tool_call.name, &tool_call.arguments, tools);
    let errors = validate_tool_arguments(&tool_call.name, &coerced, tools);
    if errors.is_empty() {
        return coerced;
    }

    let Some(client) = client else {
        tracing::warn!(
            tool = %tool_call.name,
            errors = ?errors,
            "Tool arguments failed schema validation after coercion; no repair client available"
        );
        return coerced;
    };

    if let Some(repaired) =
        repair_tool_arguments_once(client, profile, &tool_call.name, &coerced, tools, &errors).await
    {
        let repaired = coerce_tool_arguments_for_schema(&tool_call.name, &repaired, tools);
        let repaired_errors = validate_tool_arguments(&tool_call.name, &repaired, tools);
        if repaired_errors.is_empty() {
            tracing::info!(tool = %tool_call.name, "Repaired tool arguments after schema validation failure");
            return repaired;
        }
        tracing::warn!(
            tool = %tool_call.name,
            original_errors = ?errors,
            repaired_errors = ?repaired_errors,
            "Tool argument repair did not satisfy schema"
        );
    }

    coerced
}

fn api_tool_call_value_from_arguments(
    tool_call: &crate::normalize::tool_extract::ToolCall,
    arguments: serde_json::Value,
    index: Option<usize>,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": serde_json::to_string(&arguments).unwrap_or_default()
        }
    });
    if let Some(index) = index {
        value["index"] = serde_json::json!(index);
    }
    value
}

async fn api_tool_call_value(
    client: Option<&LlamaClient>,
    profile: &ModelProfile,
    tool_call: &crate::normalize::tool_extract::ToolCall,
    tools: Option<&Vec<serde_json::Value>>,
    index: Option<usize>,
) -> serde_json::Value {
    let arguments = repaired_tool_arguments(client, profile, tool_call, tools).await;
    api_tool_call_value_from_arguments(tool_call, arguments, index)
}

pub(crate) async fn build_chat_request(
    profile: &ModelProfile,
    req: ChatCompletionRequest,
    server_defaults: (&Option<f32>, &Option<f32>, &Option<i32>, &Option<u32>),
    context_limit: Option<u32>,
) -> Result<CompletionRequest, ApiErrorResponse> {
    let requested_top_p = req.requested_top_p();
    let requested_top_k = req.requested_top_k();
    let thinking_disabled = req.requested_thinking_disabled();
    let (mut messages, image_data) = normalize_api_messages(&req.messages, profile).await?;
    prepend_tool_schema_message(&mut messages, profile, req.tools.as_ref());
    prepend_reasoning_guidance_message(
        &mut messages,
        profile,
        req.reasoning.as_ref(),
        req.reasoning_effort.as_deref(),
        req.reasoning_tokens,
        thinking_disabled,
    );
    let n_predict = req
        .max_tokens
        .or(profile.default_max_output_tokens)
        .or(*server_defaults.3)
        .map(|value| value as i32);
    compact_messages_to_fit(&mut messages, profile, context_limit, n_predict);

    let mut stop = profile.stop_markers.clone();
    if let Some(user_stop) = req.stop {
        stop.extend(user_stop.into_vec());
    }

    Ok(CompletionRequest {
        prompt: crate::templates::engine::render_prompt(&messages, profile),
        n_predict,
        temperature: req
            .temperature
            .or(profile.default_temperature)
            .or(*server_defaults.0),
        top_p: requested_top_p
            .or(profile.default_top_p)
            .or(*server_defaults.1),
        top_k: requested_top_k
            .or(profile.default_top_k)
            .or(*server_defaults.2),
        min_p: req.min_p.or(profile.default_min_p),
        presence_penalty: req.presence_penalty.or(profile.default_presence_penalty),
        frequency_penalty: req.frequency_penalty,
        repeat_penalty: req.repetition_penalty,
        seed: req.seed,
        stream: req.stream,
        stop,
        special: true,
        image_data,
        grammar: None,
    })
}

fn compact_messages_to_fit(
    messages: &mut Vec<crate::templates::engine::ChatMessage>,
    profile: &ModelProfile,
    context_limit: Option<u32>,
    n_predict: Option<i32>,
) {
    let Some(context_limit) = context_limit.filter(|value| *value > 0) else {
        return;
    };
    if messages.len() <= 2 {
        return;
    }

    let output_reserve = n_predict
        .filter(|value| *value > 0)
        .map(|value| value as u32)
        .unwrap_or(512)
        .min(2048);
    let budget = context_limit
        .saturating_sub(output_reserve)
        .max(context_limit / 2);
    let mut removed = Vec::new();

    while crate::normalize::think_strip::estimate_token_count(
        &crate::templates::engine::render_prompt(messages, profile),
    ) > budget
        && messages.len() > 2
    {
        let pinned = messages
            .iter()
            .take_while(|message| message.role == "system")
            .count()
            .min(messages.len().saturating_sub(1));
        let remove_index = if pinned < messages.len().saturating_sub(1) {
            pinned
        } else {
            0
        };
        let removed_message = messages.remove(remove_index);
        removed.push((removed_message.role, removed_message.content));
    }

    if !removed.is_empty() {
        let summary = crate::context::compressor::compress_messages(&removed);
        let insert_at = messages
            .iter()
            .take_while(|message| message.role == "system")
            .count()
            .min(messages.len());
        messages.insert(
            insert_at,
            crate::templates::engine::ChatMessage {
                role: "system".to_string(),
                content: summary,
            },
        );
    }

    while crate::normalize::think_strip::estimate_token_count(
        &crate::templates::engine::render_prompt(messages, profile),
    ) > budget
        && messages.len() > 2
    {
        let pinned = messages
            .iter()
            .take_while(|message| message.role == "system")
            .count()
            .min(messages.len().saturating_sub(1));
        let remove_index = if pinned < messages.len().saturating_sub(1) {
            pinned
        } else {
            0
        };
        messages.remove(remove_index);
    }
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
    thinking_disabled: bool,
) {
    let supports_reasoning = !matches!(
        profile.think_tag_style,
        crate::models::profiles::ThinkTagStyle::None
    );
    let requested_effort = reasoning
        .and_then(|cfg| cfg.effort.as_deref())
        .or(reasoning_effort)
        .map(|value| value.trim().to_ascii_lowercase());
    let requested_reasoning_tokens = reasoning
        .and_then(|cfg| cfg.max_tokens)
        .or(reasoning_tokens);

    if !supports_reasoning
        && requested_effort.is_none()
        && requested_reasoning_tokens.is_none()
        && !thinking_disabled
    {
        return;
    }

    let mut guidance = Vec::new();
    if thinking_disabled {
        guidance.push(
            "Respond directly. Do not emit hidden reasoning, chain-of-thought, analysis/channel markers, or a 'Thinking process' section.".to_string(),
        );
    }
    match requested_effort.as_deref() {
        Some("none") => {
            guidance.push("Respond directly without emitting <think> blocks.".to_string())
        }
        Some("low") => guidance.push("Keep reasoning brief and concise.".to_string()),
        Some("high") => guidance
            .push("Use thorough step-by-step reasoning before the final answer.".to_string()),
        Some("xhigh") => {
            guidance.push("Use very detailed reasoning before the final answer.".to_string())
        }
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
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiErrorResponse> {
    if let Some(upstream) = crate::api::upstream::active_openai_provider(&state).await {
        return crate::api::upstream::proxy_json_to_openai_provider(
            upstream,
            "/chat/completions",
            body,
        )
        .await;
    }

    let original_request_body = body.clone();
    let client_request_id =
        crate::replay::preferred_client_correlation_id(&headers, &original_request_body);
    let req: ChatCompletionRequest = serde_json::from_value(body)
        .map_err(|error| ApiErrorResponse::bad_request(error.to_string()))?;
    let include_usage_details = req
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(true);
    let requested_context_size = req.requested_context_size();
    let requested_model = req.model.clone().unwrap_or_default();
    let requested_overrides = extract_runtime_load_overrides(req.options.as_ref(), &req.extra);
    let requested_tools = req.tools.clone();
    let model_name = resolve_loaded_model(
        &state,
        &requested_model,
        requested_context_size,
        requested_overrides,
    )
    .await?;
    let profile = crate::models::overrides::detect_effective_profile(&model_name);

    let (server_defaults, scheduler, llama_port, context_limit) = {
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
        context_limit,
    )
    .await?;

    ensure_runtime_vision_ready(
        &state,
        &model_name,
        &profile,
        !request.image_data.is_empty(),
    )
    .await?;
    let _permit = scheduler.acquire().await;
    {
        let mut s = state.write().await;
        s.last_prompt = Some(request.prompt.clone());
    }

    let client = LlamaClient::new(llama_port);
    let generation_started_at = chrono::Utc::now().to_rfc3339();
    let generation_started = std::time::Instant::now();
    let gen = begin_api_generation(&state, model_name.clone()).await;
    let request_id = gen.request_id.clone();

    if request.stream {
        return stream_chat_completion(
            state,
            client,
            request,
            model_name,
            profile,
            generation_started_at,
            generation_started,
            request_id,
            gen.cancel,
            include_usage_details,
            requested_tools,
            client_request_id,
            original_request_body,
            llama_port,
            context_limit,
        )
        .await;
    }

    let response = match client.complete(&request).await {
        Ok(response) => response,
        Err(e) => {
            let diagnostics = completion_failure_diagnostics(&state, &model_name, &e).await;
            finish_api_generation_for_request(&state, &request_id, "error").await;
            tracing::error!(error = %e, diagnostics = %diagnostics, "Completion failed");
            return Err(ApiErrorResponse::inference_failed(&diagnostics));
        }
    };

    let reasoning_text =
        extract_reasoning_content_with_style(&response.content, profile.think_tag_style);
    let stripped = strip_think_tags_with_style(&response.content, profile.think_tag_style);
    let (tool_calls, text) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(&stripped, &profile);
    let content = if text.is_empty() {
        None
    } else {
        Some(text.clone())
    };
    let mut api_tool_calls: Vec<serde_json::Value> = Vec::new();
    for tool_call in &tool_calls {
        api_tool_calls.push(
            api_tool_call_value(
                Some(&client),
                &profile,
                tool_call,
                requested_tools.as_ref(),
                None,
            )
            .await,
        );
    }
    if !response.content.is_empty() {
        append_live_stream_delta_for_request(&state, &request_id, "raw", &response.content).await;
    }
    if !reasoning_text.is_empty() {
        append_live_stream_delta_for_request(&state, &request_id, "reasoning", &reasoning_text)
            .await;
    }
    if !api_tool_calls.is_empty() {
        append_live_stream_delta_for_request(
            &state,
            &request_id,
            "tool_call",
            &serde_json::to_string_pretty(&api_tool_calls).unwrap_or_default(),
        )
        .await;
    }
    if let Some(text) = &content {
        append_live_stream_delta_for_request(&state, &request_id, "content", text).await;
    }
    let reasoning_tokens = summarize_reasoning_tokens(
        response.tokens_predicted,
        content.as_deref().unwrap_or(""),
        &reasoning_text,
    );
    let elapsed_ms = generation_started.elapsed().as_millis() as u64;
    let end_to_end_tps = end_to_end_tokens_per_second(response.tokens_predicted, elapsed_ms);
    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(build_parse_trace(
            &profile,
            &response.content,
            content.as_deref().unwrap_or(""),
        ));
        s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
            source: "api".to_string(),
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
    let canonical = crate::replay::build_canonical_response(
        &profile,
        &request_id,
        client_request_id,
        "chat-completions-api",
        &model_name,
        &response.content,
        content.as_deref().unwrap_or(""),
        api_tool_calls.clone(),
        &response,
        elapsed_ms,
        end_to_end_tps,
        llama_port,
        context_limit,
    );
    crate::replay::append_api_replay_record(
        "/v1/chat/completions",
        original_request_body,
        &request,
        canonical,
    )
    .await;
    finish_api_generation_for_request(&state, &request_id, "completed").await;

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
    request_id: String,
    cancel: tokio_util::sync::CancellationToken,
    include_usage_details: bool,
    requested_tools: Option<Vec<serde_json::Value>>,
    client_request_id: Option<String>,
    original_request_body: serde_json::Value,
    llama_port: u16,
    context_limit: Option<u32>,
) -> Result<Response, ApiErrorResponse> {
    let response = match client.complete_stream(&request).await {
        Ok(response) => response,
        Err(e) => {
            let diagnostics = completion_failure_diagnostics(&state, &model_name, &e).await;
            finish_api_generation_for_request(&state, &request_id, "error").await;
            tracing::error!(error = %e, diagnostics = %diagnostics, "Stream completion failed");
            return Err(ApiErrorResponse::inference_failed(&diagnostics));
        }
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let _ = streaming::consume_sse_stream(response, tx, cancel).await;
    });

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = now_unix_secs();
    let state_for_stream = state.clone();
    let buffer_tool_content = requested_tools
        .as_ref()
        .map(|tools| !tools.is_empty())
        .unwrap_or(false);

    let stream = async_stream::stream! {
        let mut raw_full_text = String::new();
        let mut first_token_at: Option<std::time::Instant> = None;
        let mut visible_tokens: u32 = 0;

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
                StreamEvent::RawDelta(raw) => {
                    raw_full_text.push_str(&raw);
                    append_live_stream_delta_for_request(&state_for_stream, &request_id, "raw", &raw).await;
                }
                StreamEvent::Token(token) => {
                    if first_token_at.is_none() {
                        first_token_at = Some(std::time::Instant::now());
                    }
                    visible_tokens = visible_tokens.saturating_add(1);
                    publish_live_generation_metrics(
                        &state_for_stream,
                        "api",
                        &model_name,
                        &request_id,
                        &generation_started_at,
                        generation_started,
                        first_token_at,
                        visible_tokens,
                    ).await;
                    if buffer_tool_content {
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "content_buffered", &token).await;
                    } else {
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "content", &token).await;
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
                }
                StreamEvent::ReasoningDelta(reasoning) => {
                    raw_full_text.push_str("<think>");
                    raw_full_text.push_str(&reasoning);
                    raw_full_text.push_str("</think>");
                    append_live_stream_delta_for_request(&state_for_stream, &request_id, "reasoning", &reasoning).await;
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
                    let parse_trace = build_parse_trace(&profile, &full_text, &stripped);
                    let elapsed_ms = generation_started.elapsed().as_millis() as u64;
                    let metrics = crate::state::RuntimePerformanceMetrics {
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
                        end_to_end_tokens_per_second: end_to_end_tokens_per_second(
                            Some(tokens_predicted),
                            elapsed_ms,
                        ),
                    };
                    {
                        let mut s = state_for_stream.write().await;
                        s.last_parse_trace = Some(parse_trace);
                        s.last_generation_metrics = Some(metrics);
                    }

                    let (stream_tool_calls, cleaned_text) =
                        crate::normalize::tool_extract::extract_tool_calls_for_profile(
                            &stripped,
                            &profile,
                        );
                    let mut api_stream_tool_calls: Vec<serde_json::Value> = Vec::new();
                    for (i, tc) in stream_tool_calls.iter().enumerate() {
                        api_stream_tool_calls.push(
                            api_tool_call_value(
                                Some(&client),
                                &profile,
                                tc,
                                requested_tools.as_ref(),
                                Some(i),
                            )
                            .await,
                        );
                    }
                    if !api_stream_tool_calls.is_empty() {
                        append_live_stream_delta_for_request(
                            &state_for_stream,
                            &request_id,
                            "tool_call",
                            &serde_json::to_string_pretty(&api_stream_tool_calls)
                                .unwrap_or_default(),
                        ).await;
                        let tool_calls_chunk = serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": { "tool_calls": api_stream_tool_calls },
                                "finish_reason": serde_json::Value::Null
                            }]
                        });
                        yield Ok::<Event, std::convert::Infallible>(
                            Event::default().data(tool_calls_chunk.to_string()),
                        );
                    } else if buffer_tool_content && !cleaned_text.trim().is_empty() {
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "content", &cleaned_text).await;
                        let chunk_json = serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": cleaned_text },
                                "finish_reason": serde_json::Value::Null
                            }]
                        });
                        yield Ok::<Event, std::convert::Infallible>(
                            Event::default().data(chunk_json.to_string()),
                        );
                    }
                    let stream_response = CompletionResponse {
                        content: full_text.clone(),
                        stop: true,
                        tokens_predicted: Some(tokens_predicted),
                        tokens_evaluated: Some(tokens_evaluated),
                        timings: Some(Timings {
                            predicted_per_second: Some(decode_tokens_per_second),
                            prompt_per_second: prompt_tokens_per_second,
                        }),
                    };
                    let end_to_end_tps = end_to_end_tokens_per_second(
                        Some(tokens_predicted),
                        elapsed_ms,
                    );
                    let canonical = crate::replay::build_canonical_response(
                        &profile,
                        &request_id,
                        client_request_id.clone(),
                        "chat-completions-api-stream",
                        &model_name,
                        &full_text,
                        &cleaned_text,
                        api_stream_tool_calls.clone(),
                        &stream_response,
                        elapsed_ms,
                        end_to_end_tps,
                        llama_port,
                        context_limit,
                    );
                    crate::replay::append_api_replay_record(
                        "/v1/chat/completions",
                        original_request_body.clone(),
                        &request,
                        canonical,
                    )
                    .await;
                    let finish_reason = if api_stream_tool_calls.is_empty() {
                        "stop"
                    } else {
                        "tool_calls"
                    };
                    let final_chunk = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
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
                    finish_api_generation_for_request(&state_for_stream, &request_id, "completed").await;
                    return;
                }
                StreamEvent::Error(error) => {
                    append_live_stream_delta_for_request(&state_for_stream, &request_id, "error", &error).await;
                    finish_api_generation_for_request(&state_for_stream, &request_id, "error").await;
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
        finish_api_generation_for_request(&state_for_stream, &request_id, "completed").await;
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
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl TextCompletionRequest {
    fn requested_context_size(&self) -> Option<u32> {
        self.context_size
            .filter(|value| *value > 0)
            .or_else(|| {
                self.options
                    .as_ref()
                    .and_then(extract_context_size_from_value)
            })
            .or_else(|| extract_context_size_from_hash_map(&self.extra))
    }
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
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiErrorResponse> {
    if let Some(upstream) = crate::api::upstream::active_openai_provider(&state).await {
        return crate::api::upstream::proxy_json_to_openai_provider(upstream, "/completions", body)
            .await;
    }

    let req: TextCompletionRequest = serde_json::from_value(body)
        .map_err(|error| ApiErrorResponse::bad_request(error.to_string()))?;
    let requested_context_size = req.requested_context_size();
    let requested_model = req.model.clone().unwrap_or_default();
    let requested_overrides = extract_runtime_load_overrides(req.options.as_ref(), &req.extra);
    let model_name = resolve_loaded_model(
        &state,
        &requested_model,
        requested_context_size,
        requested_overrides,
    )
    .await?;
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
        grammar: None,
    };

    let _permit = scheduler.acquire().await;
    {
        let mut s = state.write().await;
        s.last_prompt = Some(prompt);
    }

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
    use super::{
        api_tool_call_value, build_tool_argument_repair_request, compact_messages_to_fit,
        context_request_matches_preview, context_size_for_reload, loaded_model_matches_request,
        normalize_api_messages, validate_tool_arguments, ApiMessage, ChatCompletionRequest,
    };
    use crate::engine::process::LaunchPreview;
    use crate::models::profiles::ModelProfile;
    use crate::templates::engine::ChatMessage;

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

    #[test]
    fn reload_context_preserves_loaded_context_when_request_omits_context() {
        assert_eq!(
            context_size_for_reload(None, None, Some(32768)),
            Some(32768)
        );
    }

    #[test]
    fn reload_context_prefers_explicit_request() {
        assert_eq!(
            context_size_for_reload(Some(16384), None, Some(32768)),
            Some(16384)
        );
    }

    #[test]
    fn loaded_context_satisfies_lower_context_request() {
        let preview = test_launch_preview(Some(32768));

        assert!(context_request_matches_preview(Some(24576), Some(&preview)));
        assert!(context_request_matches_preview(Some(32768), Some(&preview)));
        assert!(!context_request_matches_preview(
            Some(65536),
            Some(&preview)
        ));
    }

    #[test]
    fn context_request_requires_known_loaded_context_when_explicit() {
        let preview = test_launch_preview(None);

        assert!(context_request_matches_preview(None, Some(&preview)));
        assert!(!context_request_matches_preview(
            Some(24576),
            Some(&preview)
        ));
        assert!(!context_request_matches_preview(Some(24576), None));
    }

    fn test_launch_preview(context_size: Option<u32>) -> LaunchPreview {
        LaunchPreview {
            server_path: String::new(),
            model_path: String::new(),
            hf_repo: None,
            hf_file: None,
            mmproj_path: None,
            backend_preference: String::new(),
            context_size,
            port: 0,
            parallel_slots: 1,
            fit_mode: None,
            cache_ram_mb: None,
            ctxcp: None,
            use_jinja: false,
            reasoning_mode: None,
            template_mode: String::new(),
            template_source: None,
            template_path: None,
            template_name: None,
            chat_template_kwargs_json: None,
            args: Vec::new(),
        }
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

        let profile = ModelProfile::detect("Qwen3.6-27B-Q4_K_M.gguf");
        let (messages, image_data) = match normalize_api_messages(&request.messages, &profile).await
        {
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
            content: Some(
                serde_json::from_str(
                    r#"[
                    { "type": "text", "text": "what is in this image?" },
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } }
                ]"#,
                )
                .expect("content should deserialize"),
            ),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            refusal: None,
        }];

        let profile = ModelProfile::detect("Qwen3.6-27B-Q4_K_M.gguf");
        let (normalized, image_data) = match normalize_api_messages(&messages, &profile).await {
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
            request
                .reasoning
                .as_ref()
                .and_then(|value| value.effort.as_deref()),
            Some("high")
        );
        assert_eq!(
            request
                .reasoning
                .as_ref()
                .and_then(|value| value.max_tokens),
            Some(256)
        );
    }

    #[test]
    fn detects_vendor_thinking_disabled_flags() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "enable_thinking": false,
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");

        assert!(request.requested_thinking_disabled());
    }

    #[test]
    fn detects_reasoning_effort_none_as_thinking_disabled() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "reasoning_effort": "none",
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");

        assert!(request.requested_thinking_disabled());
    }

    #[tokio::test]
    async fn coerces_tool_arguments_to_declared_schema_types() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "subagent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timeout_seconds": { "type": "integer" },
                        "enabled": { "type": "boolean" },
                        "score": { "type": "number" }
                    }
                }
            }
        })];
        let tool_call = crate::normalize::tool_extract::ToolCall {
            id: "call_1".to_string(),
            name: "subagent".to_string(),
            arguments: serde_json::json!({
                "timeout_seconds": "60",
                "enabled": "true",
                "score": "0.95"
            }),
            raw_text: None,
        };

        let profile = ModelProfile::detect("Qwen3.6-27B-Q4_K_M.gguf");
        let api_value = api_tool_call_value(None, &profile, &tool_call, Some(&tools), None).await;
        let args_text = api_value["function"]["arguments"]
            .as_str()
            .expect("arguments should be stringified JSON");
        let args: serde_json::Value =
            serde_json::from_str(args_text).expect("arguments should parse");

        assert_eq!(args["timeout_seconds"], serde_json::json!(60));
        assert_eq!(args["enabled"], serde_json::json!(true));
        assert_eq!(args["score"], serde_json::json!(0.95));
    }

    #[tokio::test]
    async fn schema_torture_coerces_typed_tool_arguments() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "researcher",
                "parameters": {
                    "type": "object",
                    "required": ["task", "max_tool_calls", "timeout_seconds", "online"],
                    "properties": {
                        "task": { "type": "string" },
                        "max_tool_calls": { "type": "integer" },
                        "timeout_seconds": { "type": "integer" },
                        "online": { "type": "boolean" },
                        "mode": { "type": "string", "enum": ["offline", "online"] },
                        "domains": { "type": "array", "items": { "type": "string" } },
                        "options": { "type": "object" }
                    }
                }
            }
        })];
        let tool_call = crate::normalize::tool_extract::ToolCall {
            id: "call_1".to_string(),
            name: "researcher".to_string(),
            arguments: serde_json::json!({
                "task": "find traffic data",
                "max_tool_calls": "8",
                "timeout_seconds": "180",
                "online": "true",
                "mode": "online",
                "domains": "[\"similarweb.com\",\"semrush.com\"]",
                "options": "{\"fetch\":true,\"depth\":2}"
            }),
            raw_text: None,
        };

        let profile = ModelProfile::detect("Qwen3.6-27B-Q4_K_M.gguf");
        let api_value = api_tool_call_value(None, &profile, &tool_call, Some(&tools), None).await;
        let args: serde_json::Value = serde_json::from_str(
            api_value["function"]["arguments"]
                .as_str()
                .expect("arguments should be stringified JSON"),
        )
        .expect("arguments should parse");

        assert!(validate_tool_arguments("researcher", &args, Some(&tools)).is_empty());
        assert_eq!(args["max_tool_calls"], serde_json::json!(8));
        assert_eq!(args["timeout_seconds"], serde_json::json!(180));
        assert_eq!(args["online"], serde_json::json!(true));
        assert_eq!(
            args["domains"],
            serde_json::json!(["similarweb.com", "semrush.com"])
        );
        assert_eq!(args["options"]["depth"], serde_json::json!(2));
    }

    #[test]
    fn schema_torture_flags_enum_and_required_errors() {
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "researcher",
                "parameters": {
                    "type": "object",
                    "required": ["task", "max_tool_calls"],
                    "properties": {
                        "task": { "type": "string" },
                        "max_tool_calls": { "type": "integer" },
                        "mode": { "type": "string", "enum": ["offline", "online"] }
                    }
                }
            }
        })];
        let arguments = serde_json::json!({
            "mode": "webby"
        });

        let errors = validate_tool_arguments("researcher", &arguments, Some(&tools));

        assert!(errors
            .iter()
            .any(|error| error.contains("missing required field `task`")));
        assert!(errors
            .iter()
            .any(|error| error.contains("missing required field `max_tool_calls`")));
        assert!(errors
            .iter()
            .any(|error| error.contains("field `mode` must be one of")));
    }

    #[test]
    fn repair_formatter_is_strict_and_deterministic() {
        let profile = ModelProfile::detect("Qwen3.6-27B-Q4_K_M.gguf");
        let schema = serde_json::json!({
            "type": "object",
            "required": ["max_tool_calls"],
            "properties": {
                "max_tool_calls": { "type": "integer" }
            }
        });
        let arguments = serde_json::json!({ "max_tool_calls": "8" });
        let errors =
            vec!["field `max_tool_calls` has wrong type for schema `\"integer\"`".to_string()];

        let request = build_tool_argument_repair_request(
            &profile,
            "researcher",
            &schema,
            &arguments,
            &errors,
        );

        assert_eq!(request.temperature, Some(0.0));
        assert_eq!(request.stream, false);
        assert!(request
            .prompt
            .contains("Return exactly one valid JSON object"));
        assert!(request.prompt.contains("Do not call tools"));
        assert!(request.prompt.contains("Validation errors"));
        assert!(request.prompt.contains("max_tool_calls"));
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

    #[test]
    fn matches_provider_profile_aliases_to_loaded_gguf() {
        assert!(loaded_model_matches_request(
            "Qwen3.6-27B-Q4_K_M.gguf",
            "qwen3.6-think"
        ));
        assert!(loaded_model_matches_request(
            "unsloth/Qwen3.5-27B-Instruct-Q4_K_M.gguf",
            "qwen3.5-reasoning"
        ));
    }

    #[test]
    fn compacts_messages_before_context_overflow() {
        let profile = ModelProfile::detect("gemma-4-26B-A4B-it-QAT-Q4_0.gguf");
        let mut messages = vec![ChatMessage {
            role: "system".to_string(),
            content: "Always be concise.".to_string(),
        }];
        for i in 0..16 {
            messages.push(ChatMessage {
                role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                content: "x".repeat(1200),
            });
        }

        compact_messages_to_fit(&mut messages, &profile, Some(2048), Some(256));
        let prompt = crate::templates::engine::render_prompt(&messages, &profile);

        assert!(crate::normalize::think_strip::estimate_token_count(&prompt) <= 2048);
        assert!(messages
            .iter()
            .any(|message| message.content.contains("[Earlier conversation summary]")));
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

        let profile = ModelProfile::detect("Qwen3.6-27B-Q4_K_M.gguf");
        let (normalized, _) = match normalize_api_messages(&messages, &profile).await {
            Ok(value) => value,
            Err(_) => panic!("messages should normalize"),
        };

        assert!(normalized[0].content.contains("[name:planner]"));
        assert!(!normalized[0].content.contains("[tool_calls]"));
        assert!(normalized[0].content.contains("<tool_call>"));
        assert!(normalized[0].content.contains("<function=search_docs>"));
        assert!(normalized[0]
            .content
            .contains("<parameter=query>qwen</parameter>"));
    }
}
