use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::api::errors::ApiErrorResponse;
use crate::commands::model::RuntimeLoadOverrides;
use crate::engine::client::{
    CompletionRequest, CompletionResponse, ImageData, LlamaClient, Timings,
};
use crate::engine::process::{LaunchPreview, ProcessState, SamplingDefaults};
use crate::engine::scheduler::RequestPermit;
use crate::engine::streaming::{self, StreamEvent};
use crate::models::profiles::{ModelFamily, ModelProfile, RendererType, ToolCallFormat};
use crate::normalize::images::normalize_image_payload as normalize_image_payload_shared;
use crate::normalize::think_strip::{
    extract_reasoning_content_with_style, strip_think_tags_with_style,
};
use crate::state::{
    append_live_stream_delta_for_request, begin_api_generation, finish_api_generation_for_request,
    summarize_reasoning_tokens, GenerationDropGuard, SharedState,
};

#[derive(Debug, Clone, Default)]
pub(crate) struct CompactionInfo {
    removed_messages: usize,
    remaining_messages: usize,
    context_limit: u32,
    budget: u32,
}

impl CompactionInfo {
    fn removed_messages_header(&self) -> HeaderValue {
        HeaderValue::from_str(&self.removed_messages.to_string())
            .unwrap_or_else(|_| HeaderValue::from_static("0"))
    }

    fn summary_header(&self) -> HeaderValue {
        HeaderValue::from_str(&format!(
            "removed_messages={}, remaining_messages={}, context_limit={}, budget={}",
            self.removed_messages, self.remaining_messages, self.context_limit, self.budget
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("compacted"))
    }
}

fn apply_compaction_headers(response: &mut Response, compaction: Option<&CompactionInfo>) {
    let Some(compaction) = compaction else {
        return;
    };
    let headers = response.headers_mut();
    headers.insert(
        "x-inference-bridge-compacted",
        HeaderValue::from_static("true"),
    );
    headers.insert(
        "x-inference-bridge-compacted-messages",
        compaction.removed_messages_header(),
    );
    headers.insert("x-inference-bridge-compaction", compaction.summary_header());
}

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
    #[serde(default, alias = "responseFormat")]
    pub response_format: Option<serde_json::Value>,
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

/// Maps an OpenAI `response_format` into a llama-server json_schema payload.
/// Returns None when no constraint is requested.
pub(crate) fn response_format_to_json_schema(
    response_format: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    let response_format = response_format?;
    match response_format.get("type").and_then(|value| value.as_str()) {
        Some("json_schema") => response_format
            .get("json_schema")
            .and_then(|json_schema| json_schema.get("schema"))
            .or_else(|| response_format.get("schema"))
            .cloned(),
        Some("json_object") => Some(serde_json::json!({ "type": "object" })),
        _ => None,
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
const REASONING_PRESERVE_KEYS: &[&str] = &["reasoning_preserve", "reasoningPreserve"];
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
const DRAFT_MODEL_KEYS: &[&str] = &[
    "draft_model_path",
    "draftModelPath",
    "draft_model",
    "draftModel",
    "spec_draft_model",
    "specDraftModel",
];
const SPEC_TYPE_KEYS: &[&str] = &["spec_type", "specType"];
const SPEC_DRAFT_N_MAX_KEYS: &[&str] = &[
    "spec_draft_n_max",
    "specDraftNMax",
    "spec_draft_tokens",
    "draftNMax",
];
const DRAFT_MAX_KEYS: &[&str] = &[
    "draft_max_tokens",
    "draftMaxTokens",
    "draft_max",
    "draftMax",
];
const DRAFT_MIN_KEYS: &[&str] = &[
    "draft_min_tokens",
    "draftMinTokens",
    "draft_min",
    "draftMin",
];
const DRAFT_P_MIN_KEYS: &[&str] = &["draft_p_min", "draftPMin"];
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
            None
        }
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

fn parse_f32_value(value: &serde_json::Value) -> Option<f32> {
    match value {
        serde_json::Value::Number(number) => number.as_f64().map(|value| value as f32),
        serde_json::Value::String(text) => text.trim().parse::<f32>().ok(),
        _ => None,
    }
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
        reasoning_preserve: lookup(REASONING_PRESERVE_KEYS)
            .as_ref()
            .and_then(parse_bool_value),
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
        draft_model_path: lookup(DRAFT_MODEL_KEYS)
            .as_ref()
            .and_then(parse_string_value),
        spec_type: lookup(SPEC_TYPE_KEYS).as_ref().and_then(parse_string_value),
        spec_draft_n_max: lookup(SPEC_DRAFT_N_MAX_KEYS)
            .as_ref()
            .and_then(parse_u32_value),
        draft_max_tokens: lookup(DRAFT_MAX_KEYS).as_ref().and_then(parse_u32_value),
        draft_min_tokens: lookup(DRAFT_MIN_KEYS).as_ref().and_then(parse_u32_value),
        draft_p_min: lookup(DRAFT_P_MIN_KEYS).as_ref().and_then(parse_f32_value),
        extra_args: lookup(EXTRA_ARGS_KEYS)
            .as_ref()
            .and_then(parse_string_array),
        attach_mmproj: None,
        force_reload: false,
        ..Default::default()
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
        || overrides.reasoning_preserve.is_some()
        || overrides.template_mode.is_some()
        || overrides.template_name.is_some()
        || overrides.custom_template_path.is_some()
        || overrides.chat_template_kwargs_json.is_some()
        || overrides.draft_model_path.is_some()
        || overrides.spec_type.is_some()
        || overrides.spec_draft_n_max.is_some()
        || overrides.draft_max_tokens.is_some()
        || overrides.draft_min_tokens.is_some()
        || overrides.draft_p_min.is_some()
        || overrides.attach_mmproj.is_some()
        || overrides.force_reload
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

    fn as_url(&self) -> &str {
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

fn finish_reason_from_completion(response: &CompletionResponse, has_tool_calls: bool) -> String {
    if has_tool_calls {
        return "tool_calls".to_string();
    }

    let stopped_by_limit = response.stopped_limit.unwrap_or(false)
        || response
            .stop_type
            .as_deref()
            .map(|value| value.to_ascii_lowercase().contains("limit"))
            .unwrap_or(false);

    if stopped_by_limit {
        "length".to_string()
    } else {
        "stop".to_string()
    }
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

pub(crate) fn now_unix_secs() -> u64 {
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
    if requested_tokens.len() == 1 && !is_distinctive_model_token(&requested_tokens[0]) {
        return false;
    }

    let loaded_tokens = model_match_tokens(loaded);
    requested_tokens
        .iter()
        .all(|requested| loaded_tokens.iter().any(|loaded| loaded == requested))
}

fn is_distinctive_model_token(token: &str) -> bool {
    let has_alpha = token.chars().any(|ch| ch.is_ascii_alphabetic());
    let has_digit = token.chars().any(|ch| ch.is_ascii_digit());
    let is_size_only = token
        .chars()
        .all(|ch| ch.is_ascii_digit() || matches!(ch.to_ascii_lowercase(), 'b' | 'm'));

    has_alpha && has_digit && !is_size_only
}

fn registry_model_matches_api_request(
    model: &crate::models::scanner::ScannedModel,
    requested_model: &str,
) -> bool {
    let requested = requested_model.trim();
    if requested.is_empty() {
        return false;
    }

    let requested_name = normalize_model_match_name(requested);
    let filename = normalize_model_match_name(&model.filename);
    if filename == requested_name || filename.trim_end_matches(".gguf") == requested_name {
        return true;
    }

    let requested_path = requested.replace('\\', "/").to_ascii_lowercase();
    let model_path = model
        .path
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    model_path == requested_path
}

fn requested_model_allows_api_jit_load(requested_model: &str) -> bool {
    let requested = requested_model.trim();
    if requested.is_empty() {
        return false;
    }

    let lower = requested.to_ascii_lowercase();
    lower.ends_with(".gguf") || requested.contains('/') || requested.contains('\\')
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
            overrides.reasoning_preserve,
            Some(preview.reasoning_preserve),
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
        && optional_override_matches(
            overrides.draft_model_path.as_deref(),
            Some(preview.draft_model_path.as_str()),
        )
        && optional_override_matches(
            overrides.spec_type.as_deref(),
            Some(preview.spec_type.as_str()),
        )
        && optional_override_matches(overrides.spec_draft_n_max, Some(preview.spec_draft_n_max))
        && optional_override_matches(overrides.draft_max_tokens, Some(preview.draft_max_tokens))
        && optional_override_matches(overrides.draft_min_tokens, Some(preview.draft_min_tokens))
        && optional_override_matches(overrides.draft_p_min, Some(preview.draft_p_min))
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
    if let Some(active) = s
        .active_generation
        .as_mut()
        .filter(|active| active.id == request_id)
    {
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

/// Run a caller-facing non-streaming completion over llama-server's streaming
/// transport so the desktop can observe output while it is being generated.
/// The returned value is the same buffered `CompletionResponse` shape used by
/// the existing non-streaming handlers; only the internal transport changes.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn complete_with_live_capture(
    state: &SharedState,
    client: &LlamaClient,
    request: &CompletionRequest,
    model_name: &str,
    request_id: &str,
    source: &str,
    generation_started_at: &str,
    generation_started: std::time::Instant,
    cancel: tokio_util::sync::CancellationToken,
    buffer_tool_content: bool,
) -> anyhow::Result<CompletionResponse> {
    let mut streamed_request = request.clone();
    streamed_request.stream = true;

    let response = client.complete_stream(&streamed_request).await?;
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let stream_cancel = cancel.clone();
    tokio::spawn(async move {
        let _ = streaming::consume_sse_stream(response, tx, cancel).await;
    });

    let guard = GenerationDropGuard::new(state.clone(), request_id.to_string(), stream_cancel);
    let mut first_token_at: Option<std::time::Instant> = None;
    let mut observed_tokens: u32 = 0;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::RawDelta(raw) => {
                if first_token_at.is_none() {
                    first_token_at = Some(std::time::Instant::now());
                }
                observed_tokens = observed_tokens.saturating_add(1);
                append_live_stream_delta_for_request(state, request_id, "raw", &raw).await;
                publish_live_generation_metrics(
                    state,
                    source,
                    model_name,
                    request_id,
                    generation_started_at,
                    generation_started,
                    first_token_at,
                    observed_tokens,
                )
                .await;
            }
            StreamEvent::Token(token) => {
                append_live_stream_delta_for_request(
                    state,
                    request_id,
                    if buffer_tool_content {
                        "content_buffered"
                    } else {
                        "content"
                    },
                    &token,
                )
                .await;
            }
            StreamEvent::ReasoningDelta(reasoning) => {
                append_live_stream_delta_for_request(state, request_id, "reasoning", &reasoning)
                    .await;
            }
            StreamEvent::ToolCallDelta(tool_call) => {
                append_live_stream_delta_for_request(state, request_id, "tool_call", &tool_call)
                    .await;
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
                guard.mark_completed();
                return Ok(CompletionResponse {
                    content: full_text,
                    stop: true,
                    stopped_limit,
                    stop_type,
                    tokens_predicted: Some(tokens_predicted),
                    tokens_evaluated: Some(tokens_evaluated),
                    timings: Some(Timings {
                        predicted_per_second: Some(decode_tokens_per_second),
                        prompt_per_second: prompt_tokens_per_second,
                    }),
                });
            }
            StreamEvent::Error(error) => {
                append_live_stream_delta_for_request(state, request_id, "error", &error).await;
                guard.mark_completed();
                anyhow::bail!(error);
            }
        }
    }

    guard.mark_completed();
    anyhow::bail!("llama-server stream ended without a completion event")
}

/// Native-chat equivalent of `complete_with_live_capture`. It always asks
/// llama-server for SSE internally, even when the caller needs a buffered
/// response, so Logs receives content, reasoning, and tool-call deltas live.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn complete_native_chat_with_live_capture(
    state: &SharedState,
    client: &LlamaClient,
    body: &serde_json::Value,
    model_name: &str,
    request_id: &str,
    source: &str,
    generation_started_at: &str,
    generation_started: std::time::Instant,
    cancel: tokio_util::sync::CancellationToken,
    buffer_tool_content: bool,
) -> anyhow::Result<CompletionResponse> {
    let mut streamed_body = body.clone();
    if let Some(object) = streamed_body.as_object_mut() {
        object.insert("stream".to_string(), serde_json::Value::Bool(true));
        object.insert(
            "stream_options".to_string(),
            serde_json::json!({ "include_usage": true }),
        );
    }

    let response = client.chat_completion_response(&streamed_body).await?;
    if !response.status().is_success() {
        let status = response.status();
        let response_body = response.text().await.unwrap_or_default();
        anyhow::bail!("llama-server returned {status}: {response_body}");
    }
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let stream_cancel = cancel.clone();
    tokio::spawn(async move {
        let _ = streaming::consume_chat_sse_stream(response, tx, cancel).await;
    });

    let guard = GenerationDropGuard::new(state.clone(), request_id.to_string(), stream_cancel);
    let mut first_token_at: Option<std::time::Instant> = None;
    let mut observed_tokens: u32 = 0;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::RawDelta(raw) => {
                if first_token_at.is_none() {
                    first_token_at = Some(std::time::Instant::now());
                }
                observed_tokens = observed_tokens.saturating_add(1);
                append_live_stream_delta_for_request(state, request_id, "raw", &raw).await;
                publish_live_generation_metrics(
                    state,
                    source,
                    model_name,
                    request_id,
                    generation_started_at,
                    generation_started,
                    first_token_at,
                    observed_tokens,
                )
                .await;
            }
            StreamEvent::Token(token) => {
                append_live_stream_delta_for_request(
                    state,
                    request_id,
                    if buffer_tool_content {
                        "content_buffered"
                    } else {
                        "content"
                    },
                    &token,
                )
                .await;
            }
            StreamEvent::ReasoningDelta(reasoning) => {
                append_live_stream_delta_for_request(state, request_id, "reasoning", &reasoning)
                    .await;
            }
            StreamEvent::ToolCallDelta(tool_call) => {
                append_live_stream_delta_for_request(state, request_id, "tool_call", &tool_call)
                    .await;
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
                guard.mark_completed();
                return Ok(CompletionResponse {
                    content: full_text,
                    stop: true,
                    stopped_limit,
                    stop_type,
                    tokens_predicted: Some(tokens_predicted),
                    tokens_evaluated: Some(tokens_evaluated),
                    timings: Some(Timings {
                        predicted_per_second: Some(decode_tokens_per_second),
                        prompt_per_second: prompt_tokens_per_second,
                    }),
                });
            }
            StreamEvent::Error(error) => {
                append_live_stream_delta_for_request(state, request_id, "error", &error).await;
                guard.mark_completed();
                anyhow::bail!(error);
            }
        }
    }

    guard.mark_completed();
    anyhow::bail!("llama-server native chat stream ended without a completion event")
}

pub(crate) async fn completion_failure_diagnostics(
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
        if s.image_generation_exclusive
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(ApiErrorResponse::service_unavailable(
                "Image generation is using the managed GPU runtime; chat will resume after model restoration.",
            ));
        }
        let runtime_running = matches!(s.process.state(), ProcessState::Running);
        let current_context_size = s.model_stats.as_ref().map(|stats| stats.context_size);
        let target_model = if requested_model.trim().is_empty() {
            s.loaded_model
                .clone()
                .ok_or_else(ApiErrorResponse::no_model)?
        } else if let Some(model) = s
            .model_registry
            .list()
            .iter()
            .find(|model| registry_model_matches_api_request(model, requested_model.trim()))
        {
            model.filename.clone()
        } else if s
            .loaded_model
            .as_deref()
            .map(|loaded| loaded_model_matches_request(loaded, requested_model.trim()))
            .unwrap_or(false)
        {
            s.loaded_model.clone().unwrap_or_default()
        } else if requested_model_allows_api_jit_load(requested_model) {
            requested_model.trim().to_string()
        } else {
            return Err(ApiErrorResponse::model_not_found(requested_model.trim()));
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

        // Qwen models expect tool results inside a `user` turn (the training format uses
        // <|im_start|>user\n<tool_response>...</tool_response>). The `tool` role has no
        // special meaning in the ChatML renderer and would produce an unrecognised
        // <|im_start|>tool token. Llama 3 uses its own `tool` header and is left as-is.
        let effective_role =
            if message.role == "tool" && profile.renderer_type == RendererType::QwenChat {
                "user".to_string()
            } else {
                message.role.clone()
            };
        normalized.push(crate::templates::engine::ChatMessage {
            role: effective_role,
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
            "If a tool is needed, reply with a Hermes-style JSON tool call wrapped in <tool_call> tags and no prose:\n<tool_call>{\"name\":\"TOOL_NAME\",\"arguments\":{\"PARAM_NAME\":\"VALUE\"}}</tool_call>"
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
        json_schema: None,
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
    repair_enabled: bool,
) -> serde_json::Value {
    let coerced = coerce_tool_arguments_for_schema(&tool_call.name, &tool_call.arguments, tools);
    let errors = validate_tool_arguments(&tool_call.name, &coerced, tools);
    if errors.is_empty() {
        return coerced;
    }

    if !repair_enabled {
        tracing::warn!(
            tool = %tool_call.name,
            errors = ?errors,
            "Tool arguments failed schema validation after coercion; repair disabled by config"
        );
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

    tracing::info!(
        tool = %tool_call.name,
        errors = ?errors,
        "Attempting one bounded tool argument repair pass"
    );
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

pub(crate) async fn api_tool_call_value(
    client: Option<&LlamaClient>,
    profile: &ModelProfile,
    tool_call: &crate::normalize::tool_extract::ToolCall,
    tools: Option<&Vec<serde_json::Value>>,
    index: Option<usize>,
    repair_enabled: bool,
) -> serde_json::Value {
    let arguments =
        repaired_tool_arguments(client, profile, tool_call, tools, repair_enabled).await;
    api_tool_call_value_from_arguments(tool_call, arguments, index)
}

pub(crate) async fn build_chat_request(
    profile: &ModelProfile,
    req: ChatCompletionRequest,
    server_defaults: (&Option<f32>, &Option<f32>, &Option<i32>, &Option<u32>),
    launch_defaults: &SamplingDefaults,
    context_limit: Option<u32>,
    compaction_client: Option<&LlamaClient>,
) -> Result<(CompletionRequest, Option<CompactionInfo>), ApiErrorResponse> {
    let requested_top_p = req.requested_top_p();
    let requested_top_k = req.requested_top_k();
    let thinking_disabled = req.requested_thinking_disabled();
    let json_schema = response_format_to_json_schema(req.response_format.as_ref());
    let (mut messages, image_data) = normalize_api_messages(&req.messages, profile).await?;
    let has_tools = req.tools.as_ref().map_or(false, |t| !t.is_empty());
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
        .or(*server_defaults.3)
        .or(profile.default_max_output_tokens)
        .map(|value| value as i32);
    if let (Some(context_limit), Some(n_predict)) = (context_limit, n_predict) {
        if n_predict > 0 && (n_predict as u32) >= context_limit.saturating_sub(256) {
            return Err(ApiErrorResponse::bad_request(format!(
                "Requested max output of {n_predict} tokens leaves no usable prompt budget in the active {context_limit}-token context. Reduce max_tokens or load a larger context."
            )));
        }
    }
    let compaction = compact_messages_to_fit(
        &mut messages,
        profile,
        context_limit,
        n_predict,
        compaction_client,
    )
    .await;

    let mut stop = profile.stop_markers.clone();
    if let Some(user_stop) = req.stop {
        stop.extend(user_stop.into_vec());
    }

    Ok((
        CompletionRequest {
            prompt: crate::templates::engine::render_prompt_with_tools(
                &messages, profile, has_tools,
            ),
            n_predict,
            temperature: req
                .temperature
                .or(launch_defaults.temperature)
                .or(*server_defaults.0)
                .or(profile.default_temperature),
            top_p: requested_top_p
                .or(launch_defaults.top_p)
                .or(*server_defaults.1)
                .or(profile.default_top_p),
            top_k: requested_top_k
                .or(launch_defaults.top_k)
                .or(*server_defaults.2)
                .or(profile.default_top_k),
            min_p: req
                .min_p
                .or(launch_defaults.min_p)
                .or(profile.default_min_p),
            presence_penalty: req
                .presence_penalty
                .or(launch_defaults.presence_penalty)
                .or(profile.default_presence_penalty),
            frequency_penalty: req.frequency_penalty,
            repeat_penalty: req.repetition_penalty.or(launch_defaults.repeat_penalty),
            seed: req.seed,
            stream: req.stream,
            stop,
            special: true,
            image_data,
            grammar: None,
            json_schema,
        },
        compaction,
    ))
}

pub(crate) fn uses_native_chat_api(profile: &ModelProfile) -> bool {
    matches!(profile.family, ModelFamily::Qwen3_5)
}

pub(crate) fn api_messages_have_images(messages: &[ApiMessage]) -> bool {
    messages.iter().any(|message| {
        matches!(
            message.content.as_ref(),
            Some(ApiMessageContent::Parts(parts))
                if parts.iter().any(|part| matches!(
                    part,
                    ApiContentPart::ImageUrl { .. } | ApiContentPart::InputImage { .. }
                ))
        )
    })
}

pub(crate) fn api_messages_to_native_value(messages: &[ApiMessage]) -> serde_json::Value {
    serde_json::Value::Array(
        messages
            .iter()
            .map(|message| {
                let mut value = serde_json::Map::new();
                value.insert("role".to_string(), serde_json::json!(message.role));
                if let Some(name) = message.name.as_ref() {
                    value.insert("name".to_string(), serde_json::json!(name));
                }
                if let Some(tool_call_id) = message.tool_call_id.as_ref() {
                    value.insert("tool_call_id".to_string(), serde_json::json!(tool_call_id));
                }
                if let Some(tool_calls) = message.tool_calls.as_ref() {
                    value.insert("tool_calls".to_string(), serde_json::json!(tool_calls));
                }
                if let Some(refusal) = message.refusal.as_ref() {
                    value.insert("refusal".to_string(), serde_json::json!(refusal));
                }
                let content = match message.content.as_ref() {
                    None => serde_json::Value::Null,
                    Some(ApiMessageContent::Text(text)) => serde_json::json!(text),
                    Some(ApiMessageContent::Parts(parts)) => serde_json::Value::Array(
                        parts
                            .iter()
                            .filter_map(|part| match part {
                                ApiContentPart::Text { text }
                                | ApiContentPart::InputText { text } => Some(serde_json::json!({
                                    "type": "text",
                                    "text": text,
                                })),
                                ApiContentPart::ImageUrl { image_url } => Some(serde_json::json!({
                                    "type": "image_url",
                                    "image_url": { "url": image_url.as_url() },
                                })),
                                ApiContentPart::InputImage {
                                    image_url,
                                    image_base64,
                                } => image_url
                                    .as_ref()
                                    .map(ApiImageUrl::as_url)
                                    .map(str::to_string)
                                    .or_else(|| {
                                        image_base64.as_ref().map(|data| {
                                            if data.starts_with("data:") {
                                                data.clone()
                                            } else {
                                                format!("data:image/png;base64,{data}")
                                            }
                                        })
                                    })
                                    .map(|url| {
                                        serde_json::json!({
                                            "type": "image_url",
                                            "image_url": { "url": url },
                                        })
                                    }),
                            })
                            .collect(),
                    ),
                };
                value.insert("content".to_string(), content);
                serde_json::Value::Object(value)
            })
            .collect(),
    )
}

/// Builds the request sent to llama-server's native OpenAI-compatible chat
/// endpoint. Qwen3.5/Tess must take this path so the GGUF's embedded Jinja
/// template, launch-time reasoning mode, and native tool-call grammar remain
/// authoritative.
pub(crate) fn build_native_chat_body(
    original: &serde_json::Value,
    req: &ChatCompletionRequest,
    model_name: &str,
    profile: &ModelProfile,
    server_defaults: (Option<f32>, Option<f32>, Option<i32>, Option<u32>),
    launch_defaults: &SamplingDefaults,
) -> serde_json::Value {
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), serde_json::json!(model_name));
    body.insert(
        "messages".to_string(),
        api_messages_to_native_value(&req.messages),
    );
    body.insert("stream".to_string(), serde_json::json!(req.stream));

    let resolved = [
        (
            "max_tokens",
            req.max_tokens
                .or(server_defaults.3)
                .or(profile.default_max_output_tokens)
                .map(serde_json::Value::from),
        ),
        (
            "temperature",
            req.temperature
                .or(launch_defaults.temperature)
                .or(server_defaults.0)
                .or(profile.default_temperature)
                .map(serde_json::Value::from),
        ),
        (
            "top_p",
            req.requested_top_p()
                .or(launch_defaults.top_p)
                .or(server_defaults.1)
                .or(profile.default_top_p)
                .map(serde_json::Value::from),
        ),
        (
            "top_k",
            req.requested_top_k()
                .or(launch_defaults.top_k)
                .or(server_defaults.2)
                .or(profile.default_top_k)
                .map(serde_json::Value::from),
        ),
        (
            "min_p",
            req.min_p
                .or(launch_defaults.min_p)
                .or(profile.default_min_p)
                .map(serde_json::Value::from),
        ),
        (
            "presence_penalty",
            req.presence_penalty
                .or(launch_defaults.presence_penalty)
                .or(profile.default_presence_penalty)
                .map(serde_json::Value::from),
        ),
        (
            "frequency_penalty",
            req.frequency_penalty.map(serde_json::Value::from),
        ),
        (
            "repeat_penalty",
            req.repetition_penalty
                .or(launch_defaults.repeat_penalty)
                .or((profile.family == ModelFamily::Qwen3_5).then_some(1.0))
                .map(serde_json::Value::from),
        ),
        ("seed", req.seed.map(serde_json::Value::from)),
    ];
    for (key, value) in resolved {
        if let Some(value) = value {
            body.insert(key.to_string(), value);
        }
    }

    if let Some(stop) = req.stop.as_ref() {
        body.insert(
            "stop".to_string(),
            match stop {
                StopParam::Single(value) => serde_json::json!(value),
                StopParam::Multiple(values) => serde_json::json!(values),
            },
        );
    }
    if let Some(tools) = req.tools.as_ref() {
        body.insert("tools".to_string(), serde_json::json!(tools));
    }
    if let Some(response_format) = req.response_format.as_ref() {
        body.insert("response_format".to_string(), response_format.clone());
    }
    if let Some(stream_options) = original
        .get("stream_options")
        .or_else(|| original.get("streamOptions"))
        .cloned()
    {
        body.insert("stream_options".to_string(), stream_options);
    }

    // Keep the API serial by default. If a caller explicitly opts into or out
    // of parallel calls, preserve that value exactly.
    body.insert(
        "parallel_tool_calls".to_string(),
        req.extra
            .get("parallel_tool_calls")
            .or_else(|| req.extra.get("parallelToolCalls"))
            .cloned()
            .unwrap_or(serde_json::Value::Bool(false)),
    );
    if let Some(tool_choice) = req
        .extra
        .get("tool_choice")
        .or_else(|| req.extra.get("toolChoice"))
        .cloned()
    {
        body.insert("tool_choice".to_string(), tool_choice);
    }

    for key in ["n", "user", "logprobs", "top_logprobs", "logit_bias"] {
        if let Some(value) = original.get(key).cloned() {
            body.insert(key.to_string(), value);
        }
    }

    serde_json::Value::Object(body)
}

pub(crate) fn native_replay_request(body: &serde_json::Value) -> CompletionRequest {
    let number_f32 = |key: &str| {
        body.get(key)
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32)
    };
    let stop = match body.get("stop") {
        Some(serde_json::Value::String(value)) => vec![value.clone()],
        Some(serde_json::Value::Array(values)) => values
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    };
    CompletionRequest {
        prompt: serde_json::to_string_pretty(body).unwrap_or_else(|_| body.to_string()),
        n_predict: body
            .get("max_tokens")
            .and_then(serde_json::Value::as_i64)
            .and_then(|value| i32::try_from(value).ok()),
        temperature: number_f32("temperature"),
        top_p: number_f32("top_p"),
        top_k: body
            .get("top_k")
            .and_then(serde_json::Value::as_i64)
            .and_then(|value| i32::try_from(value).ok()),
        min_p: number_f32("min_p"),
        presence_penalty: number_f32("presence_penalty"),
        frequency_penalty: number_f32("frequency_penalty"),
        repeat_penalty: number_f32("repeat_penalty"),
        seed: body.get("seed").and_then(serde_json::Value::as_i64),
        stream: body
            .get("stream")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        stop,
        special: true,
        image_data: Vec::new(),
        grammar: None,
        json_schema: body.get("response_format").cloned(),
    }
}

fn native_message_text(message: &ApiMessage) -> String {
    let mut text = match message.content.as_ref() {
        Some(ApiMessageContent::Text(text)) => text.clone(),
        Some(ApiMessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|part| match part {
                ApiContentPart::Text { text } | ApiContentPart::InputText { text } => {
                    Some(text.as_str())
                }
                ApiContentPart::ImageUrl { .. } | ApiContentPart::InputImage { .. } => {
                    Some("[image]")
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    };
    if let Some(tool_calls) = message
        .tool_calls
        .as_ref()
        .filter(|tool_calls| !tool_calls.is_empty())
    {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str("[Assistant tool calls: ");
        text.push_str(&serde_json::to_string(tool_calls).unwrap_or_default());
        text.push(']');
    }
    text
}

pub(crate) fn native_fixed_prompt_tokens(req: &ChatCompletionRequest) -> u32 {
    let tools = req
        .tools
        .as_ref()
        .map(|tools| serde_json::to_string(tools).unwrap_or_default().len() as u32 / 4)
        .unwrap_or(0);
    let response_format = req
        .response_format
        .as_ref()
        .map(|format| format.to_string().len() as u32 / 4)
        .unwrap_or(0);
    let tool_choice = req
        .extra
        .get("tool_choice")
        .or_else(|| req.extra.get("toolChoice"))
        .map(|choice| choice.to_string().len() as u32 / 4)
        .unwrap_or(0);
    tools
        .saturating_add(response_format)
        .saturating_add(tool_choice)
        .saturating_add(64)
}

fn native_messages_token_count(messages: &[ApiMessage]) -> u32 {
    messages
        .iter()
        .map(|message| {
            let mut tokens = 8_u32.saturating_add((message.role.len() as u32) / 4);
            tokens = tokens.saturating_add(match message.content.as_ref() {
                Some(ApiMessageContent::Text(text)) => (text.len() as u32 / 4).max(1),
                Some(ApiMessageContent::Parts(parts)) => parts
                    .iter()
                    .map(|part| match part {
                        ApiContentPart::Text { text } | ApiContentPart::InputText { text } => {
                            (text.len() as u32 / 4).max(1)
                        }
                        // Image data bytes are not prompt text tokens. Reserve a
                        // conservative fixed multimodal budget without counting
                        // the base64 payload itself as 4-char text tokens.
                        ApiContentPart::ImageUrl { .. } | ApiContentPart::InputImage { .. } => 1024,
                    })
                    .sum(),
                None => 0,
            });
            if let Some(tool_calls) = message.tool_calls.as_ref() {
                tokens = tokens.saturating_add(
                    (serde_json::to_string(tool_calls).unwrap_or_default().len() as u32 / 4).max(1),
                );
            }
            tokens
        })
        .sum::<u32>()
        .max(1)
}

fn native_message_is_pinned_instruction(message: &ApiMessage) -> bool {
    matches!(message.role.as_str(), "system" | "developer")
}

fn coalesce_native_instruction_messages(messages: &[ApiMessage]) -> Vec<ApiMessage> {
    let instruction_text = messages
        .iter()
        .filter(|message| native_message_is_pinned_instruction(message))
        .map(native_message_text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut normalized = messages
        .iter()
        .filter(|message| !native_message_is_pinned_instruction(message))
        .cloned()
        .collect::<Vec<_>>();
    if !instruction_text.is_empty() {
        normalized.insert(
            0,
            ApiMessage {
                role: "system".to_string(),
                content: Some(ApiMessageContent::Text(instruction_text)),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                refusal: None,
            },
        );
    }
    normalized
}

fn merge_native_compaction_summary(messages: &mut Vec<ApiMessage>, summary: String) {
    let summary = format!("[Earlier conversation summary]\n{summary}");
    if let Some(first) = messages
        .first_mut()
        .filter(|message| message.role == "system")
    {
        let existing = native_message_text(first);
        first.content = Some(ApiMessageContent::Text(if existing.is_empty() {
            summary
        } else {
            format!("{existing}\n\n{summary}")
        }));
    } else {
        messages.insert(
            0,
            ApiMessage {
                role: "system".to_string(),
                content: Some(ApiMessageContent::Text(summary)),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                refusal: None,
            },
        );
    }
}

fn native_oldest_evictable_span(messages: &[ApiMessage]) -> Option<std::ops::Range<usize>> {
    if messages.len() <= 1 {
        return None;
    }
    let pinned = messages
        .iter()
        .take_while(|message| native_message_is_pinned_instruction(message))
        .count()
        .min(messages.len().saturating_sub(1));
    let newest_turn_start = messages
        .iter()
        .rposition(|message| message.role == "user")
        .unwrap_or_else(|| messages.len().saturating_sub(1));
    let start = pinned;
    if start >= newest_turn_start {
        return None;
    }
    let mut end = start + 1;
    // A conversation turn starts at a user message and includes the following
    // assistant reply plus any native tool-call/tool-result sequence. Evict the
    // complete oldest turn so no assistant or tool result is orphaned merely
    // because removing the first message happened to satisfy the budget.
    while end < newest_turn_start && messages[end].role != "user" {
        end += 1;
    }

    Some(start..end)
}

/// Compact native structured messages without flattening tool roles, IDs, or
/// typed image parts. The newest turn and leading system messages are pinned;
/// assistant tool calls and their tool results are evicted atomically.
pub(crate) fn compact_native_messages_to_fit(
    messages: &[ApiMessage],
    context_limit: Option<u32>,
    max_output_tokens: Option<u32>,
    fixed_prompt_tokens: u32,
) -> Result<(Vec<ApiMessage>, Option<CompactionInfo>), ApiErrorResponse> {
    let mut compacted = coalesce_native_instruction_messages(messages);
    let Some(context_limit) = context_limit.filter(|value| *value > 0) else {
        return Ok((compacted, None));
    };
    let output_reserve = max_output_tokens.unwrap_or(512);
    if output_reserve >= context_limit.saturating_sub(256) {
        return Err(ApiErrorResponse::bad_request(format!(
            "Requested max output of {output_reserve} tokens leaves no usable prompt budget in the active {context_limit}-token context. Reduce max_tokens or load a larger context."
        )));
    }
    let prompt_budget = context_limit.saturating_sub(output_reserve);
    if fixed_prompt_tokens >= prompt_budget.saturating_sub(256) {
        return Err(ApiErrorResponse::bad_request(format!(
            "Tools/response-format overhead of about {fixed_prompt_tokens} tokens leaves no usable message budget in the active {context_limit}-token context. Reduce the tool schema or load a larger context."
        )));
    }
    let budget = prompt_budget.saturating_sub(fixed_prompt_tokens);
    let mut removed = Vec::new();

    while native_messages_token_count(&compacted) > budget {
        let Some(span) = native_oldest_evictable_span(&compacted) else {
            break;
        };
        removed.extend(compacted.drain(span));
    }

    if !removed.is_empty() {
        let pairs = removed
            .iter()
            .map(|message| (message.role.clone(), native_message_text(message)))
            .collect::<Vec<_>>();
        let mut summary = crate::context::compressor::compress_messages(&pairs);
        if summary.chars().count() > 2048 {
            summary = summary.chars().take(2048).collect();
            summary.push_str("...");
        }
        let without_summary = compacted.clone();
        merge_native_compaction_summary(&mut compacted, summary);
        if native_messages_token_count(&compacted) > budget {
            compacted = without_summary;
        }
    }

    if native_messages_token_count(&compacted) > budget {
        return Err(ApiErrorResponse::bad_request(format!(
            "Native chat input exceeds the {budget}-token prompt budget for the active {context_limit}-token context even after safe compaction. Reduce the newest message/image or load a larger context."
        )));
    }

    let compaction = (!removed.is_empty()).then_some(CompactionInfo {
        removed_messages: removed.len(),
        remaining_messages: compacted.len(),
        context_limit,
        budget,
    });
    Ok((compacted, compaction))
}

async fn compact_messages_to_fit(
    messages: &mut Vec<crate::templates::engine::ChatMessage>,
    profile: &ModelProfile,
    context_limit: Option<u32>,
    n_predict: Option<i32>,
    compaction_client: Option<&LlamaClient>,
) -> Option<CompactionInfo> {
    let Some(context_limit) = context_limit.filter(|value| *value > 0) else {
        return None;
    };
    if messages.len() <= 2 {
        return None;
    }

    let output_reserve = n_predict
        .filter(|value| *value > 0)
        .map(|value| value as u32)
        .unwrap_or(512);
    let budget = context_limit.saturating_sub(output_reserve);
    let mut removed = Vec::new();

    while prompt_token_count(messages, profile) > budget && messages.len() > 2 {
        let Some(span) = oldest_evictable_message_span(messages) else {
            break;
        };
        let removed_messages = messages
            .drain(span)
            .map(|message| (message.role, message.content))
            .collect::<Vec<_>>();
        removed.extend(removed_messages);
    }

    if !removed.is_empty() {
        let summary = summarize_removed_messages(compaction_client, profile, &removed).await;
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

    while prompt_token_count(messages, profile) > budget && messages.len() > 2 {
        let Some(span) = oldest_evictable_message_span(messages) else {
            break;
        };
        let extra_removed = messages
            .drain(span)
            .map(|message| (message.role, message.content))
            .collect::<Vec<_>>();
        if extra_removed.is_empty() {
            break;
        }
        removed.extend(extra_removed);
        let summary = summarize_removed_messages(compaction_client, profile, &removed).await;
        replace_existing_compaction_summary(messages, summary);
    }

    (!removed.is_empty()).then_some(CompactionInfo {
        removed_messages: removed.len(),
        remaining_messages: messages.len(),
        context_limit,
        budget,
    })
}

async fn summarize_removed_messages(
    client: Option<&LlamaClient>,
    profile: &ModelProfile,
    removed: &[(String, String)],
) -> String {
    if let Some(client) = client {
        if let Some(summary) = crate::context::compressor::summarize_message_pairs_with_client(
            client, profile, removed,
        )
        .await
        {
            return format!("[Earlier conversation summary]\n{summary}");
        }
    }

    crate::context::compressor::compress_messages(removed)
}

fn prompt_token_count(
    messages: &[crate::templates::engine::ChatMessage],
    profile: &ModelProfile,
) -> u32 {
    crate::normalize::think_strip::estimate_token_count(&crate::templates::engine::render_prompt(
        messages, profile,
    ))
}

fn oldest_evictable_message_span(
    messages: &[crate::templates::engine::ChatMessage],
) -> Option<std::ops::Range<usize>> {
    if messages.len() <= 2 {
        return None;
    }

    let pinned = messages
        .iter()
        .take_while(|message| message.role == "system")
        .count()
        .min(messages.len().saturating_sub(1));
    let last_index = messages.len().saturating_sub(1);
    let start = (pinned..last_index).find(|index| !is_compaction_summary(&messages[*index]))?;
    let mut end = start + 1;

    if message_has_tool_call(&messages[start])
        && end < last_index
        && message_is_tool_result(&messages[end])
    {
        end += 1;
    } else if message_is_tool_result(&messages[start])
        && start > pinned
        && message_has_tool_call(&messages[start - 1])
    {
        return Some((start - 1)..end);
    }

    Some(start..end)
}

fn replace_existing_compaction_summary(
    messages: &mut [crate::templates::engine::ChatMessage],
    summary_content: String,
) {
    let Some(summary) = messages
        .iter_mut()
        .find(|message| is_compaction_summary(message))
    else {
        return;
    };

    summary.content = summary_content;
}

fn is_compaction_summary(message: &crate::templates::engine::ChatMessage) -> bool {
    message.role == "system" && message.content.contains("[Earlier conversation summary]")
}

fn message_has_tool_call(message: &crate::templates::engine::ChatMessage) -> bool {
    message.role == "assistant"
        && (message.content.contains("<tool_call>")
            || message.content.contains("<|tool_call>")
            || message.content.contains("[tool_call]"))
}

fn message_is_tool_result(message: &crate::templates::engine::ChatMessage) -> bool {
    message.role == "tool"
        || message.content.contains("<tool_response>")
        || message.content.contains("<|tool_response>")
        || message.content.contains("Tool call id:")
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

#[derive(Debug, Default, PartialEq, Eq)]
struct NativeMultimodalTrace {
    content: String,
    reasoning: String,
    tool_call_deltas: Vec<serde_json::Value>,
    finish_reason: Option<String>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

fn merge_native_multimodal_chunk(value: &serde_json::Value, trace: &mut NativeMultimodalTrace) {
    if let Some(usage) = value.get("usage") {
        trace.prompt_tokens = usage
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .or(trace.prompt_tokens);
        trace.completion_tokens = usage
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .or(trace.completion_tokens);
    }

    let Some(choice) = value.get("choices").and_then(|choices| choices.get(0)) else {
        return;
    };
    if let Some(finish_reason) = choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
    {
        trace.finish_reason = Some(finish_reason.to_string());
    }
    let message = choice.get("delta").or_else(|| choice.get("message"));
    let Some(message) = message else {
        return;
    };
    if let Some(content) = message.get("content").and_then(serde_json::Value::as_str) {
        trace.content.push_str(content);
    }
    if let Some(reasoning) = message
        .get("reasoning_content")
        .or_else(|| message.get("reasoning"))
        .and_then(serde_json::Value::as_str)
    {
        trace.reasoning.push_str(reasoning);
    }
    if let Some(tool_calls) = message
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
    {
        trace.tool_call_deltas.extend(tool_calls.iter().cloned());
    }
}

fn parse_native_multimodal_trace(raw: &[u8], streaming: bool) -> NativeMultimodalTrace {
    let mut trace = NativeMultimodalTrace::default();
    let text = String::from_utf8_lossy(raw);
    if !streaming {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
            merge_native_multimodal_chunk(&value, &mut trace);
        }
        return trace;
    }

    for line in text.lines() {
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
            merge_native_multimodal_chunk(&value, &mut trace);
        }
    }
    trace
}

fn drain_native_sse_trace(
    buffer: &mut String,
    trace: &mut NativeMultimodalTrace,
) -> (String, String, Option<String>, bool) {
    let content_start = trace.content.len();
    let reasoning_start = trace.reasoning.len();
    let tool_calls_start = trace.tool_call_deltas.len();
    let mut saw_done = false;

    while let Some(newline) = buffer.find('\n') {
        let line = buffer.drain(..=newline).collect::<String>();
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        if data.is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
            merge_native_multimodal_chunk(&value, trace);
        }
    }

    let tool_call_delta = (trace.tool_call_deltas.len() > tool_calls_start).then(|| {
        serde_json::Value::Array(trace.tool_call_deltas[tool_calls_start..].to_vec()).to_string()
    });
    (
        trace.content[content_start..].to_string(),
        trace.reasoning[reasoning_start..].to_string(),
        tool_call_delta,
        saw_done,
    )
}

fn assemble_native_tool_calls(deltas: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut calls = std::collections::BTreeMap::<usize, (Option<String>, String, String)>::new();
    for (fallback_index, delta) in deltas.iter().enumerate() {
        let index = delta
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(fallback_index);
        let entry = calls.entry(index).or_default();
        if let Some(id) = delta.get("id").and_then(serde_json::Value::as_str) {
            entry.0 = Some(id.to_string());
        }
        let function = delta.get("function").unwrap_or(delta);
        if let Some(name) = function.get("name").and_then(serde_json::Value::as_str) {
            entry.1.push_str(name);
        }
        if let Some(arguments) = function
            .get("arguments")
            .and_then(serde_json::Value::as_str)
        {
            entry.2.push_str(arguments);
        }
    }
    calls
        .into_iter()
        .map(|(index, (id, name, arguments))| {
            serde_json::json!({
                "index": index,
                "id": id.unwrap_or_else(|| format!("call_native_{index}")),
                "type": "function",
                "function": { "name": name, "arguments": arguments }
            })
        })
        .collect()
}

fn normalized_native_trace_text(
    trace: &NativeMultimodalTrace,
    native_tool_calls: &[serde_json::Value],
) -> String {
    let mut normalized_raw = String::new();
    if !trace.reasoning.is_empty() {
        normalized_raw.push_str("<think>");
        normalized_raw.push_str(&trace.reasoning);
        normalized_raw.push_str("</think>\n");
    }
    normalized_raw.push_str(&trace.content);
    for tool_call in native_tool_calls {
        let function = tool_call.get("function").unwrap_or(tool_call);
        let arguments_text = function
            .get("arguments")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("{}");
        let arguments = serde_json::from_str::<serde_json::Value>(arguments_text)
            .unwrap_or_else(|_| serde_json::Value::String(arguments_text.to_string()));
        normalized_raw.push_str("<tool_call>");
        normalized_raw.push_str(
            &serde_json::json!({
                "id": tool_call.get("id").and_then(serde_json::Value::as_str).unwrap_or(""),
                "name": function.get("name").and_then(serde_json::Value::as_str).unwrap_or("tool"),
                "arguments": arguments,
            })
            .to_string(),
        );
        normalized_raw.push_str("</tool_call>");
    }
    normalized_raw
}

#[allow(clippy::too_many_arguments)]
fn build_native_replay_canonical(
    profile: &ModelProfile,
    trace: &NativeMultimodalTrace,
    model_name: &str,
    request_id: &str,
    client_request_id: Option<String>,
    generation_started: std::time::Instant,
    llama_port: u16,
    context_limit: Option<u32>,
) -> crate::replay::CanonicalModelResponse {
    let native_tool_calls = assemble_native_tool_calls(&trace.tool_call_deltas);
    let normalized_raw = normalized_native_trace_text(trace, &native_tool_calls);
    let elapsed_ms = generation_started.elapsed().as_millis() as u64;
    let end_to_end_tps = end_to_end_tokens_per_second(trace.completion_tokens, elapsed_ms);
    let response = CompletionResponse {
        content: trace.content.clone(),
        stop: true,
        stopped_limit: Some(trace.finish_reason.as_deref() == Some("length")),
        stop_type: trace.finish_reason.clone(),
        tokens_predicted: trace.completion_tokens,
        tokens_evaluated: trace.prompt_tokens,
        timings: None,
    };
    let mut canonical = crate::replay::build_canonical_response(
        profile,
        request_id,
        client_request_id,
        "chat-completions-api-native-chat",
        model_name,
        &normalized_raw,
        &trace.content,
        native_tool_calls,
        &response,
        elapsed_ms,
        end_to_end_tps,
        llama_port,
        context_limit,
    );
    canonical.backend.endpoint = "/v1/chat/completions";
    canonical
}

async fn record_native_multimodal_trace(
    state: &SharedState,
    profile: &ModelProfile,
    model_name: &str,
    request_id: &str,
    generation_started_at: &str,
    generation_started: std::time::Instant,
    raw: &[u8],
    streaming: bool,
    append_final_deltas: bool,
) -> NativeMultimodalTrace {
    let trace = parse_native_multimodal_trace(raw, streaming);
    if append_final_deltas && !trace.reasoning.is_empty() {
        append_live_stream_delta_for_request(state, request_id, "reasoning", &trace.reasoning)
            .await;
    }
    if append_final_deltas && !trace.content.is_empty() {
        append_live_stream_delta_for_request(state, request_id, "content", &trace.content).await;
    }
    if append_final_deltas && !trace.tool_call_deltas.is_empty() {
        append_live_stream_delta_for_request(
            state,
            request_id,
            "tool_call",
            &serde_json::Value::Array(trace.tool_call_deltas.clone()).to_string(),
        )
        .await;
    }

    let elapsed_ms = generation_started.elapsed().as_millis() as u64;
    let end_to_end_tps = end_to_end_tokens_per_second(trace.completion_tokens, elapsed_ms);
    let mut state = state.write().await;
    let native_tool_calls = assemble_native_tool_calls(&trace.tool_call_deltas);
    let normalized_raw = normalized_native_trace_text(&trace, &native_tool_calls);
    let base_parse_trace = crate::normalize::parse_trace::build_parse_trace(
        profile,
        &normalized_raw,
        &normalized_raw,
        Some(&trace.reasoning),
    );
    let mut parse_trace = serde_json::from_str::<serde_json::Value>(&base_parse_trace)
        .unwrap_or_else(|_| serde_json::json!({ "raw_response": trace.content }));
    parse_trace["native_tool_call_deltas"] =
        serde_json::Value::Array(trace.tool_call_deltas.clone());
    parse_trace["native_tool_calls"] = serde_json::Value::Array(native_tool_calls);
    parse_trace["finish_reason"] = trace
        .finish_reason
        .clone()
        .map(serde_json::Value::String)
        .unwrap_or(serde_json::Value::Null);
    state.last_parse_trace =
        Some(serde_json::to_string_pretty(&parse_trace).unwrap_or_else(|_| base_parse_trace));
    state.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
        source: "api-native-chat".to_string(),
        model: model_name.to_string(),
        request_id: request_id.to_string(),
        started_at: generation_started_at.to_string(),
        finished_at: chrono::Utc::now().to_rfc3339(),
        elapsed_ms,
        time_to_first_token_ms: None,
        prompt_tokens: trace.prompt_tokens,
        completion_tokens: trace.completion_tokens,
        total_tokens: match (trace.prompt_tokens, trace.completion_tokens) {
            (Some(prompt), Some(completion)) => Some(prompt + completion),
            _ => None,
        },
        prompt_tokens_per_second: None,
        decode_tokens_per_second: None,
        end_to_end_tokens_per_second: end_to_end_tps,
    });
    drop(state);
    trace
}

#[allow(clippy::too_many_arguments)]
async fn finalize_native_proxy_generation(
    state: &SharedState,
    guard: &GenerationDropGuard,
    status: StatusCode,
    profile: &ModelProfile,
    model_name: &str,
    request_id: &str,
    generation_started_at: &str,
    generation_started: std::time::Instant,
    captured: &[u8],
    streaming: bool,
    client_request_id: Option<String>,
    llama_port: u16,
    context_limit: Option<u32>,
    original_request_body: serde_json::Value,
    replay_completion_request: &CompletionRequest,
) {
    guard.mark_completed();
    finish_api_generation_for_request(
        state,
        request_id,
        if status.is_success() {
            "completed"
        } else {
            "error"
        },
    )
    .await;

    let mut trace = record_native_multimodal_trace(
        state,
        profile,
        model_name,
        request_id,
        generation_started_at,
        generation_started,
        captured,
        streaming,
        !streaming,
    )
    .await;
    if trace.finish_reason.is_none() && !status.is_success() {
        trace.finish_reason = Some("error".to_string());
    }
    let canonical = build_native_replay_canonical(
        profile,
        &trace,
        model_name,
        request_id,
        client_request_id,
        generation_started,
        llama_port,
        context_limit,
    );
    crate::replay::append_api_replay_record(
        "/v1/chat/completions",
        original_request_body,
        replay_completion_request,
        canonical,
    )
    .await;
}

#[allow(clippy::too_many_arguments)]
async fn proxy_native_multimodal_chat_completion(
    state: SharedState,
    client: LlamaClient,
    mut body: serde_json::Value,
    original_request_body: serde_json::Value,
    client_request_id: Option<String>,
    model_name: String,
    profile: ModelProfile,
    request_id: String,
    cancel: tokio_util::sync::CancellationToken,
    permit: RequestPermit,
    generation_started_at: String,
    generation_started: std::time::Instant,
    llama_port: u16,
    context_limit: Option<u32>,
) -> Result<Response, ApiErrorResponse> {
    let replay_completion_request = native_replay_request(&body);
    let streaming = body
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let client_requested_stream_usage = body
        .get("stream_options")
        .or_else(|| body.get("streamOptions"))
        .and_then(|options| options.get("include_usage"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "model".to_string(),
            serde_json::Value::String(model_name.clone()),
        );
        if !object.contains_key("max_tokens") {
            if let Some(max_tokens) = object.get("max_completion_tokens").cloned() {
                object.insert("max_tokens".to_string(), max_tokens);
            }
        }
        if streaming {
            let options = object
                .entry("stream_options".to_string())
                .or_insert_with(|| serde_json::json!({}));
            if let Some(options) = options.as_object_mut() {
                options.insert("include_usage".to_string(), serde_json::Value::Bool(true));
            }
        }
    }

    let upstream = match tokio::time::timeout(
        std::time::Duration::from_secs(600),
        client.chat_completion_response(&body),
    )
    .await
    {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            finish_api_generation_for_request(&state, &request_id, "error").await;
            return Err(ApiErrorResponse::inference_failed(&format!(
                "Native chat completion failed: {error}"
            )));
        }
        Err(_) => {
            finish_api_generation_for_request(&state, &request_id, "error").await;
            return Err(ApiErrorResponse::inference_failed(
                "Native chat completion timed out before llama-server returned response headers.",
            ));
        }
    };

    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| HeaderValue::from_bytes(value.as_bytes()).ok());
    let mut upstream_stream = upstream.bytes_stream();
    let state_for_stream = state.clone();
    let request_id_for_stream = request_id.clone();
    let model_for_stream = model_name.clone();
    let profile_for_stream = profile.clone();
    let cancel_for_stream = cancel.clone();

    let stream = async_stream::stream! {
        const TRACE_CAPTURE_LIMIT: usize = 8 * 1024 * 1024;
        let _permit = permit;
        let guard = GenerationDropGuard::new(
            state_for_stream.clone(),
            request_id_for_stream.clone(),
            cancel_for_stream.clone(),
        );
        let mut captured = Vec::new();
        let mut live_sse_buffer = String::new();
        let mut live_trace = NativeMultimodalTrace::default();
        let mut forward_buffer = String::new();
        let mut first_token_at: Option<std::time::Instant> = None;
        let mut observed_tokens: u32 = 0;
        let mut terminal_recorded = false;

        loop {
            let next = tokio::select! {
                _ = cancel_for_stream.cancelled() => {
                    guard.mark_completed();
                    finish_api_generation_for_request(
                        &state_for_stream,
                        &request_id_for_stream,
                        "cancelled",
                    ).await;
                    return;
                }
                next = upstream_stream.next() => next,
            };

            match next {
                Some(Ok(bytes)) => {
                    if captured.len() < TRACE_CAPTURE_LIMIT {
                        let remaining = TRACE_CAPTURE_LIMIT - captured.len();
                        captured.extend_from_slice(&bytes[..bytes.len().min(remaining)]);
                    }
                    append_live_stream_delta_for_request(
                        &state_for_stream,
                        &request_id_for_stream,
                        "raw",
                        &String::from_utf8_lossy(&bytes),
                    ).await;
                    if streaming {
                        live_sse_buffer.push_str(&String::from_utf8_lossy(&bytes));
                        let (content_delta, reasoning_delta, tool_call_delta, saw_done) =
                            drain_native_sse_trace(&mut live_sse_buffer, &mut live_trace);
                        let observed_delta = format!(
                            "{}{}{}",
                            content_delta,
                            reasoning_delta,
                            tool_call_delta.as_deref().unwrap_or_default()
                        );
                        if !reasoning_delta.is_empty() {
                            append_live_stream_delta_for_request(
                                &state_for_stream,
                                &request_id_for_stream,
                                "reasoning",
                                &reasoning_delta,
                            ).await;
                        }
                        if !content_delta.is_empty() {
                            append_live_stream_delta_for_request(
                                &state_for_stream,
                                &request_id_for_stream,
                                "content",
                                &content_delta,
                            ).await;
                        }
                        if let Some(tool_call_delta) = tool_call_delta {
                            append_live_stream_delta_for_request(
                                &state_for_stream,
                                &request_id_for_stream,
                                "tool_call",
                                &tool_call_delta,
                            ).await;
                        }
                        if !observed_delta.is_empty() {
                            first_token_at.get_or_insert_with(std::time::Instant::now);
                            observed_tokens = observed_tokens.saturating_add(
                                crate::normalize::think_strip::estimate_token_count(&observed_delta),
                            );
                            publish_live_generation_metrics(
                                &state_for_stream,
                                "api-native-chat",
                                &model_for_stream,
                                &request_id_for_stream,
                                &generation_started_at,
                                generation_started,
                                first_token_at,
                                observed_tokens,
                            ).await;
                        }
                        if saw_done && !terminal_recorded {
                            finalize_native_proxy_generation(
                                &state_for_stream,
                                &guard,
                                status,
                                &profile_for_stream,
                                &model_for_stream,
                                &request_id_for_stream,
                                &generation_started_at,
                                generation_started,
                                &captured,
                                streaming,
                                client_request_id.clone(),
                                llama_port,
                                context_limit,
                                original_request_body.clone(),
                                &replay_completion_request,
                            ).await;
                            terminal_recorded = true;
                        }
                    }
                    if streaming && !client_requested_stream_usage {
                        forward_buffer.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(newline) = forward_buffer.find('\n') {
                            let line = forward_buffer.drain(..=newline).collect::<String>();
                            let usage_only = line
                                .trim()
                                .strip_prefix("data:")
                                .map(str::trim)
                                .filter(|data| !data.is_empty() && *data != "[DONE]")
                                .and_then(|data| serde_json::from_str::<serde_json::Value>(data).ok())
                                .is_some_and(|value| {
                                    value.get("usage").is_some()
                                        && value
                                            .get("choices")
                                            .and_then(serde_json::Value::as_array)
                                            .is_some_and(Vec::is_empty)
                                });
                            if !usage_only {
                                yield Ok::<Bytes, std::io::Error>(Bytes::from(line));
                            }
                        }
                    } else {
                        yield Ok::<Bytes, std::io::Error>(bytes);
                    }
                }
                Some(Err(error)) => {
                    append_live_stream_delta_for_request(
                        &state_for_stream,
                        &request_id_for_stream,
                        "error",
                        &error.to_string(),
                    ).await;
                    guard.mark_completed();
                    finish_api_generation_for_request(
                        &state_for_stream,
                        &request_id_for_stream,
                        "error",
                    ).await;
                    yield Err::<Bytes, std::io::Error>(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        error.to_string(),
                    ));
                    return;
                }
                None => break,
            }
        }

        if !terminal_recorded {
            finalize_native_proxy_generation(
                &state_for_stream,
                &guard,
                status,
                &profile_for_stream,
                &model_for_stream,
                &request_id_for_stream,
                &generation_started_at,
                generation_started,
                &captured,
                streaming,
                client_request_id.clone(),
                llama_port,
                context_limit,
                original_request_body.clone(),
                &replay_completion_request,
            ).await;
        }

        if !forward_buffer.is_empty() {
            yield Ok::<Bytes, std::io::Error>(Bytes::from(forward_buffer));
        }
    };

    let mut response = Response::builder()
        .status(status)
        .header("x-inference-bridge-chat-route", "native-jinja")
        .header("x-inference-bridge-multimodal-route", "native");
    if let Some(content_type) = content_type {
        response = response.header(header::CONTENT_TYPE, content_type);
    }
    response
        .body(Body::from_stream(stream))
        .map_err(|error| ApiErrorResponse::service_unavailable(error.to_string()))
}

pub async fn chat_completions(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiErrorResponse> {
    if let Some(upstream) = crate::api::upstream::active_openai_provider(&state).await {
        return crate::api::upstream::proxy_json_to_openai_provider(
            state.clone(),
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
    let include_stream_usage = req
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let include_usage_details = false;
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
    let profile = {
        let s = state.read().await;
        s.effective_profile_for_model(&model_name)
    };

    let (
        server_defaults,
        launch_defaults,
        scheduler,
        llama_port,
        context_limit,
        tool_argument_repair_enabled,
    ) = {
        let s = state.read().await;
        (
            (
                s.config.server.default_temperature,
                s.config.server.default_top_p,
                s.config.server.default_top_k,
                s.config.server.default_max_tokens,
            ),
            s.active_sampling_defaults(),
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

    if uses_native_chat_api(&profile) {
        let has_images = api_messages_have_images(&req.messages);
        ensure_runtime_vision_ready(&state, &model_name, &profile, has_images).await?;
        let max_output_tokens = req
            .max_tokens
            .or(server_defaults.3)
            .or(profile.default_max_output_tokens);
        let (native_messages, compaction) = compact_native_messages_to_fit(
            &req.messages,
            context_limit,
            max_output_tokens,
            native_fixed_prompt_tokens(&req),
        )?;
        let mut native_body = build_native_chat_body(
            &original_request_body,
            &req,
            &model_name,
            &profile,
            server_defaults,
            &launch_defaults,
        );
        native_body["messages"] = api_messages_to_native_value(&native_messages);
        {
            let mut s = state.write().await;
            s.last_prompt = Some(
                serde_json::to_string_pretty(&native_body)
                    .unwrap_or_else(|_| native_body.to_string()),
            );
        }
        let generation_started_at = chrono::Utc::now().to_rfc3339();
        let generation_started = std::time::Instant::now();
        let gen = begin_api_generation(&state, model_name.clone()).await;
        let mut response = proxy_native_multimodal_chat_completion(
            state,
            client,
            native_body,
            original_request_body,
            client_request_id,
            model_name,
            profile,
            gen.request_id,
            gen.cancel,
            permit,
            generation_started_at,
            generation_started,
            llama_port,
            context_limit,
        )
        .await?;
        apply_compaction_headers(&mut response, compaction.as_ref());
        return Ok(response);
    }

    let (request, compaction) = build_chat_request(
        &profile,
        req,
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
    .await?;

    let has_images = !request.image_data.is_empty();
    ensure_runtime_vision_ready(&state, &model_name, &profile, has_images).await?;
    {
        let mut s = state.write().await;
        s.last_prompt = Some(request.prompt.clone());
    }

    let generation_started_at = chrono::Utc::now().to_rfc3339();
    let generation_started = std::time::Instant::now();
    let gen = begin_api_generation(&state, model_name.clone()).await;
    let request_id = gen.request_id.clone();

    if has_images {
        return proxy_native_multimodal_chat_completion(
            state,
            client,
            original_request_body.clone(),
            original_request_body,
            client_request_id,
            model_name,
            profile,
            request_id,
            gen.cancel,
            permit,
            generation_started_at,
            generation_started,
            llama_port,
            context_limit,
        )
        .await;
    }

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
            include_stream_usage,
            requested_tools,
            client_request_id,
            original_request_body,
            llama_port,
            context_limit,
            permit,
            compaction,
            tool_argument_repair_enabled,
        )
        .await;
    }

    let buffer_tool_content = requested_tools
        .as_ref()
        .map(|tools| !tools.is_empty())
        .unwrap_or(false);
    let response = match complete_with_live_capture(
        &state,
        &client,
        &request,
        &model_name,
        &request_id,
        "api",
        &generation_started_at,
        generation_started,
        gen.cancel,
        buffer_tool_content,
    )
    .await
    {
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
    let (detected_tool_calls, extracted_text) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(&stripped, &profile);
    let capability_enforcement = crate::normalize::capability_truth::enforce_tool_calls(
        detected_tool_calls,
        extracted_text,
        &crate::normalize::capability_truth::RuntimeCapabilities::from_requested_tools(
            requested_tools.as_deref(),
        ),
    );
    let tool_calls = capability_enforcement.accepted;
    let text = capability_enforcement.display_text;
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
                tool_argument_repair_enabled,
            )
            .await,
        );
    }
    if !api_tool_calls.is_empty() {
        append_live_stream_delta_for_request(
            &state,
            &request_id,
            "tool_call",
            &serde_json::to_string_pretty(&api_tool_calls).unwrap_or_default(),
        )
        .await;
    } else if buffer_tool_content {
        if let Some(text) = &content {
            append_live_stream_delta_for_request(&state, &request_id, "content", text).await;
        }
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
        s.last_parse_trace = Some(crate::normalize::parse_trace::build_parse_trace(
            &profile,
            &response.content,
            content.as_deref().unwrap_or(""),
            None,
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

    let mut response = Json(ChatCompletionResponse {
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
            finish_reason: finish_reason_from_completion(&response, !api_tool_calls.is_empty()),
        }],
        usage: build_usage(
            response.tokens_evaluated.unwrap_or(0),
            response.tokens_predicted.unwrap_or(0),
            reasoning_tokens,
            include_usage_details,
        ),
    })
    .into_response();
    apply_compaction_headers(&mut response, compaction.as_ref());
    Ok(response)
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
    include_stream_usage: bool,
    requested_tools: Option<Vec<serde_json::Value>>,
    client_request_id: Option<String>,
    original_request_body: serde_json::Value,
    llama_port: u16,
    context_limit: Option<u32>,
    permit: RequestPermit,
    compaction: Option<CompactionInfo>,
    tool_argument_repair_enabled: bool,
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

    let stream_cancel = cancel.clone();
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
        let _permit = permit;
        let guard = GenerationDropGuard::new(
            state_for_stream.clone(),
            request_id.clone(),
            stream_cancel,
        );
        let mut raw_full_text = String::new();
        let mut first_token_at: Option<std::time::Instant> = None;
        let mut visible_tokens: u32 = 0;
        let mut output_gate =
            crate::normalize::capability_truth::ToolOutputStreamGate::new(buffer_tool_content);

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
                    if let Some(visible) = output_gate.push(&token) {
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "content", &visible).await;
                        let chunk_json = serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": visible },
                                "finish_reason": serde_json::Value::Null
                            }]
                        });
                        yield Ok::<Event, std::convert::Infallible>(Event::default().data(chunk_json.to_string()));
                    } else {
                        append_live_stream_delta_for_request(&state_for_stream, &request_id, "content_buffered", &token).await;
                    }
                }
                StreamEvent::ReasoningDelta(reasoning) => {
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
                StreamEvent::ToolCallDelta(tool_call) => {
                    append_live_stream_delta_for_request(&state_for_stream, &request_id, "tool_call", &tool_call).await;
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
                    let reasoning_text =
                        extract_reasoning_content_with_style(&full_text, profile.think_tag_style);
                    let stripped = strip_think_tags_with_style(&full_text, profile.think_tag_style);
                    let reasoning_tokens = summarize_reasoning_tokens(
                        Some(tokens_predicted),
                        &stripped,
                        &reasoning_text,
                    );
                    let parse_trace = crate::normalize::parse_trace::build_parse_trace(
                        &profile, &full_text, &stripped, None,
                    );
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

                    let (detected_tool_calls, extracted_text) =
                        crate::normalize::tool_extract::extract_tool_calls_for_profile(
                            &stripped,
                            &profile,
                        );
                    let capability_enforcement =
                        crate::normalize::capability_truth::enforce_tool_calls(
                            detected_tool_calls,
                            extracted_text,
                            &crate::normalize::capability_truth::RuntimeCapabilities::from_requested_tools(
                                requested_tools.as_deref(),
                            ),
                        );
                    let stream_tool_calls = capability_enforcement.accepted;
                    let cleaned_text = capability_enforcement.display_text;
                    let mut api_stream_tool_calls: Vec<serde_json::Value> = Vec::new();
                    for (i, tc) in stream_tool_calls.iter().enumerate() {
                        api_stream_tool_calls.push(
                            api_tool_call_value(
                                Some(&client),
                                &profile,
                                tc,
                                requested_tools.as_ref(),
                                Some(i),
                                tool_argument_repair_enabled,
                            )
                            .await,
                        );
                    }
                    if output_gate.should_emit_final() && !cleaned_text.trim().is_empty() {
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
                    }
                    let stream_response = CompletionResponse {
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
                    let finish_reason = finish_reason_from_completion(
                        &stream_response,
                        !api_stream_tool_calls.is_empty(),
                    );
                    let final_chunk = serde_json::json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
                        }]
                    });
                    guard.mark_completed();
                    finish_api_generation_for_request(
                        &state_for_stream,
                        &request_id,
                        "completed",
                    )
                    .await;
                    yield Ok(Event::default().data(final_chunk.to_string()));
                    if include_stream_usage {
                        let usage_chunk = serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [],
                            "usage": build_usage(
                                tokens_evaluated,
                                tokens_predicted,
                                reasoning_tokens,
                                false,
                            )
                        });
                        yield Ok(Event::default().data(usage_chunk.to_string()));
                    }
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                StreamEvent::Error(error) => {
                    append_live_stream_delta_for_request(&state_for_stream, &request_id, "error", &error).await;
                    guard.mark_completed();
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
            s.last_parse_trace = Some(crate::normalize::parse_trace::build_parse_trace(
                &profile,
                &raw_full_text,
                &stripped,
                None,
            ));
        }
        guard.mark_completed();
        finish_api_generation_for_request(&state_for_stream, &request_id, "completed").await;
        yield Ok(Event::default().data("[DONE]"));
    };

    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response();
    apply_compaction_headers(&mut response, compaction.as_ref());
    Ok(response)
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
    #[serde(default, alias = "minP")]
    pub min_p: Option<f32>,
    #[serde(default, alias = "presencePenalty")]
    pub presence_penalty: Option<f32>,
    #[serde(default, alias = "repeatPenalty", alias = "repetition_penalty")]
    pub repeat_penalty: Option<f32>,
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
    #[serde(default, alias = "responseFormat")]
    pub response_format: Option<serde_json::Value>,
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
        return crate::api::upstream::proxy_json_to_openai_provider(
            state.clone(),
            upstream,
            "/completions",
            body,
        )
        .await;
    }

    let req: TextCompletionRequest = serde_json::from_value(body)
        .map_err(|error| ApiErrorResponse::bad_request(error.to_string()))?;
    if req.stream {
        return Err(ApiErrorResponse::bad_request(
            "`stream: true` is not supported by `/v1/completions` yet. Use `/v1/chat/completions` for streaming, or send `stream: false`.",
        ));
    }

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
    let profile = {
        let s = state.read().await;
        s.effective_profile_for_model(&model_name)
    };

    let prompt = req.prompt.unwrap_or_default();
    if prompt.is_empty() {
        return Err(ApiErrorResponse::bad_request(
            "The `prompt` field is required and must not be empty.",
        ));
    }

    let (srv_temp, srv_top_p, srv_top_k, srv_max_tokens, launch_defaults, port, scheduler) = {
        let s = state.read().await;
        (
            s.config.server.default_temperature,
            s.config.server.default_top_p,
            s.config.server.default_top_k,
            s.config.server.default_max_tokens,
            s.active_sampling_defaults(),
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
        temperature: req
            .temperature
            .or(launch_defaults.temperature)
            .or(srv_temp)
            .or(profile.default_temperature),
        top_p: req
            .top_p
            .or_else(|| req.top.as_ref().and_then(TopParam::as_top_p))
            .or(launch_defaults.top_p)
            .or(srv_top_p)
            .or(profile.default_top_p),
        top_k: req
            .top_k
            .or_else(|| req.top.as_ref().and_then(TopParam::as_top_k))
            .or(launch_defaults.top_k)
            .or(srv_top_k)
            .or(profile.default_top_k),
        min_p: req
            .min_p
            .or(launch_defaults.min_p)
            .or(profile.default_min_p),
        presence_penalty: req
            .presence_penalty
            .or(launch_defaults.presence_penalty)
            .or(profile.default_presence_penalty),
        frequency_penalty: None,
        repeat_penalty: req.repeat_penalty.or(launch_defaults.repeat_penalty),
        seed: req.seed,
        stream: false,
        stop,
        special: true,
        image_data: vec![],
        grammar: None,
        json_schema: response_format_to_json_schema(req.response_format.as_ref()),
    };

    let _permit = scheduler.acquire().await;
    {
        let mut s = state.write().await;
        s.last_prompt = Some(prompt);
    }

    let client = LlamaClient::new(port);
    let generation_started_at = chrono::Utc::now().to_rfc3339();
    let generation_started = std::time::Instant::now();
    let gen = begin_api_generation(&state, model_name.clone()).await;
    let request_id = gen.request_id.clone();
    let response = match complete_with_live_capture(
        &state,
        &client,
        &completion_req,
        &model_name,
        &request_id,
        "text-completions-api",
        &generation_started_at,
        generation_started,
        gen.cancel,
        false,
    )
    .await
    {
        Ok(response) => response,
        Err(error) => {
            finish_api_generation_for_request(&state, &request_id, "error").await;
            tracing::error!(error = %error, "Text completion failed");
            return Err(ApiErrorResponse::inference_failed(&error.to_string()));
        }
    };

    let reasoning_text =
        extract_reasoning_content_with_style(&response.content, profile.think_tag_style);
    let text = strip_think_tags_with_style(&response.content, profile.think_tag_style);
    let reasoning_tokens =
        summarize_reasoning_tokens(response.tokens_predicted, &text, &reasoning_text);
    let elapsed_ms = generation_started.elapsed().as_millis() as u64;
    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(crate::normalize::parse_trace::build_parse_trace(
            &profile,
            &response.content,
            &text,
            None,
        ));
        s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
            source: "text-completions-api".to_string(),
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
            end_to_end_tokens_per_second: end_to_end_tokens_per_second(
                response.tokens_predicted,
                elapsed_ms,
            ),
        });
    }

    let response = Json(TextCompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: now_unix_secs(),
        model: model_name,
        choices: vec![TextChoice {
            index: 0,
            text: text.clone(),
            finish_reason: finish_reason_from_completion(&response, false),
        }],
        usage: build_usage(
            response.tokens_evaluated.unwrap_or(0),
            response.tokens_predicted.unwrap_or(0),
            reasoning_tokens,
            true,
        ),
    })
    .into_response();
    finish_api_generation_for_request(&state, &request_id, "completed").await;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::{
        api_tool_call_value, build_native_chat_body, build_native_replay_canonical,
        build_tool_argument_repair_request, compact_messages_to_fit,
        compact_native_messages_to_fit, complete_with_live_capture,
        context_request_matches_preview, context_size_for_reload, drain_native_sse_trace,
        extract_runtime_load_overrides, loaded_model_matches_request, native_fixed_prompt_tokens,
        native_message_text, normalize_api_messages, parse_native_multimodal_trace,
        response_format_to_json_schema, validate_tool_arguments, ApiMessage, ApiMessageContent,
        ChatCompletionRequest, NativeMultimodalTrace,
    };
    use crate::config::AppConfig;
    use crate::engine::client::{CompletionRequest, LlamaClient};
    use crate::engine::process::{LaunchPreview, SamplingDefaults};
    use crate::models::profiles::ModelProfile;
    use crate::state::{begin_api_generation, finish_api_generation_for_request, AppState};
    use crate::templates::engine::ChatMessage;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn non_stream_completion_is_captured_before_buffered_response_finishes() {
        let app = axum::Router::new().route(
            "/completion",
            axum::routing::post(
                |axum::Json(body): axum::Json<serde_json::Value>| async move {
                    assert_eq!(body.get("stream"), Some(&serde_json::Value::Bool(true)));
                    let events = async_stream::stream! {
                        yield Ok::<_, std::convert::Infallible>(
                            axum::response::sse::Event::default().data(
                                serde_json::json!({ "content": "Hello", "stop": false }).to_string()
                            )
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                        yield Ok::<_, std::convert::Infallible>(
                            axum::response::sse::Event::default().data(
                                serde_json::json!({
                                    "content": " world",
                                    "stop": true,
                                    "tokens_predicted": 2,
                                    "tokens_evaluated": 3,
                                    "timings": {
                                        "predicted_per_second": 20.0,
                                        "prompt_per_second": 100.0
                                    }
                                }).to_string()
                            )
                        );
                    };
                    axum::response::sse::Sse::new(events)
                },
            ),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener should bind");
        let port = listener
            .local_addr()
            .expect("test listener should have an address")
            .port();
        let server = tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("test SSE server should run");
        });

        let state = Arc::new(RwLock::new(
            AppState::new(AppConfig::default()).expect("state should initialize"),
        ));
        let generation = begin_api_generation(&state, "test-model".to_string()).await;
        let request_id = generation.request_id.clone();
        let request = CompletionRequest {
            prompt: "Say hello".to_string(),
            n_predict: Some(8),
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repeat_penalty: None,
            seed: None,
            stream: false,
            stop: Vec::new(),
            special: true,
            image_data: Vec::new(),
            grammar: None,
            json_schema: None,
        };
        let request_for_task = request.clone();
        let state_for_task = state.clone();
        let request_id_for_task = request_id.clone();
        let started_at = chrono::Utc::now().to_rfc3339();
        let client = LlamaClient::new(port);
        let capture = tokio::spawn(async move {
            complete_with_live_capture(
                &state_for_task,
                &client,
                &request_for_task,
                "test-model",
                &request_id_for_task,
                "api",
                &started_at,
                std::time::Instant::now(),
                generation.cancel,
                false,
            )
            .await
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let captured_first_delta = {
                let state = state.read().await;
                let stream = state
                    .live_streams
                    .iter()
                    .find(|stream| stream.request_id == request_id)
                    .expect("request stream should exist");
                stream.raw_output == "Hello"
                    && state
                        .last_generation_metrics
                        .as_ref()
                        .and_then(|metrics| metrics.completion_tokens)
                        == Some(1)
            };
            if captured_first_delta {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "first SSE delta was not captured live"
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            !capture.is_finished(),
            "capture should expose the first delta before the buffered response completes"
        );

        let response = capture
            .await
            .expect("capture task should join")
            .expect("capture should succeed");
        assert_eq!(response.content, "Hello world");
        assert!(
            !request.stream,
            "caller-facing request semantics must stay non-streaming"
        );
        {
            let state = state.read().await;
            let stream = state
                .live_streams
                .iter()
                .find(|stream| stream.request_id == request_id)
                .expect("request stream should remain in history");
            assert_eq!(stream.raw_output, "Hello world");
            assert_eq!(stream.visible_output, "Hello world");
        }

        finish_api_generation_for_request(&state, &request_id, "completed").await;
        server.abort();
    }

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
    fn native_qwen35_body_preserves_structured_messages_and_sampler_precedence() {
        let original = serde_json::json!({
            "messages": [],
            "parallel_tool_calls": false,
            "enable_thinking": true
        });
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "Tess-4-27B-Q4_K_M.gguf",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"London\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "12 C"},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "Read this"},
                    {"type": "input_image", "image_base64": "QUFB"}
                ]}
            ],
            "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "parallel_tool_calls": false
        }))
        .expect("native request should deserialize");
        let profile = ModelProfile::detect("Tess-4-27B-Q4_K_M.gguf");
        let launch = SamplingDefaults {
            temperature: Some(0.6),
            top_p: Some(0.7),
            top_k: None,
            min_p: Some(0.02),
            presence_penalty: None,
            repeat_penalty: Some(1.1),
        };

        let body = build_native_chat_body(
            &original,
            &request,
            "Tess-4-27B-Q4_K_M.gguf",
            &profile,
            (Some(0.9), Some(0.95), Some(33), Some(100)),
            &launch,
        );

        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["top_p"], 0.7);
        assert_eq!(body["top_k"], 33);
        assert_eq!(body["min_p"], 0.02);
        assert_eq!(body["presence_penalty"], 1.5);
        assert_eq!(body["repeat_penalty"], 1.1);
        assert_eq!(body["max_tokens"], 100);
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["messages"][0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(body["messages"][1]["tool_call_id"], "call_1");
        assert_eq!(body["messages"][2]["content"][0]["type"], "text");
        assert_eq!(body["messages"][2]["content"][1]["type"], "image_url");
        assert_eq!(
            body["messages"][2]["content"][1]["image_url"]["url"],
            "data:image/png;base64,QUFB"
        );
        assert!(body.get("enable_thinking").is_none());
    }

    #[test]
    fn native_compaction_evicts_whole_oldest_turn() {
        let message = |role: &str, content: String| ApiMessage {
            role: role.to_string(),
            content: Some(ApiMessageContent::Text(content)),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            refusal: None,
        };
        let messages = vec![
            message("user", "x".repeat(420)),
            message(
                "assistant",
                "old reply that must not be orphaned".to_string(),
            ),
            message("user", "newest request".to_string()),
        ];

        let (compacted, info) =
            match compact_native_messages_to_fit(&messages, Some(100), Some(0), 0) {
                Ok(result) => result,
                Err(_) => panic!("compaction should fit the newest turn"),
            };

        assert!(info.is_some());
        assert!(compacted.iter().all(|message| message.role != "assistant"));
        assert_eq!(compacted.last().unwrap().role, "user");
    }

    #[test]
    fn native_compaction_reserves_full_tess_output_budget() {
        let messages = vec![
            ApiMessage {
                role: "user".to_string(),
                content: Some(ApiMessageContent::Text("x".repeat(120_000))),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                refusal: None,
            },
            ApiMessage {
                role: "assistant".to_string(),
                content: Some(ApiMessageContent::Text("old".to_string())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                refusal: None,
            },
            ApiMessage {
                role: "user".to_string(),
                content: Some(ApiMessageContent::Text("new".to_string())),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                refusal: None,
            },
        ];
        let (_, info) =
            match compact_native_messages_to_fit(&messages, Some(32_768), Some(8_192), 0) {
                Ok(result) => result,
                Err(_) => panic!("32K context with 8K output should retain a 24K prompt budget"),
            };
        assert_eq!(info.expect("compaction expected").budget, 24_576);
    }

    #[test]
    fn native_compaction_does_not_count_base64_as_text_tokens() {
        let messages = vec![ApiMessage {
            role: "user".to_string(),
            content: Some(ApiMessageContent::Parts(vec![
                super::ApiContentPart::Text {
                    text: "describe this".to_string(),
                },
                super::ApiContentPart::InputImage {
                    image_url: None,
                    image_base64: Some("A".repeat(200_000)),
                },
            ])),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            refusal: None,
        }];

        assert!(compact_native_messages_to_fit(&messages, Some(32_768), Some(8_192), 0).is_ok());
    }

    #[test]
    fn native_compaction_coalesces_system_developer_and_summary_for_tess() {
        let text_message = |role: &str, text: String| ApiMessage {
            role: role.to_string(),
            content: Some(ApiMessageContent::Text(text)),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            refusal: None,
        };
        let messages = vec![
            text_message("system", "Follow the safety policy.".to_string()),
            text_message("developer", "Never reveal secrets.".to_string()),
            text_message("user", "x".repeat(120_000)),
            text_message("assistant", "old".to_string()),
            text_message("user", "new request".to_string()),
        ];

        let (compacted, _) =
            match compact_native_messages_to_fit(&messages, Some(32_768), Some(8_192), 0) {
                Ok(result) => result,
                Err(_) => panic!("developer instruction and newest turn should fit"),
            };
        assert_eq!(compacted.first().unwrap().role, "system");
        assert_eq!(
            compacted
                .iter()
                .filter(|message| message.role == "system")
                .count(),
            1
        );
        assert!(compacted.iter().all(|message| message.role != "developer"));
        let instruction = native_message_text(compacted.first().unwrap());
        let system_at = instruction.find("Follow the safety policy.").unwrap();
        let developer_at = instruction.find("Never reveal secrets.").unwrap();
        let summary_at = instruction.find("[Earlier conversation summary]").unwrap();
        assert!(system_at < developer_at && developer_at < summary_at);
    }

    #[test]
    fn native_normalization_removes_developer_without_context_compaction() {
        let messages: Vec<ApiMessage> = serde_json::from_value(serde_json::json!([
            {"role": "system", "content": "system one"},
            {"role": "developer", "content": "developer two"},
            {"role": "user", "content": "hello"}
        ]))
        .expect("messages should deserialize");

        let (normalized, info) =
            match compact_native_messages_to_fit(&messages, None, Some(8_192), 0) {
                Ok(result) => result,
                Err(_) => panic!("normalization should not require an active context size"),
            };
        assert!(info.is_none());
        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0].role, "system");
        assert!(normalized.iter().all(|message| message.role != "developer"));
        assert_eq!(
            native_message_text(&normalized[0]),
            "system one\n\ndeveloper two"
        );
    }

    #[test]
    fn native_compaction_rejects_irreducibly_large_tool_schema() {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "huge_tool",
                    "description": "x".repeat(100_000),
                    "parameters": {"type": "object"}
                }
            }]
        }))
        .expect("tool request should deserialize");
        let fixed = native_fixed_prompt_tokens(&request);

        assert!(compact_native_messages_to_fit(
            &request.messages,
            Some(32_768),
            Some(8_192),
            fixed,
        )
        .is_err());
    }

    #[test]
    fn parses_native_multimodal_json_trace() {
        let raw = br#"{
            "choices":[{"message":{"content":"GREEN | HELIX 73","reasoning_content":"checked pixels"}}],
            "usage":{"prompt_tokens":252,"completion_tokens":8}
        }"#;
        assert_eq!(
            parse_native_multimodal_trace(raw, false),
            NativeMultimodalTrace {
                content: "GREEN | HELIX 73".to_string(),
                reasoning: "checked pixels".to_string(),
                tool_call_deltas: Vec::new(),
                finish_reason: None,
                prompt_tokens: Some(252),
                completion_tokens: Some(8),
            }
        );
    }

    #[test]
    fn parses_native_multimodal_sse_trace_and_usage() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"GREEN | \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"HELIX 73\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":252,\"completion_tokens\":8}}\n\n",
            "data: [DONE]\n\n",
        );
        assert_eq!(
            parse_native_multimodal_trace(raw.as_bytes(), true),
            NativeMultimodalTrace {
                content: "GREEN | HELIX 73".to_string(),
                reasoning: String::new(),
                tool_call_deltas: Vec::new(),
                finish_reason: None,
                prompt_tokens: Some(252),
                completion_tokens: Some(8),
            }
        );
    }

    #[test]
    fn native_terminal_sse_detection_survives_split_done_frame() {
        let mut trace = NativeMultimodalTrace::default();
        let mut buffer = "data: [DO".to_string();
        let (_, _, _, saw_done) = drain_native_sse_trace(&mut buffer, &mut trace);
        assert!(!saw_done);
        assert_eq!(buffer, "data: [DO");

        buffer.push_str("NE]\n\n");
        let (_, _, _, saw_done) = drain_native_sse_trace(&mut buffer, &mut trace);
        assert!(saw_done);
        assert_eq!(buffer, "\n");
    }

    #[test]
    fn native_trace_builds_complete_canonical_replay_payload() {
        let profile = ModelProfile::detect("Tess-4-27B-Q4_K_M.gguf");
        let trace = NativeMultimodalTrace {
            content: "The answer is 12:00.".to_string(),
            reasoning: "I checked the clock tool.".to_string(),
            tool_call_deltas: vec![serde_json::json!({
                "index": 0,
                "id": "call_clock",
                "type": "function",
                "function": {"name": "clock", "arguments": "{\"zone\":\"Europe/London\"}"}
            })],
            finish_reason: Some("tool_calls".to_string()),
            prompt_tokens: Some(100),
            completion_tokens: Some(20),
        };

        let canonical = build_native_replay_canonical(
            &profile,
            &trace,
            "Tess-4-27B-Q4_K_M.gguf",
            "request-native",
            Some("client-correlation".to_string()),
            std::time::Instant::now(),
            8081,
            Some(32_768),
        );

        assert_eq!(
            canonical.client_request_id.as_deref(),
            Some("client-correlation")
        );
        assert_eq!(canonical.correlation_id, "client-correlation");
        assert_eq!(canonical.visible_text, "The answer is 12:00.");
        assert_eq!(canonical.reasoning_text, "I checked the clock tool.");
        assert_eq!(canonical.tool_calls.len(), 1);
        assert_eq!(canonical.tool_calls[0]["id"], "call_clock");
        assert_eq!(canonical.usage.prompt_tokens, Some(100));
        assert_eq!(canonical.usage.completion_tokens, Some(20));
        assert_eq!(canonical.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(canonical.backend.endpoint, "/v1/chat/completions");
        assert!(canonical.raw_text.contains("<think>"));
        assert!(canonical.raw_text.contains("<tool_call>"));
    }

    #[test]
    fn response_format_json_schema_maps_to_llama_schema() {
        let response_format = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "strict": true,
                "schema": {
                    "type": "object",
                    "required": ["unit"],
                    "properties": {
                        "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
                    }
                }
            }
        });

        let schema = response_format_to_json_schema(Some(&response_format))
            .expect("schema response_format should map");
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"][0], "unit");
    }

    #[test]
    fn response_format_json_object_maps_to_object_schema() {
        let schema = response_format_to_json_schema(Some(&serde_json::json!({
            "type": "json_object"
        })))
        .expect("json_object should map");

        assert_eq!(schema, serde_json::json!({ "type": "object" }));
    }

    #[test]
    fn response_format_none_or_unknown_is_unconstrained() {
        assert!(response_format_to_json_schema(None).is_none());
        assert!(response_format_to_json_schema(Some(&serde_json::json!({
            "type": "text"
        })))
        .is_none());
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
            runtime: "llama-server".to_string(),
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
            reasoning_preserve: false,
            template_mode: String::new(),
            template_source: None,
            template_path: None,
            template_name: None,
            chat_template_kwargs_json: None,
            draft_model_path: String::new(),
            spec_type: String::new(),
            spec_draft_n_max: 0,
            draft_max_tokens: 0,
            draft_min_tokens: 0,
            draft_p_min: 0.0,
            sampling_defaults: SamplingDefaults::default(),
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
        let api_value =
            api_tool_call_value(None, &profile, &tool_call, Some(&tools), None, true).await;
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
        let api_value =
            api_tool_call_value(None, &profile, &tool_call, Some(&tools), None, true).await;
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
    fn deserializes_options_context_length_aliases() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [
                    { "role": "user", "content": "hello" }
                ],
                "options": {
                    "context_length": 32768
                }
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.requested_context_size(), Some(32768));
    }

    #[test]
    fn ignores_context_size_in_arbitrary_nested_objects() {
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

        assert_eq!(request.requested_context_size(), None);
    }

    #[test]
    fn ignores_unrelated_numeric_extra_fields_as_context_size() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "timeout": 600000,
                "top_logprobs": 5,
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.requested_context_size(), None);
    }

    #[test]
    fn parses_context_size_strings_with_k_suffix() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [
                    { "role": "user", "content": "hello" }
                ],
                "options": {
                    "contextWindow": "32k"
                }
            }"#,
        )
        .expect("request should deserialize");

        assert_eq!(request.requested_context_size(), Some(32768));
    }

    #[test]
    fn ignores_nested_runtime_load_overrides_in_unknown_objects() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "messages": [
                    { "role": "user", "content": "hello" }
                ],
                "client_metadata": {
                    "file": "surprise.gguf",
                    "fit": "aggressive",
                    "cache_ram_mb": 12345
                }
            }"#,
        )
        .expect("request should deserialize");

        let overrides = extract_runtime_load_overrides(request.options.as_ref(), &request.extra);

        assert!(overrides.hf_file.is_none());
        assert!(overrides.fit_mode.is_none());
        assert!(overrides.cache_ram_mb.is_none());
    }

    #[test]
    fn accepts_top_level_and_options_runtime_load_overrides() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "extra_args": ["--flash-attn"],
                "options": {
                    "cache_ram_mb": 4096
                },
                "messages": [
                    { "role": "user", "content": "hello" }
                ]
            }"#,
        )
        .expect("request should deserialize");

        let overrides = extract_runtime_load_overrides(request.options.as_ref(), &request.extra);

        assert_eq!(overrides.cache_ram_mb, Some(4096));
        assert_eq!(overrides.extra_args, Some(vec!["--flash-attn".to_string()]));
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
    fn rejects_overly_broad_loaded_model_aliases() {
        assert!(!loaded_model_matches_request(
            "Qwen3.6-27B-Q4_K_M.gguf",
            "qwen"
        ));
        assert!(!loaded_model_matches_request(
            "Qwen3.6-27B-Q4_K_M.gguf",
            "27b"
        ));
        assert!(loaded_model_matches_request(
            "Qwen3.6-27B-Q4_K_M.gguf",
            "qwen3.6"
        ));
    }

    #[test]
    fn cloud_model_names_do_not_trigger_api_jit_load() {
        assert!(!super::requested_model_allows_api_jit_load("gpt-4o"));
        assert!(!super::requested_model_allows_api_jit_load(
            "claude-sonnet-4-6"
        ));
        assert!(super::requested_model_allows_api_jit_load(
            "Qwen3.6-27B-Q4_K_M.gguf"
        ));
        assert!(super::requested_model_allows_api_jit_load(
            "C:\\models\\Qwen3.6-27B-Q4_K_M.gguf"
        ));
    }

    #[tokio::test]
    async fn compacts_messages_before_context_overflow() {
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

        compact_messages_to_fit(&mut messages, &profile, Some(2048), Some(256), None).await;
        let prompt = crate::templates::engine::render_prompt(&messages, &profile);

        assert!(crate::normalize::think_strip::estimate_token_count(&prompt) <= 2048);
        assert!(messages
            .iter()
            .any(|message| message.content.contains("[Earlier conversation summary]")));
    }

    #[tokio::test]
    async fn compaction_removes_tool_call_and_result_as_pair() {
        let profile = ModelProfile::detect("Qwen3.6-27B-Q4_K_M.gguf");
        let mut messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Use tools carefully.".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: format!(
                    "<tool_call>{{\"name\":\"search_docs\",\"arguments\":{{\"query\":\"{}\"}}}}</tool_call>",
                    "qwen ".repeat(400)
                ),
            },
            ChatMessage {
                role: "tool".to_string(),
                content: format!("<tool_response>{}</tool_response>", "result ".repeat(400)),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Continue from the tool result.".to_string(),
            },
        ];

        let info = compact_messages_to_fit(&mut messages, &profile, Some(256), Some(64), None)
            .await
            .expect("messages should compact");

        assert_eq!(info.removed_messages, 2);
        assert!(messages
            .iter()
            .any(|message| message.content.contains("[Earlier conversation summary]")));
        assert!(!messages
            .iter()
            .filter(|message| message.role != "system")
            .any(|message| message.content.contains("<tool_response>")));
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
