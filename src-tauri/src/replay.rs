//! Structured replay records for model responses.
//!
//! These JSONL records are intentionally append-only and easy to diff. They
//! give Mauler/HelixClaw runs something deterministic to correlate against
//! without depending on the live UI log shape.

use std::path::PathBuf;

use serde::Serialize;
use serde_json::{json, Value};

use crate::engine::client::{CompletionRequest, CompletionResponse};
use crate::models::profiles::ModelProfile;
use crate::normalize::think_strip::{
    extract_reasoning_content_with_style, strip_think_tags_with_style,
};

const CORRELATION_KEYS: &[&str] = &[
    "request_id",
    "requestId",
    "correlation_id",
    "correlationId",
    "trace_id",
    "traceId",
    "run_id",
    "runId",
];

#[derive(Debug, Clone, Serialize)]
pub struct CanonicalUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CanonicalTimings {
    pub elapsed_ms: u64,
    pub prompt_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub end_to_end_tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CanonicalBackend {
    pub provider: &'static str,
    pub endpoint: &'static str,
    pub port: u16,
    pub context_limit: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CanonicalModelResponse {
    pub object: &'static str,
    pub request_id: String,
    pub client_request_id: Option<String>,
    pub correlation_id: String,
    pub source: String,
    pub model: String,
    pub created_at: String,
    pub raw_text: String,
    pub visible_text: String,
    pub reasoning_text: String,
    pub tool_calls: Vec<Value>,
    pub finish_reason: Option<String>,
    pub usage: CanonicalUsage,
    pub timings: CanonicalTimings,
    pub backend: CanonicalBackend,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiReplayRecord {
    pub schema_version: u32,
    pub kind: &'static str,
    pub captured_at: String,
    pub request: Value,
    pub completion_request: Value,
    pub response: CanonicalModelResponse,
}

pub fn extract_client_correlation_id(body: &Value) -> Option<String> {
    find_correlation_value(body).or_else(|| {
        body.get("metadata")
            .and_then(find_correlation_value)
            .or_else(|| body.get("extra").and_then(find_correlation_value))
    })
}

pub fn extract_header_correlation_id(headers: &axum::http::HeaderMap) -> Option<String> {
    for key in [
        "x-request-id",
        "x-correlation-id",
        "x-trace-id",
        "x-mauler-request-id",
        "x-mauler-run-id",
    ] {
        if let Some(value) = headers
            .get(key)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            return Some(value.to_string());
        }
    }
    None
}

pub fn preferred_client_correlation_id(
    headers: &axum::http::HeaderMap,
    body: &Value,
) -> Option<String> {
    extract_header_correlation_id(headers).or_else(|| extract_client_correlation_id(body))
}

pub fn contains_control_marker_leak(text: &str) -> bool {
    [
        "<|tool_call>",
        "<tool_call|>",
        "<|channel>",
        "<|channel|>",
        "<channel|>",
        "<turn|>",
        "<|turn>",
        "<start_of_turn>",
        "<end_of_turn>",
        "callcall::",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

pub fn build_canonical_response(
    profile: &ModelProfile,
    request_id: &str,
    client_request_id: Option<String>,
    source: &str,
    model: &str,
    raw_text: &str,
    visible_text: &str,
    tool_calls: Vec<Value>,
    response: &CompletionResponse,
    elapsed_ms: u64,
    end_to_end_tokens_per_second: Option<f64>,
    port: u16,
    context_limit: Option<u32>,
) -> CanonicalModelResponse {
    let prompt_tokens = response.tokens_evaluated;
    let completion_tokens = response.tokens_predicted;
    let total_tokens = match (prompt_tokens, completion_tokens) {
        (Some(prompt), Some(completion)) => Some(prompt + completion),
        _ => None,
    };
    let client_request_id = client_request_id.filter(|value| !value.trim().is_empty());
    let correlation_id = client_request_id
        .clone()
        .unwrap_or_else(|| request_id.to_string());
    let raw_visible = strip_think_tags_with_style(raw_text, profile.think_tag_style);
    let visible_text = if visible_text.is_empty() {
        raw_visible
    } else {
        visible_text.to_string()
    };

    CanonicalModelResponse {
        object: "inference_bridge.canonical_model_response",
        request_id: request_id.to_string(),
        client_request_id,
        correlation_id,
        source: source.to_string(),
        model: model.to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        raw_text: raw_text.to_string(),
        visible_text,
        reasoning_text: extract_reasoning_content_with_style(raw_text, profile.think_tag_style),
        tool_calls,
        finish_reason: response.stop_type.clone(),
        usage: CanonicalUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        },
        timings: CanonicalTimings {
            elapsed_ms,
            prompt_tokens_per_second: response
                .timings
                .as_ref()
                .and_then(|timings| timings.prompt_per_second),
            decode_tokens_per_second: response
                .timings
                .as_ref()
                .and_then(|timings| timings.predicted_per_second),
            end_to_end_tokens_per_second,
        },
        backend: CanonicalBackend {
            provider: "llama.cpp",
            endpoint: "/completion",
            port,
            context_limit,
        },
    }
}

pub async fn append_api_replay_record(
    endpoint: &'static str,
    original_request: Value,
    completion_request: &CompletionRequest,
    response: CanonicalModelResponse,
) {
    let record = ApiReplayRecord {
        schema_version: 1,
        kind: endpoint,
        captured_at: chrono::Utc::now().to_rfc3339(),
        request: original_request,
        completion_request: serde_json::to_value(completion_request).unwrap_or_else(|error| {
            json!({
                "serialization_error": error.to_string()
            })
        }),
        response,
    };

    if let Err(error) = append_jsonl(&record).await {
        tracing::warn!(error = %error, "Failed to append InferenceBridge replay log");
    }
}

pub fn replay_log_path() -> PathBuf {
    dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("InferenceBridge")
        .join("replay")
        .join("api-replay.jsonl")
}

async fn append_jsonl<T: Serialize>(record: &T) -> anyhow::Result<()> {
    let path = replay_log_path();
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let line = serde_json::to_string(record)?;
    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    file.write_all(line.as_bytes()).await?;
    file.write_all(b"\n").await?;
    Ok(())
}

fn find_correlation_value(value: &Value) -> Option<String> {
    let object = value.as_object()?;
    for key in CORRELATION_KEYS {
        if let Some(found) = object.get(*key).and_then(value_as_nonempty_string) {
            return Some(found);
        }
    }
    None
}

fn value_as_nonempty_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.trim().is_empty() => Some(text.trim().to_string()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::extract_client_correlation_id;
    use serde_json::json;

    #[test]
    fn extracts_top_level_correlation_id() {
        let body = json!({ "correlation_id": "mauler-run-1" });
        assert_eq!(
            extract_client_correlation_id(&body).as_deref(),
            Some("mauler-run-1")
        );
    }

    #[test]
    fn extracts_metadata_request_id() {
        let body = json!({ "metadata": { "requestId": "turn-42" } });
        assert_eq!(
            extract_client_correlation_id(&body).as_deref(),
            Some("turn-42")
        );
    }

    #[test]
    fn header_correlation_overrides_body_correlation() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            "x-correlation-id",
            axum::http::HeaderValue::from_static("header-turn"),
        );
        let body = json!({ "metadata": { "requestId": "body-turn" } });

        assert_eq!(
            super::preferred_client_correlation_id(&headers, &body).as_deref(),
            Some("header-turn")
        );
    }

    #[test]
    fn detects_marker_leaks() {
        assert!(super::contains_control_marker_leak(
            "<|channel>thought<channel|>Final"
        ));
        assert!(super::contains_control_marker_leak(
            "<|tool_call>callcall::todotodo__createcreate{}"
        ));
        assert!(!super::contains_control_marker_leak("Final answer only."));
    }
}
