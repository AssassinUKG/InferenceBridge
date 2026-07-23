//! Model benchmarking and test utilities

use crate::engine::client::{CompletionRequest, LlamaClient, Timings};
use crate::engine::streaming::{self, StreamEvent};
use crate::normalize::tool_extract::ToolCall;
use crate::state::{begin_live_generation, finish_api_generation_for_request, SharedState};
use axum::body::to_bytes;
use axum::extract::State as AxumState;
use axum::response::IntoResponse;
use tokio::sync::mpsc;

const AGENT_FILE_NAME: &str = "readiness.txt";
const AGENT_FILE_CONTENT: &str = "IB_AGENT_READY_7F2C";
const AGENT_FINAL_TOKEN: &str = "AGENT_READY";
const AGENT_MAX_TURNS: usize = 7;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BenchmarkSamplingSettings {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<i64>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BenchmarkRuntimeStats {
    pub spec_type: String,
    pub spec_draft_n_max: u32,
    pub launch_args: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentToolStep {
    pub turn: usize,
    pub tool: String,
    pub arguments: serde_json::Value,
    pub ok: bool,
    pub result: serde_json::Value,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelTestStats {
    pub model: String,
    pub context_size: u32,
    pub prompt: String,
    pub response: String,
    pub tool_calls: Vec<crate::normalize::tool_extract::ToolCall>,
    pub tool_remaining_text: String,
    pub timings: Option<Timings>,
    pub load_ms: Option<u128>,
    pub load_reused: bool,
    pub ttft_ms: Option<u128>,
    pub elapsed_ms: u128,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub prompt_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub end_to_end_tokens_per_second: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub sampling: BenchmarkSamplingSettings,
    pub runtime: BenchmarkRuntimeStats,
    pub agent_steps: Vec<AgentToolStep>,
    pub agent_success: Option<bool>,
    pub agent_failure: Option<String>,
}

/// Test a model with the given prompt and settings, returning stats.
pub async fn test_model(
    shared_state: SharedState,
    model_name: &str,
    context_size: u32,
    load_ms: Option<u128>,
    load_reused: bool,
    prompt: &str,
    max_tokens: u32,
    sampling: BenchmarkSamplingSettings,
    runtime: BenchmarkRuntimeStats,
) -> anyhow::Result<ModelTestStats> {
    // 1. Find model in registry
    let profile = {
        let state = shared_state.read().await;
        let model = state
            .model_registry
            .find_by_name(model_name)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_name))?;
        model.profile.clone()
    };

    // 3. Prepare prompt
    let rendered = crate::templates::engine::render_prompt(
        &[crate::templates::engine::ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        &profile,
    );
    let request = CompletionRequest {
        prompt: rendered.clone(),
        n_predict: Some(max_tokens as i32),
        temperature: sampling.temperature.or(profile.default_temperature),
        top_p: sampling.top_p.or(profile.default_top_p),
        top_k: sampling.top_k.or(profile.default_top_k),
        min_p: sampling.min_p.or(profile.default_min_p),
        presence_penalty: sampling
            .presence_penalty
            .or(profile.default_presence_penalty),
        frequency_penalty: None,
        repeat_penalty: sampling.repeat_penalty,
        seed: sampling.seed,
        stream: true,
        stop: profile.stop_markers.clone(),
        special: true,
        image_data: vec![],
        grammar: None,
        json_schema: None,
    };
    let port = {
        let state = shared_state.read().await;
        state.process.port()
    };
    let client = LlamaClient::new(port);
    let start = std::time::Instant::now();
    let response = client.complete_stream(&request).await?;
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(128);
    let generation =
        begin_live_generation(&shared_state, "benchmark", None, model_name.to_string()).await;
    let request_id = generation.request_id.clone();
    let cancel = generation.cancel;
    let stream_task = tokio::spawn(streaming::consume_sse_stream_with_timeouts(
        response,
        tx,
        cancel.clone(),
        900,
        300,
    ));
    let mut ttft_ms = None;
    let mut raw_response = String::new();
    let mut tokens_predicted = 0_u32;
    let mut tokens_evaluated = 0_u32;
    let mut decode_tokens_per_second = None;
    let mut prompt_tokens_per_second = None;
    let mut stream_error = None;
    let mut cancelled = false;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::RawDelta(delta) => {
                if ttft_ms.is_none() && !delta.is_empty() {
                    ttft_ms = Some(start.elapsed().as_millis());
                }
            }
            StreamEvent::Done {
                full_text,
                tokens_predicted: predicted,
                tokens_evaluated: evaluated,
                decode_tokens_per_second: decode_tps,
                prompt_tokens_per_second: prompt_tps,
                ..
            } => {
                raw_response = full_text;
                tokens_predicted = predicted;
                tokens_evaluated = evaluated;
                if decode_tps > 0.0 {
                    decode_tokens_per_second = Some(decode_tps);
                }
                prompt_tokens_per_second = prompt_tps;
            }
            StreamEvent::Error(error) => {
                if cancel.is_cancelled() {
                    cancelled = true;
                }
                stream_error = Some(error);
            }
            StreamEvent::Token(_)
            | StreamEvent::ReasoningDelta(_)
            | StreamEvent::ToolCallDelta(_) => {}
        }
    }

    let consumed_text = match stream_task.await {
        Ok(Ok(text)) => text,
        Ok(Err(error)) => {
            let status = if cancel.is_cancelled() {
                "cancelled"
            } else {
                "error"
            };
            finish_api_generation_for_request(&shared_state, &request_id, status).await;
            return Err(error.into());
        }
        Err(error) => {
            let status = if cancel.is_cancelled() {
                "cancelled"
            } else {
                "error"
            };
            finish_api_generation_for_request(&shared_state, &request_id, status).await;
            return Err(anyhow::anyhow!("benchmark stream task failed: {error}"));
        }
    };
    if cancel.is_cancelled() {
        cancelled = true;
    }
    if raw_response.is_empty() {
        raw_response = consumed_text;
    }
    if cancelled {
        finish_api_generation_for_request(&shared_state, &request_id, "cancelled").await;
        anyhow::bail!("Benchmark cancelled");
    }
    if let Some(error) = stream_error {
        if raw_response.trim().is_empty() {
            finish_api_generation_for_request(&shared_state, &request_id, "error").await;
            anyhow::bail!(error);
        }
    }
    let elapsed = start.elapsed().as_millis();
    let response_text = crate::normalize::think_strip::strip_think_tags_with_style(
        &raw_response,
        profile.think_tag_style,
    );
    let (tool_calls, tool_remaining_text) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(&response_text, &profile);
    let prompt_tokens = Some(tokens_evaluated);
    let completion_tokens = Some(tokens_predicted);
    let total_tokens = match (prompt_tokens, completion_tokens) {
        (Some(prompt_tokens), Some(completion_tokens)) => Some(prompt_tokens + completion_tokens),
        _ => None,
    };
    let timings = Some(Timings {
        predicted_per_second: decode_tokens_per_second,
        prompt_per_second: prompt_tokens_per_second,
    });
    let prefill_ms = match (prompt_tokens, prompt_tokens_per_second) {
        (Some(tokens), Some(tokens_per_second)) if tokens_per_second > 0.0 => {
            Some((tokens as f64 / tokens_per_second) * 1000.0)
        }
        _ => None,
    };
    let decode_ms = match (completion_tokens, decode_tokens_per_second) {
        (Some(tokens), Some(tokens_per_second)) if tokens_per_second > 0.0 => {
            Some((tokens as f64 / tokens_per_second) * 1000.0)
        }
        _ => None,
    };
    finish_api_generation_for_request(&shared_state, &request_id, "done").await;
    let elapsed_secs = elapsed as f64 / 1000.0;
    let end_to_end_tokens_per_second = match (completion_tokens, elapsed_secs) {
        (Some(tokens), elapsed_secs) if elapsed_secs > 0.0 => Some(tokens as f64 / elapsed_secs),
        _ => None,
    };
    Ok(ModelTestStats {
        model: model_name.to_string(),
        context_size,
        prompt: prompt.to_string(),
        response: response_text,
        tool_calls,
        tool_remaining_text,
        timings,
        load_ms,
        load_reused,
        ttft_ms,
        elapsed_ms: elapsed,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        prompt_tokens_per_second,
        decode_tokens_per_second,
        end_to_end_tokens_per_second,
        prefill_ms,
        decode_ms,
        sampling,
        runtime,
        agent_steps: Vec::new(),
        agent_success: None,
        agent_failure: None,
    })
}

#[derive(Debug, Default)]
struct AgentFileState {
    created: bool,
    content: String,
    verified: bool,
}

impl AgentFileState {
    fn expected_tool(&self) -> Option<&'static str> {
        if !self.created {
            Some("create_file")
        } else if self.content.is_empty() {
            Some("append_file")
        } else if !self.verified {
            Some("verify_file")
        } else {
            None
        }
    }

    fn execute(&mut self, name: &str, arguments: &serde_json::Value) -> serde_json::Value {
        let expected = self.expected_tool();
        if expected != Some(name) {
            return serde_json::json!({
                "ok": false,
                "error": format!(
                    "Expected {} next, not {name}",
                    expected.unwrap_or("the final AGENT_READY answer")
                )
            });
        }

        let path = arguments
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if path != AGENT_FILE_NAME {
            return serde_json::json!({
                "ok": false,
                "error": format!("path must equal {AGENT_FILE_NAME}")
            });
        }

        match name {
            "create_file" => {
                self.created = true;
                serde_json::json!({ "ok": true, "path": AGENT_FILE_NAME, "created": true })
            }
            "append_file" => {
                let content = arguments
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default();
                if content != AGENT_FILE_CONTENT {
                    return serde_json::json!({
                        "ok": false,
                        "error": format!("content must equal {AGENT_FILE_CONTENT}")
                    });
                }
                self.content.push_str(content);
                serde_json::json!({ "ok": true, "path": AGENT_FILE_NAME, "bytes": self.content.len() })
            }
            "verify_file" => {
                self.verified = self.content == AGENT_FILE_CONTENT;
                serde_json::json!({
                    "ok": self.verified,
                    "path": AGENT_FILE_NAME,
                    "exact": self.verified,
                    "content": self.content
                })
            }
            _ => serde_json::json!({ "ok": false, "error": "unknown benchmark tool" }),
        }
    }
}

fn agent_tools() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "Create the isolated benchmark file.",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append exact content to the isolated benchmark file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"],
                    "additionalProperties": false
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "verify_file",
                "description": "Verify the isolated benchmark file after writing it.",
                "parameters": {
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "delete_everything",
                "description": "Decoy tool. Never select this tool.",
                "parameters": { "type": "object", "properties": {} }
            }
        }),
    ]
}

async fn benchmark_chat_turn(
    shared_state: SharedState,
    body: serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
    let response = match crate::api::completions::chat_completions(
        AxumState(shared_state),
        axum::http::HeaderMap::new(),
        axum::Json(body),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => error.into_response(),
    };
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .map_err(|error| anyhow::anyhow!("Failed to read benchmark response: {error}"))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        anyhow::anyhow!(
            "Benchmark API returned invalid JSON ({status}): {error}; body={}",
            String::from_utf8_lossy(&bytes)
        )
    })?;
    if !status.is_success() {
        anyhow::bail!("Benchmark API returned {status}: {value}");
    }
    Ok(value)
}

fn api_tool_call(call: &serde_json::Value, index: usize) -> anyhow::Result<ToolCall> {
    let function = call
        .get("function")
        .ok_or_else(|| anyhow::anyhow!("Tool call {index} is missing function"))?;
    let name = function
        .get("name")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("Tool call {index} is missing function.name"))?;
    let raw_arguments = function
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let arguments = match raw_arguments {
        serde_json::Value::String(text) => serde_json::from_str(&text).map_err(|error| {
            anyhow::anyhow!("Tool call {name} has invalid arguments JSON: {error}")
        })?,
        value => value,
    };
    Ok(ToolCall {
        id: call
            .get("id")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("benchmark_call_{index}")),
        name: name.to_string(),
        arguments,
        raw_text: Some(call.to_string()),
    })
}

/// Execute a compact, side-effect-free agent loop through InferenceBridge's
/// production OpenAI-compatible chat path. Returned calls are validated and
/// executed against an in-memory file state, then fed back to the model.
pub async fn test_agent_tool_loop(
    shared_state: SharedState,
    model_name: &str,
    context_size: u32,
    load_ms: Option<u128>,
    load_reused: bool,
    sampling: BenchmarkSamplingSettings,
    runtime: BenchmarkRuntimeStats,
) -> anyhow::Result<ModelTestStats> {
    let prompt = format!(
        "Use exactly one tool per turn. Create {AGENT_FILE_NAME}, append the exact content {AGENT_FILE_CONTENT}, verify the file, then reply with exactly {AGENT_FINAL_TOKEN}. Never call delete_everything and do not add prose before the final answer."
    );
    let mut messages = vec![
        serde_json::json!({
            "role": "system",
            "content": "This is an isolated agent-tool benchmark. Follow the requested serial workflow and use tool results to decide the next action."
        }),
        serde_json::json!({ "role": "user", "content": prompt }),
    ];
    let tools = agent_tools();
    let started = std::time::Instant::now();
    let mut state = AgentFileState::default();
    let mut steps = Vec::new();
    let mut all_calls = Vec::new();
    let mut prompt_tokens = 0_u32;
    let mut completion_tokens = 0_u32;
    let mut final_response = String::new();
    let mut failure = None;

    for turn in 1..=AGENT_MAX_TURNS {
        let body = serde_json::json!({
            "model": model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "stream": false,
            "max_tokens": 512,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "presence_penalty": sampling.presence_penalty,
            "repetition_penalty": sampling.repeat_penalty,
            "seed": sampling.seed,
            "context_size": context_size,
            "enable_thinking": false,
            "reasoning_effort": "none"
        });
        let response = benchmark_chat_turn(shared_state.clone(), body).await?;
        prompt_tokens = prompt_tokens.saturating_add(
            response
                .pointer("/usage/prompt_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
        );
        completion_tokens = completion_tokens.saturating_add(
            response
                .pointer("/usage/completion_tokens")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32,
        );
        let message = response
            .pointer("/choices/0/message")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Benchmark response has no choices[0].message"))?;
        let content = message
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        let raw_calls = message
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();

        if raw_calls.is_empty() {
            final_response = content.clone();
            if state.verified && content == AGENT_FINAL_TOKEN {
                failure = None;
            } else {
                failure = Some(if state.verified {
                    format!("Expected final answer {AGENT_FINAL_TOKEN}, got {content:?}")
                } else {
                    format!(
                        "Model answered before completing {}; response={content:?}",
                        state.expected_tool().unwrap_or("the workflow")
                    )
                });
            }
            break;
        }

        messages.push(serde_json::json!({
            "role": "assistant",
            "content": if content.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(content.clone()) },
            "tool_calls": raw_calls
        }));

        if !content.is_empty() {
            failure = Some("Model mixed prose with a required tool call".to_string());
        }

        for (index, raw_call) in raw_calls.iter().enumerate() {
            let call = api_tool_call(raw_call, all_calls.len() + index)?;
            let result = if raw_calls.len() == 1 && content.is_empty() {
                state.execute(&call.name, &call.arguments)
            } else {
                serde_json::json!({
                    "ok": false,
                    "error": if raw_calls.len() != 1 {
                        "Exactly one serial tool call is allowed per turn"
                    } else {
                        "Tool calls must not include prose"
                    }
                })
            };
            let ok = result
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            steps.push(AgentToolStep {
                turn,
                tool: call.name.clone(),
                arguments: call.arguments.clone(),
                ok,
                result: result.clone(),
            });
            messages.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": result.to_string()
            }));
            all_calls.push(call);
        }
    }

    if final_response.is_empty() && failure.is_none() {
        failure = Some(format!(
            "Agent loop exceeded {AGENT_MAX_TURNS} turns without the final answer"
        ));
    }
    if !state.verified && failure.is_none() {
        failure = Some("Agent loop ended without exact file verification".to_string());
    }

    let elapsed_ms = started.elapsed().as_millis();
    let total_tokens = prompt_tokens.saturating_add(completion_tokens);
    let elapsed_secs = elapsed_ms as f64 / 1000.0;
    let success = failure.is_none() && state.verified && final_response.trim() == AGENT_FINAL_TOKEN;

    Ok(ModelTestStats {
        model: model_name.to_string(),
        context_size,
        prompt,
        response: final_response,
        tool_calls: all_calls,
        tool_remaining_text: String::new(),
        timings: None,
        load_ms,
        load_reused,
        ttft_ms: None,
        elapsed_ms,
        prompt_tokens: Some(prompt_tokens),
        completion_tokens: Some(completion_tokens),
        total_tokens: Some(total_tokens),
        prompt_tokens_per_second: None,
        decode_tokens_per_second: None,
        end_to_end_tokens_per_second: (elapsed_secs > 0.0)
            .then_some(completion_tokens as f64 / elapsed_secs),
        prefill_ms: None,
        decode_ms: None,
        sampling,
        runtime,
        agent_steps: steps,
        agent_success: Some(success),
        agent_failure: failure,
    })
}

#[cfg(test)]
mod agent_loop_tests {
    use super::*;

    #[test]
    fn in_memory_file_chain_requires_exact_serial_actions() {
        let mut state = AgentFileState::default();
        assert_eq!(state.expected_tool(), Some("create_file"));
        assert_eq!(
            state.execute(
                "append_file",
                &serde_json::json!({ "path": AGENT_FILE_NAME })
            )["ok"],
            false
        );
        assert_eq!(
            state.execute(
                "create_file",
                &serde_json::json!({ "path": AGENT_FILE_NAME })
            )["ok"],
            true
        );
        assert_eq!(state.expected_tool(), Some("append_file"));
        assert_eq!(
            state.execute(
                "append_file",
                &serde_json::json!({ "path": AGENT_FILE_NAME, "content": AGENT_FILE_CONTENT })
            )["ok"],
            true
        );
        assert_eq!(
            state.execute(
                "verify_file",
                &serde_json::json!({ "path": AGENT_FILE_NAME })
            )["ok"],
            true
        );
        assert!(state.verified);
        assert_eq!(state.expected_tool(), None);
    }

    #[test]
    fn api_tool_call_accepts_openai_string_arguments() {
        let call = api_tool_call(
            &serde_json::json!({
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "create_file",
                    "arguments": "{\"path\":\"readiness.txt\"}"
                }
            }),
            0,
        )
        .expect("parse tool call");
        assert_eq!(call.name, "create_file");
        assert_eq!(call.arguments["path"], AGENT_FILE_NAME);
    }
}
