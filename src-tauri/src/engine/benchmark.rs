//! Model benchmarking and test utilities

use crate::engine::client::{CompletionRequest, LlamaClient, Timings};
use crate::engine::streaming::{self, StreamEvent};
use crate::state::{begin_live_generation, finish_api_generation_for_request, SharedState};
use tokio::sync::mpsc;

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
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<i64>,
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
        temperature: temperature.or(profile.default_temperature),
        top_p: top_p.or(profile.default_top_p),
        top_k: top_k.or(profile.default_top_k),
        min_p: profile.default_min_p,
        presence_penalty: profile.default_presence_penalty,
        frequency_penalty: None,
        repeat_penalty: None,
        seed,
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
            StreamEvent::Token(_) | StreamEvent::ReasoningDelta(_) => {}
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
    })
}
