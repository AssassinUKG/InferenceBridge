use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(serde::Deserialize)]
struct SseChunkData {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    stop: bool,
    #[serde(default)]
    stopped_limit: Option<bool>,
    #[serde(default)]
    stop_type: Option<String>,
    #[serde(default)]
    tokens_predicted: Option<u64>,
    #[serde(default)]
    tokens_evaluated: Option<u64>,
    #[serde(default)]
    timings: Option<SseTimings>,
}

#[derive(serde::Deserialize)]
struct SseTimings {
    #[serde(default)]
    predicted_per_second: Option<f64>,
    #[serde(default)]
    prompt_per_second: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    RawDelta(String),
    Token(String),
    ReasoningDelta(String),
    ToolCallDelta(String),
    Done {
        full_text: String,
        tokens_predicted: u32,
        tokens_evaluated: u32,
        decode_tokens_per_second: f64,
        prompt_tokens_per_second: Option<f64>,
        stopped_limit: Option<bool>,
        stop_type: Option<String>,
    },
    Error(String),
}

fn longest_partial_suffix(buffer: &str, candidates: &[&str]) -> usize {
    let mut longest = 0usize;
    for candidate in candidates {
        for (start, _) in buffer.char_indices() {
            let suffix = &buffer[start..];
            if !suffix.is_empty()
                && suffix.len() < candidate.len()
                && candidate.starts_with(suffix)
                && suffix.len() > longest
            {
                longest = suffix.len();
            }
        }

        if !buffer.is_empty()
            && buffer.len() < candidate.len()
            && candidate.starts_with(buffer)
            && buffer.len() > longest
        {
            longest = buffer.len();
        }
    }
    longest
}

async fn emit_parsed_content(
    tx: &mpsc::Sender<StreamEvent>,
    parser_buffer: &mut String,
    in_think: &mut bool,
    hidden_tool_close: &mut Option<&'static str>,
    hit_generation_boundary: &mut bool,
) {
    const OPEN_TAGS: [&str; 3] = ["<think>", "<|think|>", "<|channel>thought"];
    const CLOSE_TAGS: [&str; 3] = ["</think>", "<|/think|>", "<channel|>"];
    const BOUNDARY_TAGS: [&str; 8] = [
        "<turn|>",
        "<|turn>",
        "<end_of_turn>",
        "<start_of_turn>",
        "<|im_end|>",
        "<|im_start|>",
        "<|eot_id|>",
        "<|start_header_id|>",
    ];
    const TOOL_OPEN_TAGS: [(&str, Option<&str>); 3] = [
        ("<tool_call>", Some("</tool_call>")),
        ("<div class=\"tool_code\">", Some("</div>")),
        ("<div class='tool_code'>", Some("</div>")),
    ];

    loop {
        if *hit_generation_boundary {
            parser_buffer.clear();
            break;
        } else if let Some(close_tag) = hidden_tool_close {
            if let Some(close_idx) = parser_buffer.find(*close_tag) {
                let drain_len = close_idx + close_tag.len();
                parser_buffer.drain(..drain_len);
                *hidden_tool_close = None;
                continue;
            }

            let keep = longest_partial_suffix(parser_buffer, &[*close_tag]);
            if parser_buffer.len() > keep {
                parser_buffer.replace_range(..parser_buffer.len() - keep, "");
            }
            break;
        } else if *in_think {
            let next_close = CLOSE_TAGS
                .iter()
                .filter_map(|tag| parser_buffer.find(tag).map(|idx| (idx, *tag)))
                .min_by_key(|(idx, _)| *idx);

            if let Some((close_idx, close_tag)) = next_close {
                let reasoning = parser_buffer[..close_idx].to_string();
                if !reasoning.is_empty() {
                    let _ = tx.send(StreamEvent::ReasoningDelta(reasoning)).await;
                }
                let drain_len = close_idx + close_tag.len();
                parser_buffer.drain(..drain_len);
                *in_think = false;
                continue;
            }

            let keep = longest_partial_suffix(parser_buffer, &CLOSE_TAGS);
            if parser_buffer.len() > keep {
                let reasoning = parser_buffer[..parser_buffer.len() - keep].to_string();
                parser_buffer.replace_range(..parser_buffer.len() - keep, "");
                if !reasoning.is_empty() {
                    let _ = tx.send(StreamEvent::ReasoningDelta(reasoning)).await;
                }
            }
            break;
        } else {
            let next_close = CLOSE_TAGS
                .iter()
                .filter_map(|tag| parser_buffer.find(tag).map(|idx| (idx, *tag)))
                .min_by_key(|(idx, _)| *idx);
            let next_open = OPEN_TAGS
                .iter()
                .filter_map(|tag| parser_buffer.find(tag).map(|idx| (idx, *tag)))
                .min_by_key(|(idx, _)| *idx);
            let next_boundary = BOUNDARY_TAGS
                .iter()
                .filter_map(|tag| parser_buffer.find(tag).map(|idx| (idx, *tag)))
                .min_by_key(|(idx, _)| *idx);
            let next_tool_open = TOOL_OPEN_TAGS
                .iter()
                .filter_map(|(tag, close)| parser_buffer.find(tag).map(|idx| (idx, *tag, *close)))
                .min_by_key(|(idx, _, _)| *idx);

            if let Some((boundary_idx, _)) = next_boundary {
                if boundary_idx > 0 {
                    let visible = parser_buffer[..boundary_idx].to_string();
                    if !visible.is_empty() {
                        let _ = tx.send(StreamEvent::Token(visible)).await;
                    }
                }
                parser_buffer.clear();
                *hit_generation_boundary = true;
                break;
            }

            if let Some((close_idx, close_tag)) = next_close {
                let open_idx = next_open.map(|(idx, _)| idx);
                if open_idx.map_or(true, |idx| close_idx < idx) {
                    let reasoning = parser_buffer[..close_idx].to_string();
                    if !reasoning.is_empty() {
                        let _ = tx.send(StreamEvent::ReasoningDelta(reasoning)).await;
                    }
                    let drain_len = close_idx + close_tag.len();
                    parser_buffer.drain(..drain_len);
                    continue;
                }
            }

            if let Some((open_idx, open_tag)) = next_open {
                let tool_idx = next_tool_open.map(|(idx, _, _)| idx);
                if tool_idx.map_or(false, |idx| idx < open_idx) {
                    // Handled below so the earliest hidden marker wins.
                } else {
                    if open_idx > 0 {
                        let visible = parser_buffer[..open_idx].to_string();
                        let _ = tx.send(StreamEvent::Token(visible)).await;
                    }
                    let drain_len = open_idx + open_tag.len();
                    parser_buffer.drain(..drain_len);
                    *in_think = true;
                    continue;
                }
            }

            if let Some((tool_idx, tool_tag, close_tag)) = next_tool_open {
                if tool_idx > 0 {
                    let visible = parser_buffer[..tool_idx].to_string();
                    let _ = tx.send(StreamEvent::Token(visible)).await;
                }
                let drain_len = tool_idx + tool_tag.len();
                parser_buffer.drain(..drain_len);
                *hidden_tool_close = close_tag;
                if hidden_tool_close.is_none() {
                    parser_buffer.clear();
                }
                continue;
            }

            if let Some((open_idx, open_tag)) = next_open {
                if open_idx > 0 {
                    let visible = parser_buffer[..open_idx].to_string();
                    let _ = tx.send(StreamEvent::Token(visible)).await;
                }
                let drain_len = open_idx + open_tag.len();
                parser_buffer.drain(..drain_len);
                *in_think = true;
                continue;
            }

            let tool_open_tags = TOOL_OPEN_TAGS
                .iter()
                .map(|(tag, _)| *tag)
                .collect::<Vec<_>>();
            let keep = longest_partial_suffix(
                parser_buffer,
                &[
                    OPEN_TAGS.as_slice(),
                    CLOSE_TAGS.as_slice(),
                    BOUNDARY_TAGS.as_slice(),
                    tool_open_tags.as_slice(),
                ]
                .concat(),
            );
            if parser_buffer.len() > keep {
                let visible = parser_buffer[..parser_buffer.len() - keep].to_string();
                parser_buffer.replace_range(..parser_buffer.len() - keep, "");
                if !visible.is_empty() {
                    let _ = tx.send(StreamEvent::Token(visible)).await;
                }
            }
            break;
        }
    }
}

pub async fn consume_sse_stream(
    response: reqwest::Response,
    tx: mpsc::Sender<StreamEvent>,
    cancel: CancellationToken,
) -> anyhow::Result<String> {
    consume_sse_stream_with_timeouts(response, tx, cancel, 900, 300).await
}

pub async fn consume_sse_stream_with_timeouts(
    response: reqwest::Response,
    tx: mpsc::Sender<StreamEvent>,
    cancel: CancellationToken,
    first_token_timeout_secs: u64,
    inter_token_timeout_secs: u64,
) -> anyhow::Result<String> {
    let mut full_text = String::new();
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut parser_buffer = String::new();
    let mut in_think = false;
    let mut hidden_tool_close: Option<&'static str> = None;
    let mut hit_generation_boundary = false;
    let mut tokens_predicted: u32 = 0;
    let mut tokens_evaluated: u32 = 0;
    let mut decode_tokens_per_second: f64 = 0.0;
    let mut prompt_tokens_per_second: Option<f64> = None;
    let mut got_first_token = false;

    let first_token_timeout = std::time::Duration::from_secs(first_token_timeout_secs);
    let inter_token_timeout = std::time::Duration::from_secs(inter_token_timeout_secs);

    loop {
        if tx.is_closed() {
            cancel.cancel();
            return Ok(full_text);
        }

        if cancel.is_cancelled() {
            let _ = tx
                .send(StreamEvent::Done {
                    full_text: full_text.clone(),
                    tokens_predicted,
                    tokens_evaluated,
                    decode_tokens_per_second,
                    prompt_tokens_per_second,
                    stopped_limit: None,
                    stop_type: None,
                })
                .await;
            return Ok(full_text);
        }

        let timeout = if got_first_token {
            inter_token_timeout
        } else {
            first_token_timeout
        };

        let chunk = match tokio::time::timeout(timeout, stream.next()).await {
            Ok(Some(Ok(chunk))) => chunk,
            Ok(Some(Err(error))) => {
                let _ = tx.send(StreamEvent::Error(error.to_string())).await;
                return Err(error.into());
            }
            Ok(None) => break,
            Err(_) => {
                let message = if got_first_token {
                    "llama-server stopped responding (inter-token timeout)".to_string()
                } else {
                    "llama-server took too long to start generating (prompt evaluation timeout)"
                        .to_string()
                };
                if full_text.is_empty() {
                    let _ = tx.send(StreamEvent::Error(message)).await;
                } else {
                    let _ = tx
                        .send(StreamEvent::Done {
                            full_text: full_text.clone(),
                            tokens_predicted,
                            tokens_evaluated,
                            decode_tokens_per_second,
                            prompt_tokens_per_second,
                            stopped_limit: None,
                            stop_type: None,
                        })
                        .await;
                }
                return Ok(full_text);
            }
        };

        match std::str::from_utf8(&chunk) {
            Ok(s) => buffer.push_str(s),
            Err(_) => buffer.push_str(&String::from_utf8_lossy(&chunk)),
        }

        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer.drain(..line_end + 1);

            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    if !parser_buffer.is_empty() {
                        if hidden_tool_close.is_some() {
                            parser_buffer.clear();
                        } else if in_think {
                            let _ = tx
                                .send(StreamEvent::ReasoningDelta(parser_buffer.clone()))
                                .await;
                        } else {
                            let _ = tx.send(StreamEvent::Token(parser_buffer.clone())).await;
                        }
                        parser_buffer.clear();
                    }

                    let _ = tx
                        .send(StreamEvent::Done {
                            full_text: full_text.clone(),
                            tokens_predicted,
                            tokens_evaluated,
                            decode_tokens_per_second,
                            prompt_tokens_per_second,
                            stopped_limit: None,
                            stop_type: None,
                        })
                        .await;
                    return Ok(full_text);
                }

                match serde_json::from_str::<SseChunkData>(data) {
                    Ok(json) => {
                        if let Some(content) = json.content.as_deref() {
                            if !content.is_empty() {
                                full_text.push_str(content);
                                let _ = tx.send(StreamEvent::RawDelta(content.to_string())).await;
                                parser_buffer.push_str(content);
                                got_first_token = true;
                                emit_parsed_content(
                                    &tx,
                                    &mut parser_buffer,
                                    &mut in_think,
                                    &mut hidden_tool_close,
                                    &mut hit_generation_boundary,
                                )
                                .await;
                            }
                        }

                        if json.stop {
                            if let Some(value) = json.tokens_predicted {
                                tokens_predicted = value as u32;
                            }
                            if let Some(value) = json.tokens_evaluated {
                                tokens_evaluated = value as u32;
                            }
                            if let Some(value) =
                                json.timings.as_ref().and_then(|t| t.predicted_per_second)
                            {
                                decode_tokens_per_second = value;
                            }
                            if let Some(value) =
                                json.timings.as_ref().and_then(|t| t.prompt_per_second)
                            {
                                prompt_tokens_per_second = Some(value);
                            }

                            if !parser_buffer.is_empty() {
                                emit_parsed_content(
                                    &tx,
                                    &mut parser_buffer,
                                    &mut in_think,
                                    &mut hidden_tool_close,
                                    &mut hit_generation_boundary,
                                )
                                .await;
                            }

                            let _ = tx
                                .send(StreamEvent::Done {
                                    full_text: full_text.clone(),
                                    tokens_predicted,
                                    tokens_evaluated,
                                    decode_tokens_per_second,
                                    prompt_tokens_per_second,
                                    stopped_limit: json.stopped_limit,
                                    stop_type: json.stop_type.clone(),
                                })
                                .await;
                            return Ok(full_text);
                        }
                    }
                    Err(error) => {
                        tracing::warn!(data, error = %error, "Failed to parse SSE data");
                    }
                }
            }
        }
    }

    if !parser_buffer.is_empty() {
        emit_parsed_content(
            &tx,
            &mut parser_buffer,
            &mut in_think,
            &mut hidden_tool_close,
            &mut hit_generation_boundary,
        )
        .await;
    }

    if full_text.is_empty() {
        let _ = tx
            .send(StreamEvent::Error(
                "llama-server returned empty response".to_string(),
            ))
            .await;
    } else {
        let _ = tx
            .send(StreamEvent::Done {
                full_text: full_text.clone(),
                tokens_predicted,
                tokens_evaluated,
                decode_tokens_per_second,
                prompt_tokens_per_second,
                stopped_limit: None,
                stop_type: None,
            })
            .await;
    }

    Ok(full_text)
}

#[derive(Debug, Default)]
struct NativeToolCallAccumulator {
    id: Option<String>,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
struct NativeChatAccumulator {
    content: String,
    reasoning: String,
    tool_calls: std::collections::BTreeMap<usize, NativeToolCallAccumulator>,
    prompt_tokens: u32,
    completion_tokens: u32,
    decode_tokens_per_second: Option<f64>,
    prompt_tokens_per_second: Option<f64>,
    finish_reason: Option<String>,
}

fn merge_native_chat_chunk(
    value: &serde_json::Value,
    accumulator: &mut NativeChatAccumulator,
) -> (Option<String>, Option<String>, Option<String>) {
    let timings = value
        .get("timings")
        .or_else(|| value.pointer("/usage/timings"));
    accumulator.decode_tokens_per_second = timings
        .and_then(|value| value.get("predicted_per_second"))
        .or_else(|| value.get("predicted_per_second"))
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .or(accumulator.decode_tokens_per_second);
    accumulator.prompt_tokens_per_second = timings
        .and_then(|value| value.get("prompt_per_second"))
        .or_else(|| value.get("prompt_per_second"))
        .and_then(serde_json::Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .or(accumulator.prompt_tokens_per_second);

    if let Some(usage) = value.get("usage") {
        accumulator.prompt_tokens = usage
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(accumulator.prompt_tokens);
        accumulator.completion_tokens = usage
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(accumulator.completion_tokens);
    }

    let Some(choice) = value.get("choices").and_then(|choices| choices.get(0)) else {
        return (None, None, None);
    };
    if let Some(reason) = choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
    {
        accumulator.finish_reason = Some(reason.to_string());
    }
    let Some(delta) = choice.get("delta").or_else(|| choice.get("message")) else {
        return (None, None, None);
    };

    let content = delta
        .get("content")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    if let Some(content) = content.as_deref() {
        accumulator.content.push_str(content);
    }
    let reasoning = delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    if let Some(reasoning) = reasoning.as_deref() {
        accumulator.reasoning.push_str(reasoning);
    }

    let tool_call_delta = delta
        .get("tool_calls")
        .filter(|value| value.as_array().is_some_and(|calls| !calls.is_empty()))
        .map(serde_json::Value::to_string);
    if let Some(tool_calls) = delta
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
    {
        for (fallback_index, tool_call) in tool_calls.iter().enumerate() {
            let index = tool_call
                .get("index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(fallback_index);
            let entry = accumulator.tool_calls.entry(index).or_default();
            if let Some(id) = tool_call.get("id").and_then(serde_json::Value::as_str) {
                entry.id = Some(id.to_string());
            }
            let function = tool_call.get("function").unwrap_or(tool_call);
            if let Some(name) = function.get("name").and_then(serde_json::Value::as_str) {
                entry.name.push_str(name);
            }
            if let Some(arguments) = function
                .get("arguments")
                .and_then(serde_json::Value::as_str)
            {
                entry.arguments.push_str(arguments);
            }
        }
    }

    (content, reasoning, tool_call_delta)
}

fn native_decode_rate(
    accumulator: &NativeChatAccumulator,
    first_output_at: Option<&std::time::Instant>,
    stream_started: &std::time::Instant,
) -> f64 {
    if let Some(rate) = accumulator.decode_tokens_per_second {
        return rate;
    }
    if accumulator.completion_tokens == 0 {
        return 0.0;
    }
    let elapsed = first_output_at
        .map(std::time::Instant::elapsed)
        .unwrap_or_else(|| stream_started.elapsed())
        .as_secs_f64();
    if elapsed > 0.0 {
        accumulator.completion_tokens as f64 / elapsed
    } else {
        0.0
    }
}

fn native_chat_full_text(accumulator: &NativeChatAccumulator) -> String {
    let mut full_text = String::new();
    if !accumulator.reasoning.is_empty() {
        full_text.push_str("<think>");
        full_text.push_str(&accumulator.reasoning);
        full_text.push_str("</think>\n");
    }
    full_text.push_str(&accumulator.content);
    for tool_call in accumulator.tool_calls.values() {
        let arguments = serde_json::from_str::<serde_json::Value>(&tool_call.arguments)
            .unwrap_or_else(|_| serde_json::Value::String(tool_call.arguments.clone()));
        let value = serde_json::json!({
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": arguments,
        });
        full_text.push_str("<tool_call>");
        full_text.push_str(&value.to_string());
        full_text.push_str("</tool_call>");
    }
    full_text
}

/// Consume llama-server's native OpenAI chat SSE shape and expose the same
/// normalized events as the legacy `/completion` decoder. This keeps GUI,
/// Responses, and Messages streaming adapters independent of the transport.
pub async fn consume_chat_sse_stream(
    response: reqwest::Response,
    tx: mpsc::Sender<StreamEvent>,
    cancel: CancellationToken,
) -> anyhow::Result<String> {
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut accumulator = NativeChatAccumulator::default();
    let stream_started = std::time::Instant::now();
    let mut first_output_at = None;

    loop {
        if tx.is_closed() {
            cancel.cancel();
            return Ok(native_chat_full_text(&accumulator));
        }
        if cancel.is_cancelled() {
            break;
        }
        let Some(chunk) = stream.next().await else {
            break;
        };
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => {
                let _ = tx.send(StreamEvent::Error(error.to_string())).await;
                return Err(error.into());
            }
        };
        let raw = String::from_utf8_lossy(&chunk);
        let _ = tx.send(StreamEvent::RawDelta(raw.to_string())).await;
        buffer.push_str(&raw);

        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer.drain(..line_end + 1);
            let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                let full_text = native_chat_full_text(&accumulator);
                let decode_tokens_per_second =
                    native_decode_rate(&accumulator, first_output_at.as_ref(), &stream_started);
                let _ = tx
                    .send(StreamEvent::Done {
                        full_text: full_text.clone(),
                        tokens_predicted: accumulator.completion_tokens,
                        tokens_evaluated: accumulator.prompt_tokens,
                        decode_tokens_per_second,
                        prompt_tokens_per_second: accumulator.prompt_tokens_per_second,
                        stopped_limit: Some(accumulator.finish_reason.as_deref() == Some("length")),
                        stop_type: accumulator.finish_reason.clone(),
                    })
                    .await;
                return Ok(full_text);
            }
            match serde_json::from_str::<serde_json::Value>(data) {
                Ok(value) => {
                    let (content, reasoning, tool_call_delta) =
                        merge_native_chat_chunk(&value, &mut accumulator);
                    if first_output_at.is_none()
                        && (content.is_some() || reasoning.is_some() || tool_call_delta.is_some())
                    {
                        first_output_at = Some(std::time::Instant::now());
                    }
                    if let Some(content) = content {
                        let _ = tx.send(StreamEvent::Token(content)).await;
                    }
                    if let Some(reasoning) = reasoning {
                        let _ = tx.send(StreamEvent::ReasoningDelta(reasoning)).await;
                    }
                    if let Some(tool_call_delta) = tool_call_delta {
                        let _ = tx.send(StreamEvent::ToolCallDelta(tool_call_delta)).await;
                    }
                }
                Err(error) => {
                    tracing::warn!(data, error = %error, "Failed to parse native chat SSE data");
                }
            }
        }
    }

    let full_text = native_chat_full_text(&accumulator);
    let decode_tokens_per_second =
        native_decode_rate(&accumulator, first_output_at.as_ref(), &stream_started);
    let _ = tx
        .send(StreamEvent::Done {
            full_text: full_text.clone(),
            tokens_predicted: accumulator.completion_tokens,
            tokens_evaluated: accumulator.prompt_tokens,
            decode_tokens_per_second,
            prompt_tokens_per_second: accumulator.prompt_tokens_per_second,
            stopped_limit: Some(accumulator.finish_reason.as_deref() == Some("length")),
            stop_type: accumulator.finish_reason,
        })
        .await;
    Ok(full_text)
}

#[cfg(test)]
mod tests {
    use super::{
        emit_parsed_content, longest_partial_suffix, merge_native_chat_chunk,
        native_chat_full_text, native_decode_rate, NativeChatAccumulator, StreamEvent,
    };
    use tokio::sync::mpsc;

    #[test]
    fn longest_partial_suffix_handles_multibyte_unicode() {
        assert_eq!(longest_partial_suffix("\u{e9}", &["<think>"]), 0);
    }

    #[test]
    fn longest_partial_suffix_keeps_partial_tag_after_unicode_prefix() {
        assert_eq!(longest_partial_suffix("\u{e9}<thi", &["<think>"]), 4);
    }

    #[test]
    fn native_chat_accumulates_two_tool_calls_without_empty_think_tags() {
        let mut accumulator = NativeChatAccumulator::default();
        merge_native_chat_chunk(
            &serde_json::json!({
                "choices": [{
                    "delta": {
                        "reasoning_content": "",
                        "tool_calls": [
                            {"index": 0, "id": "call_weather", "function": {"name": "weather", "arguments": "{\"city\":"}},
                            {"index": 1, "id": "call_time", "function": {"name": "clock", "arguments": "{\"zone\":"}}
                        ]
                    }
                }]
            }),
            &mut accumulator,
        );
        merge_native_chat_chunk(
            &serde_json::json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": "\"London\"}"}},
                            {"index": 1, "function": {"arguments": "\"Europe/London\"}"}}
                        ]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 21, "completion_tokens": 9},
                "timings": {"predicted_per_second": 42.5, "prompt_per_second": 812.0}
            }),
            &mut accumulator,
        );

        let text = native_chat_full_text(&accumulator);
        assert!(!text.contains("<think>"));
        assert!(text.contains("\"name\":\"weather\""));
        assert!(text.contains("\"city\":\"London\""));
        assert!(text.contains("\"name\":\"clock\""));
        assert!(text.contains("\"zone\":\"Europe/London\""));
        assert_eq!(accumulator.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(accumulator.prompt_tokens, 21);
        assert_eq!(accumulator.completion_tokens, 9);
        assert_eq!(accumulator.decode_tokens_per_second, Some(42.5));
        assert_eq!(accumulator.prompt_tokens_per_second, Some(812.0));
    }

    #[test]
    fn native_chat_reads_llama_timing_only_usage_chunk() {
        let mut accumulator = NativeChatAccumulator::default();
        let result = merge_native_chat_chunk(
            &serde_json::json!({
                "choices": [],
                "usage": {
                    "completion_tokens": 162,
                    "prompt_tokens": 3014,
                    "total_tokens": 3176
                },
                "timings": {
                    "prompt_per_second": 250.5,
                    "predicted_per_second": 36.75
                }
            }),
            &mut accumulator,
        );

        assert_eq!(result, (None, None, None));
        assert_eq!(accumulator.prompt_tokens, 3014);
        assert_eq!(accumulator.completion_tokens, 162);
        assert_eq!(accumulator.prompt_tokens_per_second, Some(250.5));
        assert_eq!(accumulator.decode_tokens_per_second, Some(36.75));
    }

    #[test]
    fn native_chat_measures_a_nonzero_rate_when_timings_are_missing() {
        let accumulator = NativeChatAccumulator {
            completion_tokens: 10,
            ..NativeChatAccumulator::default()
        };
        let first_output_at = std::time::Instant::now() - std::time::Duration::from_secs(2);
        let stream_started = first_output_at;

        let rate = native_decode_rate(&accumulator, Some(&first_output_at), &stream_started);
        assert!(
            rate > 4.0 && rate <= 5.0,
            "unexpected measured rate: {rate}"
        );
    }

    #[tokio::test]
    async fn parsed_content_routes_orphan_close_prefix_to_reasoning() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut parser_buffer = "scratch notes</think>Final answer".to_string();
        let mut in_think = false;
        let mut hidden_tool_close = None;
        let mut hit_generation_boundary = false;

        emit_parsed_content(
            &tx,
            &mut parser_buffer,
            &mut in_think,
            &mut hidden_tool_close,
            &mut hit_generation_boundary,
        )
        .await;
        drop(tx);

        match rx.recv().await {
            Some(StreamEvent::ReasoningDelta(text)) => assert_eq!(text, "scratch notes"),
            other => panic!("expected reasoning delta, got {other:?}"),
        }
        match rx.recv().await {
            Some(StreamEvent::Token(text)) => assert_eq!(text, "Final answer"),
            other => panic!("expected visible token, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn parsed_content_suppresses_streamed_tool_call_markup() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut parser_buffer =
            "Let me check.\n<tool_call>_.glob(dir=\".\",pattern=\"**/*\")".to_string();
        let mut in_think = false;
        let mut hidden_tool_close = None;
        let mut hit_generation_boundary = false;

        emit_parsed_content(
            &tx,
            &mut parser_buffer,
            &mut in_think,
            &mut hidden_tool_close,
            &mut hit_generation_boundary,
        )
        .await;
        drop(tx);

        match rx.recv().await {
            Some(StreamEvent::Token(text)) => assert_eq!(text, "Let me check.\n"),
            other => panic!("expected visible preamble, got {other:?}"),
        }
        assert!(rx.recv().await.is_none());
        assert_eq!(hidden_tool_close, Some("</tool_call>"));
    }

    #[tokio::test]
    async fn parsed_content_routes_gemma_channel_markers_to_reasoning() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let mut parser_buffer =
            "<|channel>thought\nprivate notes\n<channel|>Final answer".to_string();
        let mut in_think = false;
        let mut hidden_tool_close = None;
        let mut hit_generation_boundary = false;

        emit_parsed_content(
            &tx,
            &mut parser_buffer,
            &mut in_think,
            &mut hidden_tool_close,
            &mut hit_generation_boundary,
        )
        .await;
        drop(tx);

        match rx.recv().await {
            Some(StreamEvent::ReasoningDelta(text)) => assert_eq!(text, "\nprivate notes\n"),
            other => panic!("expected reasoning delta, got {other:?}"),
        }
        match rx.recv().await {
            Some(StreamEvent::Token(text)) => assert_eq!(text, "Final answer"),
            other => panic!("expected visible token, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn parsed_content_truncates_at_gemma_turn_boundary() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let mut parser_buffer =
            "Final answer.<turn|><|turn>user\nignore this continuation".to_string();
        let mut in_think = false;
        let mut hidden_tool_close = None;
        let mut hit_generation_boundary = false;

        emit_parsed_content(
            &tx,
            &mut parser_buffer,
            &mut in_think,
            &mut hidden_tool_close,
            &mut hit_generation_boundary,
        )
        .await;
        drop(tx);

        match rx.recv().await {
            Some(StreamEvent::Token(text)) => assert_eq!(text, "Final answer."),
            other => panic!("expected visible answer before boundary, got {other:?}"),
        }
        assert!(hit_generation_boundary);
        assert!(rx.recv().await.is_none());
    }
}
