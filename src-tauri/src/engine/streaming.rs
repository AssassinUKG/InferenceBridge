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

#[cfg(test)]
mod tests {
    use super::{emit_parsed_content, longest_partial_suffix, StreamEvent};
    use tokio::sync::mpsc;

    #[test]
    fn longest_partial_suffix_handles_multibyte_unicode() {
        assert_eq!(longest_partial_suffix("\u{e9}", &["<think>"]), 0);
    }

    #[test]
    fn longest_partial_suffix_keeps_partial_tag_after_unicode_prefix() {
        assert_eq!(longest_partial_suffix("\u{e9}<thi", &["<think>"]), 4);
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
