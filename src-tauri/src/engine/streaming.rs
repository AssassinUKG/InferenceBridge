use futures_util::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(String),
    ReasoningDelta(String),
    Done {
        full_text: String,
        tokens_predicted: u32,
        tokens_evaluated: u32,
        decode_tokens_per_second: f64,
        prompt_tokens_per_second: Option<f64>,
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
) {
    const OPEN_TAGS: [&str; 2] = ["<think>", "<|think|>"];
    const CLOSE_TAGS: [&str; 2] = ["</think>", "<|/think|>"];

    loop {
        if *in_think {
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
            let next_open = OPEN_TAGS
                .iter()
                .filter_map(|tag| parser_buffer.find(tag).map(|idx| (idx, *tag)))
                .min_by_key(|(idx, _)| *idx);

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

            let keep = longest_partial_suffix(parser_buffer, &OPEN_TAGS);
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
    consume_sse_stream_with_timeouts(response, tx, cancel, 300, 120).await
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
    let mut tokens_predicted: u32 = 0;
    let mut tokens_evaluated: u32 = 0;
    let mut decode_tokens_per_second: f64 = 0.0;
    let mut prompt_tokens_per_second: Option<f64> = None;
    let mut got_first_token = false;

    let first_token_timeout = std::time::Duration::from_secs(first_token_timeout_secs);
    let inter_token_timeout = std::time::Duration::from_secs(inter_token_timeout_secs);

    loop {
        if cancel.is_cancelled() {
            let _ = tx
                .send(StreamEvent::Done {
                    full_text: full_text.clone(),
                    tokens_predicted,
                    tokens_evaluated,
                    decode_tokens_per_second,
                    prompt_tokens_per_second,
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
                        })
                        .await;
                }
                return Ok(full_text);
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer = buffer[line_end + 1..].to_string();

            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    if !parser_buffer.is_empty() {
                        if in_think {
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
                        })
                        .await;
                    return Ok(full_text);
                }

                match serde_json::from_str::<serde_json::Value>(data) {
                    Ok(json) => {
                        if let Some(content) = json["content"].as_str() {
                            if !content.is_empty() {
                                full_text.push_str(content);
                                parser_buffer.push_str(content);
                                got_first_token = true;
                                emit_parsed_content(&tx, &mut parser_buffer, &mut in_think).await;
                            }
                        }

                        if json["stop"].as_bool().unwrap_or(false) {
                            if let Some(value) = json["tokens_predicted"].as_u64() {
                                tokens_predicted = value as u32;
                            }
                            if let Some(value) = json["tokens_evaluated"].as_u64() {
                                tokens_evaluated = value as u32;
                            }
                            if let Some(value) = json
                                .get("timings")
                                .and_then(|timings| timings["predicted_per_second"].as_f64())
                            {
                                decode_tokens_per_second = value;
                            }
                            if let Some(value) = json
                                .get("timings")
                                .and_then(|timings| timings["prompt_per_second"].as_f64())
                            {
                                prompt_tokens_per_second = Some(value);
                            }

                            if !parser_buffer.is_empty() {
                                if in_think {
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
        if in_think {
            let _ = tx
                .send(StreamEvent::ReasoningDelta(parser_buffer.clone()))
                .await;
        } else {
            let _ = tx.send(StreamEvent::Token(parser_buffer.clone())).await;
        }
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
            })
            .await;
    }

    Ok(full_text)
}

#[cfg(test)]
mod tests {
    use super::longest_partial_suffix;

    #[test]
    fn longest_partial_suffix_handles_multibyte_unicode() {
        assert_eq!(longest_partial_suffix("–", &["<think>"]), 0);
    }

    #[test]
    fn longest_partial_suffix_keeps_partial_tag_after_unicode_prefix() {
        assert_eq!(longest_partial_suffix("–<thi", &["<think>"]), 4);
    }
}
