//! SSE stream consumer for llama-server's streaming completion endpoint.

use futures_util::StreamExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Events emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A new token chunk.
    Token(String),
    /// Generation is complete.
    Done {
        full_text: String,
        tokens_predicted: u32,
        tokens_evaluated: u32,
        tokens_per_second: f64,
    },
    /// An error occurred during streaming.
    Error(String),
}

/// Consume an SSE stream from llama-server and emit events.
/// Includes a per-chunk timeout to detect server hangs.
pub async fn consume_sse_stream(
    response: reqwest::Response,
    tx: mpsc::Sender<StreamEvent>,
    stop: Arc<AtomicBool>,
) -> anyhow::Result<String> {
    let mut full_text = String::new();
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut tokens_predicted: u32 = 0;
    let mut tokens_evaluated: u32 = 0;
    let mut tokens_per_second: f64 = 0.0;
    let mut got_first_token = false;

    // First token can take a while (prompt evaluation), subsequent tokens should be faster
    let first_token_timeout = std::time::Duration::from_secs(300); // 5 min for first token
    let inter_token_timeout = std::time::Duration::from_secs(120); // 2 min between tokens

    loop {
        // Check for cancellation before waiting on the next chunk
        if stop.load(Ordering::Relaxed) {
            tracing::info!("Stream cancelled by stop signal");
            let _ = tx
                .send(StreamEvent::Done {
                    full_text: full_text.clone(),
                    tokens_predicted,
                    tokens_evaluated,
                    tokens_per_second,
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
            Ok(Some(Ok(c))) => c,
            Ok(Some(Err(e))) => {
                let _ = tx.send(StreamEvent::Error(e.to_string())).await;
                return Err(e.into());
            }
            Ok(None) => break, // Stream ended
            Err(_) => {
                // Timeout waiting for next chunk
                let msg = if got_first_token {
                    "llama-server stopped responding (inter-token timeout)".to_string()
                } else {
                    "llama-server took too long to start generating (prompt evaluation timeout)"
                        .to_string()
                };
                tracing::warn!("{msg}");
                if full_text.is_empty() {
                    let _ = tx.send(StreamEvent::Error(msg)).await;
                } else {
                    let _ = tx
                        .send(StreamEvent::Done {
                            full_text: full_text.clone(),
                            tokens_predicted,
                            tokens_evaluated,
                            tokens_per_second,
                        })
                        .await;
                }
                return Ok(full_text);
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE lines
        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer = buffer[line_end + 1..].to_string();

            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    let _ = tx
                        .send(StreamEvent::Done {
                            full_text: full_text.clone(),
                            tokens_predicted,
                            tokens_evaluated,
                            tokens_per_second,
                        })
                        .await;
                    return Ok(full_text);
                }

                match serde_json::from_str::<serde_json::Value>(data) {
                    Ok(json) => {
                        if let Some(content) = json["content"].as_str() {
                            if !content.is_empty() {
                                full_text.push_str(content);
                                got_first_token = true;
                                let _ = tx.send(StreamEvent::Token(content.to_string())).await;
                            }
                        }
                        if let Some(stop) = json["stop"].as_bool() {
                            if stop {
                                if let Some(tp) = json["tokens_predicted"].as_u64() {
                                    tokens_predicted = tp as u32;
                                }
                                if let Some(te) = json["tokens_evaluated"].as_u64() {
                                    tokens_evaluated = te as u32;
                                }
                                if let Some(timings) = json.get("timings") {
                                    if let Some(tps) = timings["predicted_per_second"].as_f64() {
                                        tokens_per_second = tps;
                                    }
                                }
                                let _ = tx
                                    .send(StreamEvent::Done {
                                        full_text: full_text.clone(),
                                        tokens_predicted,
                                        tokens_evaluated,
                                        tokens_per_second,
                                    })
                                    .await;
                                return Ok(full_text);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(data, error = %e, "Failed to parse SSE data");
                    }
                }
            }
        }
    }

    // Stream ended without [DONE] — always send a terminal event
    // so the receiver doesn't hang forever waiting on rx.recv()
    if full_text.is_empty() {
        tracing::warn!("SSE stream ended with no tokens generated");
        let _ = tx
            .send(StreamEvent::Error(
                "llama-server returned empty response".to_string(),
            ))
            .await;
    } else {
        tracing::info!(
            len = full_text.len(),
            "SSE stream ended without [DONE] marker"
        );
        let _ = tx
            .send(StreamEvent::Done {
                full_text: full_text.clone(),
                tokens_predicted,
                tokens_evaluated,
                tokens_per_second,
            })
            .await;
    }
    Ok(full_text)
}
