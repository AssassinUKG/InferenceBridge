//! Context/KV cache tracker — polls llama-server /slots for usage info.

use crate::engine::client::LlamaClient;
use std::sync::atomic::{AtomicBool, Ordering};

/// Whether we've already logged that /slots is unavailable.
static SLOTS_WARNED: AtomicBool = AtomicBool::new(false);

/// Snapshot of current context usage.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextStatus {
    /// Total context window size (tokens).
    pub total_tokens: u32,
    /// Tokens currently used in KV cache.
    pub used_tokens: u32,
    /// Fill percentage (0.0 - 1.0).
    pub fill_ratio: f32,
}

impl ContextStatus {
    pub fn empty() -> Self {
        Self {
            total_tokens: 0,
            used_tokens: 0,
            fill_ratio: 0.0,
        }
    }
}

/// Reset the warned flag (call when loading a new model).
pub fn reset_slots_warning() {
    SLOTS_WARNED.store(false, Ordering::Relaxed);
}

/// Poll the llama-server /slots endpoint and return context usage.
/// Falls back to /props for total context window if /slots is unavailable.
pub async fn poll_context_status(client: &LlamaClient) -> ContextStatus {
    match client.get_slots().await {
        Ok(slots) if !slots.is_empty() => {
            let slot = &slots[0]; // Single-slot mode
                                  // Use n_past if available, otherwise fall back to next_token.n_decoded
            let used = if slot.n_past > 0 {
                slot.n_past
            } else {
                slot.next_token.as_ref().map_or(0, |nt| nt.n_decoded)
            };
            let fill = if slot.n_ctx > 0 {
                used as f32 / slot.n_ctx as f32
            } else {
                0.0
            };
            tracing::debug!(
                n_ctx = slot.n_ctx,
                used = used,
                fill_pct = format_args!("{:.1}%", fill * 100.0),
                "KV cache status"
            );
            ContextStatus {
                total_tokens: slot.n_ctx,
                used_tokens: used,
                fill_ratio: fill,
            }
        }
        Ok(_) => {
            tracing::debug!("KV poll: /slots returned empty array");
            ContextStatus::empty()
        }
        Err(e) => {
            if !SLOTS_WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(error = %e, "/slots endpoint unavailable — falling back to /props for context window");
            }
            // Fallback: use /props to at least get total context size
            match client.get_props().await {
                Ok(props) => {
                    let n_ctx = props
                        .default_generation_settings
                        .and_then(|s| s.n_ctx)
                        .unwrap_or(0);
                    if n_ctx > 0 {
                        ContextStatus {
                            total_tokens: n_ctx,
                            used_tokens: 0,
                            fill_ratio: 0.0,
                        }
                    } else {
                        ContextStatus::empty()
                    }
                }
                Err(_) => ContextStatus::empty(),
            }
        }
    }
}
