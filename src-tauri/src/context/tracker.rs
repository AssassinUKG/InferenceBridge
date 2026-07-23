use crate::engine::client::LlamaClient;
use crate::engine::process::ProcessState;
use std::sync::atomic::{AtomicBool, Ordering};

static SLOTS_WARNED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextStatus {
    pub total_tokens: u32,
    pub used_tokens: u32,
    pub fill_ratio: f32,
    pub pinned_tokens: u32,
    pub rolling_tokens: u32,
    pub compressed_tokens: u32,
    pub last_compaction_action: Option<String>,
}

impl ContextStatus {
    pub fn empty() -> Self {
        Self {
            total_tokens: 0,
            used_tokens: 0,
            fill_ratio: 0.0,
            pinned_tokens: 0,
            rolling_tokens: 0,
            compressed_tokens: 0,
            last_compaction_action: None,
        }
    }

    pub fn with_breakdown(
        mut self,
        pinned_tokens: u32,
        rolling_tokens: u32,
        compressed_tokens: u32,
        last_compaction_action: Option<String>,
    ) -> Self {
        self.pinned_tokens = pinned_tokens;
        self.rolling_tokens = rolling_tokens;
        self.compressed_tokens = compressed_tokens;
        self.last_compaction_action = last_compaction_action;
        self
    }
}

pub fn reset_slots_warning() {
    SLOTS_WARNED.store(false, Ordering::Relaxed);
}

pub fn can_poll_context(loaded: bool, process_state: ProcessState) -> bool {
    loaded && matches!(process_state, ProcessState::Running)
}

pub async fn poll_context_status(client: &LlamaClient) -> ContextStatus {
    // ProcessState::Running means the child exists, not that model loading has
    // completed. Avoid probing /slots (and its /props fallback) until the
    // server health endpoint confirms readiness; callers can retain their
    // previously stored status while this returns empty during startup.
    if !client.health().await.unwrap_or(false) {
        return ContextStatus::empty();
    }

    match client.get_slots().await {
        Ok(slots) if !slots.is_empty() => {
            let slot = &slots[0];
            SLOTS_WARNED.store(false, Ordering::Relaxed);
            let used = slot.used_tokens();
            let fill = if slot.n_ctx > 0 {
                used as f32 / slot.n_ctx as f32
            } else {
                0.0
            };

            ContextStatus {
                total_tokens: slot.n_ctx,
                used_tokens: used,
                fill_ratio: fill,
                pinned_tokens: 0,
                rolling_tokens: used,
                compressed_tokens: 0,
                last_compaction_action: None,
            }
        }
        Ok(_) => ContextStatus::empty(),
        Err(error) => {
            if !SLOTS_WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(error = %error, "/slots endpoint unavailable; falling back to /props for context window");
            }

            match client.get_props().await {
                Ok(props) => {
                    let total_tokens = props
                        .default_generation_settings
                        .and_then(|settings| settings.n_ctx)
                        .unwrap_or(0);
                    if total_tokens == 0 {
                        ContextStatus::empty()
                    } else {
                        ContextStatus {
                            total_tokens,
                            used_tokens: 0,
                            fill_ratio: 0.0,
                            pinned_tokens: 0,
                            rolling_tokens: 0,
                            compressed_tokens: 0,
                            last_compaction_action: None,
                        }
                    }
                }
                Err(_) => ContextStatus::empty(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::can_poll_context;
    use crate::engine::process::ProcessState;

    #[test]
    fn context_polling_requires_a_loaded_running_backend() {
        assert!(can_poll_context(true, ProcessState::Running));
        assert!(!can_poll_context(false, ProcessState::Running));
        assert!(!can_poll_context(true, ProcessState::Idle));
        assert!(!can_poll_context(true, ProcessState::Starting));
        assert!(!can_poll_context(true, ProcessState::Stopping));
        assert!(!can_poll_context(true, ProcessState::Error));
    }
}
