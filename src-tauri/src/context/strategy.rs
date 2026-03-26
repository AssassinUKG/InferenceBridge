//! Layered context strategy: pinned, rolling, compressed, rebuild.

/// Context layer types.
#[derive(Debug, Clone, serde::Serialize)]
pub enum ContextLayer {
    /// System prompt + tool definitions — never evicted.
    Pinned { token_count: u32 },
    /// Recent N messages — oldest evicted first.
    Rolling {
        message_ids: Vec<i64>,
        token_count: u32,
    },
    /// Summarized older messages.
    Compressed {
        summary: String,
        original_count: u32,
        token_count: u32,
    },
}

/// Decisions the context strategy can make.
#[derive(Debug)]
pub enum ContextAction {
    /// Everything fits, no action needed.
    NoAction,
    /// Compress oldest N rolling messages into a summary.
    Compress { message_count: u32 },
    /// Full context rebuild from session DB.
    Rebuild,
}

/// Given current usage and limits, decide what action to take.
pub fn decide_action(
    used_tokens: u32,
    total_tokens: u32,
    rolling_message_count: u32,
) -> ContextAction {
    if total_tokens == 0 {
        return ContextAction::NoAction;
    }
    let fill = used_tokens as f32 / total_tokens as f32;

    if fill >= 0.95 {
        // Critical: rebuild from scratch
        ContextAction::Rebuild
    } else if fill >= 0.80 && rolling_message_count > 4 {
        // High: compress oldest rolling messages
        let to_compress = (rolling_message_count / 3).max(2);
        ContextAction::Compress {
            message_count: to_compress,
        }
    } else {
        ContextAction::NoAction
    }
}
