//! Message compressor — summarizes older messages to free context space.
//! Uses the loaded model itself for summarization (self-summarize).

/// Compress a sequence of messages into a summary string.
/// For now, uses a simple heuristic (take first and last lines of each message).
/// Will be replaced with model-based summarization when the chat pipeline is ready.
pub fn compress_messages(messages: &[(String, String)]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mut summary = String::from("[Earlier conversation summary]\n");
    for (role, content) in messages {
        let trimmed = content.trim();
        let preview = if trimmed.len() > 200 {
            format!("{}...", &trimmed[..200])
        } else {
            trimmed.to_string()
        };
        summary.push_str(&format!("{}: {}\n", role, preview));
    }
    summary
}
