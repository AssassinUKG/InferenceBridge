//! Strip `<think>…</think>` chain-of-thought blocks from model output.
//! Ported from HelixClaw's llm.rs strip_think_tags().

/// Strip think tags, preserving content after the closing tag.
/// If the entire output is wrapped in think tags, preserve the inner
/// content (the model's reasoning IS the response for short queries).
pub fn strip_think_tags(text: &str) -> String {
    let tag_pairs = [("<think>", "</think>"), ("<|think|>", "<|/think|>")];

    for (open, close) in tag_pairs {
        if text.contains(open) {
            let mut remaining = text.to_string();
            let mut salvaged_inner: Option<String> = None;

            while let Some(start) = remaining.find(open) {
                let after_open = start + open.len();
                if let Some(rel_end) = remaining[after_open..].find(close) {
                    let end = after_open + rel_end;
                    let inner = remaining[after_open..end].trim().to_string();
                    if !inner.is_empty() {
                        salvaged_inner = Some(inner);
                    }
                    let close_end = end + close.len();
                    remaining = format!("{}{}", &remaining[..start], &remaining[close_end..]);
                } else {
                    // Unclosed think tag — keep the inner text as the response
                    let before = remaining[..start].trim().to_string();
                    if !before.is_empty() {
                        return before;
                    }
                    let inner = remaining[after_open..].trim().to_string();
                    if !inner.is_empty() {
                        return inner;
                    }
                    return String::new();
                }
            }

            let stripped = remaining.trim().to_string();
            if !stripped.is_empty() {
                return stripped;
            }
            // Model only produced think content — use it as the response
            if let Some(inner) = salvaged_inner {
                return inner;
            }
            return String::new();
        }
    }

    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_think_preserves_answer() {
        let raw = "<think>reasoning</think>{\"summary\":\"done\"}";
        assert_eq!(strip_think_tags(raw), "{\"summary\":\"done\"}");
    }

    #[test]
    fn pure_think_preserves_inner() {
        let raw = "<think>just reasoning no answer</think>";
        assert_eq!(strip_think_tags(raw), "just reasoning no answer");
    }

    #[test]
    fn salvages_json_inside_think() {
        let raw = "<think>{\"name\":\"test\"}</think>";
        assert!(!strip_think_tags(raw).is_empty());
    }
}
