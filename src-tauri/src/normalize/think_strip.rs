//! Strip `<think>...</think>` style reasoning blocks from model output.

use crate::models::profiles::ThinkTagStyle;

/// Strip think tags, preserving visible content after the closing tag.
/// If the entire output is wrapped in think tags, preserve the inner
/// content so short "reasoning-only" replies do not collapse to empty text.
pub fn strip_think_tags(text: &str) -> String {
    let text = strip_channel_markers(text);
    let tag_pairs = [("<think>", "</think>"), ("<|think|>", "<|/think|>")];

    for (open, close) in tag_pairs {
        if text.contains(open) {
            let mut remaining = text.clone();
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
                    let before = remaining[..start].trim().to_string();
                    let inner = remaining[after_open..].trim().to_string();
                    if !before.is_empty() && !inner.is_empty() {
                        return format!("{before} {inner}");
                    }
                    if !before.is_empty() {
                        return before;
                    }
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
            if let Some(inner) = salvaged_inner {
                return inner;
            }
            return String::new();
        }

        if let Some(end) = text.find(close) {
            let after_close = &text[end + close.len()..];
            return after_close.trim().to_string();
        }
    }

    text
}

pub fn strip_think_tags_with_style(text: &str, style: ThinkTagStyle) -> String {
    match style {
        ThinkTagStyle::None => strip_channel_markers(text),
        ThinkTagStyle::Standard | ThinkTagStyle::Qwen => strip_think_tags(text),
    }
}

/// Remove provider/model control-channel markers that should never be shown as
/// ordinary chat text, while preserving regular `<think>` blocks for the UI's
/// expandable thinking renderer.
pub fn strip_control_channel_markers(text: &str) -> String {
    strip_channel_markers(text)
}

pub fn extract_reasoning_content(text: &str) -> String {
    extract_reasoning_content_with_style(text, ThinkTagStyle::Standard)
}

pub fn extract_reasoning_content_with_style(text: &str, style: ThinkTagStyle) -> String {
    match style {
        ThinkTagStyle::None => extract_channel_reasoning(text),
        ThinkTagStyle::Standard => extract_tagged_sections(text, &[("<think>", "</think>")]),
        ThinkTagStyle::Qwen => {
            let tagged = extract_tagged_sections(
                text,
                &[("<think>", "</think>"), ("<|think|>", "<|/think|>")],
            );
            let channel = extract_channel_reasoning(text);
            match (tagged.is_empty(), channel.is_empty()) {
                (true, true) => String::new(),
                (false, true) => tagged,
                (true, false) => channel,
                (false, false) => format!("{tagged}\n{channel}"),
            }
        }
    }
}

pub fn estimate_token_count(text: &str) -> u32 {
    let chars = text.trim().chars().count();
    if chars == 0 {
        0
    } else {
        ((chars as f32) / 4.0).ceil() as u32
    }
}

fn extract_tagged_sections(text: &str, tag_pairs: &[(&str, &str)]) -> String {
    let mut extracted = Vec::new();

    for (open, close) in tag_pairs {
        let mut remaining = text;
        while let Some(start) = remaining.find(open) {
            let after_open = start + open.len();
            let rest = &remaining[after_open..];
            if let Some(rel_end) = rest.find(close) {
                let inner = rest[..rel_end].trim();
                if !inner.is_empty() {
                    extracted.push(inner.to_string());
                }
                remaining = &rest[rel_end + close.len()..];
            } else {
                let inner = rest.trim();
                if !inner.is_empty() {
                    extracted.push(inner.to_string());
                }
                break;
            }
        }

        if !text.contains(open) {
            if let Some(end) = text.find(close) {
                let inner = text[..end].trim();
                if !inner.is_empty() {
                    extracted.push(inner.to_string());
                }
            }
        }
    }

    extracted.join("\n")
}

fn strip_channel_markers(text: &str) -> String {
    let text = truncate_at_generation_boundary(text);
    const CLOSE: &str = "<channel|>";
    let mut remaining = text.to_string();
    for open in ["<|channel>thought", "<|channel|>thought"] {
        while let Some(start) = remaining.find(open) {
            let after_open = start + open.len();
            if let Some(rel_end) = remaining[after_open..].find(CLOSE) {
                let close_end = after_open + rel_end + CLOSE.len();
                remaining = format!("{}{}", &remaining[..start], &remaining[close_end..]);
            } else {
                remaining.truncate(start);
                break;
            }
        }
    }
    remaining.replace(CLOSE, "").trim().to_string()
}

/// Remove model/template turn markers that indicate the model has started
/// continuing the chat template instead of answering.  Truncate rather than
/// replace, otherwise leaked text like `<turn|><|turn>user ...` becomes visible
/// as if the assistant said it.
pub fn truncate_at_generation_boundary(text: &str) -> &str {
    const BOUNDARIES: &[&str] = &[
        "<turn|>",
        "<|turn>",
        "<end_of_turn>",
        "<start_of_turn>",
        "<|im_end|>",
        "<|im_start|>",
        "<|eot_id|>",
        "<|start_header_id|>",
    ];
    let first = BOUNDARIES
        .iter()
        .filter_map(|marker| text.find(marker))
        .min();
    match first {
        Some(idx) => &text[..idx],
        None => text,
    }
}

fn extract_channel_reasoning(text: &str) -> String {
    const CLOSE: &str = "<channel|>";
    let mut extracted = Vec::new();
    for open in ["<|channel>thought", "<|channel|>thought"] {
        let mut remaining = text;
        while let Some(start) = remaining.find(open) {
            let after_open = start + open.len();
            let rest = &remaining[after_open..];
            if let Some(rel_end) = rest.find(CLOSE) {
                let inner = rest[..rel_end].trim();
                if !inner.is_empty() {
                    extracted.push(inner.to_string());
                }
                remaining = &rest[rel_end + CLOSE.len()..];
            } else {
                let inner = rest.trim();
                if !inner.is_empty() {
                    extracted.push(inner.to_string());
                }
                break;
            }
        }
    }
    extracted.join("\n")
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

    #[test]
    fn unmatched_opening_tag_preserves_visible_prefix_and_suffix() {
        let raw = "Answer:<think>still useful trailing content";
        assert_eq!(
            strip_think_tags(raw),
            "Answer: still useful trailing content"
        );
    }

    #[test]
    fn orphan_closing_tag_strips_reasoning_prefix() {
        let raw = "private reasoning</think>Final answer";
        assert_eq!(strip_think_tags(raw), "Final answer");
    }

    #[test]
    fn orphan_closing_tag_extracts_reasoning_prefix() {
        let raw = "private reasoning</think>Final answer";
        assert_eq!(extract_reasoning_content(raw), "private reasoning");
    }

    #[test]
    fn extracts_reasoning_blocks() {
        let raw = "<think>plan A</think>Answer<think>plan B</think>";
        assert_eq!(extract_reasoning_content(raw), "plan A\nplan B");
    }

    #[test]
    fn strips_gemma_channel_thought_markers_from_visible_text() {
        let raw = "<|channel>thought\nscratch\n<channel|>{\"language\":\"typescript\"}";
        assert_eq!(
            strip_think_tags_with_style(raw, ThinkTagStyle::None),
            "{\"language\":\"typescript\"}"
        );
    }

    #[test]
    fn strips_control_channel_markers_even_when_thinking_is_visible() {
        let raw = "<|channel>thought <channel|>I'm doing great.";
        assert_eq!(strip_control_channel_markers(raw), "I'm doing great.");
    }

    #[test]
    fn strips_pipe_channel_variant_when_thinking_is_visible() {
        let raw = "<|channel|>thought scratch <channel|>I'm doing great.";
        assert_eq!(strip_control_channel_markers(raw), "I'm doing great.");
    }

    #[test]
    fn truncates_gemma_turn_continuation_from_visible_text() {
        let raw = "Final answer.<turn|><|turn>user\nignore this";
        assert_eq!(
            strip_think_tags_with_style(raw, ThinkTagStyle::None),
            "Final answer."
        );
    }

    #[test]
    fn extracts_gemma_channel_thought_as_reasoning() {
        let raw = "<|channel>thought\nscratch\n<channel|>Final";
        assert_eq!(
            extract_reasoning_content_with_style(raw, ThinkTagStyle::None),
            "scratch"
        );
    }

    #[test]
    fn estimates_tokens_from_text_length() {
        assert_eq!(estimate_token_count(""), 0);
        assert_eq!(estimate_token_count("abcd"), 1);
        assert_eq!(estimate_token_count("abcdefgh"), 2);
    }
}
