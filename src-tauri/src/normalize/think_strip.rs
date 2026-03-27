//! Strip `<think>...</think>` style reasoning blocks from model output.

use crate::models::profiles::ThinkTagStyle;

/// Strip think tags, preserving visible content after the closing tag.
/// If the entire output is wrapped in think tags, preserve the inner
/// content so short "reasoning-only" replies do not collapse to empty text.
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
            if let Some(inner) = salvaged_inner {
                return inner;
            }
            return String::new();
        }
    }

    text.to_string()
}

pub fn strip_think_tags_with_style(text: &str, style: ThinkTagStyle) -> String {
    match style {
        ThinkTagStyle::None => text.to_string(),
        ThinkTagStyle::Standard | ThinkTagStyle::Qwen => strip_think_tags(text),
    }
}

pub fn extract_reasoning_content(text: &str) -> String {
    extract_reasoning_content_with_style(text, ThinkTagStyle::Standard)
}

pub fn extract_reasoning_content_with_style(text: &str, style: ThinkTagStyle) -> String {
    match style {
        ThinkTagStyle::None => String::new(),
        ThinkTagStyle::Standard => extract_tagged_sections(text, &[("<think>", "</think>")]),
        ThinkTagStyle::Qwen => extract_tagged_sections(
            text,
            &[("<think>", "</think>"), ("<|think|>", "<|/think|>")],
        ),
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
    fn extracts_reasoning_blocks() {
        let raw = "<think>plan A</think>Answer<think>plan B</think>";
        assert_eq!(extract_reasoning_content(raw), "plan A\nplan B");
    }

    #[test]
    fn estimates_tokens_from_text_length() {
        assert_eq!(estimate_token_count(""), 0);
        assert_eq!(estimate_token_count("abcd"), 1);
        assert_eq!(estimate_token_count("abcdefgh"), 2);
    }
}
