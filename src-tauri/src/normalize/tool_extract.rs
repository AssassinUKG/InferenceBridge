//! Extract structured tool calls from model output text.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::models::profiles::{ModelProfile, ParserType};

static QWEN_FUNCTION_RE: OnceLock<regex::Regex> = OnceLock::new();
static QWEN_PARAM_RE: OnceLock<regex::Regex> = OnceLock::new();
static TOOL_CODE_DIV_RE: OnceLock<regex::Regex> = OnceLock::new();
static TOOL_CALL_EXPR_RE: OnceLock<regex::Regex> = OnceLock::new();
static SQUARE_BRACKET_RE: OnceLock<regex::Regex> = OnceLock::new();
static PARENTHESIS_RE: OnceLock<regex::Regex> = OnceLock::new();

fn qwen_function_re() -> &'static regex::Regex {
    QWEN_FUNCTION_RE.get_or_init(|| {
        regex::Regex::new(r"(?s)<function=([^>]+)>(.*?)(?:</function>|$)")
            .expect("valid Qwen function regex")
    })
}

fn qwen_param_re() -> &'static regex::Regex {
    QWEN_PARAM_RE.get_or_init(|| {
        regex::Regex::new(r"(?s)<parameter=([^>]+)>(.*?)(?:</parameter>|$)")
            .expect("valid Qwen parameter regex")
    })
}

fn tool_code_div_re() -> &'static regex::Regex {
    TOOL_CODE_DIV_RE.get_or_init(|| {
        regex::Regex::new(r#"(?s)<div\s+class=["']tool_code["']\s*>(.*?)</div>"#)
            .expect("valid tool_code div regex")
    })
}

fn tool_call_expr_re() -> &'static regex::Regex {
    TOOL_CALL_EXPR_RE.get_or_init(|| {
        regex::Regex::new(
            r#"(?s)<tool_call>\s*(?:_\.)?([a-zA-Z][a-zA-Z0-9_\-]*)\s*\((.*?)\)\s*(?:</tool_call>)?"#,
        )
        .expect("valid plain tool_call expression regex")
    })
}

fn square_bracket_re() -> &'static regex::Regex {
    SQUARE_BRACKET_RE.get_or_init(|| {
        regex::Regex::new(r"\[([a-zA-Z][a-zA-Z0-9_\-]*)\]\s*\{")
            .expect("valid square bracket tool regex")
    })
}

fn parenthesis_re() -> &'static regex::Regex {
    PARENTHESIS_RE.get_or_init(|| {
        regex::Regex::new(r"\[tool_call\]([a-zA-Z][a-zA-Z0-9_\-]*)\s*\(\s*(\{)")
            .expect("valid parenthesis tool regex")
    })
}

/// A parsed tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    /// Raw text that generated this tool call (for debugging).
    pub raw_text: Option<String>,
}

/// Extract tool calls from text using various formats.
pub fn extract_tool_calls(text: &str) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut remaining = crate::normalize::think_strip::strip_think_tags(text);

    // Try Hermes-style <tool_call>{json}</tool_call>
    extract_hermes_calls(&mut remaining, &mut calls);

    // Try Qwen-style <function=name><parameter=key>value</parameter></function>
    extract_qwen_function_calls(&mut remaining, &mut calls);

    // Try HTML/Python-ish fallbacks emitted by some agent prompts.
    extract_html_tool_code_calls(&mut remaining, &mut calls);
    extract_plain_tool_call_expressions(&mut remaining, &mut calls);

    if !calls.is_empty() {
        remove_empty_tool_wrappers(&mut remaining);
    }

    if !calls.is_empty() {
        remove_empty_tool_wrappers(&mut remaining);
    }

    (calls, remaining.trim().to_string())
}

fn remove_empty_tool_wrappers(text: &mut String) {
    *text = text
        .replace("<tool_call>", "")
        .replace("</tool_call>", "")
        .replace("<|channel>thought", "")
        .replace("<channel|>", "")
        .trim()
        .to_string();
}

pub fn extract_tool_calls_for_profile(
    text: &str,
    profile: &ModelProfile,
) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut remaining =
        crate::normalize::think_strip::strip_think_tags_with_style(text, profile.think_tag_style);

    match profile.parser_type {
        ParserType::QwenStateMachine => {
            extract_qwen_function_calls(&mut remaining, &mut calls);
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_hermes_calls(&mut remaining, &mut calls);
            }
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_html_tool_code_calls(&mut remaining, &mut calls);
                extract_plain_tool_call_expressions(&mut remaining, &mut calls);
            }
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_json_tool_objects(&mut remaining, &mut calls);
            }
            // Safety net: if llama-server didn't inject the Qwen chat template, the model may
            // fall back to text-format tool calls instead of <function=...> XML.
            // Try both known fallback formats in order.
            if calls.is_empty() && profile.allow_fallback_extraction {
                if extract_square_bracket_tool_calls(&mut remaining, &mut calls) {
                    tracing::info!(
                        "qwen safety-net: [tool_name]{{...}} format detected - \
                        check chat template config on llama-server"
                    );
                } else if extract_parenthesis_tool_calls(&mut remaining, &mut calls) {
                    tracing::info!(
                        "qwen safety-net: [tool_call]name({{...}}) format detected - \
                        check chat template config on llama-server"
                    );
                } else if extract_bare_tool_call_json(&mut remaining, &mut calls) {
                    tracing::info!(
                        "qwen safety-net: bare <tool_call>{{...}} format detected - \
                        check chat template config on llama-server"
                    );
                }
            }
        }
        ParserType::Gemma4StateMachine => {
            extract_gemma4_native_calls(&mut remaining, &mut calls);
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_json_tool_objects(&mut remaining, &mut calls);
            }
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_hermes_calls(&mut remaining, &mut calls);
                extract_plain_tool_call_expressions(&mut remaining, &mut calls);
            }
        }
        ParserType::HermesFallback => {
            extract_hermes_calls(&mut remaining, &mut calls);
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_qwen_function_calls(&mut remaining, &mut calls);
            }
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_html_tool_code_calls(&mut remaining, &mut calls);
                extract_plain_tool_call_expressions(&mut remaining, &mut calls);
            }
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_json_tool_objects(&mut remaining, &mut calls);
            }
        }
        ParserType::NativeApi => {
            if profile.allow_fallback_extraction {
                extract_hermes_calls(&mut remaining, &mut calls);
                if calls.is_empty() {
                    extract_qwen_function_calls(&mut remaining, &mut calls);
                }
                if calls.is_empty() {
                    extract_html_tool_code_calls(&mut remaining, &mut calls);
                    extract_plain_tool_call_expressions(&mut remaining, &mut calls);
                }
                if calls.is_empty() {
                    extract_json_tool_objects(&mut remaining, &mut calls);
                }
            } else if extract_json_tool_objects(&mut remaining, &mut calls) {
                tracing::info!(
                    "native-api safety-net: content-only JSON tool call detected - \
                    check chat_format/template support on llama-server"
                );
            }
        }
    }

    if !calls.is_empty() {
        remove_empty_tool_wrappers(&mut remaining);
    }

    (calls, remaining.trim().to_string())
}

fn extract_hermes_calls(text: &mut String, calls: &mut Vec<ToolCall>) {
    while let Some(start) = text.find("<tool_call>") {
        let after = start + "<tool_call>".len();
        if let Some(end) = text[after..].find("</tool_call>") {
            let json_str = text[after..after + end].trim();
            if let Some(value) = super::json_repair::repair_json(json_str) {
                if let (Some(name), Some(args)) = (
                    value.get("name").and_then(|v| v.as_str()),
                    value.get("arguments"),
                ) {
                    calls.push(ToolCall {
                        id: uuid::Uuid::new_v4().to_string(),
                        name: name.to_string(),
                        arguments: args.clone(),
                        raw_text: Some(text[start..after + end + "</tool_call>".len()].to_string()),
                    });
                }
            }
            let remove_end = after + end + "</tool_call>".len();
            *text = format!("{}{}", &text[..start], &text[remove_end..]);
        } else {
            break;
        }
    }
}

fn extract_html_tool_code_calls(text: &mut String, calls: &mut Vec<ToolCall>) -> bool {
    let mut removals = Vec::new();

    for cap in tool_code_div_re().captures_iter(text) {
        let full_match = cap.get(0).unwrap();
        let body = cap.get(1).map(|m| m.as_str()).unwrap_or("").trim();
        let Some(value) = super::json_repair::repair_json(body) else {
            continue;
        };

        let Some(name) = value
            .get("tool_name")
            .or_else(|| value.get("name"))
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };

        let arguments = value
            .get("tool_argument")
            .or_else(|| value.get("tool_arguments"))
            .or_else(|| value.get("arguments"))
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

        calls.push(ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            arguments,
            raw_text: Some(full_match.as_str().to_string()),
        });
        removals.push((full_match.start(), full_match.end()));
    }

    if removals.is_empty() {
        return false;
    }

    let mut new_text = text.clone();
    for (start, end) in removals.into_iter().rev() {
        new_text.replace_range(start..end, "");
    }
    *text = new_text;
    true
}

fn extract_plain_tool_call_expressions(text: &mut String, calls: &mut Vec<ToolCall>) -> bool {
    let mut removals = Vec::new();

    for cap in tool_call_expr_re().captures_iter(text) {
        let full_match = cap.get(0).unwrap();
        let Some(name_match) = cap.get(1) else {
            continue;
        };
        let args_text = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        let Some(arguments) = parse_call_expression_arguments(args_text) else {
            continue;
        };

        calls.push(ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: name_match.as_str().to_string(),
            arguments,
            raw_text: Some(full_match.as_str().to_string()),
        });
        removals.push((full_match.start(), full_match.end()));
    }

    if removals.is_empty() {
        return false;
    }

    let mut new_text = text.clone();
    for (start, end) in removals.into_iter().rev() {
        new_text.replace_range(start..end, "");
    }
    *text = new_text;
    true
}

fn extract_json_tool_objects(text: &mut String, calls: &mut Vec<ToolCall>) -> bool {
    if extract_json_tool_fences(text, calls) {
        return true;
    }
    extract_standalone_json_tool_object(text, calls)
}

fn extract_json_tool_fences(text: &mut String, calls: &mut Vec<ToolCall>) -> bool {
    let mut found_any = false;
    loop {
        let Some(fence_start) = text.find("```") else {
            break;
        };
        let body_start = fence_start + 3;
        let Some(rel_fence_end) = text[body_start..].find("```") else {
            break;
        };
        let fence_end = body_start + rel_fence_end + 3;
        let body = &text[body_start..body_start + rel_fence_end];
        let body = body
            .trim_start()
            .strip_prefix("json")
            .unwrap_or(body.trim_start())
            .trim_start();
        let Some(rel_brace) = body.find('{').or_else(|| body.find('[')) else {
            break;
        };
        let Some(json_len) = balanced_json_value_len(&body[rel_brace..]) else {
            break;
        };
        let json_str = &body[rel_brace..rel_brace + json_len];
        let Some(value) = super::json_repair::repair_json(json_str) else {
            break;
        };
        let inferred = infer_bare_tool_calls(value);
        if inferred.is_empty() {
            break;
        }
        for (name, arguments) in inferred {
            calls.push(ToolCall {
                id: uuid::Uuid::new_v4().to_string(),
                name,
                arguments,
                raw_text: Some(text[fence_start..fence_end].to_string()),
            });
        }
        *text = format!("{}{}", &text[..fence_start], &text[fence_end..]);
        found_any = true;
    }
    found_any
}

fn extract_standalone_json_tool_object(text: &mut String, calls: &mut Vec<ToolCall>) -> bool {
    let trimmed_start = text.trim_start();
    let leading = text.len() - trimmed_start.len();
    let Some(json_len) = balanced_json_value_len(trimmed_start) else {
        return false;
    };
    let json_str = &trimmed_start[..json_len];
    let Some(value) = super::json_repair::repair_json(json_str) else {
        return false;
    };
    let inferred = infer_bare_tool_calls(value);
    if inferred.is_empty() {
        return false;
    }
    let remove_end = leading + json_len;
    for (name, arguments) in inferred {
        calls.push(ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            arguments,
            raw_text: Some(text[leading..remove_end].to_string()),
        });
    }
    *text = format!("{}{}", &text[..leading], &text[remove_end..]);
    true
}

fn parse_call_expression_arguments(args_text: &str) -> Option<serde_json::Value> {
    let trimmed = args_text.trim();
    if trimmed.is_empty() {
        return Some(serde_json::Value::Object(serde_json::Map::new()));
    }
    if let Some(value) = super::json_repair::repair_json(trimmed) {
        return Some(value);
    }

    let mut args = serde_json::Map::new();
    for part in split_top_level_commas(trimmed) {
        let (key, value) = part.split_once('=')?;
        let key = key.trim().trim_matches(|ch| ch == '"' || ch == '\'');
        if key.is_empty() {
            return None;
        }
        args.insert(key.to_string(), parse_call_expression_value(value.trim()));
    }
    Some(serde_json::Value::Object(args))
}

fn split_top_level_commas(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote.is_some() && ch == '\\' {
            escaped = true;
            continue;
        }
        if let Some(current_quote) = quote {
            if ch == current_quote {
                quote = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => quote = Some(ch),
            '{' | '[' | '(' => depth += 1,
            '}' | ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(text[start..idx].trim());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }

    if start <= text.len() {
        let tail = text[start..].trim();
        if !tail.is_empty() {
            parts.push(tail);
        }
    }
    parts
}

fn parse_call_expression_value(text: &str) -> serde_json::Value {
    let trimmed = text.trim();
    if let Some(value) = super::json_repair::repair_json(trimmed) {
        return value;
    }
    if let Some(value) = trimmed
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
    {
        return serde_json::Value::String(value.to_string());
    }
    match trimmed {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        "null" | "None" | "none" => serde_json::Value::Null,
        _ => trimmed
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(serde_json::Value::Number)
            .unwrap_or_else(|| serde_json::Value::String(trimmed.to_string())),
    }
}

fn extract_qwen_function_calls(text: &mut String, calls: &mut Vec<ToolCall>) {
    let mut removals = Vec::new();

    for cap in qwen_function_re().captures_iter(text) {
        let full_match = cap.get(0).unwrap();
        let outer_name = cap.get(1).unwrap().as_str().trim();
        let body = cap.get(2).unwrap().as_str();

        // Parse parameters from <parameter=key>value</parameter> pairs
        let mut args = serde_json::Map::new();
        for param_cap in qwen_param_re().captures_iter(body) {
            let key = param_cap.get(1).unwrap().as_str().trim();
            let raw_value = param_cap.get(2).unwrap().as_str();
            let normalized = trim_xml_line_padding(raw_value);
            let trimmed = normalized.trim();
            let parse_as_json = (trimmed.starts_with('{') && trimmed.ends_with('}'))
                || (trimmed.starts_with('[') && trimmed.ends_with(']'))
                || (trimmed.starts_with('"') && trimmed.ends_with('"'));
            let json_val = if parse_as_json {
                serde_json::from_str(trimmed)
                    .unwrap_or_else(|_| serde_json::Value::String(normalized.to_string()))
            } else {
                serde_json::Value::String(normalized.to_string())
            };
            args.insert(key.to_string(), json_val);
        }

        let (name, arguments) = unwrap_meta_tool_call(outer_name, args);
        calls.push(ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            arguments,
            raw_text: Some(full_match.as_str().to_string()),
        });
        removals.push((full_match.start(), full_match.end()));
    }

    if removals.is_empty() {
        return;
    }

    let mut new_text = text.clone();
    for (start, end) in removals.into_iter().rev() {
        new_text.replace_range(start..end, "");
    }

    *text = new_text;
}

fn extract_gemma4_native_calls(text: &mut String, calls: &mut Vec<ToolCall>) -> bool {
    let mut found_any = false;
    loop {
        let Some(start) = text
            .find("<|tool_call>")
            .or_else(|| text.find("<tool_call>"))
        else {
            break;
        };
        let open_len = if text[start..].starts_with("<|tool_call>") {
            "<|tool_call>".len()
        } else {
            "<tool_call>".len()
        };
        let body_start = start + open_len;
        let close_rel = text[body_start..]
            .find("<tool_call|>")
            .or_else(|| text[body_start..].find("</tool_call>"));
        let body_end = close_rel.map(|rel| body_start + rel).unwrap_or(text.len());
        let close_end = close_rel
            .map(|rel| {
                let close_start = body_start + rel;
                if text[close_start..].starts_with("<tool_call|>") {
                    close_start + "<tool_call|>".len()
                } else {
                    close_start + "</tool_call>".len()
                }
            })
            .unwrap_or(body_end);
        let body = text[body_start..body_end].trim();
        let Some((name, arguments)) = parse_gemma4_native_body(body) else {
            *text = format!("{}{}", &text[..start], &text[close_end..]);
            found_any = true;
            continue;
        };
        calls.push(ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            arguments,
            raw_text: Some(text[start..close_end].to_string()),
        });
        *text = format!("{}{}", &text[..start], &text[close_end..]);
        found_any = true;
    }
    found_any
}

fn parse_gemma4_native_body(body: &str) -> Option<(String, serde_json::Value)> {
    let rest = body.trim().strip_prefix("call:")?;
    let brace_start = rest.find('{')?;
    let name = rest[..brace_start].trim();
    if name.is_empty() {
        return None;
    }
    let args_text = &rest[brace_start..];
    let args_len = balanced_gemma4_brace_len(args_text)?;
    let args = parse_gemma4_argument_object(&args_text[..args_len]);
    Some((name.to_string(), serde_json::Value::Object(args)))
}

fn balanced_gemma4_brace_len(text: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_quote = false;
    let mut escaped = false;
    for (idx, ch) in text.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if in_quote && ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_quote = !in_quote;
            continue;
        }
        if in_quote {
            continue;
        }
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(idx + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_gemma4_argument_object(text: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut args = serde_json::Map::new();
    let inner = text.trim().trim_start_matches('{').trim_end_matches('}');
    for part in split_top_level_commas(inner) {
        let Some((key, value)) = part.split_once(':') else {
            continue;
        };
        let key = key.trim().trim_matches(|ch| ch == '"' || ch == '\'');
        if key.is_empty() {
            continue;
        }
        args.insert(key.to_string(), parse_gemma4_argument_value(value));
    }
    args
}

fn parse_gemma4_argument_value(text: &str) -> serde_json::Value {
    let trimmed = text.trim().trim_end_matches(',');
    if let Some(value) = trimmed
        .strip_prefix("<|\"|>")
        .and_then(|value| value.strip_suffix("<|\"|>"))
    {
        return serde_json::Value::String(value.to_string());
    }
    parse_call_expression_value(trimmed)
}

fn trim_xml_line_padding(value: &str) -> &str {
    value.trim_matches(|ch| ch == '\r' || ch == '\n')
}

/// Finds the byte length of a balanced `{...}` JSON object starting at the beginning of `text`.
/// Returns `None` if the text doesn't start with `{` or braces never balance.
fn balanced_json_object_len(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if b == b'\\' && in_string {
            escaped = true;
            continue;
        }
        match b {
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn balanced_json_value_len(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let first = *bytes.first()?;
    let closing = match first {
        b'{' => b'}',
        b'[' => b']',
        _ => return None,
    };
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if b == b'\\' && in_string {
            escaped = true;
            continue;
        }
        match b {
            b'"' => in_string = !in_string,
            b'{' | b'[' if !in_string => depth += 1,
            b'}' | b']' if !in_string => {
                depth -= 1;
                if b == closing && depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parses `[tool_name]{...}` text-format tool calls.
/// Used as a safety net when llama-server doesn't inject the Qwen chat template and the model
/// falls back to square-bracket notation instead of `<function=...>` XML.
/// Returns `true` if at least one call was extracted.
fn extract_square_bracket_tool_calls(text: &mut String, calls: &mut Vec<ToolCall>) -> bool {
    let mut found_any = false;
    loop {
        let (mat_start, name, brace_start) = {
            let Some(cap) = square_bracket_re().captures(text) else {
                break;
            };
            let mat = cap.get(0).unwrap();
            let name = cap.get(1).unwrap().as_str().to_string();
            let mat_start = mat.start();
            // mat ends just after the opening `{`, so brace_start is mat.end() - 1
            let brace_start = mat.end() - 1;
            (mat_start, name, brace_start)
        };
        let json_slice = &text[brace_start..];
        let Some(json_len) = balanced_json_object_len(json_slice) else {
            break;
        };
        let body_end = brace_start + json_len;
        let json_str = &text[brace_start..body_end];
        if let Some(args) = super::json_repair::repair_json(json_str) {
            let raw_text = text[mat_start..body_end].to_string();
            calls.push(ToolCall {
                id: uuid::Uuid::new_v4().to_string(),
                name,
                arguments: args,
                raw_text: Some(raw_text),
            });
            *text = format!("{}{}", &text[..mat_start], &text[body_end..]);
            found_any = true;
        } else {
            break;
        }
    }
    found_any
}

/// Parses `[tool_call]name({...})` text-format tool calls.
/// Qwen models emit this parenthesis-style format when the chat template isn't injected and
/// the model falls back to a Python/JS function-call convention.
/// Example: `[tool_call]submit_helix_graph({"nodes": [...]})`
/// Returns `true` if at least one call was extracted.
fn extract_parenthesis_tool_calls(text: &mut String, calls: &mut Vec<ToolCall>) -> bool {
    // Matches: [tool_call]name( where name is an identifier and ( opens the arg list
    let mut found_any = false;
    loop {
        let (mat_start, name, brace_start) = {
            let Some(cap) = parenthesis_re().captures(text) else {
                break;
            };
            let mat = cap.get(0).unwrap();
            let name = cap.get(1).unwrap().as_str().to_string();
            let mat_start = mat.start();
            // cap group 2 is the `{` - brace_start is its byte offset
            let brace_start = cap.get(2).unwrap().start();
            (mat_start, name, brace_start)
        };
        let json_slice = &text[brace_start..];
        let Some(json_len) = balanced_json_object_len(json_slice) else {
            break;
        };
        let body_end = brace_start + json_len;
        // Consume the closing `)` that wraps the JSON if present
        let call_end = {
            let after = text[body_end..].trim_start();
            if after.starts_with(')') {
                body_end + (text[body_end..].len() - after.len()) + 1
            } else {
                body_end
            }
        };
        let json_str = &text[brace_start..body_end];
        if let Some(args) = super::json_repair::repair_json(json_str) {
            let raw_text = text[mat_start..call_end].to_string();
            calls.push(ToolCall {
                id: uuid::Uuid::new_v4().to_string(),
                name,
                arguments: args,
                raw_text: Some(raw_text),
            });
            *text = format!("{}{}", &text[..mat_start], &text[call_end..]);
            found_any = true;
        } else {
            break;
        }
    }
    found_any
}

fn extract_bare_tool_call_json(text: &mut String, calls: &mut Vec<ToolCall>) -> bool {
    let mut found_any = false;
    loop {
        let Some(start) = text.find("<tool_call>") else {
            break;
        };
        let after = start + "<tool_call>".len();
        let Some(rel_brace) = text[after..].find('{') else {
            break;
        };
        let brace_start = after + rel_brace;
        let Some(json_len) = balanced_json_object_len(&text[brace_start..]) else {
            break;
        };
        let body_end = brace_start + json_len;
        let json_str = &text[brace_start..body_end];
        let Some(value) = super::json_repair::repair_json(json_str) else {
            break;
        };
        let Some((name, arguments)) = infer_bare_tool_call(value) else {
            break;
        };
        let close_end = text[body_end..]
            .find("</tool_call>")
            .map(|rel| body_end + rel + "</tool_call>".len())
            .unwrap_or(body_end);
        let raw_text = text[start..close_end].to_string();
        calls.push(ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            arguments,
            raw_text: Some(raw_text),
        });
        *text = format!("{}{}", &text[..start], &text[close_end..]);
        found_any = true;
    }
    found_any
}

fn infer_bare_tool_call(value: serde_json::Value) -> Option<(String, serde_json::Value)> {
    if value.get("type").and_then(|value| value.as_str()) == Some("function") {
        if let Some(function) = value.get("function").and_then(|value| value.as_object()) {
            let name = function
                .get("name")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())?;
            let arguments = function
                .get("arguments")
                .or_else(|| function.get("parameters"))
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
            return Some((name.to_string(), arguments));
        }
    }

    if let Some(name) = value
        .get("name")
        .or_else(|| value.get("tool"))
        .or_else(|| value.get("tool_name"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let arguments = value
            .get("arguments")
            .or_else(|| value.get("input"))
            .or_else(|| value.get("tool_argument"))
            .or_else(|| value.get("tool_arguments"))
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
        return Some((name.to_string(), arguments));
    }

    let serde_json::Value::Object(args) = value else {
        return None;
    };

    let inferred_name = if args
        .get("pattern")
        .and_then(|value| value.as_str())
        .map(looks_like_glob_pattern)
        .unwrap_or(false)
    {
        Some("glob")
    } else if args
        .get("command")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some()
    {
        Some("shell")
    } else if args.len() == 1
        && args
            .get("path")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_some()
    {
        Some("read_file")
    } else {
        None
    }?;

    Some((inferred_name.to_string(), serde_json::Value::Object(args)))
}

fn infer_bare_tool_calls(value: serde_json::Value) -> Vec<(String, serde_json::Value)> {
    match value {
        serde_json::Value::Array(items) => {
            items.into_iter().filter_map(infer_bare_tool_call).collect()
        }
        other => infer_bare_tool_call(other).into_iter().collect(),
    }
}

fn looks_like_glob_pattern(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && (trimmed.contains('*')
            || trimmed.contains('?')
            || trimmed.contains('[')
            || trimmed.contains('{'))
}

fn unwrap_meta_tool_call(
    outer_name: &str,
    mut args: serde_json::Map<String, serde_json::Value>,
) -> (String, serde_json::Value) {
    if outer_name != "tool_call" {
        return (outer_name.to_string(), serde_json::Value::Object(args));
    }

    let Some(inner_name) = args
        .get("name")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
    else {
        return (outer_name.to_string(), serde_json::Value::Object(args));
    };

    if let Some(nested_arguments) = args.remove("arguments") {
        let arguments = match nested_arguments {
            serde_json::Value::String(raw) => {
                let trimmed = raw.trim();
                let looks_like_json = (trimmed.starts_with('{') && trimmed.ends_with('}'))
                    || (trimmed.starts_with('[') && trimmed.ends_with(']'));
                if looks_like_json {
                    serde_json::from_str(trimmed).unwrap_or_else(|_| serde_json::Value::String(raw))
                } else {
                    serde_json::Value::String(raw)
                }
            }
            other => other,
        };
        return (inner_name.to_string(), arguments);
    }

    args.remove("name");
    (inner_name.to_string(), serde_json::Value::Object(args))
}

#[cfg(test)]
mod tests {
    use crate::models::profiles::{
        ModelFamily, ModelProfile, ParserType, RendererType, ThinkTagStyle, ToolCallFormat,
    };

    use super::{extract_tool_calls, extract_tool_calls_for_profile};

    fn qwen_profile() -> ModelProfile {
        ModelProfile {
            family: ModelFamily::Qwen3_5,
            tool_call_format: ToolCallFormat::QwenXml,
            think_tag_style: ThinkTagStyle::Qwen,
            interleaved_think_tool: true,
            supports_parallel_tools: true,
            supports_vision: false,
            default_max_output_tokens: None,
            default_context_window: None,
            max_context_window: None,
            parser_type: ParserType::QwenStateMachine,
            renderer_type: RendererType::QwenChat,
            stop_markers: vec![],
            allow_fallback_extraction: true,
            default_presence_penalty: None,
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            default_min_p: None,
            disable_thinking_for_tools: true,
        }
    }

    #[test]
    fn qwen_profile_extracts_qwen_xml() {
        let text = "<function=get_weather><parameter=city>London</parameter></function>";
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(remaining.is_empty());
    }

    #[test]
    fn qwen_profile_extracts_function_cut_at_end_of_text() {
        let text = "<tool_call>\n<function=glob>\n<parameter=pattern>**/*.go</parameter>";
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "glob");
        assert_eq!(calls[0].arguments["pattern"], "**/*.go");
        assert!(remaining.trim().is_empty(), "remaining={remaining:?}");
    }

    #[test]
    fn legacy_extractor_still_supports_qwen_and_hermes() {
        let text = "<tool_call>{\"name\":\"ping\",\"arguments\":{\"x\":1}}</tool_call>";
        let (calls, remaining) = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert!(remaining.is_empty());
    }

    #[test]
    fn qwen_xml_preserves_parameter_whitespace() {
        let text =
            "<function=echo><parameter=text>  keep surrounding spaces  </parameter></function>";
        let (calls, _) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(
            calls[0].arguments["text"],
            serde_json::Value::String("  keep surrounding spaces  ".to_string())
        );
    }

    #[test]
    fn qwen_xml_trims_template_line_padding() {
        let text = "<function=read_file><parameter=path>\nmain.go\n</parameter></function>";
        let (calls, _) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls[0].arguments["path"], "main.go");
    }

    #[test]
    fn qwen_xml_removes_only_each_matched_block_once() {
        let text = "<function=alpha><parameter=x>1</parameter></function>\nkeep\n<function=alpha><parameter=x>1</parameter></function>";
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 2);
        assert_eq!(remaining, "keep");
    }

    #[test]
    fn ignores_tool_calls_inside_think_blocks() {
        let text = "<think><function=secret><parameter=x>1</parameter></function></think>Answer";
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert!(calls.is_empty());
        assert_eq!(remaining, "Answer");
    }

    #[test]
    fn qwen_tool_call_wrapper_uses_inner_tool_name_with_flattened_args() {
        let text = "<function=tool_call><parameter=name>submit_helix_graph</parameter><parameter=nodes>[{\"id\":\"plan\"}]</parameter></function>";
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "submit_helix_graph");
        assert_eq!(calls[0].arguments["nodes"][0]["id"], "plan");
        assert!(remaining.is_empty());
    }

    #[test]
    fn qwen_tool_call_wrapper_uses_inner_tool_name_with_nested_arguments() {
        let text = "<function=tool_call><parameter=name>agent_message</parameter><parameter=arguments>{\"agent_id\":\"Barry\",\"task\":\"Build the app\"}</parameter></function>";
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "agent_message");
        assert_eq!(calls[0].arguments["agent_id"], "Barry");
        assert_eq!(calls[0].arguments["task"], "Build the app");
        assert!(remaining.is_empty());
    }

    #[test]
    fn qwen_safety_net_parses_square_bracket_format() {
        // Model emitted [tool_name]{...} because llama-server didn't inject the chat template
        let text = "[agent_message]{\"agent_id\":\"Barry\",\"task\":\"Do something\"}";
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "agent_message");
        assert_eq!(calls[0].arguments["agent_id"], "Barry");
        assert_eq!(calls[0].arguments["task"], "Do something");
        assert!(remaining.is_empty());
    }

    #[test]
    fn qwen_safety_net_parses_square_bracket_with_surrounding_text() {
        let text = "Sure, I'll call that.\n[write_file]{\"path\":\"foo.txt\",\"content\":\"hello\"}\nDone.";
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].arguments["path"], "foo.txt");
        assert!(remaining.contains("Sure"));
        assert!(remaining.contains("Done."));
    }

    #[test]
    fn qwen_xml_takes_priority_over_square_bracket() {
        // XML format should be used when present; safety net should not interfere
        let text = "<function=get_weather><parameter=city>London</parameter></function>";
        let (calls, _) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn qwen_safety_net_parses_parenthesis_format() {
        // Garry-style: [tool_call]name({json}) with parentheses wrapping JSON
        let text = r#"[tool_call]submit_helix_graph({"nodes":[{"id":"t1","agent_id":"Barry"}]})"#;
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "submit_helix_graph");
        assert_eq!(calls[0].arguments["nodes"][0]["id"], "t1");
        assert_eq!(calls[0].arguments["nodes"][0]["agent_id"], "Barry");
        assert!(remaining.is_empty());
    }

    #[test]
    fn qwen_safety_net_parenthesis_with_surrounding_text() {
        let text = "I'll dispatch the tasks now.\n[tool_call]agent_message({\"agent_id\":\"Barry\",\"task\":\"Build it\"})\nDone.";
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "agent_message");
        assert_eq!(calls[0].arguments["agent_id"], "Barry");
        assert!(remaining.contains("I'll dispatch"));
        assert!(remaining.contains("Done."));
    }

    #[test]
    fn qwen_safety_net_parses_html_tool_code_div() {
        let text = r#"Let me check.
<div class="tool_code">{"tool_name":"glob","tool_argument":{"pattern":"*"}}</div>
Done."#;
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "glob");
        assert_eq!(calls[0].arguments["pattern"], "*");
        assert!(remaining.contains("Let me check."));
        assert!(remaining.contains("Done."));
        assert!(!remaining.contains("tool_code"));
    }

    #[test]
    fn qwen_safety_net_parses_plain_xml_tool_call_expression() {
        let text = r#"Let me check what is in the current folder.
<tool_call>_.glob(dir=".",pattern="**/*")"#;
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "glob");
        assert_eq!(calls[0].arguments["dir"], ".");
        assert_eq!(calls[0].arguments["pattern"], "**/*");
        assert!(remaining.contains("Let me check"));
        assert!(!remaining.contains("<tool_call>"));
    }

    #[test]
    fn qwen_safety_net_repairs_bare_tool_call_json_with_inferred_tool() {
        let text = r#"Let me inspect.
<tool_call>
{"pattern":"**/*.go"}"#;
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "glob");
        assert_eq!(calls[0].arguments["pattern"], "**/*.go");
        assert!(remaining.contains("Let me inspect."));
        assert!(!remaining.contains("<tool_call>"));
    }

    #[test]
    fn qwen_safety_net_parses_fenced_tool_input_json() {
        let text = r#"```json
{
  "tool": "glob",
  "input": {
    "pattern": "**/*.go"
  }
}
```"#;
        let (calls, remaining) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "glob");
        assert_eq!(calls[0].arguments["pattern"], "**/*.go");
        assert!(remaining.is_empty());
    }

    #[test]
    fn native_api_safety_net_parses_content_only_openai_tool_array() {
        let mut profile = qwen_profile();
        profile.parser_type = ParserType::NativeApi;
        profile.tool_call_format = ToolCallFormat::NativeApi;
        profile.renderer_type = RendererType::GemmaChat;
        profile.think_tag_style = ThinkTagStyle::None;
        profile.allow_fallback_extraction = false;

        let text = r#"<|channel>thought
<channel|>```json
[
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "parameters": {
        "path": "package.json"
      }
    }
  }
]
```"#;
        let (calls, remaining) = extract_tool_calls_for_profile(text, &profile);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "package.json");
        assert!(remaining.is_empty(), "remaining={remaining:?}");
    }

    #[test]
    fn gemma4_profile_extracts_native_tool_call() {
        let profile = ModelProfile::detect("google/gemma-4-12B-it");
        let text = r#"<|channel>thought
<channel|><|tool_call>call:web_search{query:<|"|>Gemma 4 12B tool calling<|"|>,limit:3}<tool_call|>"#;

        let (calls, remaining) = extract_tool_calls_for_profile(text, &profile);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].arguments["query"], "Gemma 4 12B tool calling");
        assert_eq!(calls[0].arguments["limit"], 3.0);
        assert!(!remaining.contains("tool_call"));
    }

    #[test]
    fn gemma4_profile_extracts_plain_quoted_native_tool_call() {
        let profile = ModelProfile::detect("gemma-4-26B-A4B-it-QAT-Q4_0.gguf");
        let text = r#"<|tool_call>call:noop{status: "success"}<tool_call|>"#;

        let (calls, remaining) = extract_tool_calls_for_profile(text, &profile);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "noop");
        assert_eq!(calls[0].arguments["status"], "success");
        assert!(remaining.trim().is_empty(), "remaining={remaining:?}");
    }

    #[test]
    fn gemma4_profile_suppresses_malformed_native_tool_blob() {
        let profile = ModelProfile::detect("gemma-4-26B-A4B-it-QAT-Q4_0.gguf");
        let text = r#"<|channel>thought
<channel|><|tool_call><|tool_call>callcall::todotodo__createcreate{{itemsitems:[:[<|"|><|"|>InspectInspect project project structure structure<|"|><|"|>]}}"#;

        let (calls, remaining) = extract_tool_calls_for_profile(text, &profile);

        assert!(calls.is_empty());
        assert!(remaining.trim().is_empty(), "remaining={remaining:?}");
    }

    #[test]
    fn qwen_xml_takes_priority_over_parenthesis() {
        // XML wins - parenthesis safety net should not fire when XML is present
        let text = "<function=get_weather><parameter=city>London</parameter></function>";
        let (calls, _) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }
}
