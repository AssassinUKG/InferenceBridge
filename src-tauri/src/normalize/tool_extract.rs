//! Extract structured tool calls from model output text.

use serde::{Deserialize, Serialize};

use crate::models::profiles::{ModelProfile, ParserType};

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
    let mut remaining = text.to_string();

    // Try Hermes-style <tool_call>{json}</tool_call>
    extract_hermes_calls(&mut remaining, &mut calls);

    // Try Qwen-style <function=name><parameter=key>value</parameter></function>
    extract_qwen_function_calls(&mut remaining, &mut calls);

    (calls, remaining.trim().to_string())
}

pub fn extract_tool_calls_for_profile(text: &str, profile: &ModelProfile) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut remaining = text.to_string();

    match profile.parser_type {
        ParserType::QwenStateMachine => {
            extract_qwen_function_calls(&mut remaining, &mut calls);
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_hermes_calls(&mut remaining, &mut calls);
            }
        }
        ParserType::HermesFallback => {
            extract_hermes_calls(&mut remaining, &mut calls);
            if calls.is_empty() && profile.allow_fallback_extraction {
                extract_qwen_function_calls(&mut remaining, &mut calls);
            }
        }
        ParserType::NativeApi => {
            if profile.allow_fallback_extraction {
                extract_hermes_calls(&mut remaining, &mut calls);
                if calls.is_empty() {
                    extract_qwen_function_calls(&mut remaining, &mut calls);
                }
            }
        }
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

fn extract_qwen_function_calls(text: &mut String, calls: &mut Vec<ToolCall>) {
    let re = regex::Regex::new(r"<function=([^>]+)>(.*?)</function>").unwrap();
    let mut new_text = text.clone();

    for cap in re.captures_iter(text) {
        let full_match = cap.get(0).unwrap();
        let name = cap.get(1).unwrap().as_str().trim();
        let body = cap.get(2).unwrap().as_str();

        // Parse parameters from <parameter=key>value</parameter> pairs
        let param_re = regex::Regex::new(r"<parameter=([^>]+)>(.*?)</parameter>").unwrap();
        let mut args = serde_json::Map::new();
        for param_cap in param_re.captures_iter(body) {
            let key = param_cap.get(1).unwrap().as_str().trim();
            let value = param_cap.get(2).unwrap().as_str().trim();
            // Try to parse as JSON value, fall back to string
            let json_val = serde_json::from_str(value)
                .unwrap_or_else(|_| serde_json::Value::String(value.to_string()));
            args.insert(key.to_string(), json_val);
        }

        calls.push(ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(),
            arguments: serde_json::Value::Object(args),
            raw_text: Some(full_match.as_str().to_string()),
        });

        new_text = new_text.replace(full_match.as_str(), "");
    }

    *text = new_text;
}

#[cfg(test)]
mod tests {
    use crate::models::profiles::{ModelFamily, ModelProfile, ParserType, RendererType, ThinkTagStyle, ToolCallFormat};

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
            split_tool_calling: true,
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
    fn legacy_extractor_still_supports_qwen_and_hermes() {
        let text = "<tool_call>{\"name\":\"ping\",\"arguments\":{\"x\":1}}</tool_call>";
        let (calls, remaining) = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert!(remaining.is_empty());
    }
}
