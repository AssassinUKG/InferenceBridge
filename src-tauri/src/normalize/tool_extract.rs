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
    let mut remaining = crate::normalize::think_strip::strip_think_tags(text);

    // Try Hermes-style <tool_call>{json}</tool_call>
    extract_hermes_calls(&mut remaining, &mut calls);

    // Try Qwen-style <function=name><parameter=key>value</parameter></function>
    extract_qwen_function_calls(&mut remaining, &mut calls);

    (calls, remaining.trim().to_string())
}

pub fn extract_tool_calls_for_profile(text: &str, profile: &ModelProfile) -> (Vec<ToolCall>, String) {
    let mut calls = Vec::new();
    let mut remaining =
        crate::normalize::think_strip::strip_think_tags_with_style(text, profile.think_tag_style);

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
        let outer_name = cap.get(1).unwrap().as_str().trim();
        let body = cap.get(2).unwrap().as_str();

        // Parse parameters from <parameter=key>value</parameter> pairs
        let param_re = regex::Regex::new(r"<parameter=([^>]+)>(.*?)</parameter>").unwrap();
        let mut args = serde_json::Map::new();
        for param_cap in param_re.captures_iter(body) {
            let key = param_cap.get(1).unwrap().as_str().trim();
            let raw_value = param_cap.get(2).unwrap().as_str();
            let trimmed = raw_value.trim();
            let parse_as_json = (trimmed.starts_with('{') && trimmed.ends_with('}'))
                || (trimmed.starts_with('[') && trimmed.ends_with(']'))
                || (trimmed.starts_with('"') && trimmed.ends_with('"'));
            let json_val = if parse_as_json {
                serde_json::from_str(trimmed)
                    .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()))
            } else {
                serde_json::Value::String(raw_value.to_string())
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

        new_text = new_text.replace(full_match.as_str(), "");
    }

    *text = new_text;
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
                    serde_json::from_str(trimmed)
                        .unwrap_or_else(|_| serde_json::Value::String(raw))
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

    #[test]
    fn qwen_xml_preserves_parameter_whitespace() {
        let text = "<function=echo><parameter=text>  keep surrounding spaces  </parameter></function>";
        let (calls, _) = extract_tool_calls_for_profile(text, &qwen_profile());
        assert_eq!(
            calls[0].arguments["text"],
            serde_json::Value::String("  keep surrounding spaces  ".to_string())
        );
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
}


