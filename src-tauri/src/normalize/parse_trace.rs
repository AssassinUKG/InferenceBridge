use crate::models::profiles::ModelProfile;
use crate::normalize::events::parse_normalized_for_profile;
use crate::normalize::think_strip::extract_reasoning_content_with_style;

pub fn build_parse_trace(
    profile: &ModelProfile,
    raw: &str,
    stripped: &str,
    reasoning_override: Option<&str>,
) -> String {
    let (tool_calls, visible_text) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(stripped, profile);
    let normalized = parse_normalized_for_profile(raw, profile);
    let reasoning_text = reasoning_override.map_or_else(
        || extract_reasoning_content_with_style(raw, profile.think_tag_style),
        ToOwned::to_owned,
    );
    serde_json::to_string_pretty(&serde_json::json!({
        "parser_type": format!("{:?}", profile.parser_type),
        "tool_call_format": format!("{:?}", profile.tool_call_format),
        "think_tag_style": format!("{:?}", profile.think_tag_style),
        "raw_response": raw,
        "reasoning_text": reasoning_text,
        "stripped_response": stripped,
        "visible_text": visible_text,
        "tool_calls": tool_calls,
        "normalized": normalized,
    }))
    .unwrap_or_else(|_| "Failed to serialize parse trace".to_string())
}
