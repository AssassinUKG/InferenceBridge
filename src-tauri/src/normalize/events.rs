use serde::{Deserialize, Serialize};

use crate::models::profiles::ModelProfile;
use crate::normalize::think_strip::{
    extract_reasoning_content_with_style, strip_think_tags_with_style,
};
use crate::normalize::tool_extract::{extract_tool_calls_for_profile, ToolCall};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormalizedEventKind {
    VisibleText,
    Reasoning,
    ToolCall,
    ToolResult,
    FinalText,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedToolCall {
    pub id: String,
    pub namespace: Option<String>,
    pub name: String,
    pub arguments: serde_json::Value,
    pub raw_span: Option<String>,
    pub target_channel: Option<String>,
}

impl From<ToolCall> for NormalizedToolCall {
    fn from(call: ToolCall) -> Self {
        let (namespace, name) = split_tool_namespace(&call.name);
        Self {
            id: call.id,
            namespace,
            name,
            arguments: call.arguments,
            raw_span: call.raw_text,
            target_channel: Some("commentary".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedStreamEvent {
    pub kind: NormalizedEventKind,
    pub text: Option<String>,
    pub tool_call: Option<NormalizedToolCall>,
    pub raw_span: Option<String>,
    pub parser_stage: String,
    pub decision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedParseOutput {
    pub events: Vec<NormalizedStreamEvent>,
    pub visible_text: String,
    pub reasoning_text: String,
    pub tool_calls: Vec<NormalizedToolCall>,
    pub parser_type: String,
    pub decisions: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingParserState {
    PlainText,
    Reasoning,
    ToolJson,
}

#[derive(Debug, Default)]
pub struct NormalizedStreamingParser {
    buffer: String,
}

impl NormalizedStreamingParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_chunk(
        &mut self,
        chunk: &str,
        profile: &ModelProfile,
    ) -> Option<NormalizedParseOutput> {
        self.buffer.push_str(chunk);
        if has_incomplete_control_span(&self.buffer) {
            return None;
        }
        Some(parse_normalized_for_profile(&self.buffer, profile))
    }

    pub fn finish(&mut self, profile: &ModelProfile) -> NormalizedParseOutput {
        let output = parse_normalized_for_profile(&self.buffer, profile);
        self.buffer.clear();
        output
    }
}

pub fn parse_normalized_for_profile(raw: &str, profile: &ModelProfile) -> NormalizedParseOutput {
    let parser_type = format!("{:?}", profile.parser_type);
    let reasoning_text = extract_reasoning_content_with_style(raw, profile.think_tag_style);
    let stripped = strip_think_tags_with_style(raw, profile.think_tag_style);
    let parse_result = parse_with_staged_pipeline(&stripped, profile);
    let tool_calls = parse_result.tool_calls;
    let visible_text = parse_result.visible_text;
    let tool_stage = parse_result.stage;
    let mut decisions = parse_result.decisions;

    let mut events = Vec::new();

    if !reasoning_text.trim().is_empty() {
        decisions.push("reasoning extracted from profile think/channel style".to_string());
        events.push(NormalizedStreamEvent {
            kind: NormalizedEventKind::Reasoning,
            text: Some(reasoning_text.clone()),
            tool_call: None,
            raw_span: reasoning_raw_span(raw, &reasoning_text),
            parser_stage: "strict_profile".to_string(),
            decision: "extracted reasoning channel".to_string(),
        });
    }

    let normalized_calls = tool_calls
        .into_iter()
        .map(NormalizedToolCall::from)
        .collect::<Vec<_>>();

    for call in &normalized_calls {
        decisions.push(format!(
            "tool call extracted via profile parser: {}{}",
            call.namespace
                .as_ref()
                .map(|ns| format!("{ns}."))
                .unwrap_or_default(),
            call.name
        ));
        events.push(NormalizedStreamEvent {
            kind: NormalizedEventKind::ToolCall,
            text: None,
            tool_call: Some(call.clone()),
            raw_span: call.raw_span.clone(),
            parser_stage: tool_stage.to_string(),
            decision: match tool_stage {
                ParserStage::StrictProfile => "extracted structured tool call",
                ParserStage::Recovery => "recovered structured tool call",
                ParserStage::Fallback => "fallback extracted structured tool call",
            }
            .to_string(),
        });
    }

    if !visible_text.trim().is_empty() {
        let visible_kind =
            if tool_stage == ParserStage::Fallback && contains_control_marker(&visible_text) {
                NormalizedEventKind::Unknown
            } else {
                NormalizedEventKind::VisibleText
            };
        decisions.push(match visible_kind {
            NormalizedEventKind::Unknown => {
                "unparsed control-looking text preserved as unknown".to_string()
            }
            _ => "visible text preserved after staged parsing".to_string(),
        });
        events.push(NormalizedStreamEvent {
            kind: visible_kind,
            text: Some(visible_text.clone()),
            tool_call: None,
            raw_span: Some(visible_text.clone()),
            parser_stage: if normalized_calls.is_empty() {
                "strict_profile".to_string()
            } else {
                tool_stage.to_string()
            },
            decision: "preserved user-visible text".to_string(),
        });
    }

    if events.is_empty() && !raw.trim().is_empty() {
        decisions.push("no structured parser match; raw text preserved as unknown".to_string());
        events.push(NormalizedStreamEvent {
            kind: NormalizedEventKind::Unknown,
            text: Some(raw.trim().to_string()),
            tool_call: None,
            raw_span: Some(raw.to_string()),
            parser_stage: "fallback".to_string(),
            decision: "preserved raw text".to_string(),
        });
    }

    NormalizedParseOutput {
        events,
        visible_text,
        reasoning_text,
        tool_calls: normalized_calls,
        parser_type,
        decisions,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserStage {
    StrictProfile,
    Recovery,
    Fallback,
}

impl ParserStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::StrictProfile => "strict_profile",
            Self::Recovery => "recovery",
            Self::Fallback => "fallback",
        }
    }
}

impl std::fmt::Display for ParserStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

struct StagedParseResult {
    tool_calls: Vec<ToolCall>,
    visible_text: String,
    stage: ParserStage,
    decisions: Vec<String>,
}

fn parse_with_staged_pipeline(stripped: &str, profile: &ModelProfile) -> StagedParseResult {
    let mut strict_profile = profile.clone();
    strict_profile.allow_fallback_extraction = false;
    let (strict_calls, strict_visible) = extract_tool_calls_for_profile(stripped, &strict_profile);

    if !strict_calls.is_empty() {
        return StagedParseResult {
            tool_calls: strict_calls,
            visible_text: strict_visible,
            stage: ParserStage::StrictProfile,
            decisions: vec![format!(
                "strict profile parser matched {:?}",
                profile.parser_type
            )],
        };
    }

    if profile.allow_fallback_extraction {
        let (recovered_calls, recovered_visible) =
            extract_tool_calls_for_profile(stripped, profile);
        if !recovered_calls.is_empty() {
            return StagedParseResult {
                tool_calls: recovered_calls,
                visible_text: recovered_visible,
                stage: ParserStage::Recovery,
                decisions: vec![format!(
                    "strict profile parser found no tool calls; recovery parser matched {:?}",
                    profile.parser_type
                )],
            };
        }
    }

    StagedParseResult {
        tool_calls: Vec::new(),
        visible_text: strict_visible,
        stage: ParserStage::Fallback,
        decisions: vec![
            "strict profile parser found no tool calls; preserving visible text or raw fallback"
                .to_string(),
        ],
    }
}

fn split_tool_namespace(name: &str) -> (Option<String>, String) {
    if let Some((namespace, tool_name)) = name.split_once('.') {
        let namespace = namespace.trim();
        let tool_name = tool_name.trim();
        if !namespace.is_empty() && !tool_name.is_empty() {
            return (Some(namespace.to_string()), tool_name.to_string());
        }
    }
    (None, name.to_string())
}

fn contains_control_marker(text: &str) -> bool {
    [
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<|tool_call>",
        "<tool_call|>",
        "<|channel>",
        "<channel|>",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn reasoning_raw_span(raw: &str, reasoning_text: &str) -> Option<String> {
    let trimmed = reasoning_text.trim();
    if trimmed.is_empty() {
        return None;
    }
    raw.find(trimmed)
        .map(|start| raw[start..start + trimmed.len()].to_string())
        .or_else(|| Some(reasoning_text.to_string()))
}

fn has_incomplete_control_span(text: &str) -> bool {
    matches!(
        detect_streaming_state(text),
        StreamingParserState::Reasoning | StreamingParserState::ToolJson
    )
}

fn detect_streaming_state(text: &str) -> StreamingParserState {
    let last_think_open = text.rfind("<think>");
    let last_think_close = text.rfind("</think>");
    if last_think_open.is_some() && last_think_open > last_think_close {
        return StreamingParserState::Reasoning;
    }

    let last_qwen_think_open = text.rfind("<|think|>");
    let last_qwen_think_close = text.rfind("<|/think|>");
    if last_qwen_think_open.is_some() && last_qwen_think_open > last_qwen_think_close {
        return StreamingParserState::Reasoning;
    }

    let last_tool_open = text.rfind("<tool_call>");
    let last_tool_close = text.rfind("</tool_call>");
    if last_tool_open.is_some() && last_tool_open > last_tool_close {
        return StreamingParserState::ToolJson;
    }

    let last_function_open = text.rfind("<function=");
    let last_function_close = text.rfind("</function>");
    if last_function_open.is_some() && last_function_open > last_function_close {
        return StreamingParserState::ToolJson;
    }

    StreamingParserState::PlainText
}

#[cfg(test)]
mod tests {
    use super::{parse_normalized_for_profile, NormalizedEventKind, NormalizedStreamingParser};
    use crate::models::profiles::ModelProfile;

    #[test]
    fn emits_reasoning_tool_and_visible_events_for_hermes_output() {
        let profile = ModelProfile::detect("Hermes-3-Llama-3.1-8B.Q4_K_M.gguf");
        let raw = r#"<think>check weather</think><tool_call>{"name":"weather.get","arguments":{"city":"London"}}</tool_call>It is mild."#;

        let parsed = parse_normalized_for_profile(raw, &profile);

        assert!(parsed
            .events
            .iter()
            .any(|event| event.kind == NormalizedEventKind::Reasoning));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].namespace.as_deref(), Some("weather"));
        assert_eq!(parsed.tool_calls[0].name, "get");
        assert_eq!(parsed.visible_text, "It is mild.");
    }

    #[test]
    fn emits_tool_call_for_qwen_xml_output() {
        let profile = ModelProfile::detect("Qwen3-8B-Q4_K_M.gguf");
        let raw = "<function=search><parameter=query>llama.cpp mtp</parameter></function>Done";

        let parsed = parse_normalized_for_profile(raw, &profile);

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "search");
        assert_eq!(parsed.tool_calls[0].arguments["query"], "llama.cpp mtp");
        assert_eq!(parsed.visible_text, "Done");
        assert!(parsed.events.iter().any(|event| {
            event.kind == NormalizedEventKind::ToolCall && event.parser_stage == "strict_profile"
        }));
    }

    #[test]
    fn labels_qwen_safety_net_tool_calls_as_recovery() {
        let profile = ModelProfile::detect("Qwen3-8B-Q4_K_M.gguf");
        let raw = r#"Sure.
[search]{"query":"llama.cpp mtp"}
Done."#;

        let parsed = parse_normalized_for_profile(raw, &profile);

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "search");
        assert_eq!(parsed.tool_calls[0].arguments["query"], "llama.cpp mtp");
        assert!(parsed.events.iter().any(|event| {
            event.kind == NormalizedEventKind::ToolCall && event.parser_stage == "recovery"
        }));
    }

    #[test]
    fn streaming_parser_waits_for_split_tool_call_boundary() {
        let profile = ModelProfile::detect("Hermes-3-Llama-3.1-8B.Q4_K_M.gguf");
        let mut parser = NormalizedStreamingParser::new();

        assert!(parser
            .push_chunk("<tool_call>{\"name\":\"lookup\",\"arguments\":{", &profile)
            .is_none());
        let parsed = parser
            .push_chunk("\"id\":7}}</tool_call>ok", &profile)
            .expect("complete tool call should parse");

        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "lookup");
        assert_eq!(parsed.visible_text, "ok");
    }

    #[test]
    fn plain_unmatched_text_remains_visible_text() {
        let profile = ModelProfile::detect("generic-model.gguf");

        let parsed = parse_normalized_for_profile("   ???   ", &profile);

        assert_eq!(parsed.events.len(), 1);
        assert_eq!(parsed.events[0].kind, NormalizedEventKind::VisibleText);
        assert_eq!(parsed.events[0].text.as_deref(), Some("???"));
    }

    #[test]
    fn unknown_event_preserves_unparsed_control_text() {
        let profile = ModelProfile::detect("generic-model.gguf");

        let parsed = parse_normalized_for_profile("<tool_call>{not json", &profile);

        assert_eq!(parsed.events.len(), 1);
        assert_eq!(parsed.events[0].kind, NormalizedEventKind::Unknown);
        assert_eq!(parsed.events[0].parser_stage, "fallback");
    }
}
