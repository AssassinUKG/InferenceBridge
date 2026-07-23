use std::collections::HashSet;

use serde::Serialize;

use super::tool_extract::ToolCall;

/// Capabilities that the current request can actually execute.
///
/// Desktop chat does not execute tools. API requests may expose an explicit set
/// of tools supplied by the caller; only those names are considered available.
#[derive(Debug, Clone, Default)]
pub struct RuntimeCapabilities {
    available_tools: HashSet<String>,
    image_generation: bool,
}

impl RuntimeCapabilities {
    pub fn desktop_chat() -> Self {
        Self::default()
    }

    pub fn from_requested_tools(tools: Option<&[serde_json::Value]>) -> Self {
        let available_tools = tools
            .into_iter()
            .flatten()
            .filter_map(requested_tool_name)
            .map(normalize_tool_name)
            .collect::<HashSet<_>>();
        let image_generation = available_tools
            .iter()
            .any(|name| is_image_generation_tool(name));
        Self {
            available_tools,
            image_generation,
        }
    }

    fn supports_tool(&self, name: &str) -> bool {
        self.available_tools.contains(&normalize_tool_name(name))
    }
}

/// Short-circuit requests for capabilities the desktop runtime cannot provide.
/// This avoids spending prompt tokens asking a language model about an app-level
/// fact that the runtime already knows with certainty.
pub fn unavailable_request_response(
    input: &str,
    capabilities: &RuntimeCapabilities,
) -> Option<String> {
    if !capabilities.image_generation && looks_like_image_generation_request(input) {
        return Some(
            "Image generation is not connected to this InferenceBridge runtime, so I can’t create or return an image here. No image was created. You can still ask for help writing an image prompt, or analyse an attached image when the loaded model supports vision."
                .to_string(),
        );
    }
    None
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MissingCapability {
    ImageGeneration,
    ToolExecution,
}

#[derive(Debug, Clone)]
pub struct RejectedToolCall {
    pub call: ToolCall,
    pub capability: MissingCapability,
    pub message: String,
}

impl RejectedToolCall {
    pub fn result_json(&self) -> String {
        serde_json::json!({
            "status": "rejected",
            "code": "capability_unavailable",
            "capability": self.capability,
            "tool": self.call.name,
            "message": self.message,
        })
        .to_string()
    }
}

#[derive(Debug, Clone)]
pub struct CapabilityEnforcement {
    pub accepted: Vec<ToolCall>,
    pub rejected: Vec<RejectedToolCall>,
    pub display_text: String,
}

/// Enforce capability truth after model output parsing and before anything is
/// presented, persisted as assistant content, or returned as an API tool call.
pub fn enforce_tool_calls(
    calls: Vec<ToolCall>,
    visible_text: String,
    capabilities: &RuntimeCapabilities,
) -> CapabilityEnforcement {
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    for call in calls {
        if capabilities.supports_tool(&call.name) {
            accepted.push(call);
            continue;
        }

        let capability = if is_image_generation_tool(&call.name) {
            MissingCapability::ImageGeneration
        } else {
            MissingCapability::ToolExecution
        };
        let message = rejection_message(&call.name, capability);
        tracing::warn!(
            tool = %call.name,
            ?capability,
            "Rejected model tool call because the capability is unavailable"
        );
        rejected.push(RejectedToolCall {
            call,
            capability,
            message,
        });
    }

    let mut display_parts = Vec::new();
    let visible_text = visible_text.trim();
    if has_meaningful_visible_text(visible_text) {
        display_parts.push(visible_text.to_string());
    }
    display_parts.extend(rejected.iter().map(|item| item.message.clone()));

    CapabilityEnforcement {
        accepted,
        rejected,
        display_text: display_parts.join("\n\n"),
    }
}

fn requested_tool_name(value: &serde_json::Value) -> Option<String> {
    value
        .get("function")
        .and_then(|function| function.get("name"))
        .or_else(|| value.get("name"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

fn normalize_tool_name(name: impl AsRef<str>) -> String {
    name.as_ref().trim().to_ascii_lowercase()
}

fn is_image_generation_tool(name: &str) -> bool {
    let normalized = normalize_tool_name(name).replace([' ', '-'], "_");
    [
        "dall",
        "text2im",
        "text_to_image",
        "generate_image",
        "image_generation",
        "image.generate",
        "image_gen",
        "create_image",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
        && !normalized.contains("image_search")
}

fn rejection_message(name: &str, capability: MissingCapability) -> String {
    match capability {
        MissingCapability::ImageGeneration => format!(
            "Image generation is not available in this InferenceBridge runtime, so no image was created. The model attempted to use the unavailable tool `{name}`."
        ),
        MissingCapability::ToolExecution => format!(
            "The model attempted to use the unavailable tool `{name}`. InferenceBridge did not execute it."
        ),
    }
}

fn has_meaningful_visible_text(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric)
}

fn looks_like_image_generation_request(input: &str) -> bool {
    let normalized = input
        .to_ascii_lowercase()
        .replace(['\n', '\r', '\t', '-', '_'], " ");
    let normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

    let direct_phrases = [
        "generate an image",
        "generate a image",
        "generate images",
        "generate a picture",
        "generate pictures",
        "create an image",
        "create a image",
        "create images",
        "create a picture",
        "create pictures",
        "make an image",
        "make a image",
        "make images",
        "make a picture",
        "make pictures",
        "produce an image",
        "produce images",
        "draw me",
        "illustrate this",
        "render an image",
    ];
    if direct_phrases
        .iter()
        .any(|phrase| normalized.contains(phrase))
    {
        return true;
    }

    let asks_ability = [
        "can you",
        "could you",
        "are you able to",
        "do you",
        "will you",
    ]
    .iter()
    .any(|phrase| normalized.contains(phrase));
    let mentions_images = [" image", " images", " picture", " pictures", " artwork"]
        .iter()
        .any(|term| normalized.contains(term));
    let generation_verb = [
        " make",
        " create",
        " generate",
        " draw",
        " produce",
        " render",
    ]
    .iter()
    .any(|term| normalized.contains(term));

    asks_ability && mentions_images && generation_verb
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamGateMode {
    Undecided,
    Passthrough,
    Buffered,
}

/// Prevent structured tool envelopes from flashing as ordinary assistant text
/// while preserving token-by-token streaming for normal prose.
#[derive(Debug, Clone)]
pub struct ToolOutputStreamGate {
    mode: StreamGateMode,
    pending: String,
}

impl ToolOutputStreamGate {
    pub fn new(force_buffer: bool) -> Self {
        Self {
            mode: if force_buffer {
                StreamGateMode::Buffered
            } else {
                StreamGateMode::Undecided
            },
            pending: String::new(),
        }
    }

    pub fn push(&mut self, delta: &str) -> Option<String> {
        match self.mode {
            StreamGateMode::Passthrough => Some(delta.to_string()),
            StreamGateMode::Buffered => None,
            StreamGateMode::Undecided => {
                self.pending.push_str(delta);
                let Some(first) = self.pending.chars().find(|ch| !ch.is_whitespace()) else {
                    return None;
                };
                if matches!(first, '{' | '[' | '<' | '`') {
                    self.mode = StreamGateMode::Buffered;
                    None
                } else {
                    self.mode = StreamGateMode::Passthrough;
                    Some(std::mem::take(&mut self.pending))
                }
            }
        }
    }

    pub fn should_emit_final(&self) -> bool {
        self.mode != StreamGateMode::Passthrough
    }
}

impl Default for ToolOutputStreamGate {
    fn default() -> Self {
        Self::new(false)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: "call-1".to_string(),
            name: name.to_string(),
            arguments: json!({"prompt": "a lighthouse"}),
            raw_text: None,
        }
    }

    #[test]
    fn desktop_rejects_invented_image_generation() {
        let result = enforce_tool_calls(
            vec![call("dalle.text2im")],
            ".".to_string(),
            &RuntimeCapabilities::desktop_chat(),
        );

        assert!(result.accepted.is_empty());
        assert_eq!(result.rejected.len(), 1);
        assert_eq!(
            result.rejected[0].capability,
            MissingCapability::ImageGeneration
        );
        assert!(result.display_text.contains("no image was created"));
        assert!(!result.display_text.starts_with('.'));
    }

    #[test]
    fn api_only_accepts_tools_declared_by_the_caller() {
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "get_weather" }
        })];
        let capabilities = RuntimeCapabilities::from_requested_tools(Some(&tools));
        let result = enforce_tool_calls(
            vec![call("get_weather"), call("shell")],
            String::new(),
            &capabilities,
        );

        assert_eq!(result.accepted.len(), 1);
        assert_eq!(result.accepted[0].name, "get_weather");
        assert_eq!(result.rejected.len(), 1);
        assert_eq!(result.rejected[0].call.name, "shell");
    }

    #[test]
    fn anthropic_flat_tool_names_are_supported() {
        let tools = vec![json!({ "name": "lookup_docs" })];
        let capabilities = RuntimeCapabilities::from_requested_tools(Some(&tools));
        assert!(capabilities.supports_tool("LOOKUP_DOCS"));
    }

    #[test]
    fn image_generation_requests_are_answered_by_runtime_truth() {
        let capabilities = RuntimeCapabilities::desktop_chat();
        assert!(unavailable_request_response("Can you make images?", &capabilities).is_some());
        assert!(unavailable_request_response(
            "Please generate an image of a lighthouse",
            &capabilities
        )
        .is_some());
        assert!(
            unavailable_request_response("Can you analyse this image?", &capabilities).is_none()
        );
        assert!(
            unavailable_request_response("Write a prompt for an image", &capabilities).is_none()
        );
    }

    #[test]
    fn stream_gate_buffers_structured_candidates_only() {
        let mut json_gate = ToolOutputStreamGate::default();
        assert_eq!(json_gate.push(" \n"), None);
        assert_eq!(json_gate.push("{\"action\":"), None);
        assert!(json_gate.should_emit_final());

        let mut prose_gate = ToolOutputStreamGate::default();
        assert_eq!(prose_gate.push("Hello"), Some("Hello".to_string()));
        assert_eq!(prose_gate.push(" world"), Some(" world".to_string()));
        assert!(!prose_gate.should_emit_final());
    }
}
