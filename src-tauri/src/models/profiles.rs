//! Model capability profiles — ported from HelixClaw's model_profiles.rs.
//!
//! Maps model families to their tool-call format, think-tag support,
//! parser type, and other behavioral flags.

use serde::{Deserialize, Serialize};

/// Coarse model family, detected from the model name string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFamily {
    Qwen3_5,
    Qwen3,
    Qwen2_5,
    DeepSeekR1,
    Llama3,
    Phi,
    Mistral,
    Generic,
}

/// How the model emits tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallFormat {
    /// Model uses the provider's native tool_calls API field.
    NativeApi,
    /// Hermes-style XML: `<tool_call>{json}</tool_call>`.
    HermesXml,
    /// Qwen-specific: may emit tool calls in text, possibly inside think blocks.
    QwenXml,
}

/// How the model emits chain-of-thought / reasoning blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkTagStyle {
    None,
    Standard,
    Qwen,
}

/// Which parser to use for output normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParserType {
    NativeApi,
    HermesFallback,
    QwenStateMachine,
}

/// Which template renderer to use for chat formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RendererType {
    ChatML,
    QwenChat,
    Llama3Chat,
}

/// Complete capability profile for a model family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub family: ModelFamily,
    pub tool_call_format: ToolCallFormat,
    pub think_tag_style: ThinkTagStyle,
    pub interleaved_think_tool: bool,
    pub supports_parallel_tools: bool,
    pub default_max_output_tokens: Option<u32>,
    pub default_context_window: Option<u32>,
    pub max_context_window: Option<u32>,
    pub parser_type: ParserType,
    pub renderer_type: RendererType,
    pub stop_markers: Vec<String>,
    pub allow_fallback_extraction: bool,
    pub default_presence_penalty: Option<f32>,
    pub default_temperature: Option<f32>,
    pub default_top_p: Option<f32>,
    pub default_top_k: Option<i32>,
    pub default_min_p: Option<f32>,
    pub disable_thinking_for_tools: bool,
    pub split_tool_calling: bool,
}

impl ModelProfile {
    /// Detect model family and return a complete capability profile.
    pub fn detect(model_name: &str) -> Self {
        let m = model_name.to_lowercase();

        if m.contains("qwen3.5") {
            return Self::qwen3_5(&m);
        }
        if m.contains("qwen3") {
            return Self::qwen3(&m);
        }
        if m.contains("qwen2.5") {
            return Self::qwen2_5();
        }
        if m.contains("deepseek") && (m.contains("r1") || m.contains("reasoning")) {
            return Self::deepseek_r1();
        }
        if m.contains("llama") && (m.contains("3.") || m.contains("3-") || m.contains("3:")) {
            return Self::llama3();
        }
        if m.contains("phi-3") || m.contains("phi-4") || m.contains("phi3") || m.contains("phi4") {
            return Self::phi();
        }
        if m.contains("mistral") || m.contains("mixtral") || m.contains("nemo") {
            return Self::mistral();
        }
        Self::generic()
    }

    fn qwen3_5(_m: &str) -> Self {
        Self {
            family: ModelFamily::Qwen3_5,
            tool_call_format: ToolCallFormat::QwenXml,
            think_tag_style: ThinkTagStyle::Qwen,
            interleaved_think_tool: true,
            supports_parallel_tools: true,
            default_max_output_tokens: Some(8192),
            default_context_window: Some(8192),
            max_context_window: Some(262144),
            parser_type: ParserType::QwenStateMachine,
            renderer_type: RendererType::QwenChat,
            stop_markers: vec!["</tool_call>".into(), "</function>".into()],
            allow_fallback_extraction: true,
            default_presence_penalty: Some(1.5),
            default_temperature: Some(0.15),
            default_top_p: Some(1.0),
            default_top_k: Some(-1),
            default_min_p: Some(0.0),
            disable_thinking_for_tools: true,
            split_tool_calling: true,
        }
    }

    fn qwen3(m: &str) -> Self {
        Self {
            family: ModelFamily::Qwen3,
            tool_call_format: ToolCallFormat::QwenXml,
            think_tag_style: ThinkTagStyle::Standard,
            interleaved_think_tool: true,
            supports_parallel_tools: m.contains("14b") || m.contains("30b") || m.contains("32b"),
            default_max_output_tokens: Some(8192),
            default_context_window: Some(8192),
            max_context_window: Some(131072),
            parser_type: ParserType::QwenStateMachine,
            renderer_type: RendererType::QwenChat,
            stop_markers: vec!["</tool_call>".into(), "</function>".into()],
            allow_fallback_extraction: true,
            default_presence_penalty: Some(1.5),
            default_temperature: Some(0.15),
            default_top_p: Some(1.0),
            default_top_k: Some(-1),
            default_min_p: Some(0.0),
            disable_thinking_for_tools: true,
            split_tool_calling: true,
        }
    }

    fn qwen2_5() -> Self {
        Self {
            family: ModelFamily::Qwen2_5,
            tool_call_format: ToolCallFormat::NativeApi,
            think_tag_style: ThinkTagStyle::None,
            interleaved_think_tool: false,
            supports_parallel_tools: false,
            default_max_output_tokens: Some(4096),
            default_context_window: Some(8192),
            max_context_window: Some(32768),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![],
            allow_fallback_extraction: false,
            // Alibaba recommended non-thinking config for Qwen2.5 chat
            default_presence_penalty: Some(1.05),
            default_temperature: Some(0.7),
            default_top_p: Some(0.8),
            default_top_k: Some(20),
            default_min_p: None,
            disable_thinking_for_tools: false,
            split_tool_calling: false,
        }
    }

    fn deepseek_r1() -> Self {
        Self {
            family: ModelFamily::DeepSeekR1,
            tool_call_format: ToolCallFormat::NativeApi,
            think_tag_style: ThinkTagStyle::Standard,
            interleaved_think_tool: false,
            supports_parallel_tools: true,
            default_max_output_tokens: Some(8192),
            default_context_window: None,
            max_context_window: Some(131072),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![], // Let model finish reasoning naturally
            allow_fallback_extraction: false,
            // DeepSeek R1 always reasons; official recommended: temp=0.6, top_p=0.95
            // top_k deliberately disabled — limits hurt long reasoning chains
            default_presence_penalty: None,
            default_temperature: Some(0.6),
            default_top_p: Some(0.95),
            default_top_k: Some(-1),
            default_min_p: Some(0.0),
            disable_thinking_for_tools: false,
            split_tool_calling: false,
        }
    }

    fn llama3() -> Self {
        Self {
            family: ModelFamily::Llama3,
            tool_call_format: ToolCallFormat::NativeApi,
            think_tag_style: ThinkTagStyle::None,
            interleaved_think_tool: false,
            supports_parallel_tools: true,
            default_max_output_tokens: Some(4096),
            default_context_window: None,
            max_context_window: Some(131072),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::Llama3Chat,
            stop_markers: vec![],
            allow_fallback_extraction: false,
            // Meta recommended for Llama 3: temp=0.6, top_p=0.9
            // Presence penalty 1.05 helps avoid repetition
            default_presence_penalty: Some(1.05),
            default_temperature: Some(0.6),
            default_top_p: Some(0.9),
            default_top_k: Some(-1),
            default_min_p: None,
            disable_thinking_for_tools: false,
            split_tool_calling: false,
        }
    }

    fn phi() -> Self {
        Self {
            family: ModelFamily::Phi,
            tool_call_format: ToolCallFormat::NativeApi,
            think_tag_style: ThinkTagStyle::None,
            interleaved_think_tool: false,
            supports_parallel_tools: false,
            default_max_output_tokens: Some(4096),
            default_context_window: None,
            max_context_window: Some(131072),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![],
            allow_fallback_extraction: false,
            // Phi-3/4 recommended: temp=0.7, top_p=0.9 for balanced chat
            default_presence_penalty: None,
            default_temperature: Some(0.7),
            default_top_p: Some(0.9),
            default_top_k: Some(-1),
            default_min_p: None,
            disable_thinking_for_tools: false,
            split_tool_calling: true,
        }
    }

    fn mistral() -> Self {
        Self {
            family: ModelFamily::Mistral,
            tool_call_format: ToolCallFormat::NativeApi,
            think_tag_style: ThinkTagStyle::None,
            interleaved_think_tool: false,
            supports_parallel_tools: true,
            default_max_output_tokens: Some(4096),
            default_context_window: None,
            max_context_window: Some(32768),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![],
            allow_fallback_extraction: false,
            // Mistral recommended: temp=0.7, top_p=0.9
            // Presence penalty 1.05 reduces repetition in longer outputs
            default_presence_penalty: Some(1.05),
            default_temperature: Some(0.7),
            default_top_p: Some(0.9),
            default_top_k: Some(-1),
            default_min_p: None,
            disable_thinking_for_tools: false,
            split_tool_calling: false,
        }
    }

    fn generic() -> Self {
        Self {
            family: ModelFamily::Generic,
            tool_call_format: ToolCallFormat::NativeApi,
            think_tag_style: ThinkTagStyle::None,
            interleaved_think_tool: false,
            supports_parallel_tools: true,
            default_max_output_tokens: None,
            default_context_window: None,
            max_context_window: Some(131072),
            parser_type: ParserType::HermesFallback,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![],
            allow_fallback_extraction: false,
            // Conservative defaults that work well across most GGUF models
            default_presence_penalty: None,
            default_temperature: Some(0.7),
            default_top_p: Some(0.9),
            default_top_k: Some(-1),
            default_min_p: None,
            disable_thinking_for_tools: false,
            split_tool_calling: false,
        }
    }

    /// Returns true if this model uses think tags (any style).
    pub fn has_think_tags(&self) -> bool {
        !matches!(self.think_tag_style, ThinkTagStyle::None)
    }

    /// Returns a system-prompt suffix for think-tag guidance.
    pub fn think_guidance_suffix(&self) -> Option<&'static str> {
        match self.family {
            ModelFamily::Qwen3_5 | ModelFamily::Qwen3 => Some(
                "\n\n# Output Format (STRICT)\n\
                - You MUST produce a tool call OR a text response on EVERY turn\n\
                - If you reason in <think> tags, you MUST ALWAYS follow the closing </think> with either a tool call or a text answer\n\
                - NEVER stop after </think> — there must ALWAYS be content after it\n\
                - An empty response after reasoning is NEVER acceptable",
            ),
            _ => None,
        }
    }
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Qwen3_5 => write!(f, "Qwen3.5"),
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen2_5 => write!(f, "Qwen2.5"),
            Self::DeepSeekR1 => write!(f, "DeepSeek-R1"),
            Self::Llama3 => write!(f, "Llama3"),
            Self::Phi => write!(f, "Phi"),
            Self::Mistral => write!(f, "Mistral"),
            Self::Generic => write!(f, "Generic"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_qwen3_5() {
        let p = ModelProfile::detect("qwen3.5-35b-q4_k_m");
        assert_eq!(p.family, ModelFamily::Qwen3_5);
        assert_eq!(p.parser_type, ParserType::QwenStateMachine);
        assert!(p.disable_thinking_for_tools);
    }

    #[test]
    fn detect_generic() {
        let p = ModelProfile::detect("unknown-model-v2");
        assert_eq!(p.family, ModelFamily::Generic);
    }
}
