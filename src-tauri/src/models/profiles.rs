use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFamily {
    Qwen3_5,
    Qwen3,
    Qwen2_5,
    DeepSeekR1,
    Llama3,
    Phi,
    Mistral,
    Gemma,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallFormat {
    NativeApi,
    HermesXml,
    QwenXml,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkTagStyle {
    None,
    Standard,
    Qwen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParserType {
    NativeApi,
    HermesFallback,
    QwenStateMachine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RendererType {
    ChatML,
    QwenChat,
    Llama3Chat,
    GemmaChat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub family: ModelFamily,
    pub tool_call_format: ToolCallFormat,
    pub think_tag_style: ThinkTagStyle,
    pub interleaved_think_tool: bool,
    pub supports_parallel_tools: bool,
    pub supports_vision: bool,
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
    pub fn detect(model_name: &str) -> Self {
        let lower = model_name.to_lowercase();
        let supports_vision = Self::infer_vision_support(&lower);

        let mut profile = if lower.contains("qwen3.5") {
            Self::qwen3_5()
        } else if lower.contains("qwen3") {
            Self::qwen3(&lower)
        } else if lower.contains("qwen2.5") {
            Self::qwen2_5()
        } else if lower.contains("deepseek") && (lower.contains("r1") || lower.contains("reasoning")) {
            Self::deepseek_r1()
        } else if lower.contains("gemma") {
            Self::gemma()
        } else if lower.contains("llama") && (lower.contains("3.") || lower.contains("3-") || lower.contains("3:")) {
            Self::llama3()
        } else if lower.contains("phi-3")
            || lower.contains("phi-4")
            || lower.contains("phi3")
            || lower.contains("phi4")
        {
            Self::phi()
        } else if lower.contains("mistral") || lower.contains("mixtral") || lower.contains("nemo") {
            Self::mistral()
        } else {
            Self::generic()
        };

        profile.supports_vision = supports_vision;
        profile
    }

    fn infer_vision_support(model_name: &str) -> bool {
        let normalized = model_name.replace('_', "-");
        model_name.contains("vision")
            || model_name.contains("llava")
            || model_name.contains("multimodal")
            || model_name.contains("qwen2.5-vl")
            || model_name.contains("-vl")
            || model_name.contains("_vl")
            // Qwen3.5 35B-A3B multimodal variants are commonly published as plain GGUF
            // filenames without an explicit `vision` or `-vl` marker.
            || normalized.contains("qwen3.5-35b-a3b")
    }

    fn qwen3_5() -> Self {
        Self {
            family: ModelFamily::Qwen3_5,
            tool_call_format: ToolCallFormat::QwenXml,
            think_tag_style: ThinkTagStyle::Qwen,
            interleaved_think_tool: true,
            supports_parallel_tools: true,
            supports_vision: false,
            default_max_output_tokens: Some(8192),
            default_context_window: Some(8192),
            max_context_window: Some(262144),
            parser_type: ParserType::QwenStateMachine,
            renderer_type: RendererType::QwenChat,
            stop_markers: vec!["</tool_call>".into(), "</function>".into()],
            allow_fallback_extraction: true,
            default_presence_penalty: Some(1.5),
            default_temperature: Some(0.6),
            default_top_p: Some(0.95),
            default_top_k: Some(20),
            default_min_p: Some(0.0),
            disable_thinking_for_tools: true,
            split_tool_calling: true,
        }
    }

    fn qwen3(model_name: &str) -> Self {
        Self {
            family: ModelFamily::Qwen3,
            tool_call_format: ToolCallFormat::QwenXml,
            think_tag_style: ThinkTagStyle::Standard,
            interleaved_think_tool: true,
            supports_parallel_tools: model_name.contains("14b")
                || model_name.contains("30b")
                || model_name.contains("32b"),
            supports_vision: false,
            default_max_output_tokens: Some(8192),
            default_context_window: Some(8192),
            max_context_window: Some(131072),
            parser_type: ParserType::QwenStateMachine,
            renderer_type: RendererType::QwenChat,
            stop_markers: vec!["</tool_call>".into(), "</function>".into()],
            allow_fallback_extraction: true,
            default_presence_penalty: Some(1.5),
            default_temperature: Some(0.6),
            default_top_p: Some(0.95),
            default_top_k: Some(20),
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
            supports_vision: false,
            default_max_output_tokens: Some(4096),
            default_context_window: Some(8192),
            max_context_window: Some(32768),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![],
            allow_fallback_extraction: false,
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
            supports_vision: false,
            default_max_output_tokens: Some(8192),
            default_context_window: None,
            max_context_window: Some(131072),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![],
            allow_fallback_extraction: false,
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
            supports_vision: false,
            default_max_output_tokens: Some(4096),
            default_context_window: None,
            max_context_window: Some(131072),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::Llama3Chat,
            stop_markers: vec![],
            allow_fallback_extraction: false,
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
            supports_vision: false,
            default_max_output_tokens: Some(4096),
            default_context_window: None,
            max_context_window: Some(131072),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![],
            allow_fallback_extraction: false,
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
            supports_vision: false,
            default_max_output_tokens: Some(4096),
            default_context_window: None,
            max_context_window: Some(32768),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![],
            allow_fallback_extraction: false,
            default_presence_penalty: Some(1.05),
            default_temperature: Some(0.7),
            default_top_p: Some(0.9),
            default_top_k: Some(-1),
            default_min_p: None,
            disable_thinking_for_tools: false,
            split_tool_calling: false,
        }
    }

    fn gemma() -> Self {
        Self {
            family: ModelFamily::Gemma,
            tool_call_format: ToolCallFormat::NativeApi,
            think_tag_style: ThinkTagStyle::None,
            interleaved_think_tool: false,
            supports_parallel_tools: false,
            supports_vision: false,
            default_max_output_tokens: Some(4096),
            default_context_window: Some(8192),
            max_context_window: Some(131072),
            parser_type: ParserType::NativeApi,
            renderer_type: RendererType::GemmaChat,
            stop_markers: vec![],
            allow_fallback_extraction: false,
            default_presence_penalty: Some(1.0),
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
            supports_vision: false,
            default_max_output_tokens: None,
            default_context_window: None,
            max_context_window: Some(131072),
            parser_type: ParserType::HermesFallback,
            renderer_type: RendererType::ChatML,
            stop_markers: vec![],
            allow_fallback_extraction: false,
            default_presence_penalty: None,
            default_temperature: Some(0.7),
            default_top_p: Some(0.9),
            default_top_k: Some(-1),
            default_min_p: None,
            disable_thinking_for_tools: false,
            split_tool_calling: false,
        }
    }

    pub fn has_think_tags(&self) -> bool {
        !matches!(self.think_tag_style, ThinkTagStyle::None)
    }

    pub fn think_guidance_suffix(&self) -> Option<&'static str> {
        match self.family {
            ModelFamily::Qwen3_5 | ModelFamily::Qwen3 => Some(
                "\n\n# Output Format (STRICT)\n\
                - You MUST produce a tool call OR a text response on EVERY turn\n\
                - If you reason in <think> tags, you MUST ALWAYS follow the closing </think> with either a tool call or a text answer\n\
                - NEVER stop after </think>; there must ALWAYS be content after it\n\
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
            Self::Gemma => write!(f, "Gemma"),
            Self::Generic => write!(f, "Generic"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_qwen3_5() {
        let profile = ModelProfile::detect("qwen3.5-35b-q4_k_m");
        assert_eq!(profile.family, ModelFamily::Qwen3_5);
        assert_eq!(profile.parser_type, ParserType::QwenStateMachine);
        assert!(profile.disable_thinking_for_tools);
    }

    #[test]
    fn detect_gemma() {
        let profile = ModelProfile::detect("gemma-3-12b-it-q4_k_m.gguf");
        assert_eq!(profile.family, ModelFamily::Gemma);
        assert_eq!(profile.renderer_type, RendererType::GemmaChat);
    }

    #[test]
    fn detect_vision_support() {
        let profile = ModelProfile::detect("qwen2.5-vl-7b-instruct-q4.gguf");
        assert!(profile.supports_vision);
    }

    #[test]
    fn detect_qwen3_5_35b_a3b_as_vision_capable() {
        let profile = ModelProfile::detect("Qwen3.5-35B-A3B-Q4_K_M.gguf");
        assert!(profile.supports_vision);
    }

    #[test]
    fn detect_qwen3_5_35b_a3b_with_repo_style_tokens_as_vision_capable() {
        let profile =
            ModelProfile::detect("HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive");
        assert!(profile.supports_vision);
    }
}
