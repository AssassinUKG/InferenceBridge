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
    DiffusionGemma,
    Gemma4,
    Gemma,
    Generic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallFormat {
    NativeApi,
    HermesXml,
    QwenXml,
    Gemma4Native,
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
    Gemma4StateMachine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RendererType {
    ChatML,
    QwenChat,
    Llama3Chat,
    GemmaChat,
    Gemma4Chat,
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
}

impl ModelProfile {
    pub fn detect(model_name: &str) -> Self {
        Self::detect_with_arch(model_name, None)
    }

    /// Detect the model profile, using the GGUF `general.architecture` string as
    /// a fallback when the filename alone doesn't match a known family.
    ///
    /// The filename still wins when it carries finer-grained signal (e.g.
    /// `qwen3.5` vs `qwen3`, or a `-vl` vision marker). The architecture only
    /// rescues models whose names are unrecognized — so a renamed or obscurely
    /// named GGUF gets a real profile instead of falling through to `Generic`.
    pub fn detect_with_arch(model_name: &str, architecture: Option<&str>) -> Self {
        let mut profile = Self::detect_by_name(model_name);

        // If the filename told us nothing, try to recover the family from the
        // embedded architecture identifier.
        if profile.family == ModelFamily::Generic {
            if let Some(arch_profile) = architecture.and_then(Self::detect_by_architecture) {
                let preserved_vision = profile.supports_vision;
                profile = arch_profile;
                profile.supports_vision = profile.supports_vision || preserved_vision;
            }
        }

        profile
    }

    /// Map a GGUF `general.architecture` value to a base profile. Conservative:
    /// only maps to coarse families (the filename handles finer variants).
    fn detect_by_architecture(architecture: &str) -> Option<Self> {
        let arch = architecture.to_lowercase();
        let profile = if arch.starts_with("qwen3") {
            Self::qwen3(&arch)
        } else if arch.starts_with("qwen2") || arch.starts_with("qwen") {
            Self::qwen2_5()
        } else if arch.starts_with("gemma") {
            Self::gemma()
        } else if arch.starts_with("phi") {
            Self::phi()
        } else if arch.starts_with("mistral") {
            Self::mistral()
        } else if arch.starts_with("llama") {
            // Note: many Mistral/derivative GGUFs also report `llama` here; the
            // filename matcher catches those before we reach this fallback.
            Self::llama3()
        } else {
            return None;
        };
        Some(profile)
    }

    fn detect_by_name(model_name: &str) -> Self {
        let lower = model_name.to_lowercase();
        let supports_vision = Self::infer_vision_support(&lower);

        let mut profile = if lower.contains("qwen3.5") {
            Self::qwen3_5()
        } else if lower.contains("qwen3") {
            Self::qwen3(&lower)
        } else if lower.contains("qwen2.5") {
            Self::qwen2_5()
        } else if lower.contains("deepseek")
            && (lower.contains("r1") || lower.contains("reasoning"))
        {
            Self::deepseek_r1()
        } else if lower.contains("diffusiongemma") || lower.contains("diffusion-gemma") {
            Self::diffusion_gemma()
        } else if lower.contains("gemma-4") || lower.contains("gemma4") {
            Self::gemma4()
        } else if lower.contains("gemma") {
            Self::gemma()
        } else if lower.contains("llama")
            && (lower.contains("3.")
                || lower.contains("3-")
                || lower.contains("3:")
                || lower.contains("3_"))
        {
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

        profile.supports_vision = profile.supports_vision || supports_vision;
        profile
    }

    fn infer_vision_support(model_name: &str) -> bool {
        let lower = model_name.to_lowercase();
        let normalized = lower.replace('_', "-");
        lower.contains("vision")
            || lower.contains("llava")
            || lower.contains("multimodal")
            || lower.contains("-vl")
            || lower.contains("_vl")
            // MiniCPM-V series
            || lower.contains("minicpm-v")
            || lower.contains("minicpmv")
            // InternVL series
            || lower.contains("internvl")
            // Moondream
            || lower.contains("moondream")
            // SmolVLM
            || lower.contains("smolvlm")
            // Pixtral (Mistral vision)
            || lower.contains("pixtral")
            // BakLLaVA
            || lower.contains("bakllava")
            // CogVLM
            || lower.contains("cogvlm")
            // Idefics
            || lower.contains("idefics")
            // Qwen2-VL and Qwen2.5-VL (also caught by -vl above, but explicit for clarity)
            || normalized.contains("qwen2-vl")
            // Qwen3.5 35B-A3B multimodal variants are commonly published as plain GGUF
            // filenames without an explicit `vision` or `-vl` marker.
            || normalized.contains("qwen3.5-35b-a3b")
            // DiffusionGemma is a multimodal diffusion model with image understanding support.
            || normalized.contains("diffusiongemma")
            || normalized.contains("diffusion-gemma")
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
            default_context_window: None, // let llama-server use model metadata
            max_context_window: Some(262144),
            parser_type: ParserType::QwenStateMachine,
            renderer_type: RendererType::QwenChat,
            stop_markers: vec!["</tool_call>".into(), "</function>".into()],
            allow_fallback_extraction: true,
            default_presence_penalty: Some(1.3),
            default_temperature: Some(0.7),
            default_top_p: Some(0.8),
            default_top_k: Some(20),
            default_min_p: Some(0.05),
            disable_thinking_for_tools: true,
        }
    }

    fn qwen3(model_name: &str) -> Self {
        Self {
            family: ModelFamily::Qwen3,
            tool_call_format: ToolCallFormat::QwenXml,
            think_tag_style: ThinkTagStyle::Qwen,
            interleaved_think_tool: true,
            supports_parallel_tools: model_name.contains("14b")
                || model_name.contains("27b")
                || model_name.contains("30b")
                || model_name.contains("32b"),
            supports_vision: false,
            default_max_output_tokens: Some(8192),
            default_context_window: None, // let llama-server use model metadata
            // GGUF context_length overrides this at runtime; 262144 is the
            // native max for Qwen3.6-generation models (Unsloth/HF model cards).
            max_context_window: Some(262144),
            parser_type: ParserType::QwenStateMachine,
            renderer_type: RendererType::QwenChat,
            stop_markers: vec!["</tool_call>".into(), "</function>".into()],
            allow_fallback_extraction: true,
            default_presence_penalty: Some(1.5),
            // Qwen3.6 model card: thinking general temp=1.0/top_p=0.95/top_k=20,
            // coding temp=0.6/top_p=0.95/top_k=20, instruct/non-thinking
            // temp=0.7/top_p=0.8/top_k=20/presence_penalty=1.5.
            // We use the instruct defaults as the general case.
            default_temperature: Some(0.7),
            default_top_p: Some(0.8),
            default_top_k: Some(20),
            default_min_p: Some(0.0),
            disable_thinking_for_tools: true,
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
            default_context_window: None, // let llama-server use model metadata
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
            default_context_window: None, // let llama-server use model metadata
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
        }
    }

    fn gemma4() -> Self {
        Self {
            family: ModelFamily::Gemma4,
            tool_call_format: ToolCallFormat::Gemma4Native,
            think_tag_style: ThinkTagStyle::None,
            interleaved_think_tool: true,
            supports_parallel_tools: false,
            supports_vision: true,
            default_max_output_tokens: Some(8192),
            default_context_window: None, // let llama-server use model metadata
            max_context_window: Some(262144),
            parser_type: ParserType::Gemma4StateMachine,
            renderer_type: RendererType::Gemma4Chat,
            stop_markers: vec![
                "<turn|>".into(),
                "<|turn>".into(),
                "<end_of_turn>".into(),
                "<start_of_turn>".into(),
                "<tool_call|>".into(),
            ],
            allow_fallback_extraction: true,
            default_presence_penalty: Some(1.0),
            default_temperature: Some(1.0),
            default_top_p: Some(0.95),
            default_top_k: Some(64),
            default_min_p: Some(0.0),
            disable_thinking_for_tools: true,
        }
    }

    fn diffusion_gemma() -> Self {
        Self {
            family: ModelFamily::DiffusionGemma,
            tool_call_format: ToolCallFormat::Gemma4Native,
            think_tag_style: ThinkTagStyle::None,
            interleaved_think_tool: true,
            supports_parallel_tools: false,
            supports_vision: true,
            default_max_output_tokens: Some(2048),
            default_context_window: None, // let the diffusion runner use model metadata
            max_context_window: Some(262144),
            parser_type: ParserType::Gemma4StateMachine,
            renderer_type: RendererType::Gemma4Chat,
            stop_markers: vec![
                "<turn|>".into(),
                "<|turn>".into(),
                "<end_of_turn>".into(),
                "<start_of_turn>".into(),
                "<tool_call|>".into(),
            ],
            allow_fallback_extraction: true,
            default_presence_penalty: Some(1.0),
            default_temperature: Some(1.0),
            default_top_p: Some(0.95),
            default_top_k: Some(64),
            default_min_p: Some(0.0),
            disable_thinking_for_tools: true,
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
            Self::DiffusionGemma => write!(f, "DiffusionGemma"),
            Self::Gemma4 => write!(f, "Gemma4"),
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
    fn detect_gemma4_12b() {
        let profile = ModelProfile::detect("google/gemma-4-12B-it");
        assert_eq!(profile.family, ModelFamily::Gemma4);
        assert_eq!(profile.renderer_type, RendererType::Gemma4Chat);
        assert_eq!(profile.parser_type, ParserType::Gemma4StateMachine);
        assert_eq!(profile.tool_call_format, ToolCallFormat::Gemma4Native);
        assert!(profile.supports_vision);
        assert_eq!(profile.max_context_window, Some(262144));
    }

    #[test]
    fn detect_diffusion_gemma_before_regular_gemma() {
        let profile = ModelProfile::detect(
            "unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        );
        assert_eq!(profile.family, ModelFamily::DiffusionGemma);
        assert_eq!(profile.renderer_type, RendererType::Gemma4Chat);
        assert_eq!(profile.parser_type, ParserType::Gemma4StateMachine);
        assert!(profile.supports_vision);
        assert_eq!(profile.max_context_window, Some(262144));
        assert_eq!(profile.default_max_output_tokens, Some(2048));
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

    #[test]
    fn arch_rescues_unknown_filename() {
        // A renamed file the name matcher can't place would be Generic...
        assert_eq!(
            ModelProfile::detect("my-cool-merge-v2-q4_k_m.gguf").family,
            ModelFamily::Generic
        );
        // ...but the embedded architecture recovers the real family.
        let profile = ModelProfile::detect_with_arch("my-cool-merge-v2-q4_k_m.gguf", Some("qwen3"));
        assert_eq!(profile.family, ModelFamily::Qwen3);
        assert_eq!(profile.parser_type, ParserType::QwenStateMachine);
    }

    #[test]
    fn name_wins_over_arch_when_recognized() {
        // Filename says qwen3.5 (finer than the `qwen3` arch string); name wins.
        let profile = ModelProfile::detect_with_arch("Qwen3.5-35B-A3B-Q4_K_M.gguf", Some("qwen3"));
        assert_eq!(profile.family, ModelFamily::Qwen3_5);
    }

    #[test]
    fn arch_maps_common_families() {
        assert_eq!(
            ModelProfile::detect_with_arch("unknown.gguf", Some("gemma2")).family,
            ModelFamily::Gemma
        );
        assert_eq!(
            ModelProfile::detect_with_arch("unknown.gguf", Some("phi3")).family,
            ModelFamily::Phi
        );
        assert_eq!(
            ModelProfile::detect_with_arch("unknown.gguf", Some("llama")).family,
            ModelFamily::Llama3
        );
        // Truly unknown architecture stays Generic.
        assert_eq!(
            ModelProfile::detect_with_arch("unknown.gguf", Some("rwkv")).family,
            ModelFamily::Generic
        );
    }
}
