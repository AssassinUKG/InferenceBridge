//! Model-specific template patches.
//!
//! These patches are applied after template rendering to fix known issues
//! with specific model families (e.g., Qwen think-tag handling).

use crate::models::profiles::{ModelFamily, ModelProfile};

/// Apply model-specific patches to a rendered prompt.
pub fn apply_patches(prompt: &str, profile: &ModelProfile, has_tools: bool) -> String {
    match profile.family {
        ModelFamily::Qwen3_5 | ModelFamily::Qwen3 => patch_qwen(prompt, profile, has_tools),
        _ => prompt.to_string(),
    }
}

fn patch_qwen(prompt: &str, profile: &ModelProfile, has_tools: bool) -> String {
    let mut result = prompt.to_string();

    // Skip think guidance when tools are present and the profile disables thinking for tools.
    // The tool-schema system message already instructs the model not to emit <think> blocks;
    // adding the "NEVER stop after </think>" suffix would contradict it.
    let skip_think_guidance = has_tools && profile.disable_thinking_for_tools;

    if !skip_think_guidance {
        if let Some(suffix) = profile.think_guidance_suffix() {
            if !result.contains("Output Format (STRICT)") {
                if let Some(sys_end) = result.find("<|im_end|>") {
                    result.insert_str(sys_end, suffix);
                }
            }
        }
    }

    result
}
