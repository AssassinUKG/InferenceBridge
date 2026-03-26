//! Model-specific template patches.
//!
//! These patches are applied after template rendering to fix known issues
//! with specific model families (e.g., Qwen think-tag handling).

use crate::models::profiles::{ModelFamily, ModelProfile};

/// Apply model-specific patches to a rendered prompt.
pub fn apply_patches(prompt: &str, profile: &ModelProfile) -> String {
    match profile.family {
        ModelFamily::Qwen3_5 | ModelFamily::Qwen3 => patch_qwen(prompt, profile),
        _ => prompt.to_string(),
    }
}

fn patch_qwen(prompt: &str, profile: &ModelProfile) -> String {
    let mut result = prompt.to_string();

    // Add think guidance suffix to the system message if present
    if let Some(suffix) = profile.think_guidance_suffix() {
        // Only if not already present
        if !result.contains("Output Format (STRICT)") {
            // Find the end of the first system message
            if let Some(sys_end) = result.find("<|im_end|>") {
                result.insert_str(sys_end, suffix);
            }
        }
    }

    result
}
