//! Template engine: render conversation messages into a raw prompt string.

use crate::models::profiles::{ModelProfile, RendererType};

/// A message in the conversation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Render a list of messages into a raw prompt string using the model's template.
pub fn render_prompt(messages: &[ChatMessage], profile: &ModelProfile) -> String {
    match profile.renderer_type {
        RendererType::ChatML => render_chatml(messages, profile),
        RendererType::QwenChat => render_qwen_chat(messages, profile),
        RendererType::Llama3Chat => render_llama3_chat(messages, profile),
    }
}

fn render_chatml(messages: &[ChatMessage], profile: &ModelProfile) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    // Apply think guidance if available
    if let Some(suffix) = profile.think_guidance_suffix() {
        // Find last system message and append suffix
        if let Some(pos) = prompt.rfind("<|im_start|>system\n") {
            if let Some(end) = prompt[pos..].find("<|im_end|>") {
                let insert_pos = pos + end;
                prompt.insert_str(insert_pos, suffix);
            }
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn render_qwen_chat(messages: &[ChatMessage], profile: &ModelProfile) -> String {
    // Qwen uses ChatML format but with specific handling for tool calls
    // and think-tag guidance
    render_chatml(messages, profile)
}

fn render_llama3_chat(messages: &[ChatMessage], _profile: &ModelProfile) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|begin_of_text|>");
    for msg in messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}
