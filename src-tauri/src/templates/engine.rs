use crate::models::profiles::{ModelProfile, RendererType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub fn render_prompt(messages: &[ChatMessage], profile: &ModelProfile) -> String {
    render_prompt_with_tools(messages, profile, false)
}

pub fn render_prompt_with_tools(
    messages: &[ChatMessage],
    profile: &ModelProfile,
    has_tools: bool,
) -> String {
    let rendered = match profile.renderer_type {
        RendererType::ChatML => render_chatml(messages),
        RendererType::QwenChat => render_chatml(messages),
        RendererType::Llama3Chat => render_llama3_chat(messages),
        RendererType::GemmaChat => render_gemma_chat(messages),
        RendererType::Gemma4Chat => render_gemma4_chat(messages),
    };

    super::patches::apply_patches(&rendered, profile, has_tools)
}

fn render_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn render_llama3_chat(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for msg in messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

fn render_gemma_chat(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

fn render_gemma4_chat(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = if msg.role == "assistant" {
            "model"
        } else {
            msg.role.as_str()
        };
        prompt.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, msg.content));
    }
    prompt.push_str("<|turn>model\n");
    prompt
}
