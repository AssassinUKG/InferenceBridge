use crate::engine::client::{CompletionRequest, LlamaClient};
use crate::models::profiles::ModelProfile;
use crate::state::SharedState;
use crate::templates::engine::{render_prompt, ChatMessage};

pub fn compress_messages(messages: &[(String, String)]) -> String {
    if messages.is_empty() {
        return String::new();
    }

    let mut summary = String::from("[Earlier conversation summary]\n");
    for (role, content) in messages {
        let trimmed = content.trim();
        let preview = if trimmed.chars().count() > 200 {
            format!("{}...", trimmed.chars().take(200).collect::<String>())
        } else {
            trimmed.to_string()
        };
        summary.push_str(&format!("{role}: {preview}\n"));
    }
    summary
}

pub async fn summarize_message_pairs_with_client(
    client: &LlamaClient,
    profile: &ModelProfile,
    messages: &[(String, String)],
) -> Option<String> {
    if messages.is_empty() {
        return None;
    }

    let prompt_messages = summarization_prompt_messages(messages);
    let prompt = render_prompt(&prompt_messages, profile);
    let request = CompletionRequest {
        prompt,
        n_predict: Some(384),
        temperature: Some(0.2),
        top_p: Some(0.9),
        top_k: Some(-1),
        min_p: profile.default_min_p,
        presence_penalty: None,
        frequency_penalty: None,
        repeat_penalty: None,
        seed: None,
        stream: false,
        stop: profile.stop_markers.clone(),
        special: true,
        image_data: vec![],
        grammar: None,
        json_schema: None,
    };

    match client.complete(&request).await {
        Ok(response) => {
            let summary = crate::normalize::think_strip::strip_think_tags(&response.content);
            non_empty_summary(summary.trim())
        }
        Err(error) => {
            tracing::warn!(error = %error, "Context compaction summarization request failed");
            None
        }
    }
}

fn non_empty_summary(summary: &str) -> Option<String> {
    (!summary.trim().is_empty()).then(|| summary.trim().to_string())
}

fn summarization_prompt_messages(messages: &[(String, String)]) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: "system".to_string(),
            content: "Summarize the earlier conversation into a compact memory that preserves facts, decisions, unresolved questions, and user preferences. Keep it concise and factual.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: messages
                .iter()
                .map(|(role, content)| format!("{role}: {content}"))
                .collect::<Vec<_>>()
                .join("\n\n"),
        },
    ]
}

pub async fn summarize_messages_with_model(
    state: SharedState,
    session_id: &str,
    messages: &[(String, String)],
) -> Result<Option<String>, String> {
    if messages.is_empty() {
        return Ok(None);
    }

    {
        let s = state.read().await;
        if s.active_generation.is_some() {
            tracing::info!("Skipping context summarization because a generation is active");
            return Ok(None);
        }
    }

    let (model_name, port) = {
        let s = state.read().await;
        let Some(model_name) = s.loaded_model.clone() else {
            return Ok(None);
        };
        (model_name, s.process.port())
    };

    let profile = crate::models::overrides::detect_effective_profile(&model_name);
    let prompt_messages = summarization_prompt_messages(messages);
    let prompt = render_prompt(&prompt_messages, &profile);

    let request = CompletionRequest {
        prompt,
        n_predict: Some(384),
        temperature: Some(0.2),
        top_p: Some(0.9),
        top_k: Some(-1),
        min_p: profile.default_min_p,
        presence_penalty: None,
        frequency_penalty: None,
        repeat_penalty: None,
        seed: None,
        stream: false,
        stop: profile.stop_markers.clone(),
        special: true,
        image_data: vec![],
        grammar: None,
        json_schema: None,
    };

    let client = LlamaClient::new(port);
    match client.complete(&request).await {
        Ok(response) => {
            let summary = crate::normalize::think_strip::strip_think_tags(&response.content);
            if summary.trim().is_empty() {
                return Ok(None);
            }

            let token_estimate = (summary.len() as u32 / 4).max(1);
            let s = state.read().await;
            let db = s.session_db.lock().map_err(|e| e.to_string())?;
            db.add_context_snapshot(session_id, &summary, token_estimate)
                .map_err(|e| e.to_string())?;
            Ok(Some(summary))
        }
        Err(error) => {
            tracing::warn!(error = %error, "Context summarization request failed");
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::compress_messages;

    #[test]
    fn compress_messages_truncates_on_char_boundary() {
        let content = format!("{}😀 trailing text", "x".repeat(199));
        let summary = compress_messages(&[("tool".to_string(), content)]);

        assert!(summary.contains("[Earlier conversation summary]"));
        assert!(summary.contains("😀"));
    }
}
