use tauri::Emitter;

use crate::context::{compressor, strategy, tracker};
use crate::engine::client::{CompletionRequest, ImageData, LlamaClient};
use crate::engine::streaming::{self, StreamEvent};
use crate::models::profiles::{ModelFamily, ModelProfile};
use crate::normalize::images::normalize_inline_image_payload;
use crate::session::db::MessageInfo;
use crate::state::{
    append_live_stream_delta_for_request, begin_live_generation, cancel_all_generations,
    finish_generation_for_request, GenerationHandle, SharedState,
};
use crate::templates::engine::{render_prompt, ChatMessage};
use serde_json::json;

#[derive(Debug, Clone)]
struct PreparedMessage {
    role: String,
    content: String,
    image_base64: Option<String>,
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn launch_preview_matches_model(model_name: &str, preview_model_path: &str) -> bool {
    let requested = model_name.trim().to_ascii_lowercase();
    let preview_name = std::path::Path::new(preview_model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(preview_model_path)
        .trim()
        .to_ascii_lowercase();

    preview_name == requested
        || preview_name.trim_end_matches(".gguf") == requested
        || preview_name == requested.trim_end_matches(".gguf")
        || (!requested.is_empty() && preview_name.contains(&requested))
}

fn apply_thinking_preference(
    mut messages: Vec<PreparedMessage>,
    profile: &ModelProfile,
    show_thinking: Option<bool>,
) -> Vec<PreparedMessage> {
    if profile.family == ModelFamily::Gemma4 {
        match show_thinking {
            Some(true) => {
                if let Some(system) = messages.iter_mut().find(|message| message.role == "system") {
                    if !system.content.trim_start().starts_with("<|think|>") {
                        system.content = format!("<|think|>\n{}", system.content);
                    }
                } else {
                    messages.insert(
                        0,
                        PreparedMessage {
                            role: "system".to_string(),
                            content:
                                "<|think|>\nYou may use internal reasoning before the final answer."
                                    .to_string(),
                            image_base64: None,
                        },
                    );
                }
            }
            Some(false) => {
                messages.insert(
                    0,
                    PreparedMessage {
                        role: "system".to_string(),
                        content: "Respond directly. Do not emit hidden reasoning or thought-channel content.".to_string(),
                        image_base64: None,
                    },
                );
            }
            None => {}
        }
        return messages;
    }

    let Some(last) = messages.last_mut().filter(|message| message.role == "user") else {
        return messages;
    };

    match (profile.family, show_thinking) {
        // Qwen3.5/Tess reasoning is selected at llama-server launch with
        // --reasoning. The GUI switch is display-only and must never rewrite
        // the native embedded-template input.
        (ModelFamily::Qwen3_5, _) => {}
        (ModelFamily::Qwen3, Some(true)) => {}
        (ModelFamily::Qwen3, Some(false)) => {
            last.content = format!(
                "Respond directly without emitting <think> blocks. Return only the final answer.\n\n{}",
                last.content
            );
        }
        (_, Some(true)) if profile.has_think_tags() => {
            last.content = format!("/think\n{}", last.content);
        }
        (_, Some(false)) if profile.has_think_tags() => {
            last.content = format!("/no_think\n{}", last.content);
        }
        _ => {}
    }

    messages
}

fn prepare_messages(
    db_messages: &[MessageInfo],
    profile: &ModelProfile,
    show_thinking: Option<bool>,
) -> Vec<PreparedMessage> {
    let messages = db_messages
        .iter()
        .map(|message| {
            let image_base64 = message.image_base64.as_deref().and_then(|image| {
                match normalize_inline_image_payload(image) {
                    Ok(normalized) => Some(normalized),
                    Err(error) => {
                        tracing::warn!(message_id = message.id, %error, "Skipping invalid stored image payload");
                        None
                    }
                }
            });

            PreparedMessage {
                role: message.role.clone(),
                content: message.content.clone().unwrap_or_default(),
                image_base64,
            }
        })
        .collect();

    apply_thinking_preference(messages, profile, show_thinking)
}

async fn begin_generation(
    state: &SharedState,
    source: &str,
    session_id: Option<String>,
    model: String,
) -> GenerationHandle {
    begin_live_generation(state, source, session_id, model).await
}

fn build_openai_content_parts(text: &str, image_base64: Option<&str>) -> serde_json::Value {
    let mut parts = Vec::new();
    if !text.trim().is_empty() {
        parts.push(json!({
            "type": "text",
            "text": text,
        }));
    }
    if let Some(image_base64) = image_base64 {
        parts.push(json!({
            "type": "image_url",
            "image_url": {
                "url": format!("data:image/png;base64,{image_base64}"),
            },
        }));
    }
    serde_json::Value::Array(parts)
}

fn build_openai_messages(messages: &[PreparedMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|message| {
            json!({
                "role": message.role,
                "content": if let Some(image_base64) = message.image_base64.as_deref() {
                    build_openai_content_parts(
                        &strip_legacy_image_markers(&message.content),
                        Some(image_base64),
                    )
                } else {
                    serde_json::Value::String(message.content.clone())
                },
            })
        })
        .collect()
}

fn prepared_to_api_messages(
    messages: &[PreparedMessage],
) -> Vec<crate::api::completions::ApiMessage> {
    messages
        .iter()
        .map(|message| {
            let content = if let Some(image_base64) = message.image_base64.as_ref() {
                let mut parts = Vec::new();
                let text = strip_legacy_image_markers(&message.content);
                if !text.trim().is_empty() {
                    parts.push(crate::api::completions::ApiContentPart::Text { text });
                }
                parts.push(crate::api::completions::ApiContentPart::ImageUrl {
                    image_url: crate::api::completions::ApiImageUrl::Object {
                        url: format!("data:image/png;base64,{image_base64}"),
                    },
                });
                crate::api::completions::ApiMessageContent::Parts(parts)
            } else {
                crate::api::completions::ApiMessageContent::Text(message.content.clone())
            };
            crate::api::completions::ApiMessage {
                role: message.role.clone(),
                content: Some(content),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                refusal: None,
            }
        })
        .collect()
}

fn strip_legacy_image_markers(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with("[img-") && trimmed.ends_with(']'))
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn openai_response_text(value: &serde_json::Value) -> String {
    value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .unwrap_or_default()
        .to_string()
}

fn openai_usage_tokens(value: &serde_json::Value, key: &str) -> Option<u32> {
    value
        .get("usage")
        .and_then(|usage| usage.get(key))
        .and_then(|tokens| tokens.as_u64())
        .and_then(|tokens| u32::try_from(tokens).ok())
}

#[cfg(test)]
mod tests {
    use super::{
        build_openai_content_parts, build_openai_messages, prepare_messages,
        prepared_to_api_messages, strip_legacy_image_markers,
    };
    use crate::models::profiles::ModelProfile;
    use crate::session::db::MessageInfo;

    fn message(id: i64, role: &str, content: &str, image_base64: Option<&str>) -> MessageInfo {
        MessageInfo {
            id,
            role: role.to_string(),
            content: Some(content.to_string()),
            display_content: None,
            reasoning_content: None,
            image_base64: image_base64.map(str::to_string),
            image_path: None,
            image_metadata: None,
            token_count: Some(0),
            tokens_evaluated: None,
            tokens_predicted: None,
            created_at: "2026-07-14T00:00:00Z".to_string(),
            tool_calls: Vec::new(),
        }
    }

    #[test]
    fn strips_legacy_image_markers_for_openai_image_parts() {
        assert_eq!(
            strip_legacy_image_markers("[img-1]\nwhat is this?"),
            "what is this?"
        );
    }

    #[test]
    fn builds_openai_image_content_parts_without_marker_text() {
        let parts = build_openai_content_parts("what is this?", Some("QUFB"));
        let array = parts.as_array().expect("content should be parts array");
        assert_eq!(array.len(), 2);
        assert_eq!(array[0]["type"], "text");
        assert_eq!(array[1]["type"], "image_url");
    }

    #[test]
    fn gemma_thinking_guidance_does_not_shift_pasted_image_to_assistant() {
        let history = vec![
            message(1, "user", "hello", None),
            message(2, "assistant", "Hello!", None),
            message(3, "user", "how are you today?", None),
            message(4, "assistant", "I'm doing great.", None),
            message(
                -1,
                "user",
                "what is in this image?",
                Some("data:image/png;base64,QUFBQQ=="),
            ),
        ];
        let profile = ModelProfile::detect("Gemma4-26B-A4B-QAT-Q4_K_M.gguf");

        let prepared = prepare_messages(&history, &profile, Some(false));
        let openai_messages = build_openai_messages(&prepared);

        assert_eq!(openai_messages.len(), history.len() + 1);
        let roles = openai_messages
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            vec!["system", "user", "assistant", "user", "assistant", "user"]
        );
        assert_eq!(openai_messages[1]["content"], "hello");
        assert_eq!(openai_messages[2]["content"], "Hello!");
        assert_eq!(openai_messages[3]["content"], "how are you today?");
        assert_eq!(openai_messages[4]["content"], "I'm doing great.");

        let image_owners = openai_messages
            .iter()
            .filter(|message| {
                message["content"]
                    .as_array()
                    .is_some_and(|parts| parts.iter().any(|part| part["type"] == "image_url"))
            })
            .collect::<Vec<_>>();
        assert_eq!(image_owners.len(), 1);
        assert_eq!(image_owners[0]["role"], "user");

        let final_parts = openai_messages.last().unwrap()["content"]
            .as_array()
            .expect("final user message should contain multimodal parts");
        assert_eq!(final_parts[0]["type"], "text");
        assert_eq!(final_parts[0]["text"], "what is in this image?");
        assert_eq!(final_parts[1]["type"], "image_url");
        assert_eq!(
            final_parts[1]["image_url"]["url"],
            "data:image/png;base64,QUFBQQ=="
        );
        assert!(openai_messages
            .iter()
            .filter(|message| message["role"] == "assistant")
            .all(|message| message["content"].is_string()));
    }

    #[test]
    fn qwen35_show_thinking_is_display_only() {
        let history = vec![message(1, "user", "Explain this directly", None)];
        let profile = ModelProfile::detect("Tess-4-27B-Q4_K_M.gguf");

        let hidden = prepare_messages(&history, &profile, Some(false));
        let shown = prepare_messages(&history, &profile, Some(true));

        assert_eq!(hidden[0].content, "Explain this directly");
        assert_eq!(shown[0].content, "Explain this directly");
    }

    #[test]
    fn qwen35_native_history_retains_older_image_for_readiness_checks() {
        let history = vec![
            message(
                1,
                "user",
                "inspect this",
                Some("data:image/png;base64,QUFBQQ=="),
            ),
            message(2, "assistant", "I can see it.", None),
            message(3, "user", "now explain the result", None),
        ];
        let profile = ModelProfile::detect("Tess-4-27B-Q4_K_M.gguf");
        let prepared = prepare_messages(&history, &profile, Some(false));

        assert!(prepared
            .iter()
            .any(|message| message.image_base64.is_some()));
        let native = prepared_to_api_messages(&prepared);
        assert!(crate::api::completions::api_messages_have_images(&native));
    }
}

async fn schedule_context_follow_up(state: SharedState, session_id: String) {
    let (loaded, running, port, app_handle) = {
        let s = state.read().await;
        (
            s.loaded_model.is_some(),
            matches!(
                s.process.state(),
                crate::engine::process::ProcessState::Running
            ),
            s.process.port(),
            s.app_handle.clone(),
        )
    };

    if !loaded || !running {
        return;
    }

    let client = LlamaClient::new(port);
    let mut status = tracker::poll_context_status(&client).await;

    let (messages, latest_snapshot) = {
        let s = state.read().await;
        let db = match s.session_db.lock() {
            Ok(db) => db,
            Err(_) => return,
        };
        let messages = match db.get_messages(&session_id) {
            Ok(messages) => messages,
            Err(_) => return,
        };
        let latest_snapshot = db.latest_context_snapshot(&session_id).ok().flatten();
        (messages, latest_snapshot)
    };

    let rolling_message_count = messages
        .iter()
        .filter(|message| matches!(message.role.as_str(), "user" | "assistant"))
        .count() as u32;
    let compressed_tokens = latest_snapshot
        .as_ref()
        .map(|snap| snap.kv_tokens)
        .unwrap_or(0);
    let rolling_tokens = status.used_tokens.saturating_sub(compressed_tokens);

    let action = strategy::decide_action(
        status.used_tokens,
        status.total_tokens,
        rolling_message_count,
    );
    let action_label = match action {
        strategy::ContextAction::NoAction => None,
        strategy::ContextAction::Compress { message_count } => Some(format!(
            "Context nearing capacity; compressing {message_count} older messages."
        )),
        strategy::ContextAction::Rebuild => {
            Some("Context critically full; a rebuild is recommended.".to_string())
        }
    };

    status = status.with_breakdown(0, rolling_tokens, compressed_tokens, action_label.clone());

    {
        let mut s = state.write().await;
        s.last_context_status = Some(status.clone());
    }

    if let Some(handle) = app_handle.clone() {
        if let Some(action_text) = action_label.clone() {
            let _ = handle.emit(
                "context-pressure",
                serde_json::json!({
                    "sessionId": session_id,
                    "fillRatio": status.fill_ratio,
                    "usedTokens": status.used_tokens,
                    "totalTokens": status.total_tokens,
                    "action": action_text,
                }),
            );
        }
    }

    if let strategy::ContextAction::Compress { message_count } = action {
        tokio::spawn(async move {
            let candidates = {
                let s = state.read().await;
                let db = match s.session_db.lock() {
                    Ok(db) => db,
                    Err(_) => return,
                };
                match db.get_messages(&session_id) {
                    Ok(messages) => messages,
                    Err(_) => return,
                }
            };

            let compressible = candidates
                .iter()
                .filter(|message| matches!(message.role.as_str(), "user" | "assistant"))
                .take(message_count as usize)
                .map(|message| {
                    (
                        message.role.clone(),
                        message.content.clone().unwrap_or_default(),
                    )
                })
                .collect::<Vec<_>>();

            if compressible.is_empty() {
                return;
            }

            let summary = match compressor::summarize_messages_with_model(
                state.clone(),
                &session_id,
                &compressible,
            )
            .await
            {
                Ok(Some(summary)) => summary,
                Ok(None) => compressor::compress_messages(&compressible),
                Err(_) => compressor::compress_messages(&compressible),
            };

            let summary_tokens = (summary.len() as u32 / 4).max(1);
            {
                let s = state.read().await;
                if let Ok(db) = s.session_db.lock() {
                    let _ = db.add_context_snapshot(&session_id, &summary, summary_tokens);
                };
            }

            let mut s = state.write().await;
            if let Some(current) = s.last_context_status.clone() {
                let pinned_tokens = current.pinned_tokens;
                let rolling_tokens = current.rolling_tokens;
                let compressed_tokens = current.compressed_tokens;
                s.last_context_status = Some(current.with_breakdown(
                    pinned_tokens,
                    rolling_tokens.saturating_sub(summary_tokens),
                    compressed_tokens.saturating_add(summary_tokens),
                    Some(format!(
                        "Compressed {message_count} older messages into a memory snapshot."
                    )),
                ));
            }
        });
    }
}

#[tauri::command]
pub async fn send_message(
    state: tauri::State<'_, SharedState>,
    app: tauri::AppHandle,
    session_id: String,
    content: String,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    repeat_penalty: Option<f32>,
    max_tokens: Option<u32>,
    seed: Option<i64>,
    image_base64: Option<String>,
    show_thinking: Option<bool>,
) -> Result<String, String> {
    tracing::info!(session = %session_id, content_len = content.len(), "send_message called");
    {
        let app_state = state.read().await;
        if app_state
            .image_generation_exclusive
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(
                "Image generation is using the GPU. Chat will resume automatically after the model is restored."
                    .to_string(),
            );
        }
    }

    let normalized_inline_image = image_base64
        .as_deref()
        .map(normalize_inline_image_payload)
        .transpose()
        .map_err(|e| format!("Invalid pasted image: {e}"))?;
    let live_input = match (content.trim().is_empty(), image_base64.is_some()) {
        (true, true) => "[image attached]".to_string(),
        (false, true) => format!("{content}\n[image attached]"),
        _ => content.clone(),
    };

    let (
        model_name,
        profile,
        vision_runtime_ready,
        server_defaults,
        launch_defaults,
        context_limit,
    ) = {
        let s = state.read().await;
        let Some(model_name) = s.loaded_model.clone() else {
            return Err("No model loaded".to_string());
        };
        let profile = s.effective_profile_for_model(&model_name);
        let vision_runtime_ready = s
            .last_launch_preview
            .as_ref()
            .filter(|preview| launch_preview_matches_model(&model_name, &preview.model_path))
            .and_then(|preview| preview.mmproj_path.as_ref())
            .is_some();
        let server_defaults = (
            s.config.server.default_temperature,
            s.config.server.default_top_p,
            s.config.server.default_top_k,
            s.config.server.default_max_tokens,
        );
        let context_limit = s
            .model_stats
            .as_ref()
            .map(|stats| stats.context_size)
            .or_else(|| {
                s.last_launch_preview
                    .as_ref()
                    .and_then(|preview| preview.context_size)
            })
            .or(profile.default_context_window);
        (
            model_name,
            profile,
            vision_runtime_ready,
            server_defaults,
            s.active_sampling_defaults(),
            context_limit,
        )
    };

    if normalized_inline_image.is_some() && !profile.supports_vision {
        return Err(format!(
            "The loaded model `{model_name}` does not advertise vision support. Load a Vision-capable model first."
        ));
    }

    if normalized_inline_image.is_some() && !vision_runtime_ready {
        return Err(format!(
            "The loaded model `{model_name}` was started without a matching mmproj sidecar, so pasted images will not be seen correctly. Reload a vision-ready model first."
        ));
    }

    let mut db_messages = {
        let s = state.read().await;
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        db.get_messages(&session_id).map_err(|e| e.to_string())?
    };
    db_messages.push(crate::session::db::MessageInfo {
        id: -1,
        role: "user".to_string(),
        content: Some(content.clone()),
        display_content: None,
        reasoning_content: None,
        image_base64: image_base64.clone(),
        image_path: None,
        image_metadata: None,
        token_count: Some(0),
        tokens_evaluated: None,
        tokens_predicted: None,
        created_at: now_rfc3339(),
        tool_calls: Vec::new(),
    });

    let prepared_messages = prepare_messages(&db_messages, &profile, show_thinking);
    let has_any_image = prepared_messages
        .iter()
        .any(|message| message.image_base64.is_some());
    if has_any_image && !profile.supports_vision {
        return Err(format!(
            "The loaded model `{model_name}` does not advertise vision support, but this conversation contains image input."
        ));
    }
    if has_any_image && !vision_runtime_ready {
        return Err(format!(
            "This conversation contains image input, but `{model_name}` was started without a matching mmproj sidecar. Reload a vision-ready model before continuing."
        ));
    }

    // Persist the accepted user turn before inference starts. This preserves it
    // across generation errors/cancellation and lets the UI reconcile safely.
    {
        let s = state.read().await;
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        db.add_message(&session_id, "user", &content, 0, image_base64.as_deref())
            .map_err(|e| e.to_string())?;
    }

    if let Some(capability_response) =
        crate::normalize::capability_truth::unavailable_request_response(
            &content,
            &crate::normalize::capability_truth::RuntimeCapabilities::desktop_chat(),
        )
    {
        let assistant_message_id = {
            let s = state.read().await;
            let db = s.session_db.lock().map_err(|e| e.to_string())?;
            db.add_message(&session_id, "assistant", &capability_response, 0, None)
                .map_err(|e| e.to_string())?
        };
        {
            let s = state.read().await;
            let db = s.session_db.lock().map_err(|e| e.to_string())?;
            db.update_message_presentation(assistant_message_id, Some(&capability_response), None)
                .map_err(|e| e.to_string())?;
        }
        let _ = app.emit("stream-token", &capability_response);
        let _ = app.emit("stream-done", 0.0f64);
        return Ok(capability_response);
    }

    let mut image_data = Vec::new();
    let mut next_image_id = 1u32;
    let messages: Vec<ChatMessage> = prepared_messages
        .iter()
        .map(|message| {
            let mut message_content = message.content.clone();
            if let Some(image_base64) = &message.image_base64 {
                let image_id = next_image_id;
                next_image_id += 1;
                let marker = format!("[img-{image_id}]");
                message_content = if message_content.trim().is_empty() {
                    marker
                } else {
                    format!("{marker}\n{message_content}")
                };

                image_data.push(ImageData {
                    data: image_base64.clone(),
                    id: image_id,
                });
            }

            ChatMessage {
                role: message.role.clone(),
                content: message_content,
            }
        })
        .collect();

    if normalized_inline_image.is_some() && profile.family != ModelFamily::Qwen3_5 {
        let openai_messages = build_openai_messages(&prepared_messages);

        {
            let mut s = state.write().await;
            s.last_prompt = Some(
                serde_json::to_string_pretty(&json!({
                    "endpoint": "/v1/chat/completions",
                    "messages": openai_messages,
                }))
                .unwrap_or_default(),
            );
        }

        let generation_started_at = now_rfc3339();
        let generation_started = std::time::Instant::now();
        let generation = begin_generation(
            state.inner(),
            "gui",
            Some(session_id.clone()),
            model_name.clone(),
        )
        .await;
        append_live_stream_delta_for_request(
            state.inner(),
            &generation.request_id,
            "input",
            &live_input,
        )
        .await;
        let port = {
            let s = state.read().await;
            s.process.port()
        };
        let scheduler = {
            let s = state.read().await;
            s.request_scheduler.clone()
        };
        let _permit = scheduler.acquire().await;
        let client = LlamaClient::new(port);
        let request = json!({
            "model": model_name,
            "messages": openai_messages,
            "stream": false,
            "max_tokens": max_tokens.or(server_defaults.3).or(profile.default_max_output_tokens),
            "temperature": temperature.or(launch_defaults.temperature).or(server_defaults.0).or(profile.default_temperature),
            "top_p": top_p.or(launch_defaults.top_p).or(server_defaults.1).or(profile.default_top_p),
            "top_k": top_k.or(launch_defaults.top_k).or(server_defaults.2).or(profile.default_top_k),
            "min_p": min_p.or(launch_defaults.min_p).or(profile.default_min_p),
            "presence_penalty": presence_penalty.or(launch_defaults.presence_penalty).or(profile.default_presence_penalty),
            "repeat_penalty": repeat_penalty.or(launch_defaults.repeat_penalty),
        });
        let response = match client.chat_completion(&request).await {
            Ok(response) => response,
            Err(error) => {
                finish_generation_for_request(state.inner(), &generation.request_id, "error").await;
                return Err(format!("Vision completion failed: {error}"));
            }
        };
        let raw_text = openai_response_text(&response);
        if !raw_text.is_empty() {
            append_live_stream_delta_for_request(
                state.inner(),
                &generation.request_id,
                "raw",
                &raw_text,
            )
            .await;
        }
        let stripped = if show_thinking == Some(true) {
            crate::normalize::think_strip::strip_control_channel_markers(&raw_text)
        } else {
            crate::normalize::think_strip::strip_think_tags_with_style(
                &raw_text,
                profile.think_tag_style,
            )
        };
        let reasoning_content = crate::normalize::think_strip::extract_reasoning_content_with_style(
            &raw_text,
            profile.think_tag_style,
        );
        let presentation_source = crate::normalize::think_strip::strip_think_tags_with_style(
            &raw_text,
            profile.think_tag_style,
        );
        let (detected_tool_calls, extracted_display_content) =
            crate::normalize::tool_extract::extract_tool_calls_for_profile(
                &presentation_source,
                &profile,
            );
        let capability_enforcement = crate::normalize::capability_truth::enforce_tool_calls(
            detected_tool_calls,
            extracted_display_content,
            &crate::normalize::capability_truth::RuntimeCapabilities::desktop_chat(),
        );
        let display_content = capability_enforcement.display_text.clone();
        let stored_content = if capability_enforcement.rejected.is_empty() {
            stripped.clone()
        } else {
            display_content.clone()
        };
        if !display_content.is_empty() {
            append_live_stream_delta_for_request(
                state.inner(),
                &generation.request_id,
                "content",
                &display_content,
            )
            .await;
            let _ = app.emit("stream-token", &display_content);
        }
        let prompt_tokens = openai_usage_tokens(&response, "prompt_tokens");
        let completion_tokens = openai_usage_tokens(&response, "completion_tokens");
        {
            let mut s = state.write().await;
            s.last_parse_trace = Some(crate::normalize::parse_trace::build_parse_trace(
                &profile,
                &raw_text,
                &stripped,
                Some(""),
            ));
            s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
                source: "gui-vision".to_string(),
                model: s
                    .loaded_model
                    .clone()
                    .unwrap_or_else(|| "Unknown".to_string()),
                request_id: generation.request_id.clone(),
                started_at: generation_started_at,
                finished_at: now_rfc3339(),
                elapsed_ms: generation_started.elapsed().as_millis() as u64,
                time_to_first_token_ms: None,
                prompt_tokens,
                completion_tokens,
                total_tokens: match (prompt_tokens, completion_tokens) {
                    (Some(prompt), Some(completion)) => Some(prompt + completion),
                    _ => None,
                },
                prompt_tokens_per_second: None,
                decode_tokens_per_second: None,
                end_to_end_tokens_per_second: match (
                    completion_tokens,
                    generation_started.elapsed().as_millis() as u64,
                ) {
                    (Some(tokens), elapsed_ms) if elapsed_ms > 0 => {
                        Some(tokens as f64 / (elapsed_ms as f64 / 1000.0))
                    }
                    _ => None,
                },
            });
        }
        let final_status = if generation.cancel.is_cancelled() {
            "cancelled"
        } else {
            "completed"
        };
        finish_generation_for_request(state.inner(), &generation.request_id, final_status).await;
        let assistant_message_id = {
            let s = state.read().await;
            let db = s.session_db.lock().map_err(|e| e.to_string())?;
            db.add_message(&session_id, "assistant", &stored_content, 0, None)
                .map_err(|e| e.to_string())?
        };
        {
            let s = state.read().await;
            let db = s.session_db.lock().map_err(|e| e.to_string())?;
            db.update_message_presentation(
                assistant_message_id,
                Some(&display_content),
                (!reasoning_content.trim().is_empty()).then_some(reasoning_content.as_str()),
            )
            .map_err(|e| e.to_string())?;
            for call in &capability_enforcement.accepted {
                db.add_tool_call(
                    assistant_message_id,
                    &call.id,
                    &call.name,
                    &call.arguments.to_string(),
                    None,
                )
                .map_err(|e| e.to_string())?;
            }
            for rejected in &capability_enforcement.rejected {
                let result = rejected.result_json();
                db.add_tool_call(
                    assistant_message_id,
                    &rejected.call.id,
                    &rejected.call.name,
                    &rejected.call.arguments.to_string(),
                    Some(&result),
                )
                .map_err(|e| e.to_string())?;
            }
            db.update_message_generation_stats(
                assistant_message_id,
                completion_tokens.unwrap_or_else(|| {
                    crate::normalize::think_strip::estimate_token_count(&stripped)
                }),
                prompt_tokens,
                completion_tokens,
            )
            .map_err(|e| e.to_string())?;
        }
        let _ = app.emit("stream-done", 0.0f64);
        return Ok(display_content);
    }

    let use_native_chat = profile.family == ModelFamily::Qwen3_5;
    let (completion_request, native_chat_body) = if use_native_chat {
        let native_messages = prepared_to_api_messages(&prepared_messages);
        let (native_messages, _) = crate::api::completions::compact_native_messages_to_fit(
            &native_messages,
            context_limit,
            max_tokens
                .or(server_defaults.3)
                .or(profile.default_max_output_tokens),
            0,
        )
        .map_err(|error| error.1 .0.error.message)?;
        let mut body = json!({
            "model": model_name,
            "messages": crate::api::completions::api_messages_to_native_value(&native_messages),
            "stream": true,
            "stream_options": { "include_usage": true },
            "parallel_tool_calls": false,
            "max_tokens": max_tokens.or(server_defaults.3).or(profile.default_max_output_tokens),
            "temperature": temperature.or(launch_defaults.temperature).or(server_defaults.0).or(profile.default_temperature),
            "top_p": top_p.or(launch_defaults.top_p).or(server_defaults.1).or(profile.default_top_p),
            "top_k": top_k.or(launch_defaults.top_k).or(server_defaults.2).or(profile.default_top_k),
            "min_p": min_p.or(launch_defaults.min_p).or(profile.default_min_p),
            "presence_penalty": presence_penalty.or(launch_defaults.presence_penalty).or(profile.default_presence_penalty),
            "repeat_penalty": repeat_penalty.or(launch_defaults.repeat_penalty).or(Some(1.0_f32)),
            "seed": seed,
        });
        if let Some(object) = body.as_object_mut() {
            object.retain(|_, value| !value.is_null());
        }
        {
            let mut s = state.write().await;
            s.last_prompt =
                Some(serde_json::to_string_pretty(&body).unwrap_or_else(|_| body.to_string()));
        }
        (None, Some(body))
    } else {
        let prompt = render_prompt(&messages, &profile);
        {
            let mut s = state.write().await;
            s.last_prompt = Some(prompt.clone());
        }
        (
            Some(CompletionRequest {
                prompt,
                n_predict: max_tokens
                    .or(server_defaults.3)
                    .or(profile.default_max_output_tokens)
                    .map(|value| value as i32),
                temperature: temperature
                    .or(launch_defaults.temperature)
                    .or(server_defaults.0)
                    .or(profile.default_temperature),
                top_p: top_p
                    .or(launch_defaults.top_p)
                    .or(server_defaults.1)
                    .or(profile.default_top_p),
                top_k: top_k
                    .or(launch_defaults.top_k)
                    .or(server_defaults.2)
                    .or(profile.default_top_k),
                min_p: min_p.or(launch_defaults.min_p).or(profile.default_min_p),
                presence_penalty: presence_penalty
                    .or(launch_defaults.presence_penalty)
                    .or(profile.default_presence_penalty),
                frequency_penalty: None,
                repeat_penalty: repeat_penalty.or(launch_defaults.repeat_penalty),
                seed,
                stream: true,
                stop: profile.stop_markers.clone(),
                special: true,
                image_data,
                grammar: None,
                json_schema: None,
            }),
            None,
        )
    };

    let generation_started_at = now_rfc3339();
    let generation_started = std::time::Instant::now();

    let generation = begin_generation(
        state.inner(),
        "gui",
        Some(session_id.clone()),
        model_name.clone(),
    )
    .await;
    append_live_stream_delta_for_request(
        state.inner(),
        &generation.request_id,
        "input",
        &live_input,
    )
    .await;

    let port = {
        let s = state.read().await;
        s.process.port()
    };
    let scheduler = {
        let s = state.read().await;
        s.request_scheduler.clone()
    };
    let _permit = scheduler.acquire().await;
    let client = LlamaClient::new(port);
    let response = match if let Some(body) = native_chat_body.as_ref() {
        client.chat_completion_response(body).await
    } else {
        client
            .complete_stream(
                completion_request
                    .as_ref()
                    .expect("completion request should exist for compatibility route"),
            )
            .await
    } {
        Ok(response) => response,
        Err(error) => {
            finish_generation_for_request(state.inner(), &generation.request_id, "error").await;
            return Err(format!("Completion failed: {error}"));
        }
    };
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        finish_generation_for_request(state.inner(), &generation.request_id, "error").await;
        return Err(format!(
            "llama-server native chat returned {status}: {body}"
        ));
    }

    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let stream_cancel = generation.cancel.clone();
    tokio::spawn(async move {
        let result = if use_native_chat {
            streaming::consume_chat_sse_stream(response, tx, stream_cancel).await
        } else {
            streaming::consume_sse_stream(response, tx, stream_cancel).await
        };
        let _ = result;
    });

    let mut full_text = String::new();
    let mut reasoning_text = String::new();
    let mut tokens_predicted = None;
    let mut tokens_evaluated = None;
    let mut decode_tokens_per_second = None;
    let mut prompt_tokens_per_second = None;
    let mut output_gate = crate::normalize::capability_truth::ToolOutputStreamGate::default();

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::RawDelta(raw) => {
                append_live_stream_delta_for_request(
                    state.inner(),
                    &generation.request_id,
                    "raw",
                    &raw,
                )
                .await;
            }
            StreamEvent::Token(token) => {
                if let Some(visible) = output_gate.push(&token) {
                    append_live_stream_delta_for_request(
                        state.inner(),
                        &generation.request_id,
                        "content",
                        &visible,
                    )
                    .await;
                    let _ = app.emit("stream-token", &visible);
                } else {
                    append_live_stream_delta_for_request(
                        state.inner(),
                        &generation.request_id,
                        "content_buffered",
                        &token,
                    )
                    .await;
                }
                tokio::task::yield_now().await;
            }
            StreamEvent::ReasoningDelta(reasoning) => {
                reasoning_text.push_str(&reasoning);
                append_live_stream_delta_for_request(
                    state.inner(),
                    &generation.request_id,
                    "reasoning",
                    &reasoning,
                )
                .await;
                if show_thinking == Some(true) {
                    let _ = app.emit("stream-thinking", &reasoning);
                }
                tokio::task::yield_now().await;
            }
            StreamEvent::ToolCallDelta(tool_call) => {
                append_live_stream_delta_for_request(
                    state.inner(),
                    &generation.request_id,
                    "tool_call",
                    &tool_call,
                )
                .await;
            }
            StreamEvent::Done {
                full_text: text,
                tokens_predicted: predicted,
                tokens_evaluated: evaluated,
                decode_tokens_per_second: decode_tps,
                prompt_tokens_per_second: prompt_tps,
                ..
            } => {
                full_text = text;
                tokens_predicted = Some(predicted);
                tokens_evaluated = Some(evaluated);
                let elapsed_seconds = generation_started.elapsed().as_secs_f64();
                let resolved_decode_tps = if decode_tps.is_finite() && decode_tps > 0.0 {
                    decode_tps
                } else if predicted > 0 && elapsed_seconds > 0.0 {
                    // Native OpenAI streams may omit llama.cpp timing extensions.
                    // Report measured request throughput instead of a false 0.0.
                    predicted as f64 / elapsed_seconds
                } else {
                    0.0
                };
                decode_tokens_per_second =
                    (resolved_decode_tps > 0.0).then_some(resolved_decode_tps);
                prompt_tokens_per_second = prompt_tps;
                break;
            }
            StreamEvent::Error(error) => {
                append_live_stream_delta_for_request(
                    state.inner(),
                    &generation.request_id,
                    "error",
                    &error,
                )
                .await;
                let _ = app.emit("stream-error", &error);
                finish_generation_for_request(state.inner(), &generation.request_id, "error").await;
                return Err(error);
            }
        }
    }

    let final_status = if generation.cancel.is_cancelled() {
        "cancelled"
    } else {
        "completed"
    };
    finish_generation_for_request(state.inner(), &generation.request_id, final_status).await;

    let stripped = if show_thinking == Some(true) && profile.family != ModelFamily::Qwen3_5 {
        crate::normalize::think_strip::strip_control_channel_markers(&full_text)
    } else {
        crate::normalize::think_strip::strip_think_tags_with_style(
            &full_text,
            profile.think_tag_style,
        )
    };
    let presentation_source = crate::normalize::think_strip::strip_think_tags_with_style(
        &full_text,
        profile.think_tag_style,
    );
    let (detected_tool_calls, extracted_display_content) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(
            &presentation_source,
            &profile,
        );
    let capability_enforcement = crate::normalize::capability_truth::enforce_tool_calls(
        detected_tool_calls,
        extracted_display_content,
        &crate::normalize::capability_truth::RuntimeCapabilities::desktop_chat(),
    );
    let display_content = capability_enforcement.display_text.clone();
    let stored_content = if capability_enforcement.rejected.is_empty() {
        stripped.clone()
    } else {
        display_content.clone()
    };
    if output_gate.should_emit_final() && !display_content.is_empty() {
        append_live_stream_delta_for_request(
            state.inner(),
            &generation.request_id,
            "content",
            &display_content,
        )
        .await;
        let _ = app.emit("stream-token", &display_content);
    }
    let _ = app.emit("stream-done", decode_tokens_per_second.unwrap_or_default());
    let parse_trace = crate::normalize::parse_trace::build_parse_trace(
        &profile,
        &full_text,
        &stripped,
        Some(&reasoning_text),
    );
    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(parse_trace);
    }

    let assistant_message_id = {
        let s = state.read().await;
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        db.add_message(&session_id, "assistant", &stored_content, 0, None)
            .map_err(|e| e.to_string())?
    };

    {
        let s = state.read().await;
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        db.update_message_presentation(
            assistant_message_id,
            Some(&display_content),
            (!reasoning_text.trim().is_empty()).then_some(reasoning_text.as_str()),
        )
        .map_err(|e| e.to_string())?;
        for call in &capability_enforcement.accepted {
            db.add_tool_call(
                assistant_message_id,
                &call.id,
                &call.name,
                &call.arguments.to_string(),
                None,
            )
            .map_err(|e| e.to_string())?;
        }
        for rejected in &capability_enforcement.rejected {
            let result = rejected.result_json();
            db.add_tool_call(
                assistant_message_id,
                &rejected.call.id,
                &rejected.call.name,
                &rejected.call.arguments.to_string(),
                Some(&result),
            )
            .map_err(|e| e.to_string())?;
        }
        let predicted = tokens_predicted.unwrap_or_else(|| (stripped.len() as u32 / 4).max(1));
        db.update_message_generation_stats(
            assistant_message_id,
            predicted,
            tokens_evaluated,
            tokens_predicted,
        )
        .map_err(|e| e.to_string())?;
    }

    {
        let mut s = state.write().await;
        s.last_generation_metrics = Some(crate::state::RuntimePerformanceMetrics {
            source: "gui".to_string(),
            model: s
                .loaded_model
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
            request_id: generation.request_id.clone(),
            started_at: generation_started_at,
            finished_at: now_rfc3339(),
            elapsed_ms: generation_started.elapsed().as_millis() as u64,
            time_to_first_token_ms: None,
            prompt_tokens: tokens_evaluated,
            completion_tokens: tokens_predicted,
            total_tokens: match (tokens_evaluated, tokens_predicted) {
                (Some(prompt), Some(completion)) => Some(prompt + completion),
                _ => None,
            },
            prompt_tokens_per_second,
            decode_tokens_per_second,
            end_to_end_tokens_per_second: match (
                tokens_predicted,
                generation_started.elapsed().as_millis() as u64,
            ) {
                (Some(tokens), elapsed_ms) if elapsed_ms > 0 => {
                    Some(tokens as f64 / (elapsed_ms as f64 / 1000.0))
                }
                _ => None,
            },
        });
    }

    tokio::spawn(schedule_context_follow_up(
        state.inner().clone(),
        session_id,
    ));

    Ok(display_content)
}

#[tauri::command]
pub async fn stop_generation(state: tauri::State<'_, SharedState>) -> Result<(), String> {
    let count = cancel_all_generations(state.inner()).await;
    tracing::info!(count, "stop_generation: cancellation token fired");
    Ok(())
}
