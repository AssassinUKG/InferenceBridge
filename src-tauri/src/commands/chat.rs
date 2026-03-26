use tauri::Emitter;

use crate::context::{compressor, strategy, tracker};
use crate::engine::client::{CompletionRequest, ImageData, LlamaClient};
use crate::engine::streaming::{self, StreamEvent};
use crate::models::profiles::{ModelFamily, ModelProfile};
use crate::state::{GenerationRequest, SharedState};
use crate::templates::engine::{render_prompt, ChatMessage};

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn build_parse_trace(profile: &ModelProfile, raw: &str, stripped: &str, reasoning: &str) -> String {
    let (tool_calls, visible_text) =
        crate::normalize::tool_extract::extract_tool_calls_for_profile(stripped, profile);
    serde_json::to_string_pretty(&serde_json::json!({
        "parser_type": format!("{:?}", profile.parser_type),
        "tool_call_format": format!("{:?}", profile.tool_call_format),
        "think_tag_style": format!("{:?}", profile.think_tag_style),
        "raw_response": raw,
        "reasoning_text": reasoning,
        "stripped_response": stripped,
        "visible_text": visible_text,
        "tool_calls": tool_calls,
    }))
    .unwrap_or_else(|_| "Failed to serialize parse trace".to_string())
}

fn apply_thinking_preference(
    mut messages: Vec<ChatMessage>,
    profile: &ModelProfile,
    show_thinking: Option<bool>,
) -> Vec<ChatMessage> {
    let Some(last) = messages.last_mut().filter(|message| message.role == "user") else {
        return messages;
    };

    match (profile.family, show_thinking) {
        (ModelFamily::Qwen3_5 | ModelFamily::Qwen3, Some(true)) => {}
        (ModelFamily::Qwen3_5 | ModelFamily::Qwen3, Some(false)) => {
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

async fn begin_generation(
    state: &SharedState,
    source: &str,
    session_id: Option<String>,
    model: String,
) -> tokio_util::sync::CancellationToken {
    let mut s = state.write().await;
    s.generation_cancel.cancel();
    s.generation_cancel = tokio_util::sync::CancellationToken::new();
    s.active_generation = Some(GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        source: source.to_string(),
        session_id,
        model,
        started_at: now_rfc3339(),
        status: "running".to_string(),
    });
    s.generation_cancel.clone()
}

async fn finish_generation(state: &SharedState, status: &str) {
    let mut s = state.write().await;
    if let Some(active) = s.active_generation.as_mut() {
        active.status = status.to_string();
    }
    s.active_generation = None;
}

async fn schedule_context_follow_up(state: SharedState, session_id: String) {
    let (loaded, running, port, app_handle) = {
        let s = state.read().await;
        (
            s.loaded_model.is_some(),
            matches!(s.process.state(), crate::engine::process::ProcessState::Running),
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
    let compressed_tokens = latest_snapshot.as_ref().map(|snap| snap.kv_tokens).unwrap_or(0);
    let rolling_tokens = status.used_tokens.saturating_sub(compressed_tokens);

    let action = strategy::decide_action(status.used_tokens, status.total_tokens, rolling_message_count);
    let action_label = match action {
        strategy::ContextAction::NoAction => None,
        strategy::ContextAction::Compress { message_count } => {
            Some(format!("Context nearing capacity; compressing {message_count} older messages."))
        }
        strategy::ContextAction::Rebuild => Some("Context critically full; a rebuild is recommended.".to_string()),
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
                    Some(format!("Compressed {message_count} older messages into a memory snapshot.")),
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
    max_tokens: Option<u32>,
    seed: Option<i64>,
    image_base64: Option<String>,
    show_thinking: Option<bool>,
) -> Result<String, String> {
    tracing::info!(session = %session_id, content_len = content.len(), "send_message called");

    let (model_name, profile) = {
        let s = state.read().await;
        let Some(model_name) = s.loaded_model.clone() else {
            return Err("No model loaded".to_string());
        };
        let profile = crate::models::overrides::detect_effective_profile(&model_name);
        (model_name, profile)
    };

    if image_base64.is_some() && !profile.supports_vision {
        return Err(format!(
            "The loaded model `{model_name}` does not advertise vision support. Load a Vision-capable model first."
        ));
    }

    {
        let s = state.read().await;
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        db.add_message(&session_id, "user", &content, 0, image_base64.as_deref())
            .map_err(|e| e.to_string())?;
    }

    let db_messages = {
        let s = state.read().await;
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        db.get_messages(&session_id).map_err(|e| e.to_string())?
    };

    let mut image_data = Vec::new();
    let mut next_image_id = 1u32;
    let messages: Vec<ChatMessage> = db_messages
        .iter()
        .map(|message| {
            let mut message_content = message.content.clone().unwrap_or_default();
            if let Some(image_uri) = &message.image_base64 {
                let image_id = next_image_id;
                next_image_id += 1;
                let marker = format!("[img-{image_id}]");
                message_content = if message_content.trim().is_empty() {
                    marker
                } else {
                    format!("{marker}\n{message_content}")
                };

                let raw = image_uri.split(',').nth(1).unwrap_or(image_uri);
                image_data.push(ImageData {
                    data: raw.to_string(),
                    id: image_id,
                });
            }

            ChatMessage {
                role: message.role.clone(),
                content: message_content,
            }
        })
        .collect();

    let messages = apply_thinking_preference(messages, &profile, show_thinking);

    let prompt = render_prompt(&messages, &profile);
    {
        let mut s = state.write().await;
        s.last_prompt = Some(prompt.clone());
    }

    let request = CompletionRequest {
        prompt,
        n_predict: max_tokens
            .or(profile.default_max_output_tokens)
            .map(|value| value as i32),
        temperature: temperature.or(profile.default_temperature),
        top_p: top_p.or(profile.default_top_p),
        top_k: top_k.or(profile.default_top_k),
        min_p: profile.default_min_p,
        presence_penalty: profile.default_presence_penalty,
        frequency_penalty: None,
        repeat_penalty: None,
        seed,
        stream: true,
        stop: profile.stop_markers.clone(),
        special: true,
        image_data,
    };

    let cancel = begin_generation(
        state.inner(),
        "gui",
        Some(session_id.clone()),
        model_name.clone(),
    )
    .await;

    let port = {
        let s = state.read().await;
        s.process.port()
    };
    let client = LlamaClient::new(port);
    let response = client
        .complete_stream(&request)
        .await
        .map_err(|e| format!("Completion failed: {e}"))?;

    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let _ = streaming::consume_sse_stream(response, tx, cancel).await;
    });

    let mut full_text = String::new();
    let mut reasoning_text = String::new();
    let mut tokens_predicted = None;
    let mut tokens_evaluated = None;
    let mut tokens_per_second = None;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token(token) => {
                let _ = app.emit("stream-token", &token);
                tokio::task::yield_now().await;
            }
            StreamEvent::ReasoningDelta(reasoning) => {
                reasoning_text.push_str(&reasoning);
                let _ = app.emit("stream-thinking", &reasoning);
                tokio::task::yield_now().await;
            }
            StreamEvent::Done {
                full_text: text,
                tokens_per_second: tps,
                tokens_predicted: predicted,
                tokens_evaluated: evaluated,
            } => {
                full_text = text;
                tokens_predicted = Some(predicted);
                tokens_evaluated = Some(evaluated);
                tokens_per_second = Some(tps);
                let _ = app.emit("stream-done", tps);
                break;
            }
            StreamEvent::Error(error) => {
                let _ = app.emit("stream-error", &error);
                finish_generation(state.inner(), "error").await;
                return Err(error);
            }
        }
    }

    let stripped = if profile.has_think_tags() && show_thinking != Some(true) {
        crate::normalize::think_strip::strip_think_tags_with_style(
            &full_text,
            profile.think_tag_style,
        )
    } else {
        full_text.clone()
    };
    let parse_trace = build_parse_trace(&profile, &full_text, &stripped, &reasoning_text);
    {
        let mut s = state.write().await;
        s.last_parse_trace = Some(parse_trace);
    }

    let assistant_message_id = {
        let s = state.read().await;
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        db.add_message(&session_id, "assistant", &stripped, 0, None)
            .map_err(|e| e.to_string())?
    };

    {
        let s = state.read().await;
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
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
        s.model_stats = Some(crate::state::ModelStats {
            model: model_name,
            context_size: s
                .last_context_status
                .as_ref()
                .map(|status| status.total_tokens)
                .unwrap_or(0),
            tokens_per_sec: tokens_per_second.unwrap_or(0.0) as f32,
            memory_mb: 0,
        });
    }

    finish_generation(state.inner(), "completed").await;
    tokio::spawn(schedule_context_follow_up(
        state.inner().clone(),
        session_id,
    ));

    Ok(stripped)
}

#[tauri::command]
pub async fn stop_generation(state: tauri::State<'_, SharedState>) -> Result<(), String> {
    let cancel = {
        let s = state.read().await;
        s.generation_cancel.clone()
    };
    cancel.cancel();
    finish_generation(state.inner(), "cancelled").await;
    tracing::info!("stop_generation: cancellation token fired");
    Ok(())
}
