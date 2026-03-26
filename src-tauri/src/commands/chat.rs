//! Tauri commands for chat / generation.

use crate::engine::client::{CompletionRequest, ImageData, LlamaClient};
use crate::engine::streaming;
use crate::state::SharedState;
use crate::templates::engine::{render_prompt, ChatMessage};
use std::sync::atomic::Ordering;
use tauri::Emitter;

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
    // show_thinking: If Some(true): preserve think tags in response. If Some(false): disable thinking for Qwen models.
    // If None: default (strip think tags from display, don't modify prompt).
    show_thinking: Option<bool>,
) -> Result<String, String> {
    tracing::info!(session = %session_id, content_len = content.len(), "send_message called");

    let s = state.read().await;

    if s.loaded_model.is_none() {
        tracing::warn!("send_message: No model loaded");
        return Err("No model loaded".to_string());
    }

    // Save user message to DB (text only for now)
    {
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        let _ = db
            .add_message(&session_id, "user", &content, 0)
            .map_err(|e| e.to_string())?;
    }

    // If image is present, log its size (it will be injected into the prompt below)
    if let Some(ref img) = image_base64 {
        tracing::info!(
            len = img.len(),
            "Received image_base64 for vision model"
        );
    }

    // Build message history from DB
    let db_messages = {
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        db.get_messages(&session_id).map_err(|e| e.to_string())?
    };

    let messages: Vec<ChatMessage> = db_messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();

    // Detect model profile
    let model_name = s.loaded_model.as_deref().unwrap_or("");
    let profile = crate::models::profiles::ModelProfile::detect(model_name);

    // If an image was attached, prepend [img-1] to the last user message so the
    // vision model knows where to find the image.  DB always stores clean text.
    let messages = if let Some(_) = &image_base64 {
        let mut msgs = messages;
        if let Some(last) = msgs.last_mut().filter(|m| m.role == "user") {
            last.content = format!("[img-1]\n{}", last.content);
        }
        msgs
    } else {
        messages
    };

    // For Qwen-style thinking models: inject /think or /no_think into the last
    // user message before rendering — this controls whether the model reasons.
    // We modify the in-memory array only; the DB stores the clean original.
    let messages = if profile.has_think_tags() {
        let mut msgs = messages;
        if let Some(last) = msgs.last_mut().filter(|m| m.role == "user") {
            match show_thinking {
                Some(true) => last.content = format!("/think\n{}", last.content),
                Some(false) => last.content = format!("/no_think\n{}", last.content),
                None => {} // default: model decides
            }
        }
        msgs
    } else {
        messages
    };

    // Render prompt using template
    let prompt = render_prompt(&messages, &profile);

    // Build completion request
    let request = CompletionRequest {
        prompt,
        n_predict: max_tokens
            .or(profile.default_max_output_tokens)
            .map(|t| t as i32),
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
        // Strip the data-URI prefix (data:image/jpeg;base64,<data>) so
        // llama-server receives the raw base64 bytes.
        image_data: image_base64
            .as_deref()
            .map(|uri| {
                let raw = uri.split(',').nth(1).unwrap_or(uri);
                vec![ImageData { data: raw.to_string(), id: 1 }]
            })
            .unwrap_or_default(),
    };

    let port = s.process.port();
    let client = LlamaClient::new(port);
    // Reset stop flag and clone it for the stream consumer
    s.generation_stop.store(false, Ordering::Relaxed);
    let stop_flag = s.generation_stop.clone();
    drop(s); // Release lock before streaming

    tracing::info!(
        port,
        prompt_len = request.prompt.len(),
        "Sending completion request to llama-server"
    );

    // Stream completion
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let response = client.complete_stream(&request).await.map_err(|e| {
        tracing::error!(error = %e, "Completion request failed");
        format!("Completion failed: {e}")
    })?;

    tracing::info!(status = %response.status(), "Got streaming response from llama-server");

    // Spawn stream consumer
    tokio::spawn(async move {
        let _ = streaming::consume_sse_stream(response, tx, stop_flag).await;
    });

    // Forward events to frontend
    let mut full_text = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            streaming::StreamEvent::Token(token) => {
                full_text.push_str(&token);
                let _ = app.emit("stream-token", &token);
                // Yield to the tokio runtime so the WebView IPC queue can flush
                // before the next token is emitted. Without this, all tokens from
                // a single TCP chunk are emitted in one burst and arrive at the
                // frontend together instead of one at a time.
                tokio::task::yield_now().await;
            }
            streaming::StreamEvent::Done {
                full_text: text,
                tokens_per_second,
                ..
            } => {
                full_text = text;
                let _ = app.emit("stream-done", tokens_per_second);
                break;
            }
            streaming::StreamEvent::Error(e) => {
                let _ = app.emit("stream-error", &e);
                return Err(e);
            }
        }
    }

    tracing::info!(response_len = full_text.len(), "Generation complete");
    tracing::debug!(raw_response = %full_text, "Raw llama-server response");

    // Normalize output — strip think tags unless the user wants to see thinking
    let stripped = if profile.has_think_tags() && show_thinking != Some(true) {
        crate::normalize::think_strip::strip_think_tags(&full_text)
    } else {
        full_text.clone()
    };
    tracing::info!(stripped_len = stripped.len(), "After think-tag stripping");

    if stripped.is_empty() {
        tracing::warn!(
            "Model produced empty response after stripping — raw was {} chars",
            full_text.len()
        );
    }

    // Save assistant response to DB
    {
        let s = state.read().await;
        let db = s.session_db.lock().map_err(|e| e.to_string())?;
        let _ = db.add_message(&session_id, "assistant", &stripped, 0);
    }

    Ok(stripped)
}

#[tauri::command]
pub async fn stop_generation(state: tauri::State<'_, SharedState>) -> Result<(), String> {
    let s = state.read().await;
    s.generation_stop.store(true, Ordering::Relaxed);
    tracing::info!("stop_generation: cancellation flag set");
    Ok(())
}
