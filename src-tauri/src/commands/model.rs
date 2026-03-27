/// Emit a model-load-progress event to the GUI (no-op if handle is None / headless).
fn emit_load_progress_payload(
    handle: &Option<tauri::AppHandle>,
    progress: &crate::state::LoadProgress,
) {
    use tauri::Emitter;
    if let Some(h) = handle {
        let _ = h.emit("model-load-progress", progress.clone());
    }
}

fn emit_load_progress(
    handle: &Option<tauri::AppHandle>,
    stage: &str,
    msg: &str,
    progress: f32,
    done: bool,
    error: Option<String>,
) {
    emit_load_progress_payload(
        handle,
        &crate::state::LoadProgress {
            stage: stage.to_string(),
            message: msg.to_string(),
            progress,
            done,
            error,
        },
    );
}

fn done_model_load_state(
    transition: &crate::state::ModelLoadState,
    error: Option<&str>,
) -> crate::state::ModelLoadState {
    if let Some(error) = error {
        return crate::state::ModelLoadState::Error(error.to_string());
    }

    match transition {
        crate::state::ModelLoadState::Unloading => crate::state::ModelLoadState::Idle,
        _ => crate::state::ModelLoadState::Loaded,
    }
}

fn load_transition_for_request(
    current_model: Option<&str>,
    requested_model: &str,
) -> crate::state::ModelLoadState {
    match current_model {
        Some(current) if !names_match(current, requested_model) => {
            crate::state::ModelLoadState::Swapping
        }
        _ => crate::state::ModelLoadState::Loading,
    }
}

fn names_match(left: &str, right: &str) -> bool {
    let left = left.to_lowercase();
    let right = right.to_lowercase();
    left == right
        || left.trim_end_matches(".gguf") == right
        || left == right.trim_end_matches(".gguf")
}

fn normalize_requested_context_size(context_size: Option<u32>) -> Option<u32> {
    context_size.filter(|value| *value > 0)
}

/// Resolve the context size to pass to llama-server.
///
/// Returns `None` when no explicit override exists — the server should use the
/// model's native context window in that case.  Only returns `Some` when the
/// user, API request, or server config explicitly sets a value.
fn resolve_launch_context_size(
    requested_context_size: Option<u32>,
    server_default_ctx_size: Option<u32>,
    _profile_default_ctx_size: Option<u32>,
    fallback_ctx_size: u32,
) -> Option<u32> {
    // Explicit request always wins.
    if let Some(requested) = normalize_requested_context_size(requested_context_size) {
        return Some(requested);
    }

    // Server config override (non-zero, different from the generic fallback).
    let server_default_ctx_size = server_default_ctx_size.filter(|value| *value > 0);
    let fallback_ctx_size = fallback_ctx_size.max(1);

    if let Some(server_default) = server_default_ctx_size {
        if server_default != fallback_ctx_size {
            return Some(server_default);
        }
    }

    // No explicit override — enforce the configured fallback so llama-server
    // never silently uses its built-in 4 096-token default.  Users who want the
    // model's native context window should set  in the
    // config to a value equal to the model's training context length.
    Some(fallback_ctx_size)
}

#[cfg(test)]
mod tests {
    use super::resolve_launch_context_size;

    #[test]
    fn prefers_requested_context_size() {
        assert_eq!(
            resolve_launch_context_size(Some(32768), Some(8192), Some(32768), 8192),
            Some(32768)
        );
    }

    #[test]
    fn returns_none_when_no_explicit_override() {
        // No request, server default matches fallback, profile has a value
        // → None so llama-server uses model metadata.
        assert_eq!(
            resolve_launch_context_size(None, Some(8192), Some(32768), 8192),
            None
        );
    }

    #[test]
    fn preserves_explicit_server_default_when_it_differs_from_fallback() {
        assert_eq!(
            resolve_launch_context_size(None, Some(16384), Some(32768), 8192),
            Some(16384)
        );
    }

    #[test]
    fn returns_none_when_nothing_specified() {
        assert_eq!(
            resolve_launch_context_size(None, None, None, 8192),
            None
        );
    }
}

async fn publish_model_load_progress(
    state: &SharedState,
    transition: crate::state::ModelLoadState,
    stage: &str,
    msg: &str,
    progress: f32,
    done: bool,
    error: Option<String>,
) {
    let payload = crate::state::LoadProgress {
        stage: stage.to_string(),
        message: msg.to_string(),
        progress,
        done,
        error: error.clone(),
    };

    let app_handle = {
        let mut s = state.write().await;
        s.model_load_state = if done {
            done_model_load_state(&transition, error.as_deref())
        } else {
            transition.clone()
        };
        s.model_load_progress = Some(payload.clone());
        s.app_handle.clone()
    };

    emit_load_progress_payload(&app_handle, &payload);
}

fn model_load_state_label(state: &crate::state::ModelLoadState) -> String {
    match state {
        crate::state::ModelLoadState::Idle => "Idle".to_string(),
        crate::state::ModelLoadState::Loading => "Loading".to_string(),
        crate::state::ModelLoadState::Swapping => "Swapping".to_string(),
        crate::state::ModelLoadState::Unloading => "Unloading".to_string(),
        crate::state::ModelLoadState::Loaded => "Loaded".to_string(),
        crate::state::ModelLoadState::Error(_) => "Error".to_string(),
    }
}

async fn store_launch_preview(state: &SharedState, preview: LaunchPreview) {
    let mut s = state.write().await;
    s.last_launch_preview = Some(preview);
}

fn empty_context_status(total_tokens: u32) -> crate::context::tracker::ContextStatus {
    crate::context::tracker::ContextStatus {
        total_tokens,
        used_tokens: 0,
        fill_ratio: 0.0,
        pinned_tokens: 0,
        rolling_tokens: 0,
        compressed_tokens: 0,
        last_compaction_action: None,
    }
}

fn clear_runtime_after_backend_exit(
    state: &mut crate::state::AppState,
    error_message: Option<String>,
) {
    if let Some(loaded) = state.loaded_model.take() {
        state.previous_model = Some(loaded);
    }
    state.model_stats = None;
    state.last_context_status = Some(crate::context::tracker::ContextStatus::empty());

    if let Some(error_message) = error_message {
        state.model_load_state = crate::state::ModelLoadState::Error(error_message.clone());
        state.model_load_progress = Some(crate::state::LoadProgress {
            stage: "error".to_string(),
            message: error_message.clone(),
            progress: 0.0,
            done: true,
            error: Some(error_message),
        });
    } else {
        state.model_load_state = crate::state::ModelLoadState::Idle;
        state.model_load_progress = None;
    }
}

fn effective_profile_info_from_state(
    state: &crate::state::AppState,
    requested_model: Option<&str>,
) -> Result<EffectiveProfileInfo, String> {
    let resolved_model = requested_model
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| state.loaded_model.clone())
        .or_else(|| state.model_registry.list().first().map(|model| model.filename.clone()))
        .ok_or_else(|| "No model is available to resolve an effective profile".to_string())?;

    Ok(EffectiveProfileInfo {
        requested_model: requested_model
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        override_entry: effective_override(&resolved_model),
        profile: detect_effective_profile(&resolved_model),
        resolved_model: Some(resolved_model),
    })
}

pub fn get_effective_profile_for_shared(
    state: &crate::state::AppState,
    model_name: Option<&str>,
) -> Result<EffectiveProfileInfo, String> {
    effective_profile_info_from_state(state, model_name)
}

/// Core backend model loading logic, used by both the REST API and headless mode.
///
/// Emits `model-load-progress` Tauri events when an `app_handle` is stored in
/// AppState (GUI mode). In headless/API-only mode the handle is `None` and the
/// function is silent toward the frontend.
pub async fn backend_load_model(
    state: SharedState,
    model_name: String,
    context_size: Option<u32>,
) -> Result<String, String> {
    // Serialize concurrent model-load requests with a per-app mutex.
    //
    // Without this, two simultaneous requests for the same (or different) model both
    // enter the load path, the second one calls shutdown() + kill() on the process the
    // first one just started, and they race indefinitely.  With the mutex the second
    // caller waits, then coalesces — if the first load already loaded the right model
    // it returns immediately without touching the process at all.
    let load_mutex = {
        let s = state.read().await;
        s.model_load_mutex.clone()
    };
    let _load_guard = load_mutex.lock().await;

    // Coalesce: a concurrent load may have already satisfied this request.
    {
        let s = state.read().await;
        if let Some(loaded) = &s.loaded_model {
            let req = model_name.trim().to_ascii_lowercase();
            let cur = loaded.to_ascii_lowercase();
            let name_ok = cur == req
                || cur.trim_end_matches(".gguf") == req
                || cur == req.trim_end_matches(".gguf")
                || (!req.is_empty() && cur.contains(&req));
            let ctx_ok = context_size
                .map(|req_ctx| {
                    s.last_launch_preview
                        .as_ref()
                        .and_then(|p| p.context_size)
                        == Some(req_ctx)
                })
                .unwrap_or(true);
            if name_ok && ctx_ok {
                tracing::info!(
                    model = %model_name,
                    "Coalesced: model already loaded by concurrent request, skipping load"
                );
                return Ok(loaded.clone());
            }
        }
    }

    let transition = {
        let s = state.read().await;
        load_transition_for_request(s.loaded_model.as_deref(), &model_name)
    };

    // Grab the app handle once (no lock held after this point).
    let app_handle = {
        let s = state.read().await;
        s.app_handle.clone()
    };

    publish_model_load_progress(
        &state,
        transition.clone(),
        "resolving",
        "Resolving model...",
        0.0,
        false,
        None,
    )
    .await;

    // Phase 1: Resolve model info (brief lock) + claim loading generation
    let (config, model_filename, my_generation, launch_preview) = {
        let model = {
            let s = state.read().await;
            s.model_registry.find_by_name(&model_name).cloned()
        };

        let model = if let Some(model) = model {
            model
        } else {
            let scan_dirs = {
                let s = state.read().await;
                s.config.models.scan_dirs.clone()
            };
            let scanned = tokio::task::spawn_blocking(move || scanner::scan_all(&scan_dirs))
                .await
                .map_err(|e| format!("Failed to rescan models: {e}"))?;

            let mut s = state.write().await;
            s.model_registry.update(scanned);
            s.model_registry
                .find_by_name(&model_name)
                .cloned()
                .ok_or_else(|| {
                    let msg = format!("Model not found: {model_name}");
                    let payload = crate::state::LoadProgress {
                        stage: "error".to_string(),
                        message: msg.clone(),
                        progress: 0.0,
                        done: true,
                        error: Some(msg.clone()),
                    };
                    emit_load_progress_payload(&app_handle, &payload);
                    msg
                })?
        };

        let mut s = state.write().await;

        let ctx = resolve_launch_context_size(
            context_size,
            s.config.server.default_ctx_size,
            model.profile.default_context_window,
            s.config.models.default_context,
        );

        // Use a provisional context size for pre-launch state.
        // If ctx is None (server decides), use 0 as placeholder — will be
        // updated from /props or /slots after health check succeeds.
        let provisional_ctx = ctx.unwrap_or(0);
        s.last_context_status = Some(empty_context_status(provisional_ctx));
        s.model_stats = Some(crate::state::ModelStats {
            model: model.filename.clone(),
            context_size: provisional_ctx,
            tokens_per_sec: 0.0,
            memory_mb: 0,
        });

        let config = LaunchConfig {
            model_path: model.path.clone(),
            context_size: ctx,
            gpu_layers: s.config.process.gpu_layers,
            threads: s.config.process.threads,
            threads_batch: s.config.process.threads_batch,
            port: 8801,
            backend_preference: s.config.process.backend_preference.clone(),
            batch_size: s.config.process.batch_size,
            ubatch_size: s.config.process.ubatch_size,
            flash_attn: s.config.process.flash_attn,
            use_mmap: s.config.process.use_mmap,
            use_mlock: s.config.process.use_mlock,
            cont_batching: s.config.process.cont_batching,
            parallel_slots: s.config.process.parallel_slots,
            main_gpu: s.config.process.main_gpu,
            defrag_thold: s.config.process.defrag_thold,
            rope_freq_scale: s.config.process.rope_freq_scale,
        };

        LlamaProcess::validate_launch_config(&config).map_err(|e| {
            let msg = format!("Invalid launch configuration: {e}");
            let payload = crate::state::LoadProgress {
                stage: "error".to_string(),
                message: msg.clone(),
                progress: 0.0,
                done: true,
                error: Some(msg.clone()),
            };
            emit_load_progress_payload(&app_handle, &payload);
            msg
        })?;

        let preview = s.process.build_args_preview(&config).map_err(|e| {
            let msg = format!("Could not build launch preview: {e}");
            let payload = crate::state::LoadProgress {
                stage: "error".to_string(),
                message: msg.clone(),
                progress: 0.0,
                done: true,
                error: Some(msg.clone()),
            };
            emit_load_progress_payload(&app_handle, &payload);
            msg
        })?;

        // Bump generation so older in-flight loads won't overwrite us
        s.loading_generation += 1;
        let gen = s.loading_generation;
        tracing::info!(
            model = %model.filename,
            generation = gen,
            ctx_size = ctx,
            "Phase 1: Resolved model (API/backend)"
        );

        (config, model.filename.clone(), gen, preview)
    }; // write lock released

    store_launch_preview(&state, launch_preview.clone()).await;

    let size_info = match config.context_size {
        Some(ctx) => format!("{} (ctx: {})", model_filename, ctx),
        None => format!("{} (ctx: auto)", model_filename),
    };

    // Phase 2: Launch process (brief write lock, then released)
    //
    // Pre-launch cleanup: kill any stale port occupants BEFORE acquiring the write lock.
    // The WMIC scan inside kill_all_managed_processes() can take 1-3 seconds on Windows.
    // Running it under the write lock would freeze every concurrent reader for that duration.
    #[cfg(windows)]
    {
        let port = config.port;
        tokio::task::spawn_blocking(move || {
            crate::engine::process::LlamaProcess::clear_stale_port_processes(port);
        })
        .await
        .ok();
    }

    publish_model_load_progress(
        &state,
        transition.clone(),
        "launching",
        &format!("Launching llama-server for {}...", model_filename),
        0.05,
        false,
        None,
    )
    .await;
    {
        let mut s = state.write().await;
        s.process.launch(config).await.map_err(|e| {
            let msg = format!("Failed to launch llama-server: {e}");
            emit_load_progress(&app_handle, "error", &msg, 0.0, true, Some(msg.clone()));
            msg
        })?;
    } // write lock released

    // Phase 3: Wait for llama-server /health (NO lock held — UI can poll freely)
    emit_load_progress(
        &app_handle,
        "loading",
        &format!("Loading {} into memory...", model_filename),
        0.1,
        false,
        None,
    );
    tracing::info!("Phase 3: Waiting for llama-server health check on port 8801...");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();
    let health_url = "http://127.0.0.1:8801/health";
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(300); // 5 min for large models
    let mut attempt = 0u32;

    loop {
        if start.elapsed() > timeout {
            let msg = format!("llama-server did not become healthy within {:?}", timeout);
            tracing::error!("{msg}");
            emit_load_progress(&app_handle, "error", &msg, 0.0, true, Some(msg.clone()));
            let mut s = state.write().await;
            let _ = s.process.shutdown().await;
            clear_runtime_after_backend_exit(&mut s, Some(msg.clone()));
            return Err(msg);
        }

        // Check if process crashed.
        //
        // Use poll_exited() (non-blocking, no sleep) under the write lock so the lock
        // is not held during the 100ms stderr-flush wait.  That 100ms sleep previously
        // blocked every concurrent reader on every crash detection.
        let process_exited = {
            let mut s = state.write().await;
            s.process.poll_exited()
        };
        if process_exited {
            // Give the background stderr-reader task a moment to drain — outside the lock.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let mut s = state.write().await;
            let stderr = s.process.last_stderr().await;
            let last_lines: String = stderr
                .iter()
                .rev()
                .take(10)
                .rev()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            let msg = format!(
                "llama-server crashed on startup.\n{}",
                if last_lines.is_empty() {
                    "No stderr output captured.".to_string()
                } else {
                    last_lines
                }
            );
            tracing::error!("{msg}");
            emit_load_progress(&app_handle, "error", &msg, 0.0, true, Some(msg.clone()));
            clear_runtime_after_backend_exit(&mut s, Some(msg.clone()));
            return Err(msg);
        }

        match client.get(health_url).send().await {
            Ok(resp) => {
                let status_code = resp.status();
                let body = resp.text().await.unwrap_or_default();

                if status_code.is_success() {
                    if body.contains("ok") || body.contains("\"status\":\"ok\"") {
                        tracing::info!("llama-server healthy after {}s", start.elapsed().as_secs());
                        break;
                    }
                    if body.contains("loading") {
                        let elapsed = start.elapsed().as_secs();
                        let progress = (0.1 + (elapsed as f32 * 0.01)).min(0.9);
                        emit_load_progress(
                            &app_handle,
                            "loading",
                            &format!("Loading model into GPU memory... ({}s)", elapsed),
                            progress,
                            false,
                            None,
                        );
                    } else {
                        break; // 200 with unknown body — assume ready
                    }
                } else if status_code.as_u16() == 503 {
                    let elapsed = start.elapsed().as_secs();
                    let progress = (0.1 + (elapsed as f32 * 0.01)).min(0.9);
                    emit_load_progress(
                        &app_handle,
                        "loading",
                        &format!("Loading model weights... ({}s)", elapsed),
                        progress,
                        false,
                        None,
                    );
                } else {
                    let elapsed = start.elapsed().as_secs();
                    emit_load_progress(
                        &app_handle,
                        "starting",
                        &format!(
                            "Waiting for llama-server (HTTP {})... ({}s)",
                            status_code, elapsed
                        ),
                        0.08,
                        false,
                        None,
                    );
                }
            }
            Err(_) => {
                let elapsed = start.elapsed().as_secs();
                let progress = (0.02 + (elapsed as f32 * 0.005)).min(0.09);
                emit_load_progress(
                    &app_handle,
                    "starting",
                    &format!("Waiting for llama-server to respond... ({}s)", elapsed),
                    progress,
                    false,
                    None,
                );
            }
        }

        attempt += 1;
        if attempt < 5 || attempt % 10 == 0 {
            tracing::debug!(attempt, "Health check attempt");
        }
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    }

    // Phase 4: Mark as loaded (brief write lock)
    {
        let mut s = state.write().await;
        if s.loading_generation != my_generation {
            tracing::warn!(
                model = %model_filename,
                our_gen = my_generation,
                current_gen = s.loading_generation,
                "Stale load — a newer swap is in progress, discarding"
            );
            return Ok(format!("Superseded by newer swap (gen {})", my_generation));
        }
        if let Some(prev) = s.loaded_model.take() {
            if prev != model_filename {
                s.previous_model = Some(prev);
            }
        }
        s.loaded_model = Some(model_filename.clone());
        s.process.set_state_running();
        s.last_known_good_config = Some(launch_preview);
        s.last_startup_duration_ms = Some(start.elapsed().as_millis() as u64);
    }

    crate::context::tracker::reset_slots_warning();

    // Phase 4b: Discover actual context size from the running server.
    // When we didn't pass --ctx-size, llama-server picks the model's native
    // window.  Query /slots (or /props) to learn the real n_ctx.
    {
        let llama_client = crate::engine::client::LlamaClient::new(8801);
        let actual_ctx = match llama_client.get_slots().await {
            Ok(slots) if !slots.is_empty() => Some(slots[0].n_ctx),
            _ => match llama_client.get_props().await {
                Ok(props) => props
                    .default_generation_settings
                    .and_then(|s| s.n_ctx)
                    .filter(|v| *v > 0),
                Err(_) => None,
            },
        };

        if let Some(real_ctx) = actual_ctx {
            let mut s = state.write().await;
            s.last_context_status = Some(empty_context_status(real_ctx));
            if let Some(stats) = s.model_stats.as_mut() {
                stats.context_size = real_ctx;
            }
            // Sync the real context back into last_launch_preview so that
            // resolve_loaded_model can correctly detect context mismatches on
            // subsequent API requests (e.g. if no --ctx-size was explicitly
            // requested, preview.context_size was None and any explicit-ctx
            // request would always trigger an unnecessary reload).
            if let Some(preview) = s.last_launch_preview.as_mut() {
                preview.context_size = Some(real_ctx);
            }
            tracing::info!(real_ctx, "Discovered actual context size from running server");
        }
    }

    let result = format!("Loaded {size_info}");
    publish_model_load_progress(
        &state,
        transition.clone(),
        "ready",
        &result,
        1.0,
        true,
        None,
    )
    .await;
    tracing::info!("{result}");
    Ok(result)
}
// Tauri commands for model management.

use crate::engine::download;
use crate::engine::process::{LaunchConfig, LaunchPreview, LlamaProcess};
use crate::models::overrides::{detect_effective_profile, effective_override};
use crate::models::scanner;
use crate::state::{
    EffectiveProfileInfo, GenerationRequest, LoadProgress, RuntimePerformanceMetrics, SharedState,
};
fn command_no_window(program: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

#[tauri::command]
pub async fn list_models(state: tauri::State<'_, SharedState>) -> Result<Vec<ModelInfo>, String> {
    let s = state.read().await;
    Ok(s.model_registry
        .list()
        .iter()
        .map(|m| {
            use crate::models::profiles::{ThinkTagStyle, ToolCallFormat};
            let supports_tools = !matches!(m.profile.tool_call_format, ToolCallFormat::NativeApi)
                || m.profile.supports_parallel_tools;
            let supports_reasoning = !matches!(m.profile.think_tag_style, ThinkTagStyle::None);
            ModelInfo {
                filename: m.filename.clone(),
                path: m.path.to_string_lossy().to_string(),
                size_gb: m.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                family: m.profile.family.to_string(),
                supports_tools,
                supports_reasoning,
                supports_vision: m.profile.supports_vision,
                context_window: m.profile.default_context_window,
                max_context_window: m.profile.max_context_window,
                max_output_tokens: m.profile.default_max_output_tokens,
                quant: extract_quant(&m.filename),
                tool_call_format: format!("{:?}", m.profile.tool_call_format),
                think_tag_style: format!("{:?}", m.profile.think_tag_style),
            }
        })
        .collect())
}

#[tauri::command]
pub async fn scan_models(state: tauri::State<'_, SharedState>) -> Result<usize, String> {
    let mut s = state.write().await;
    let models = scanner::scan_all(&s.config.models.scan_dirs);
    let count = models.len();
    s.model_registry.update(models);
    Ok(count)
}

#[tauri::command]
pub async fn load_model(
    _app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
    model_name: String,
    context_size: Option<u32>,
) -> Result<String, String> {
    backend_load_model(state.inner().clone(), model_name, context_size).await
}

pub async fn backend_unload_model(state: SharedState) -> Result<String, String> {
    let unloaded_model = {
        let s = state.read().await;
        s.loaded_model.clone()
    };

    if unloaded_model.is_none() {
        let mut s = state.write().await;
        clear_runtime_after_backend_exit(&mut s, None);
        s.model_load_progress = None;
        s.last_context_status = Some(crate::context::tracker::ContextStatus::empty());
        return Ok("No model was loaded.".to_string());
    }

    publish_model_load_progress(
        &state,
        crate::state::ModelLoadState::Unloading,
        "unloading",
        "Unloading model...",
        0.0,
        false,
        None,
    )
    .await;

    {
        let mut s = state.write().await;
        s.generation_cancel.cancel();
        s.active_generation = None;
        s.process.shutdown().await.map_err(|error| {
            let message = format!("Failed to shut down llama-server: {error}");
            clear_runtime_after_backend_exit(&mut s, Some(message.clone()));
            message
        })?;
        clear_runtime_after_backend_exit(&mut s, None);
        s.model_load_progress = None;
    }

    crate::context::tracker::reset_slots_warning();

    let result = format!(
        "Unloaded {}",
        unloaded_model.unwrap_or_else(|| "model".to_string())
    );
    publish_model_load_progress(
        &state,
        crate::state::ModelLoadState::Unloading,
        "ready",
        &result,
        1.0,
        true,
        None,
    )
    .await;
    Ok(result)
}

#[tauri::command]
pub async fn unload_model(state: tauri::State<'_, SharedState>) -> Result<String, String> {
    backend_unload_model(state.inner().clone()).await
}

#[tauri::command]
pub async fn get_process_status(
    state: tauri::State<'_, SharedState>,
) -> Result<ProcessStatusInfo, String> {
    collect_process_status(state.inner().clone()).await
}

pub async fn collect_process_status(state: SharedState) -> Result<ProcessStatusInfo, String> {
    {
        let mut s = state.write().await;
        if s.process.check_crashed().await {
            clear_runtime_after_backend_exit(
                &mut s,
                Some("llama-server exited unexpectedly.".to_string()),
            );
        }
    }

    // Extract everything we need from the lock, then drop it before any I/O
    let (
        process_state,
        server_path,
        is_running,
        port,
        loaded_model,
        previous_model,
        crash_count,
        detected_arc,
        api_state,
        api_error,
        api_url,
        api_port,
        last_launch_preview,
        startup_duration_ms,
        parallel_slots,
        scheduler_snapshot,
        model_load_state,
        model_load_progress,
        active_generation,
        last_generation_metrics,
    ) = {
        let s = state.read().await;
        let server_path = s.process.find_server_binary();
        let process_state = s.process.state();
        let is_running = process_state != crate::engine::process::ProcessState::Idle;
        (
            process_state,
            server_path,
            is_running,
            s.process.port(),
            s.loaded_model.clone(),
            s.previous_model.clone(),
            s.process.crash_count(),
            s.process.detected_backend(),
            format!("{:?}", s.api_server_state),
            s.api_server_error.clone(),
            crate::api::server::reachable_api_url(&s.config.server.host, s.config.server.port),
            s.config.server.port,
            s.last_launch_preview.clone(),
            s.last_startup_duration_ms,
            Some(s.config.process.parallel_slots),
            s.request_scheduler.snapshot(),
            s.model_load_state.clone(),
            s.model_load_progress.clone(),
            s.active_generation.clone(),
            s.last_generation_metrics.clone(),
        )
    }; // read lock released before any async I/O

    let effective_loaded_model = if loaded_model.is_some() {
        loaded_model.clone()
    } else if is_running {
        last_launch_preview.as_ref().and_then(|preview| {
            std::path::Path::new(&preview.model_path)
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
        })
    } else {
        None
    };

    let state_str = format!("{process_state:?}");
    let mut effective_model_load_state = model_load_state.clone();
    let mut effective_model_load_progress = model_load_progress.clone();

    if matches!(process_state, crate::engine::process::ProcessState::Running)
        && effective_loaded_model.is_some()
    {
        effective_model_load_state = crate::state::ModelLoadState::Loaded;
        effective_model_load_progress = None;
    }

    let model_load_state_label = model_load_state_label(&effective_model_load_state);

    let transition_active = matches!(
        process_state,
        crate::engine::process::ProcessState::Starting
            | crate::engine::process::ProcessState::Stopping
    ) || matches!(
        effective_model_load_state,
        crate::state::ModelLoadState::Loading
            | crate::state::ModelLoadState::Swapping
            | crate::state::ModelLoadState::Unloading
    ) || effective_model_load_progress
        .as_ref()
        .map(|progress| !progress.done)
        .unwrap_or(false);

    let api_reachable = probe_public_api(&api_url).await;

    if api_reachable && (api_state != "Running" || api_error.is_some()) {
        let mut s = state.write().await;
        s.api_server_state = crate::state::ApiServerState::Running;
        s.api_server_error = None;
    }

    // Get backend from live detection (parsed from server stderr on startup)
    let backend = if is_running {
        let guard = detected_arc.lock().await;
        Some(
            guard
                .clone()
                .unwrap_or_else(|| detect_backend_from_path(server_path.as_deref())),
        )
    } else {
        None
    };

    // Try managed binary version file first, then query live server /props
    let server_version = download::current_version().or_else(|| {
        if !is_running {
            return None;
        }
        let client = crate::engine::client::LlamaClient::new(port);
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { client.get_props().await.ok().and_then(|p| p.build_info) })
        })
    });

    let effective_api_state = if api_reachable {
        "Running".to_string()
    } else if transition_active {
        "Starting".to_string()
    } else if api_state == "Running" {
        "Error".to_string()
    } else {
        api_state.clone()
    };

    let api_port_owner = if effective_api_state == "Error" && !api_reachable && !transition_active {
        detect_api_port_owner(api_port)
    } else {
        None
    };

    let effective_api_error = if api_reachable {
        None
    } else if transition_active {
        None
    } else if effective_api_state == "Error" {
        if api_port_owner.is_some() {
            api_error.clone().or_else(|| {
                Some(format!(
                    "The public API is not currently reachable on {api_url}. Another process appears to be holding port {api_port}."
                ))
            })
        } else {
            Some(format!(
                "The public API is not currently reachable on {api_url}. No active listener is holding port {api_port}. Retry API to start it again."
            ))
        }
    } else {
        api_error.clone()
    };

    let slot_count = if is_running {
        let client = crate::engine::client::LlamaClient::new(port);
        client.get_props().await.ok().and_then(|props| props.total_slots)
    } else {
        None
    };

    Ok(ProcessStatusInfo {
        state: state_str,
        model: effective_loaded_model,
        previous_model,
        crash_count,
        server_version,
        server_path: server_path.map(|p| p.to_string_lossy().to_string()),
        backend,
        api_state: effective_api_state,
        api_error: effective_api_error,
        api_url,
        api_reachable,
        api_port_owner,
        startup_duration_ms,
        parallel_slots,
        slot_count,
        active_requests: scheduler_snapshot.active,
        queued_requests: scheduler_snapshot.queued,
        scheduler_limit: Some(scheduler_snapshot.limit),
        last_launch_preview,
        model_load_state: model_load_state_label,
        model_load_progress: effective_model_load_progress,
        active_generation,
        last_generation_metrics,
    })
}

async fn probe_public_api(api_url: &str) -> bool {
    let health_url = format!("{}/health", api_url.trim_end_matches('/'));
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1200))
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };

    match client.get(&health_url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// Detect backend by checking the binary's directory for CUDA/Vulkan DLLs.
fn detect_backend_from_path(path: Option<&std::path::Path>) -> String {
    let Some(path) = path else {
        return "Unknown".to_string();
    };
    let dir = path.parent().unwrap_or(path);

    // Check for CUDA DLLs next to the binary
    let cuda_indicators = [
        "ggml-cuda.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudart64_12.dll",
    ];
    for dll in &cuda_indicators {
        if dir.join(dll).exists() {
            return "CUDA".to_string();
        }
    }

    // Check for Vulkan DLL
    if dir.join("ggml-vulkan.dll").exists() {
        return "Vulkan".to_string();
    }

    // Check path string as last resort
    let ps = path.to_string_lossy().to_lowercase();
    if ps.contains("cuda") {
        "CUDA".to_string()
    } else if ps.contains("vulkan") {
        "Vulkan".to_string()
    } else {
        "GPU".to_string()
    }
}

fn detect_api_port_owner(port: u16) -> Option<ApiPortOwnerInfo> {
    #[cfg(windows)]
    {
        return detect_api_port_owner_windows(port);
    }

    #[allow(unreachable_code)]
    None
}

#[cfg(windows)]
fn detect_api_port_owner_windows(port: u16) -> Option<ApiPortOwnerInfo> {
    let current_pid = std::process::id();
    let output = command_no_window("netstat")
        .args(["-ano", "-p", "tcp"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let port_suffix = format!(":{port}");
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let columns: Vec<&str> = line.split_whitespace().collect();
        if columns.len() < 5 {
            continue;
        }

        let proto = columns[0];
        let local_addr = columns[1];
        let state = columns[3];
        let Ok(pid) = columns[4].parse::<u32>() else {
            continue;
        };

        if !proto.eq_ignore_ascii_case("TCP")
            || !local_addr.ends_with(&port_suffix)
            || state != "LISTENING"
        {
            continue;
        }

        let name = process_name_for_pid(pid);
        let lower = name.as_deref().unwrap_or_default().to_lowercase();
        let (kind, killable) = if pid == current_pid {
            ("self".to_string(), false)
        } else if lower.contains("llama-server") {
            ("llama-server".to_string(), true)
        } else if lower.contains("inference-bridge") {
            ("inference-bridge".to_string(), true)
        } else if lower.is_empty() {
            return None;
        } else {
            ("other".to_string(), false)
        };

        return Some(ApiPortOwnerInfo {
            pid,
            name,
            kind,
            killable,
        });
    }

    None
}

#[cfg(windows)]
fn process_name_for_pid(pid: u32) -> Option<String> {
    let filter = format!("PID eq {pid}");
    let output = command_no_window("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    if line.is_empty() || line.starts_with("INFO:") {
        return None;
    }

    let parts: Vec<&str> = line.split(',').collect();
    let name = parts.first()?.trim_matches('"').trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Info about a running llama-server process on this machine.
#[derive(serde::Serialize)]
pub struct ExternalProcess {
    pub pid: u32,
    pub name: String,
    pub command_line: String,
    pub memory_mb: f64,
}

#[tauri::command]
pub async fn list_llama_processes() -> Result<Vec<ExternalProcess>, String> {
    // Use WMIC to list all llama-server processes
    let output = command_no_window("wmic")
        .args([
            "process",
            "where",
            "name like '%llama%server%'",
            "get",
            "ProcessId,Name,CommandLine,WorkingSetSize",
            "/FORMAT:CSV",
        ])
        .output()
        .map_err(|e| format!("Failed to query processes: {e}"))?;

    if !output.status.success() {
        // Fallback: use tasklist
        return list_llama_processes_tasklist();
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut processes = Vec::new();
    for line in text.lines().skip(1) {
        // CSV: Node,CommandLine,Name,ProcessId,WorkingSetSize
        // CommandLine can itself contain commas, so split from the RIGHT to safely
        // extract the fixed-position trailing fields.
        let parts_rev: Vec<&str> = line.rsplitn(5, ',').collect();
        // rsplitn gives: [WorkingSetSize, ProcessId, Name, CommandLine..., Node] (rev order)
        if parts_rev.len() >= 5 {
            let mem_bytes: f64 = parts_rev[0].trim().parse().unwrap_or(0.0);
            let pid: u32 = parts_rev[1].trim().parse().unwrap_or(0);
            let name = parts_rev[2].trim().to_string();
            let cmd = parts_rev[3].trim().to_string(); // may include commas from original
            if pid > 0
                && (name.to_lowercase().contains("llama") || cmd.to_lowercase().contains("llama"))
            {
                processes.push(ExternalProcess {
                    pid,
                    name,
                    command_line: cmd,
                    memory_mb: mem_bytes / (1024.0 * 1024.0),
                });
            }
        }
    }
    Ok(processes)
}

fn list_llama_processes_tasklist() -> Result<Vec<ExternalProcess>, String> {
    let output = command_no_window("tasklist")
        .args(["/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"])
        .output()
        .map_err(|e| format!("Failed to query processes: {e}"))?;

    let text = String::from_utf8_lossy(&output.stdout);
    let mut processes = Vec::new();
    for line in text.lines() {
        // CSV: "name","pid","session","session#","mem"
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 5 {
            let name = parts[0].trim_matches('"').to_string();
            let pid: u32 = parts[1].trim_matches('"').trim().parse().unwrap_or(0);
            let mem_str = parts[4]
                .trim_matches('"')
                .trim()
                .replace(" K", "")
                .replace(",", "")
                .replace(".", "");
            let mem_kb: f64 = mem_str.parse().unwrap_or(0.0);
            if pid > 0 {
                processes.push(ExternalProcess {
                    pid,
                    name,
                    command_line: String::new(),
                    memory_mb: mem_kb / 1024.0,
                });
            }
        }
    }
    Ok(processes)
}

#[tauri::command]
pub async fn kill_process(pid: u32) -> Result<String, String> {
    let output = command_no_window("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output()
        .map_err(|e| format!("Failed to kill process: {e}"))?;

    if output.status.success() {
        tracing::info!(pid = pid, "Killed process");
        Ok(format!("Killed process {pid}"))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Failed to kill PID {pid}: {stderr}"))
    }
}

#[tauri::command]
pub async fn kill_all_llama_processes() -> Result<String, String> {
    let procs = list_llama_processes().await?;
    if procs.is_empty() {
        return Ok("No llama-server processes found".to_string());
    }

    let mut killed = 0u32;
    let mut errors = Vec::new();
    for proc in &procs {
        match kill_process(proc.pid).await {
            Ok(_) => killed += 1,
            Err(e) => errors.push(e),
        }
    }

    if errors.is_empty() {
        Ok(format!("Killed {killed} process(es)"))
    } else {
        Ok(format!("Killed {killed}, errors: {}", errors.join("; ")))
    }
}

/// GPU VRAM usage stats queried from nvidia-smi.
#[derive(serde::Serialize)]
pub struct GpuStats {
    pub name: String,
    pub used_mb: u64,
    /// Dedicated VRAM (the fast on-board memory, from nvidia-smi memory.total).
    pub dedicated_mb: u64,
    pub free_mb: u64,
    /// Total system RAM — shown as the "overflow/spill" zone beyond dedicated VRAM.
    pub system_ram_mb: u64,
}

fn get_system_ram_mb() -> u64 {
    let output = command_no_window("wmic")
        .args(["OS", "get", "TotalVisibleMemorySize", "/VALUE"])
        .output();
    if let Ok(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            if let Some(val) = line.strip_prefix("TotalVisibleMemorySize=") {
                if let Ok(kb) = val.trim().parse::<u64>() {
                    return kb / 1024;
                }
            }
        }
    }
    0
}

#[tauri::command]
pub async fn get_gpu_stats() -> Result<GpuStats, String> {
    let output = command_no_window("nvidia-smi")
        .args([
            "--query-gpu=name,memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| format!("nvidia-smi not available: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("nvidia-smi error: {stderr}"));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next().ok_or("No output from nvidia-smi")?;
    let parts: Vec<&str> = line.splitn(4, ',').collect();
    if parts.len() < 4 {
        return Err(format!("Unexpected nvidia-smi output: {line}"));
    }

    Ok(GpuStats {
        name: parts[0].trim().to_string(),
        used_mb: parts[1].trim().parse().unwrap_or(0),
        dedicated_mb: parts[2].trim().parse().unwrap_or(0),
        free_mb: parts[3].trim().parse().unwrap_or(0),
        system_ram_mb: get_system_ram_mb(),
    })
}

/// Extract quantization level from a GGUF filename (e.g. "Q4_K_M", "Q5_0", "IQ4_XS").
fn extract_quant(filename: &str) -> Option<String> {
    let upper = filename.to_uppercase();
    // Match common GGUF quant patterns: Q4_K_M, Q5_0, IQ4_XS, F16, etc.
    let re = regex::Regex::new(
        r"[_.-]((?:I?Q\d+_[A-Z0-9]+(?:_[A-Z]+)?)|F(?:16|32)|BF16)(?:[_.-]|\.GGUF$)",
    )
    .ok()?;
    re.captures(&upper).map(|c| c[1].to_string())
}

#[derive(serde::Serialize)]
pub struct ModelInfo {
    pub filename: String,
    pub path: String,
    pub size_gb: f64,
    pub family: String,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub supports_vision: bool,
    pub context_window: Option<u32>,
    pub max_context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub quant: Option<String>,
    pub tool_call_format: String,
    pub think_tag_style: String,
}

#[derive(serde::Serialize)]
pub struct ProcessStatusInfo {
    pub state: String,
    pub model: Option<String>,
    pub previous_model: Option<String>,
    pub model_load_state: String,
    pub model_load_progress: Option<LoadProgress>,
    pub active_generation: Option<GenerationRequest>,
    pub crash_count: u32,
    pub server_version: Option<String>,
    pub server_path: Option<String>,
    pub backend: Option<String>,
    pub api_state: String,
    pub api_error: Option<String>,
    pub api_url: String,
    pub api_reachable: bool,
    pub api_port_owner: Option<ApiPortOwnerInfo>,
    pub startup_duration_ms: Option<u64>,
    pub parallel_slots: Option<u32>,
    pub slot_count: Option<u32>,
    pub active_requests: usize,
    pub queued_requests: usize,
    pub scheduler_limit: Option<u32>,
    pub last_launch_preview: Option<LaunchPreview>,
    pub last_generation_metrics: Option<RuntimePerformanceMetrics>,
}

#[derive(Clone, serde::Serialize)]
pub struct ApiPortOwnerInfo {
    pub pid: u32,
    pub name: Option<String>,
    pub kind: String,
    pub killable: bool,
}

#[derive(serde::Serialize)]
pub struct ServerInfo {
    pub found: bool,
    pub path: Option<String>,
    pub version: Option<String>,
}

#[tauri::command]
pub async fn get_effective_profile(
    state: tauri::State<'_, SharedState>,
    model_name: Option<String>,
) -> Result<EffectiveProfileInfo, String> {
    let s = state.read().await;
    effective_profile_info_from_state(&s, model_name.as_deref())
}

#[tauri::command]
pub async fn reload_last_known_good(
    state: tauri::State<'_, SharedState>,
) -> Result<String, String> {
    let preview = {
        let s = state.read().await;
        s.last_known_good_config.clone()
    }
    .ok_or_else(|| "No last known good launch configuration has been recorded yet".to_string())?;

    let model_name = std::path::Path::new(&preview.model_path)
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .ok_or_else(|| "Last known good config does not contain a valid model path".to_string())?;

    if !std::path::Path::new(&preview.model_path).exists() {
        return Err(format!(
            "The last known good model path no longer exists: {}",
            preview.model_path
        ));
    }

    backend_load_model(state.inner().clone(), model_name, None).await
}

#[tauri::command]
pub async fn check_llama_server(
    state: tauri::State<'_, SharedState>,
) -> Result<ServerInfo, String> {
    let s = state.read().await;
    let path = s.process.find_server_binary();
    let version = download::current_version();
    Ok(ServerInfo {
        found: path.is_some(),
        path: path.map(|p| p.to_string_lossy().to_string()),
        version,
    })
}

#[tauri::command]
pub async fn update_llama_server(app: tauri::AppHandle) -> Result<String, String> {
    match download::check_for_update().await {
        Ok(Some((tag, url, size))) => {
            match download::download_llama_server(&app, &url, &tag, size).await {
                Ok(path) => Ok(format!("Updated to {} at {}", tag, path.display())),
                Err(e) => Err(format!("Download failed: {e}")),
            }
        }
        Ok(None) => Ok("Already up-to-date".to_string()),
        Err(e) => Err(format!("Update check failed: {e}")),
    }
}

/// Swap to a different model. If no model_name is given, swaps back to the
/// previously loaded model. This is the same as load_model but with
/// swap-back semantics.
#[tauri::command]
pub async fn swap_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
    model_name: Option<String>,
    context_size: Option<u32>,
) -> Result<String, String> {
    // Determine target: explicit name or previous model
    let target = match model_name {
        Some(name) => name,
        None => {
            let s = state.read().await;
            s.previous_model
                .clone()
                .ok_or_else(|| "No previous model to swap back to".to_string())?
        }
    };

    tracing::info!(target = %target, "Hot-swap requested");

    let _ = app;
    backend_load_model(state.inner().clone(), target, context_size).await
}

