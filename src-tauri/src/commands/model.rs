/// Emit a model-load-progress event to the GUI (no-op if handle is None / headless).
fn emit_load_progress(
    handle: &Option<tauri::AppHandle>,
    stage: &str,
    msg: &str,
    progress: f32,
    done: bool,
    error: Option<String>,
) {
    use tauri::Emitter;
    if let Some(h) = handle {
        let _ = h.emit(
            "model-load-progress",
            crate::state::LoadProgress {
                stage: stage.to_string(),
                message: msg.to_string(),
                progress,
                done,
                error,
            },
        );
    }
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
    // Grab the app handle once (no lock held after this point).
    let app_handle = {
        let s = state.read().await;
        s.app_handle.clone()
    };

    emit_load_progress(
        &app_handle,
        "resolving",
        "Resolving model...",
        0.0,
        false,
        None,
    );

    // Phase 1: Resolve model info (brief lock) + claim loading generation
    let (config, model_filename, my_generation) = {
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
                    emit_load_progress(&app_handle, "error", &msg, 0.0, true, Some(msg.clone()));
                    msg
                })?
        };

        let mut s = state.write().await;

        let ctx = context_size
            .or(model.profile.default_context_window)
            .unwrap_or(s.config.models.default_context);

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

        // Bump generation so older in-flight loads won't overwrite us
        s.loading_generation += 1;
        let gen = s.loading_generation;
        tracing::info!(
            model = %model.filename,
            generation = gen,
            ctx_size = ctx,
            "Phase 1: Resolved model (API/backend)"
        );

        (config, model.filename.clone(), gen)
    }; // write lock released

    let ctx = config.context_size;
    let size_info = format!("{} (ctx: {})", model_filename, ctx);

    // Phase 2: Launch process (brief write lock, then released)
    emit_load_progress(
        &app_handle,
        "launching",
        &format!("Launching llama-server for {}...", model_filename),
        0.05,
        false,
        None,
    );
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
            return Err(msg);
        }

        // Check if process crashed
        {
            let mut s = state.write().await;
            if s.process.check_crashed().await {
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
                return Err(msg);
            }
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
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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
    }

    crate::context::tracker::reset_slots_warning();

    let result = format!("Loaded {size_info}");
    emit_load_progress(&app_handle, "ready", &result, 1.0, true, None);
    tracing::info!("{result}");
    Ok(result)
}
// Tauri commands for model management.

use crate::engine::download;
use crate::engine::process::{LaunchConfig, LlamaProcess};
use crate::models::scanner;
use crate::state::{LoadProgress, SharedState};
use tauri::Emitter;

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
            let supports_vision = model_supports_vision(&m.filename);
            ModelInfo {
                filename: m.filename.clone(),
                path: m.path.to_string_lossy().to_string(),
                size_gb: m.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                family: m.profile.family.to_string(),
                supports_tools,
                supports_reasoning,
                supports_vision,
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
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
    model_name: String,
    context_size: Option<u32>,
) -> Result<String, String> {
    let emit = |stage: &str, msg: &str, progress: f32| {
        let _ = app.emit(
            "model-load-progress",
            LoadProgress {
                stage: stage.to_string(),
                message: msg.to_string(),
                progress,
                done: false,
                error: None,
            },
        );
    };

    let emit_done = |msg: &str| {
        let _ = app.emit(
            "model-load-progress",
            LoadProgress {
                stage: "ready".to_string(),
                message: msg.to_string(),
                progress: 1.0,
                done: true,
                error: None,
            },
        );
    };

    let emit_error = |msg: &str| {
        let _ = app.emit(
            "model-load-progress",
            LoadProgress {
                stage: "error".to_string(),
                message: msg.to_string(),
                progress: 0.0,
                done: true,
                error: Some(msg.to_string()),
            },
        );
    };

    // Phase 1: Resolve model info (brief lock) + claim loading generation
    emit("resolving", "Resolving model...", 0.0);
    let (config, model_filename, my_generation) = {
        let mut s = state.write().await;
        let model = s
            .model_registry
            .find_by_name(&model_name)
            .ok_or_else(|| {
                let msg = format!("Model not found: {model_name}");
                emit_error(&msg);
                msg
            })?
            .clone();

        let ctx = context_size
            .or(model.profile.default_context_window)
            .unwrap_or(s.config.models.default_context);

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

        // Bump generation so older in-flight loads won't overwrite us
        s.loading_generation += 1;
        let gen = s.loading_generation;
        tracing::info!(
            model = %model.filename,
            generation = gen,
            ctx_size = ctx,
            "Phase 1: Resolved model"
        );

        (config, model.filename.clone(), gen)
    }; // write lock released

    let ctx = config.context_size;
    let size_info = format!("{} (ctx: {})", model_filename, ctx);

    // Phase 2: Ensure llama-server binary exists, download CUDA if needed, then launch
    emit(
        "launching",
        &format!("Launching llama-server for {}...", model_filename),
        0.05,
    );
    {
        let mut s = state.write().await;
        let backend_pref = s.config.process.backend_preference.to_lowercase();
        let found = s.process.find_server_binary_with_preference(&backend_pref);
        let has_managed_cuda = LlamaProcess::managed_binary_dir()
            .join("llama-server.exe")
            .exists();

        // Auto-download based on backend preference.
        // auto: prefer managed CUDA over system binaries for best performance.
        let should_download = match backend_pref.as_str() {
            "cuda" => found.is_none(),
            "cpu" | "avx2" => found.is_none(),
            _ => found.is_none() || (!has_managed_cuda && found.is_some()),
        };

        if should_download {
            let reason = if found.is_none() {
                "llama-server not found"
            } else {
                "Preferred backend build not found — downloading"
            };
            // Release the write lock before downloading
            drop(s);

            emit(
                "downloading",
                &format!("{reason}. Checking for download..."),
                0.02,
            );

            let pattern = match backend_pref.as_str() {
                "cpu" | "avx2" => download::asset_pattern_for("cpu"),
                "cuda" => download::asset_pattern_for("cuda"),
                _ => download::asset_pattern_for("cuda"), // auto prefers CUDA
            };

            match download::find_release_asset(&pattern).await {
                Ok((tag, url, size)) => {
                    tracing::info!(
                        version = %tag,
                        backend = %backend_pref,
                        reason,
                        "Downloading preferred llama-server build"
                    );
                    match download::download_llama_server(&app, &url, &tag, size).await {
                        Ok(server_path) => {
                            // Re-acquire write lock to set the path
                            let mut s = state.write().await;
                            s.process.set_server_path(server_path);
                        }
                        Err(e) => {
                            // If download fails but we have a WinGet binary, use it as fallback
                            if found.is_some() {
                                tracing::warn!(error = %e, "CUDA download failed, falling back to existing binary");
                            } else {
                                let msg = format!("Failed to download llama-server: {e}");
                                emit_error(&msg);
                                return Err(msg);
                            }
                        }
                    }

                    // For CUDA/auto, also try to install CUDA runtime DLLs when available.
                    if backend_pref != "cpu" && backend_pref != "avx2" {
                        match download::find_cuda_runtime_asset().await {
                            Ok(Some((rt_tag, rt_url, rt_size))) => {
                                tracing::info!(
                                    version = %rt_tag,
                                    "Also downloading CUDA runtime DLLs"
                                );
                                let _ = download::download_llama_server(
                                    &app, &rt_url, &rt_tag, rt_size,
                                )
                                .await;
                            }
                            Ok(None) => tracing::debug!("No separate CUDA runtime package found"),
                            Err(e) => {
                                tracing::warn!(error = %e, "Could not check for CUDA runtime package")
                            }
                        }
                    }
                }
                Err(e) => {
                    if found.is_none() {
                        let msg = "llama-server binary not found and no update available. \
                                   Install llama.cpp manually or run: winget install ggml.llamacpp"
                            .to_string();
                        tracing::error!(error = %e, backend = %backend_pref, "Backend download failed");
                        emit_error(&msg);
                        return Err(msg);
                    }
                    tracing::warn!(error = %e, "Backend download failed, using existing binary");
                }
            }

            // Re-acquire write lock for launch
            let mut s = state.write().await;
            emit(
                "launching",
                &format!("Launching llama-server for {}...", model_filename),
                0.05,
            );
            s.process.launch(config).await.map_err(|e| {
                let msg = format!("Failed to launch llama-server: {e}");
                emit_error(&msg);
                msg
            })?;
        } else {
            s.process.launch(config).await.map_err(|e| {
                let msg = format!("Failed to launch llama-server: {e}");
                emit_error(&msg);
                msg
            })?;
        }
    } // write lock released

    // Phase 3: Wait for healthy (NO lock held - UI can poll freely)
    emit(
        "loading",
        &format!("Loading {} into memory...", model_filename),
        0.1,
    );
    tracing::info!("Phase 3: Waiting for llama-server health check on port 8801...");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap_or_default();
    let health_url = format!("http://127.0.0.1:8801/health");
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(300); // 5 min for large models
    let mut attempt = 0u32;

    loop {
        if start.elapsed() > timeout {
            let msg = format!("llama-server did not become healthy within {:?}", timeout);
            tracing::error!("{msg}");
            emit_error(&msg);
            // Try to clean up
            let mut s = state.write().await;
            let _ = s.process.shutdown().await;
            return Err(msg);
        }

        // Check if the process has died
        {
            let mut s = state.write().await;
            if s.process.check_crashed().await {
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
                emit_error(&msg);
                return Err(msg);
            }
        }

        match client.get(&health_url).send().await {
            Ok(resp) => {
                let status_code = resp.status();
                let body = resp.text().await.unwrap_or_default();

                if attempt < 5 || attempt % 10 == 0 {
                    tracing::debug!(
                        attempt,
                        status = %status_code,
                        body = %body,
                        "Health check response"
                    );
                }

                if status_code.is_success() {
                    if body.contains("ok") || body.contains("\"status\":\"ok\"") {
                        tracing::info!(
                            "llama-server is healthy after {}s",
                            start.elapsed().as_secs()
                        );
                        break; // Fully ready
                    }
                    // llama-server returns {"status":"loading model"} while loading
                    if body.contains("loading") {
                        let elapsed = start.elapsed().as_secs();
                        let progress = (0.1 + (elapsed as f32 * 0.01)).min(0.9);
                        emit(
                            "loading",
                            &format!("Loading model into GPU memory... ({}s)", elapsed),
                            progress,
                        );
                    } else {
                        // 200 but unknown body — assume ready
                        tracing::info!(
                            body = %body,
                            "Got 200 with unknown body, assuming ready"
                        );
                        break;
                    }
                } else if status_code.as_u16() == 503 {
                    // Server is up but loading weights
                    let elapsed = start.elapsed().as_secs();
                    let progress = (0.1 + (elapsed as f32 * 0.01)).min(0.9);
                    emit(
                        "loading",
                        &format!("Loading model weights... ({}s)", elapsed),
                        progress,
                    );
                    if attempt % 10 == 0 {
                        tracing::info!("llama-server loading weights... ({elapsed}s)");
                    }
                } else {
                    let elapsed = start.elapsed().as_secs();
                    emit(
                        "starting",
                        &format!(
                            "Waiting for llama-server (HTTP {})... ({}s)",
                            status_code, elapsed
                        ),
                        0.08,
                    );
                    tracing::warn!(
                        status = %status_code,
                        body = %body,
                        "Unexpected health check response"
                    );
                }
            }
            Err(e) => {
                // Server not up yet — connection refused is normal during startup
                let elapsed = start.elapsed().as_secs();
                let progress = (0.02 + (elapsed as f32 * 0.005)).min(0.09);

                if attempt < 5 {
                    tracing::debug!(
                        attempt,
                        error = %e,
                        "Health check connection failed (expected during startup)"
                    );
                    emit(
                        "starting",
                        &format!("Starting llama-server process... ({}s)", elapsed),
                        progress,
                    );
                } else if attempt == 5 {
                    tracing::info!(
                        "Still waiting for llama-server to accept connections ({elapsed}s)..."
                    );
                    emit(
                        "starting",
                        &format!("Waiting for llama-server to respond... ({}s)", elapsed),
                        progress,
                    );
                } else if attempt % 20 == 0 {
                    tracing::warn!(
                        elapsed_s = elapsed,
                        error = %e,
                        "llama-server still not responding"
                    );
                    emit(
                        "starting",
                        &format!("Waiting for llama-server to respond... ({}s)", elapsed),
                        progress,
                    );
                } else {
                    emit(
                        "starting",
                        &format!("Waiting for llama-server to respond... ({}s)", elapsed),
                        progress,
                    );
                }
            }
        }

        attempt += 1;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    // Phase 4: Mark as loaded (brief write lock)
    // Check generation — if another load started after us, we lost the race.
    {
        let mut s = state.write().await;
        if s.loading_generation != my_generation {
            tracing::warn!(
                model = %model_filename,
                our_gen = my_generation,
                current_gen = s.loading_generation,
                "Stale load completed — a newer swap is in progress, discarding"
            );
            return Ok(format!(
                "Superseded by newer swap (loaded {} for gen {})",
                model_filename, my_generation
            ));
        }
        // Track previous model for swap-back
        if let Some(prev) = s.loaded_model.take() {
            if prev != model_filename {
                tracing::info!(previous = %prev, new = %model_filename, "Tracking previous model for swap-back");
                s.previous_model = Some(prev);
            }
        }
        s.loaded_model = Some(model_filename.clone());
        s.process.set_state_running();
    }

    crate::context::tracker::reset_slots_warning();

    let result = format!("Loaded {size_info}");
    emit_done(&result);
    tracing::info!("{result}");
    Ok(result)
}

pub async fn backend_unload_model(state: SharedState) -> Result<String, String> {
    let mut s = state.write().await;
    s.process
        .shutdown()
        .await
        .map_err(|e| format!("Failed to unload: {e}"))?;

    let unloaded = s.loaded_model.take();
    if let Some(name) = unloaded.clone() {
        s.previous_model = Some(name.clone());
    }
    s.model_load_state = crate::state::ModelLoadState::Idle;
    s.model_load_progress = None;
    s.model_stats = None;

    Ok(match unloaded {
        Some(name) if !name.is_empty() => format!("Unloaded {name}"),
        _ => "Unloaded active model".to_string(),
    })
}

#[tauri::command]
pub async fn unload_model(state: tauri::State<'_, SharedState>) -> Result<String, String> {
    backend_unload_model(state.inner().clone()).await
}

#[tauri::command]
pub async fn get_process_status(
    state: tauri::State<'_, SharedState>,
) -> Result<ProcessStatusInfo, String> {
    // Extract everything we need from the lock, then drop it before any I/O
    let (
        state_str,
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
    ) = {
        let s = state.read().await;
        let server_path = s.process.find_server_binary();
        let is_running = s.process.state() != crate::engine::process::ProcessState::Idle;
        (
            format!("{:?}", s.process.state()),
            server_path,
            is_running,
            s.process.port(),
            s.loaded_model.clone(),
            s.previous_model.clone(),
            s.process.crash_count(),
            s.process.detected_backend(),
            format!("{:?}", s.api_server_state),
            s.api_server_error.clone(),
            format!(
                "http://{}:{}/v1",
                s.config.server.host, s.config.server.port
            ),
            s.config.server.port,
        )
    }; // read lock released before any async I/O

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

    let api_port_owner = if api_state == "Error" {
        detect_api_port_owner(api_port)
    } else {
        None
    };

    Ok(ProcessStatusInfo {
        state: state_str,
        model: loaded_model,
        previous_model,
        crash_count,
        server_version,
        server_path: server_path.map(|p| p.to_string_lossy().to_string()),
        backend,
        api_state,
        api_error,
        api_url,
        api_port_owner,
    })
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
            ("unknown".to_string(), false)
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
        .args(["/PID", &pid.to_string(), "/F"])
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

fn model_supports_vision(filename: &str) -> bool {
    let name = filename.to_lowercase();
    name.contains("vision")
        || name.contains("llava")
        || name.contains("multimodal")
        || name.contains("qwen2.5-vl")
        || name.contains("-vl")
        || name.contains("_vl")
}

#[derive(serde::Serialize)]
pub struct ProcessStatusInfo {
    pub state: String,
    pub model: Option<String>,
    pub previous_model: Option<String>,
    pub crash_count: u32,
    pub server_version: Option<String>,
    pub server_path: Option<String>,
    pub backend: Option<String>,
    pub api_state: String,
    pub api_error: Option<String>,
    pub api_url: String,
    pub api_port_owner: Option<ApiPortOwnerInfo>,
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

    // Delegate to load_model which handles shutdown + relaunch atomically
    load_model(app, state, target, context_size).await
}
