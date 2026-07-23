use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tauri::Emitter;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::image_generation::{
    build_preview, capability_status, configured_output_dir, output_png_dimensions, resolve_job,
    ImageGenerationCapabilityStatus, ImageGenerationPreview, ImageGenerationProgress,
    ImageGenerationRequest, ImageGenerationResult, NativeProgressParser, ResolvedImageJob,
};
use crate::state::SharedState;

const LOG_TAIL_BYTES: usize = 128 * 1024;
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, serde::Serialize)]
pub struct DetectedImageLabSetup {
    pub runner_path: String,
    pub transformer_path: String,
    pub text_encoder_path: String,
    pub vae_path: String,
    pub output_dir: String,
}

#[tauri::command]
pub async fn detect_image_lab_setup() -> Result<DetectedImageLabSetup, String> {
    let mut search_roots = Vec::new();
    if let Ok(current_dir) = std::env::current_dir() {
        search_roots.push(current_dir);
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            search_roots.push(parent.to_path_buf());
        }
    }
    let candidates = search_roots
        .iter()
        .flat_map(|root| root.ancestors())
        .map(|ancestor| ancestor.join("image-gen-lab"))
        .collect::<Vec<_>>();
    let lab_root = candidates
        .into_iter()
        .find(|candidate| candidate.is_dir())
        .ok_or_else(|| "The tested image-gen-lab folder was not found".to_string())?;
    let runtime_root = lab_root.join("artifacts").join("runtime");
    let models_root = lab_root
        .join("artifacts")
        .join("models")
        .join("qwen-image-2512");
    let runner_path = find_named_file(&runtime_root, "sd-cli.exe")
        .ok_or_else(|| "The tested sd-cli.exe runtime was not found".to_string())?;
    let transformer_path = models_root.join("qwen-image-2512-Q6_K.gguf");
    let text_encoder_path = models_root.join("Qwen2.5-VL-7B-Instruct.Q8_0.gguf");
    let vae_path = models_root.join("qwen_image_vae.safetensors");
    for (label, path) in [
        ("Qwen-Image transformer", &transformer_path),
        ("Qwen text encoder", &text_encoder_path),
        ("Qwen image VAE", &vae_path),
    ] {
        if !path.is_file() {
            return Err(format!("{label} was not found at {}", path.display()));
        }
    }
    Ok(DetectedImageLabSetup {
        runner_path: runner_path.to_string_lossy().to_string(),
        transformer_path: transformer_path.to_string_lossy().to_string(),
        text_encoder_path: text_encoder_path.to_string_lossy().to_string(),
        vae_path: vae_path.to_string_lossy().to_string(),
        output_dir: lab_root
            .join("ib-generated-images")
            .to_string_lossy()
            .to_string(),
    })
}

#[tauri::command]
pub async fn get_image_generation_status(
    state: tauri::State<'_, SharedState>,
) -> Result<ImageGenerationCapabilityStatus, String> {
    let app_state = state.read().await;
    Ok(capability_status(
        &app_state.config.image_generation,
        app_state.image_generation_progress.clone(),
    ))
}

#[tauri::command]
pub async fn preview_image_generation(
    request: ImageGenerationRequest,
    state: tauri::State<'_, SharedState>,
) -> Result<ImageGenerationPreview, String> {
    let app_state = state.read().await;
    build_preview(&app_state.config.image_generation, &request)
}

#[tauri::command]
pub async fn cancel_image_generation(state: tauri::State<'_, SharedState>) -> Result<(), String> {
    let app_state = state.read().await;
    let Some(progress) = app_state.image_generation_progress.as_ref() else {
        return Ok(());
    };
    if !progress.done {
        app_state.image_generation_cancel.cancel();
    }
    Ok(())
}

#[tauri::command]
pub async fn read_generated_image_data_url(
    path: String,
    state: tauri::State<'_, SharedState>,
) -> Result<String, String> {
    use base64::Engine;

    let output_root = {
        let app_state = state.read().await;
        configured_output_dir(&app_state.config.image_generation)
    };
    let canonical_root = tokio::fs::canonicalize(&output_root)
        .await
        .map_err(|error| format!("Image output folder is unavailable: {error}"))?;
    let canonical_path = tokio::fs::canonicalize(path.trim())
        .await
        .map_err(|error| format!("Generated image is unavailable: {error}"))?;
    if !canonical_path.starts_with(&canonical_root)
        || canonical_path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| !extension.eq_ignore_ascii_case("png"))
            .unwrap_or(true)
    {
        return Err("Generated image path is outside the configured image folder".to_string());
    }
    let metadata = tokio::fs::metadata(&canonical_path)
        .await
        .map_err(|error| format!("Could not inspect generated image: {error}"))?;
    if metadata.len() > 50 * 1024 * 1024 {
        return Err("Generated image exceeds the 50 MiB display limit".to_string());
    }
    let bytes = tokio::fs::read(canonical_path)
        .await
        .map_err(|error| format!("Could not read generated image: {error}"))?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

#[tauri::command]
pub async fn generate_image(
    request: ImageGenerationRequest,
    state: tauri::State<'_, SharedState>,
) -> Result<ImageGenerationResult, String> {
    let shared_state = state.inner().clone();
    let (image_mutex, model_mutex, exclusive) = {
        let app_state = shared_state.read().await;
        (
            app_state.image_generation_mutex.clone(),
            app_state.model_load_mutex.clone(),
            app_state.image_generation_exclusive.clone(),
        )
    };

    let _image_guard = image_mutex
        .try_lock()
        .map_err(|_| "An image generation job is already running".to_string())?;
    let config = {
        let app_state = shared_state.read().await;
        app_state.config.image_generation.clone()
    };
    let session_id = request
        .session_id
        .clone()
        .filter(|session_id| !session_id.trim().is_empty());

    let output_dir = configured_output_dir(&config);
    let job_id = uuid::Uuid::new_v4().to_string();
    let output_path = output_dir.join(format!("{job_id}.png"));
    let resolved = resolve_job(&config, &request, output_path)?;
    tokio::fs::create_dir_all(&output_dir)
        .await
        .map_err(|error| format!("Failed to create image output directory: {error}"))?;

    // Wait for any model transition already in flight, then reserve the managed
    // GPU lifecycle before releasing the model lock again. New model loads
    // re-check this reservation after acquiring the same lock.
    let model_barrier = model_mutex.lock().await;
    let restore_snapshot = {
        let app_state = shared_state.read().await;
        if app_state.active_generation.is_some() {
            return Err(
                "Wait for the current chat response to finish before generating an image"
                    .to_string(),
            );
        }
        if app_state.process.has_child() && app_state.loaded_model.is_none() {
            return Err(
                "The chat runtime is still shutting down. Wait for it to finish, then try again."
                    .to_string(),
            );
        }
        if app_state.loaded_model.is_some() && !config.automatic_model_swap_enabled {
            return Err(
                "Automatic chat model swapping is safety-locked until its recovery tests pass. Unload the chat model first, then generate the image."
                    .to_string(),
            );
        }
        if app_state.loaded_model.is_some() && app_state.last_model_restore.is_none() {
            return Err(
                "The loaded chat model predates exact restore snapshots. Reload it once, then try image generation again."
                    .to_string(),
            );
        }
        app_state
            .loaded_model
            .as_ref()
            .and(app_state.last_model_restore.clone())
    };
    if exclusive
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err("The GPU is already reserved for image generation".to_string());
    }
    let _exclusive_guard = ImageExclusiveGuard(exclusive);
    drop(model_barrier);

    let cancel = CancellationToken::new();
    if let Some(session_id) = session_id.as_deref() {
        persist_image_prompt(&shared_state, session_id, &resolved.prompt).await?;
    }
    let progress = ImageGenerationProgress::starting(
        job_id.clone(),
        resolved.bundle.id.clone(),
        resolved.profile.id.clone(),
        resolved.profile.steps,
    );
    {
        let mut app_state = shared_state.write().await;
        app_state.image_generation_cancel = cancel.clone();
    }
    publish_progress(&shared_state, progress).await;

    if restore_snapshot.is_some() {
        update_running_stage(
            &shared_state,
            "unloading_chat",
            "Unloading chat model to free GPU memory",
            0.01,
        )
        .await;
        if let Err(error) = crate::commands::model::backend_unload_model(shared_state.clone()).await
        {
            let message = format!("Could not unload the chat model: {error}");
            let started = Instant::now();
            let final_progress = finished_progress(
                &shared_state,
                "failed",
                "failed",
                &message,
                started,
                None,
                Some(message.clone()),
            )
            .await;
            publish_progress(&shared_state, final_progress).await;
            let failed_result =
                result_for(&job_id, &resolved, "failed", started, None, Some(message));
            return restore_chat_after_image(shared_state.clone(), failed_result, restore_snapshot)
                .await;
        }
    }

    let native_result = match run_native_job(
        shared_state.clone(),
        job_id.clone(),
        resolved.clone(),
        cancel,
    )
    .await
    {
        Ok(result) => result,
        Err(error) => {
            let message = format!("Image runner failed: {error}");
            let final_progress = finished_progress(
                &shared_state,
                "failed",
                "failed",
                &message,
                Instant::now(),
                None,
                Some(message.clone()),
            )
            .await;
            publish_progress(&shared_state, final_progress).await;
            result_for(
                &job_id,
                &resolved,
                "failed",
                Instant::now(),
                None,
                Some(message),
            )
        }
    };

    let final_result =
        restore_chat_after_image(shared_state.clone(), native_result, restore_snapshot).await?;
    if let (Some(session_id), Some(output_path)) =
        (session_id.as_deref(), final_result.output_path.as_deref())
    {
        if let Err(error) =
            persist_generated_image(&shared_state, session_id, &final_result, output_path).await
        {
            tracing::warn!(%error, session_id, "Generated image could not be attached to the chat");
        }
    }
    Ok(final_result)
}

struct ImageExclusiveGuard(Arc<AtomicBool>);

impl Drop for ImageExclusiveGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

async fn restore_chat_after_image(
    state: SharedState,
    mut image_result: ImageGenerationResult,
    restore_snapshot: Option<crate::commands::model::ModelRestoreSnapshot>,
) -> Result<ImageGenerationResult, String> {
    let Some(snapshot) = restore_snapshot else {
        return Ok(image_result);
    };
    let final_before_restore = {
        let app_state = state.read().await;
        app_state.image_generation_progress.clone()
    };
    update_running_stage(
        &state,
        "restoring_chat",
        "Image job finished — restoring the chat model",
        0.99,
    )
    .await;

    match crate::commands::model::backend_restore_model_snapshot(state.clone(), snapshot).await {
        Ok(_) => {
            if let Some(mut progress) = final_before_restore {
                progress.message = match progress.status.as_str() {
                    "completed" => "Image ready — chat model restored".to_string(),
                    "cancelled" => "Image cancelled — chat model restored".to_string(),
                    _ => "Image failed — chat model restored".to_string(),
                };
                progress.updated_at = chrono::Utc::now().to_rfc3339();
                publish_progress(&state, progress).await;
            }
        }
        Err(restore_error) => {
            let message = format!(
                "{} Chat model restoration failed: {restore_error}",
                image_result
                    .error
                    .as_deref()
                    .unwrap_or("The image job finished.")
            );
            image_result.status = "failed".to_string();
            image_result.error = Some(message.clone());
            let mut progress = final_before_restore.unwrap_or_else(|| {
                ImageGenerationProgress::starting(
                    image_result.job_id.clone(),
                    image_result.bundle_id.clone(),
                    image_result.profile_id.clone(),
                    image_result.steps,
                )
            });
            progress.status = "failed".to_string();
            progress.stage = "failed".to_string();
            progress.message = "Chat model restoration failed".to_string();
            progress.done = true;
            progress.error = Some(message);
            progress.eta_seconds = None;
            progress.updated_at = chrono::Utc::now().to_rfc3339();
            publish_progress(&state, progress).await;
        }
    }
    Ok(image_result)
}

async fn update_running_stage(state: &SharedState, stage: &str, message: &str, progress: f32) {
    let current = {
        let app_state = state.read().await;
        app_state.image_generation_progress.clone()
    };
    if let Some(mut current) = current {
        current.status = "running".to_string();
        current.stage = stage.to_string();
        current.message = message.to_string();
        current.progress = progress;
        current.done = false;
        current.error = None;
        current.eta_seconds = None;
        current.updated_at = chrono::Utc::now().to_rfc3339();
        publish_progress(state, current).await;
    }
}

async fn persist_image_prompt(
    state: &SharedState,
    session_id: &str,
    prompt: &str,
) -> Result<(), String> {
    let app_handle = {
        let app_state = state.read().await;
        let db = app_state
            .session_db
            .lock()
            .map_err(|error| error.to_string())?;
        db.add_message(session_id, "user", prompt, 0, None)
            .map_err(|error| error.to_string())?;
        app_state.app_handle.clone()
    };
    if let Some(app_handle) = app_handle {
        let _ = app_handle.emit("session-messages-changed", session_id.to_string());
    }
    Ok(())
}

async fn persist_generated_image(
    state: &SharedState,
    session_id: &str,
    result: &ImageGenerationResult,
    output_path: &str,
) -> Result<(), String> {
    let content = format!(
        "Generated image — {}×{}, {} steps, seed {}",
        result.width, result.height, result.steps, result.seed
    );
    let metadata = serde_json::json!({
        "job_id": result.job_id,
        "bundle_id": result.bundle_id,
        "bundle_name": result.bundle_name,
        "quantization": result.quantization,
        "profile_id": result.profile_id,
        "prompt": result.prompt,
        "negative_prompt": result.negative_prompt,
        "seed": result.seed,
        "width": result.width,
        "height": result.height,
        "steps": result.steps,
        "cfg_scale": result.cfg_scale,
        "sampling_method": result.sampling_method,
        "elapsed_seconds": result.elapsed_seconds,
        "file_size_bytes": result.file_size_bytes,
        "completed_at": chrono::Utc::now().to_rfc3339(),
    })
    .to_string();
    let app_handle = {
        let app_state = state.read().await;
        let db = app_state
            .session_db
            .lock()
            .map_err(|error| error.to_string())?;
        db.add_generated_image_message(session_id, &content, output_path, &metadata)
            .map_err(|error| error.to_string())?;
        app_state.app_handle.clone()
    };
    if let Some(app_handle) = app_handle {
        let _ = app_handle.emit("session-messages-changed", session_id.to_string());
    }
    Ok(())
}

async fn run_native_job(
    state: SharedState,
    job_id: String,
    resolved: ResolvedImageJob,
    cancel: CancellationToken,
) -> Result<ImageGenerationResult, String> {
    let started = Instant::now();
    let mut command = Command::new(&resolved.runner_path);
    command
        .args(&resolved.arguments)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.as_std_mut().creation_flags(0x08000000);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let message = format!("Failed to start image runner: {error}");
            publish_progress(
                &state,
                finished_progress(
                    &state,
                    "failed",
                    "failed",
                    &message,
                    started,
                    None,
                    Some(message.clone()),
                )
                .await,
            )
            .await;
            return Ok(result_for(
                &job_id,
                &resolved,
                "failed",
                started,
                None,
                Some(message),
            ));
        }
    };

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Image runner stdout was not captured".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Image runner stderr was not captured".to_string())?;
    let stdout_task = tokio::spawn(read_log_tail(stdout));
    let progress_state = state.clone();
    let stderr_task = tokio::spawn(async move {
        let mut stream = stderr;
        let mut parser = NativeProgressParser::default();
        let mut log_tail = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let count = match stream.read(&mut buffer).await {
                Ok(0) => break,
                Ok(count) => count,
                Err(error) => {
                    append_log_tail(
                        &mut log_tail,
                        format!("\n[stderr read failed: {error}]").as_bytes(),
                    );
                    break;
                }
            };
            append_log_tail(&mut log_tail, &buffer[..count]);
            let text = String::from_utf8_lossy(&buffer[..count]);
            if let Some(parsed) = parser.push(&text) {
                let mut progress = {
                    let app_state = progress_state.read().await;
                    app_state.image_generation_progress.clone()
                };
                if let Some(ref mut progress) = progress {
                    progress.apply_step(parsed, started.elapsed().as_secs_f64());
                    publish_progress(&progress_state, progress.clone()).await;
                }
            }
        }
        String::from_utf8_lossy(&log_tail).into_owned()
    });

    let deadline = Duration::from_secs(resolved.profile.timeout_seconds);
    let process_outcome = loop {
        if cancel.is_cancelled() {
            let _ = child.kill().await;
            let _ = child.wait().await;
            break ProcessOutcome::Cancelled;
        }
        if started.elapsed() >= deadline {
            let _ = child.kill().await;
            let _ = child.wait().await;
            break ProcessOutcome::TimedOut;
        }
        match child.try_wait() {
            Ok(Some(status)) => break ProcessOutcome::Exited(status),
            Ok(None) => tokio::time::sleep(PROCESS_POLL_INTERVAL).await,
            Err(error) => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                break ProcessOutcome::WaitFailed(error.to_string());
            }
        }
    };

    let stdout_log = stdout_task
        .await
        .unwrap_or_else(|error| format!("[stdout task failed: {error}]"));
    let stderr_log = stderr_task
        .await
        .unwrap_or_else(|error| format!("[stderr task failed: {error}]"));

    match process_outcome {
        ProcessOutcome::Cancelled => {
            remove_partial_output(&resolved.output_path).await;
            let message = "Image generation cancelled".to_string();
            let final_progress = finished_progress(
                &state,
                "cancelled",
                "cancelled",
                &message,
                started,
                None,
                None,
            )
            .await;
            publish_progress(&state, final_progress).await;
            Ok(result_for(
                &job_id,
                &resolved,
                "cancelled",
                started,
                None,
                None,
            ))
        }
        ProcessOutcome::TimedOut => {
            remove_partial_output(&resolved.output_path).await;
            let message = format!(
                "Image generation timed out after {} seconds",
                resolved.profile.timeout_seconds
            );
            let final_progress = finished_progress(
                &state,
                "failed",
                "failed",
                &message,
                started,
                None,
                Some(message.clone()),
            )
            .await;
            publish_progress(&state, final_progress).await;
            Ok(result_for(
                &job_id,
                &resolved,
                "failed",
                started,
                None,
                Some(message),
            ))
        }
        ProcessOutcome::WaitFailed(error) => {
            remove_partial_output(&resolved.output_path).await;
            let message = format!("Failed while waiting for image runner: {error}");
            let final_progress = finished_progress(
                &state,
                "failed",
                "failed",
                &message,
                started,
                None,
                Some(message.clone()),
            )
            .await;
            publish_progress(&state, final_progress).await;
            Ok(result_for(
                &job_id,
                &resolved,
                "failed",
                started,
                None,
                Some(message),
            ))
        }
        ProcessOutcome::Exited(status) if !status.success() => {
            remove_partial_output(&resolved.output_path).await;
            let log = useful_error_log(&stderr_log, &stdout_log);
            let message = format!(
                "Image runner exited with {}{}",
                status,
                if log.is_empty() {
                    String::new()
                } else {
                    format!(": {log}")
                }
            );
            let final_progress = finished_progress(
                &state,
                "failed",
                "failed",
                &message,
                started,
                None,
                Some(message.clone()),
            )
            .await;
            publish_progress(&state, final_progress).await;
            Ok(result_for(
                &job_id,
                &resolved,
                "failed",
                started,
                None,
                Some(message),
            ))
        }
        ProcessOutcome::Exited(_) => {
            let mut saving = {
                let app_state = state.read().await;
                app_state
                    .image_generation_progress
                    .clone()
                    .expect("image progress exists while the job is running")
            };
            saving.stage = "saving".to_string();
            saving.message = "Checking generated image".to_string();
            saving.progress = 0.98;
            saving.elapsed_seconds = started.elapsed().as_secs_f64();
            saving.eta_seconds = None;
            saving.updated_at = chrono::Utc::now().to_rfc3339();
            publish_progress(&state, saving).await;

            let validation =
                output_png_dimensions(&resolved.output_path).and_then(|(width, height)| {
                    if (width, height) == (resolved.profile.width, resolved.profile.height) {
                        Ok(())
                    } else {
                        Err(format!(
                            "Generated image dimensions were {width}x{height}; expected {}x{}",
                            resolved.profile.width, resolved.profile.height
                        ))
                    }
                });
            if let Err(error) = validation {
                remove_partial_output(&resolved.output_path).await;
                let final_progress = finished_progress(
                    &state,
                    "failed",
                    "failed",
                    &error,
                    started,
                    None,
                    Some(error.clone()),
                )
                .await;
                publish_progress(&state, final_progress).await;
                return Ok(result_for(
                    &job_id,
                    &resolved,
                    "failed",
                    started,
                    None,
                    Some(error),
                ));
            }

            let output = resolved.output_path.to_string_lossy().to_string();
            let final_progress = finished_progress(
                &state,
                "completed",
                "completed",
                "Image ready",
                started,
                Some(output.clone()),
                None,
            )
            .await;
            publish_progress(&state, final_progress).await;
            Ok(result_for(
                &job_id,
                &resolved,
                "completed",
                started,
                Some(output),
                None,
            ))
        }
    }
}

enum ProcessOutcome {
    Exited(std::process::ExitStatus),
    Cancelled,
    TimedOut,
    WaitFailed(String),
}

async fn publish_progress(state: &SharedState, progress: ImageGenerationProgress) {
    let app_handle = {
        let mut app_state = state.write().await;
        app_state.image_generation_progress = Some(progress.clone());
        app_state.app_handle.clone()
    };
    if let Some(app_handle) = app_handle {
        let _ = app_handle.emit("image-generation-progress", progress);
    }
}

async fn finished_progress(
    state: &SharedState,
    status: &str,
    stage: &str,
    message: &str,
    started: Instant,
    output_path: Option<String>,
    error: Option<String>,
) -> ImageGenerationProgress {
    let mut progress = {
        let app_state = state.read().await;
        app_state
            .image_generation_progress
            .clone()
            .unwrap_or_else(|| {
                ImageGenerationProgress::starting(
                    "unknown".to_string(),
                    "unknown".to_string(),
                    "unknown".to_string(),
                    0,
                )
            })
    };
    progress.status = status.to_string();
    progress.stage = stage.to_string();
    progress.message = message.to_string();
    progress.progress = if status == "completed" {
        1.0
    } else {
        progress.progress
    };
    progress.elapsed_seconds = started.elapsed().as_secs_f64();
    progress.eta_seconds = None;
    progress.updated_at = chrono::Utc::now().to_rfc3339();
    progress.done = true;
    progress.error = error;
    progress.output_path = output_path;
    progress
}

fn result_for(
    job_id: &str,
    resolved: &ResolvedImageJob,
    status: &str,
    started: Instant,
    output_path: Option<String>,
    error: Option<String>,
) -> ImageGenerationResult {
    let file_size_bytes = output_path
        .as_deref()
        .and_then(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len());
    ImageGenerationResult {
        job_id: job_id.to_string(),
        status: status.to_string(),
        bundle_id: resolved.bundle.id.clone(),
        bundle_name: resolved.bundle.name.clone(),
        quantization: resolved.bundle.quantization.clone(),
        profile_id: resolved.profile.id.clone(),
        prompt: resolved.prompt.clone(),
        negative_prompt: resolved.negative_prompt.clone(),
        seed: resolved.seed,
        width: resolved.profile.width,
        height: resolved.profile.height,
        steps: resolved.profile.steps,
        cfg_scale: resolved.profile.cfg_scale,
        sampling_method: resolved.profile.sampling_method.clone(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        file_size_bytes,
        output_path,
        error,
    }
}

async fn read_log_tail<R>(mut stream: R) -> String
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut log_tail = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        match stream.read(&mut buffer).await {
            Ok(0) => break,
            Ok(count) => append_log_tail(&mut log_tail, &buffer[..count]),
            Err(error) => {
                append_log_tail(
                    &mut log_tail,
                    format!("\n[stream read failed: {error}]").as_bytes(),
                );
                break;
            }
        }
    }
    String::from_utf8_lossy(&log_tail).into_owned()
}

fn append_log_tail(log: &mut Vec<u8>, chunk: &[u8]) {
    log.extend_from_slice(chunk);
    if log.len() > LOG_TAIL_BYTES {
        let excess = log.len() - LOG_TAIL_BYTES;
        log.drain(..excess);
    }
}

fn useful_error_log(stderr: &str, stdout: &str) -> String {
    let source = if stderr.trim().is_empty() {
        stdout
    } else {
        stderr
    };
    source
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or_default()
        .trim()
        .chars()
        .take(1_000)
        .collect()
}

async fn remove_partial_output(path: &PathBuf) {
    if tokio::fs::metadata(path).await.is_ok() {
        let _ = tokio::fs::remove_file(path).await;
    }
}

fn find_named_file(root: &std::path::Path, filename: &str) -> Option<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.eq_ignore_ascii_case(filename))
            {
                return Some(path);
            }
            if path.is_dir() {
                pending.push(path);
            }
        }
    }
    None
}
