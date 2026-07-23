use std::path::PathBuf;
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
pub async fn generate_image(
    request: ImageGenerationRequest,
    state: tauri::State<'_, SharedState>,
) -> Result<ImageGenerationResult, String> {
    let shared_state = state.inner().clone();
    let (image_mutex, model_mutex) = {
        let app_state = shared_state.read().await;
        (
            app_state.image_generation_mutex.clone(),
            app_state.model_load_mutex.clone(),
        )
    };

    let _image_guard = image_mutex
        .try_lock()
        .map_err(|_| "An image generation job is already running".to_string())?;
    let _model_guard = model_mutex.lock().await;

    let (config, chat_runtime_busy) = {
        let app_state = shared_state.read().await;
        (
            app_state.config.image_generation.clone(),
            app_state.loaded_model.is_some()
                || app_state.process.has_child()
                || app_state.active_generation.is_some(),
        )
    };
    if chat_runtime_busy {
        return Err(
            "Unload the active chat model before generating an image. Automatic swap-and-restore is not enabled until recovery testing is complete."
                .to_string(),
        );
    }

    let output_dir = configured_output_dir(&config);
    let job_id = uuid::Uuid::new_v4().to_string();
    let output_path = output_dir.join(format!("{job_id}.png"));
    let resolved = resolve_job(&config, &request, output_path)?;
    tokio::fs::create_dir_all(&output_dir)
        .await
        .map_err(|error| format!("Failed to create image output directory: {error}"))?;
    let cancel = CancellationToken::new();
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

    run_native_job(shared_state, job_id, resolved, cancel).await
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
    ImageGenerationResult {
        job_id: job_id.to_string(),
        status: status.to_string(),
        bundle_id: resolved.bundle.id.clone(),
        profile_id: resolved.profile.id.clone(),
        prompt: resolved.prompt.clone(),
        seed: resolved.seed,
        width: resolved.profile.width,
        height: resolved.profile.height,
        steps: resolved.profile.steps,
        elapsed_seconds: started.elapsed().as_secs_f64(),
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
