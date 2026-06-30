//! Tauri command for model benchmarking

use crate::engine::benchmark::{test_model, ModelTestStats};
use crate::state::SharedState;
use std::time::Duration;
use tauri::State;

const BENCHMARK_LOAD_TIMEOUT_SECS: u64 = 240;
const BENCHMARK_TEST_TIMEOUT_SECS: u64 = 240;

fn benchmark_model_names_match(loaded: &str, requested: &str) -> bool {
    let normalize = |value: &str| {
        std::path::Path::new(value)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| value.to_string())
            .trim_end_matches(".gguf")
            .to_ascii_lowercase()
    };
    normalize(loaded) == normalize(requested)
}

async fn benchmark_crash_tail(shared_state: &SharedState) -> Option<String> {
    let (stderr, exit_status, preview) = {
        let s = shared_state.read().await;
        (
            s.process.last_stderr().await,
            s.process.last_exit_status(),
            s.last_launch_preview.clone(),
        )
    };
    let report_path = crate::commands::model::write_llama_crash_report(
        "benchmark",
        exit_status.as_deref(),
        preview.as_ref(),
        &stderr,
    );
    let tail = stderr
        .iter()
        .rev()
        .take(80)
        .rev()
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    Some(format!(
        "Exit: {}. Crash report: {}\n{}",
        exit_status.as_deref().unwrap_or("unknown"),
        report_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "not written".to_string()),
        if tail.trim().is_empty() {
            "No llama-server stderr was captured.".to_string()
        } else {
            tail
        }
    ))
}

async fn mark_benchmark_crash_if_needed(shared_state: &SharedState) -> Option<String> {
    let crashed = {
        let mut s = shared_state.write().await;
        s.process.poll_exited()
            || (matches!(
                s.process.state(),
                crate::engine::process::ProcessState::Error
            ) && !s.process.has_child())
    };

    if crashed {
        let tail = benchmark_crash_tail(shared_state)
            .await
            .unwrap_or_else(|| "No llama-server stderr was captured.".to_string());
        Some(format!("llama-server crashed during benchmark.\n{tail}"))
    } else {
        None
    }
}

#[tauri::command]
pub async fn run_model_test(
    shared_state: State<'_, SharedState>,
    model_name: String,
    context_size: u32,
    prompt: String,
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<i64>,
) -> Result<ModelTestStats, String> {
    let load_start = std::time::Instant::now();
    let shared = shared_state.inner().clone();
    let possibly_reused = {
        let s = shared.read().await;
        let model_matches = s.loaded_model.as_deref().map_or(false, |loaded| {
            benchmark_model_names_match(loaded, &model_name)
        });
        let context_ok = s
            .model_stats
            .as_ref()
            .map_or(false, |stats| stats.context_size >= context_size);
        model_matches
            && context_ok
            && matches!(
                s.process.state(),
                crate::engine::process::ProcessState::Running
            )
    };
    let load_result = tokio::time::timeout(
        Duration::from_secs(BENCHMARK_LOAD_TIMEOUT_SECS),
        crate::commands::model::backend_load_model_with_overrides(
            shared.clone(),
            model_name.clone(),
            Some(context_size),
            crate::commands::model::RuntimeLoadOverrides {
                attach_mmproj: Some(false),
                ..Default::default()
            },
        ),
    )
    .await
    .map_err(|_| {
        format!(
            "Model load timed out after {BENCHMARK_LOAD_TIMEOUT_SECS}s before benchmark. Try a smaller model/quant or lower context."
        )
    })?;

    load_result.map_err(|e| format!("Model load failed before benchmark: {e}"))?;

    if let Some(crash) = mark_benchmark_crash_if_needed(&shared).await {
        return Err(crash);
    }

    let elapsed_load_ms = load_start.elapsed().as_millis();
    let load_reused = possibly_reused && elapsed_load_ms < 1_000;
    let load_ms = if load_reused { 0 } else { elapsed_load_ms };

    let test_result = tokio::time::timeout(
        Duration::from_secs(BENCHMARK_TEST_TIMEOUT_SECS),
        test_model(
            shared.clone(),
            &model_name,
            context_size,
            Some(load_ms),
            load_reused,
            &prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
        ),
    )
    .await;

    match test_result {
        Ok(Ok(stats)) => Ok(stats),
        Ok(Err(error)) => {
            if let Some(crash) = mark_benchmark_crash_if_needed(&shared).await {
                Err(crash)
            } else {
                Err(format!("Model test failed: {error}"))
            }
        }
        Err(_) => {
            let crash = mark_benchmark_crash_if_needed(&shared).await;
            Err(crash.unwrap_or_else(|| {
                format!(
                    "Model test timed out after {BENCHMARK_TEST_TIMEOUT_SECS}s. Try lower context, fewer output tokens, or a smaller quant."
                )
            }))
        }
    }
}
