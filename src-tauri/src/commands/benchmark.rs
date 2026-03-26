//! Tauri command for model benchmarking

use crate::engine::benchmark::{test_model, ModelTestStats};
use crate::state::SharedState;
use tauri::State;

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
    test_model(
        shared_state.inner().clone(),
        &model_name,
        context_size,
        &prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        seed,
    )
    .await
    .map_err(|e| format!("Model test failed: {e}"))
}
