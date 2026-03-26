//! Tauri commands for context/KV management.

use crate::context::tracker;
use crate::engine::client::LlamaClient;
use crate::state::SharedState;

#[tauri::command]
pub async fn get_context_status(
    state: tauri::State<'_, SharedState>,
) -> Result<tracker::ContextStatus, String> {
    let s = state.read().await;
    if s.loaded_model.is_none() {
        return Ok(tracker::ContextStatus::empty());
    }
    // Only poll if server is actually running
    if !matches!(
        s.process.state(),
        crate::engine::process::ProcessState::Running
    ) {
        return Ok(tracker::ContextStatus::empty());
    }
    let client = LlamaClient::new(s.process.port());
    Ok(tracker::poll_context_status(&client).await)
}
