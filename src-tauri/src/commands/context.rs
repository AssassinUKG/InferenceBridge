use crate::context::tracker;
use crate::engine::client::LlamaClient;
use crate::state::SharedState;

#[tauri::command]
pub async fn get_context_status(
    state: tauri::State<'_, SharedState>,
) -> Result<tracker::ContextStatus, String> {
    let (loaded, running, port, stored) = {
        let s = state.read().await;
        (
            s.loaded_model.is_some(),
            matches!(s.process.state(), crate::engine::process::ProcessState::Running),
            s.process.port(),
            s.last_context_status.clone(),
        )
    };

    if !loaded || !running {
        return Ok(stored.unwrap_or_else(tracker::ContextStatus::empty));
    }

    let client = LlamaClient::new(port);
    let polled = tracker::poll_context_status(&client).await;

    if let Some(stored) = stored {
        Ok(polled.with_breakdown(
            stored.pinned_tokens,
            stored.rolling_tokens,
            stored.compressed_tokens,
            stored.last_compaction_action,
        ))
    } else {
        Ok(polled)
    }
}
