//! Tauri commands for session management.

use crate::session::db::SessionInfo;
use crate::state::SharedState;

#[tauri::command]
pub async fn create_session(
    state: tauri::State<'_, SharedState>,
    name: String,
) -> Result<String, String> {
    // The frontend-supplied name is only a display hint. Allocate the final
    // count-based title atomically in the database so rapid creates and stale
    // UI state cannot produce gaps or duplicates.
    let _ = name;
    let s = state.read().await;
    let model_id = s.loaded_model.as_deref();
    let db = s.session_db.lock().map_err(|e| e.to_string())?;
    db.create_automatic_session(model_id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn list_sessions(
    state: tauri::State<'_, SharedState>,
) -> Result<Vec<SessionInfo>, String> {
    let s = state.read().await;
    let db = s.session_db.lock().map_err(|e| e.to_string())?;
    db.list_sessions().map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn delete_session(
    state: tauri::State<'_, SharedState>,
    session_id: String,
) -> Result<(), String> {
    let s = state.read().await;
    let db = s.session_db.lock().map_err(|e| e.to_string())?;
    db.delete_session(&session_id).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn rename_session(
    state: tauri::State<'_, SharedState>,
    session_id: String,
    name: String,
) -> Result<(), String> {
    let s = state.read().await;
    let db = s.session_db.lock().map_err(|e| e.to_string())?;
    db.rename_session(&session_id, &name)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn set_session_pinned(
    state: tauri::State<'_, SharedState>,
    session_id: String,
    pinned: bool,
) -> Result<(), String> {
    let s = state.read().await;
    let db = s.session_db.lock().map_err(|e| e.to_string())?;
    db.set_session_pinned(&session_id, pinned)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_session_messages(
    state: tauri::State<'_, SharedState>,
    session_id: String,
) -> Result<Vec<crate::session::db::MessageInfo>, String> {
    let s = state.read().await;
    let db = s.session_db.lock().map_err(|e| e.to_string())?;
    db.get_messages(&session_id).map_err(|e| e.to_string())
}
