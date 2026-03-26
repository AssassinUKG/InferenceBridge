//! Export/import sessions to/from JSONL (HelixClaw compatible).

use super::db::SessionDb;
use anyhow::Result;
use std::io::Write;
use std::path::Path;

/// Export a session to a JSONL file.
pub fn export_to_jsonl(db: &SessionDb, session_id: &str, output: &Path) -> Result<()> {
    let messages = db.get_messages(session_id)?;
    let mut file = std::fs::File::create(output)?;
    for msg in &messages {
        let json = serde_json::json!({
            "role": msg.role,
            "content": msg.content,
            "image_base64": msg.image_base64,
            "token_count": msg.token_count,
            "created_at": msg.created_at,
        });
        writeln!(file, "{}", serde_json::to_string(&json)?)?;
    }
    Ok(())
}

/// Import messages from a JSONL file into a session.
pub fn import_from_jsonl(db: &SessionDb, session_id: &str, input: &Path) -> Result<u32> {
    let content = std::fs::read_to_string(input)?;
    let mut count = 0u32;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
            let role = json["role"].as_str().unwrap_or("user");
            let content = json["content"].as_str().unwrap_or("");
            let image_base64 = json["image_base64"].as_str();
            let tokens = json["token_count"].as_u64().unwrap_or(0) as u32;
            db.add_message(session_id, role, content, tokens, image_base64)?;
            count += 1;
        }
    }
    Ok(count)
}
