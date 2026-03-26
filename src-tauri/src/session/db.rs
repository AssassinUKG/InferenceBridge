//! SQLite session persistence.

use anyhow::Result;
use rusqlite::Connection;
use std::path::PathBuf;

/// Session database stores chat sessions, messages, and tool calls.
pub struct SessionDb {
    conn: Connection,
}

impl SessionDb {
    /// Open or create the database at the standard location.
    pub fn open() -> Result<Self> {
        let path = Self::db_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&path)?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn db_path() -> PathBuf {
        crate::config::app_support_dir().join("sessions.db")
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                name        TEXT,
                model_id    TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                role        TEXT NOT NULL,
                content     TEXT,
                image_base64 TEXT,
                token_count INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS tool_calls (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id  INTEGER NOT NULL REFERENCES messages(id),
                call_id     TEXT,
                name        TEXT NOT NULL,
                arguments   TEXT,
                result      TEXT
            );

            CREATE TABLE IF NOT EXISTS context_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                snapshot    TEXT NOT NULL,
                kv_tokens   INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_tool_calls_message ON tool_calls(message_id);
            ",
        )?;
        self.ensure_messages_image_column()?;
        Ok(())
    }

    fn ensure_messages_image_column(&self) -> Result<()> {
        let mut stmt = self.conn.prepare("PRAGMA table_info(messages)")?;
        let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
        let has_image_column = columns.filter_map(|col| col.ok()).any(|col| col == "image_base64");
        if !has_image_column {
            self.conn
                .execute("ALTER TABLE messages ADD COLUMN image_base64 TEXT", [])?;
        }
        Ok(())
    }

    /// Create a new session and return its ID.
    pub fn create_session(&self, name: &str, model_id: Option<&str>) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        self.conn.execute(
            "INSERT INTO sessions (id, name, model_id) VALUES (?1, ?2, ?3)",
            rusqlite::params![id, name, model_id],
        )?;
        Ok(id)
    }

    /// List all sessions, newest first.
    pub fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, model_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(SessionInfo {
                id: row.get(0)?,
                name: row.get(1)?,
                model_id: row.get(2)?,
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Add a message to a session.
    pub fn add_message(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        token_count: u32,
        image_base64: Option<&str>,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, image_base64, token_count) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![session_id, role, content, image_base64, token_count],
        )?;
        let msg_id = self.conn.last_insert_rowid();
        self.conn.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?1",
            rusqlite::params![session_id],
        )?;
        Ok(msg_id)
    }

    /// Get all messages for a session, in order.
    pub fn get_messages(&self, session_id: &str) -> Result<Vec<MessageInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, role, content, image_base64, token_count, created_at FROM messages WHERE session_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map(rusqlite::params![session_id], |row| {
            Ok(MessageInfo {
                id: row.get(0)?,
                role: row.get(1)?,
                content: row.get(2)?,
                image_base64: row.get(3)?,
                token_count: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Add a tool call to a message.
    pub fn add_tool_call(
        &self,
        message_id: i64,
        call_id: &str,
        name: &str,
        arguments: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO tool_calls (message_id, call_id, name, arguments) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![message_id, call_id, name, arguments],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Delete a session and all its messages/tool calls.
    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        self.conn.execute_batch(&format!(
            "DELETE FROM tool_calls WHERE message_id IN (SELECT id FROM messages WHERE session_id = '{}');
             DELETE FROM context_snapshots WHERE session_id = '{}';
             DELETE FROM messages WHERE session_id = '{}';
             DELETE FROM sessions WHERE id = '{}';",
            session_id, session_id, session_id, session_id
        ))?;
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub name: Option<String>,
    pub model_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MessageInfo {
    pub id: i64,
    pub role: String,
    pub content: Option<String>,
    pub image_base64: Option<String>,
    pub token_count: Option<u32>,
    pub created_at: String,
}
