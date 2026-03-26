use anyhow::Result;
use rusqlite::Connection;
use std::path::PathBuf;

pub struct SessionDb {
    conn: Connection,
}

impl SessionDb {
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
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT NOT NULL REFERENCES sessions(id),
                role              TEXT NOT NULL,
                content           TEXT,
                image_base64      TEXT,
                token_count       INTEGER DEFAULT 0,
                tokens_evaluated  INTEGER,
                tokens_predicted  INTEGER,
                created_at        TEXT NOT NULL DEFAULT (datetime('now'))
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
            CREATE INDEX IF NOT EXISTS idx_context_snapshots_session ON context_snapshots(session_id);
            ",
        )?;
        self.ensure_message_columns()?;
        Ok(())
    }

    fn ensure_message_columns(&self) -> Result<()> {
        let mut stmt = self.conn.prepare("PRAGMA table_info(messages)")?;
        let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
        let names = columns.filter_map(|col| col.ok()).collect::<Vec<_>>();

        if !names.iter().any(|col| col == "image_base64") {
            self.conn
                .execute("ALTER TABLE messages ADD COLUMN image_base64 TEXT", [])?;
        }
        if !names.iter().any(|col| col == "tokens_evaluated") {
            self.conn
                .execute("ALTER TABLE messages ADD COLUMN tokens_evaluated INTEGER", [])?;
        }
        if !names.iter().any(|col| col == "tokens_predicted") {
            self.conn
                .execute("ALTER TABLE messages ADD COLUMN tokens_predicted INTEGER", [])?;
        }

        Ok(())
    }

    pub fn create_session(&self, name: &str, model_id: Option<&str>) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        self.conn.execute(
            "INSERT INTO sessions (id, name, model_id) VALUES (?1, ?2, ?3)",
            rusqlite::params![id, name, model_id],
        )?;
        Ok(id)
    }

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
        Ok(rows.filter_map(|row| row.ok()).collect())
    }

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
        let message_id = self.conn.last_insert_rowid();
        self.conn.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?1",
            rusqlite::params![session_id],
        )?;
        Ok(message_id)
    }

    pub fn update_message_generation_stats(
        &self,
        message_id: i64,
        token_count: u32,
        tokens_evaluated: Option<u32>,
        tokens_predicted: Option<u32>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE messages SET token_count = ?2, tokens_evaluated = ?3, tokens_predicted = ?4 WHERE id = ?1",
            rusqlite::params![message_id, token_count, tokens_evaluated, tokens_predicted],
        )?;
        Ok(())
    }

    pub fn get_messages(&self, session_id: &str) -> Result<Vec<MessageInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, role, content, image_base64, token_count, tokens_evaluated, tokens_predicted, created_at FROM messages WHERE session_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map(rusqlite::params![session_id], |row| {
            Ok(MessageInfo {
                id: row.get(0)?,
                role: row.get(1)?,
                content: row.get(2)?,
                image_base64: row.get(3)?,
                token_count: row.get(4)?,
                tokens_evaluated: row.get(5)?,
                tokens_predicted: row.get(6)?,
                created_at: row.get(7)?,
            })
        })?;
        Ok(rows.filter_map(|row| row.ok()).collect())
    }

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

    pub fn add_context_snapshot(&self, session_id: &str, snapshot: &str, kv_tokens: u32) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO context_snapshots (session_id, snapshot, kv_tokens) VALUES (?1, ?2, ?3)",
            rusqlite::params![session_id, snapshot, kv_tokens],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn latest_context_snapshot(&self, session_id: &str) -> Result<Option<ContextSnapshotInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, snapshot, kv_tokens, created_at FROM context_snapshots WHERE session_id = ?1 ORDER BY id DESC LIMIT 1",
        )?;
        let mut rows = stmt.query(rusqlite::params![session_id])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(ContextSnapshotInfo {
                id: row.get(0)?,
                session_id: row.get(1)?,
                snapshot: row.get(2)?,
                kv_tokens: row.get(3)?,
                created_at: row.get(4)?,
            }));
        }
        Ok(None)
    }

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
    pub tokens_evaluated: Option<u32>,
    pub tokens_predicted: Option<u32>,
    pub created_at: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextSnapshotInfo {
    pub id: i64,
    pub session_id: String,
    pub snapshot: String,
    pub kv_tokens: u32,
    pub created_at: String,
}
