use anyhow::Result;
use rusqlite::Connection;
use std::path::PathBuf;

const AUTOMATIC_CHAT_PREFIX: &str = "Chat ";

fn is_legacy_automatic_chat_name(name: &str) -> bool {
    let Some(number) = name.strip_prefix(AUTOMATIC_CHAT_PREFIX) else {
        return false;
    };

    !number.is_empty()
        && !number.starts_with('0')
        && number.bytes().all(|character| character.is_ascii_digit())
}

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
                automatic_name INTEGER NOT NULL DEFAULT 0,
                pinned      INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id        TEXT NOT NULL REFERENCES sessions(id),
                role              TEXT NOT NULL,
                content           TEXT,
                display_content   TEXT,
                reasoning_content TEXT,
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
        self.ensure_session_columns()?;
        self.ensure_message_columns()?;
        self.renumber_automatic_sessions()?;
        Ok(())
    }

    fn ensure_session_columns(&self) -> Result<()> {
        let names = {
            let mut stmt = self.conn.prepare("PRAGMA table_info(sessions)")?;
            let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
            columns.collect::<rusqlite::Result<Vec<_>>>()?
        };

        if !names.iter().any(|column| column == "automatic_name") {
            let transaction = self.conn.unchecked_transaction()?;
            transaction.execute(
                "ALTER TABLE sessions ADD COLUMN automatic_name INTEGER NOT NULL DEFAULT 0",
                [],
            )?;

            // Existing versions only created automatic chats as exact `Chat N`
            // titles. Mark those legacy rows once; all future rows carry an
            // explicit marker so custom names remain untouched.
            let legacy_rows = {
                let mut stmt = transaction.prepare("SELECT id, name FROM sessions")?;
                let rows = stmt.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };

            for (id, name) in legacy_rows {
                if name.as_deref().is_some_and(is_legacy_automatic_chat_name) {
                    transaction.execute(
                        "UPDATE sessions SET automatic_name = 1 WHERE id = ?1",
                        rusqlite::params![id],
                    )?;
                }
            }
            transaction.commit()?;
        }

        if !names.iter().any(|column| column == "pinned") {
            self.conn.execute(
                "ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }

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
            self.conn.execute(
                "ALTER TABLE messages ADD COLUMN tokens_evaluated INTEGER",
                [],
            )?;
        }
        if !names.iter().any(|col| col == "tokens_predicted") {
            self.conn.execute(
                "ALTER TABLE messages ADD COLUMN tokens_predicted INTEGER",
                [],
            )?;
        }
        if !names.iter().any(|col| col == "display_content") {
            self.conn
                .execute("ALTER TABLE messages ADD COLUMN display_content TEXT", [])?;
        }
        if !names.iter().any(|col| col == "reasoning_content") {
            self.conn
                .execute("ALTER TABLE messages ADD COLUMN reasoning_content TEXT", [])?;
        }

        Ok(())
    }

    pub fn create_session(&self, name: &str, model_id: Option<&str>) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        self.conn.execute(
            "INSERT INTO sessions (id, name, model_id, automatic_name) VALUES (?1, ?2, ?3, 0)",
            rusqlite::params![id, name, model_id],
        )?;
        Ok(id)
    }

    pub fn create_automatic_session(&self, model_id: Option<&str>) -> Result<String> {
        let transaction = self.conn.unchecked_transaction()?;
        let automatic_count = Self::renumber_automatic_sessions_in(&transaction)?;
        let id = uuid::Uuid::new_v4().to_string();
        let name = format!("{AUTOMATIC_CHAT_PREFIX}{}", automatic_count + 1);
        transaction.execute(
            "INSERT INTO sessions (id, name, model_id, automatic_name) VALUES (?1, ?2, ?3, 1)",
            rusqlite::params![id, name, model_id],
        )?;
        transaction.commit()?;
        Ok(id)
    }

    fn renumber_automatic_sessions_in(transaction: &rusqlite::Transaction<'_>) -> Result<usize> {
        let session_ids = {
            let mut stmt = transaction.prepare(
                "SELECT id FROM sessions WHERE automatic_name = 1 ORDER BY created_at ASC, rowid ASC",
            )?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };

        for (index, session_id) in session_ids.iter().enumerate() {
            let name = format!("{AUTOMATIC_CHAT_PREFIX}{}", index + 1);
            transaction.execute(
                "UPDATE sessions SET name = ?2 WHERE id = ?1 AND name IS NOT ?2",
                rusqlite::params![session_id, name],
            )?;
        }

        Ok(session_ids.len())
    }

    fn renumber_automatic_sessions(&self) -> Result<()> {
        let transaction = self.conn.unchecked_transaction()?;
        Self::renumber_automatic_sessions_in(&transaction)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, model_id, pinned, created_at, updated_at FROM sessions ORDER BY pinned DESC, updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(SessionInfo {
                id: row.get(0)?,
                name: row.get(1)?,
                model_id: row.get(2)?,
                pinned: row.get::<_, i64>(3)? != 0,
                created_at: row.get(4)?,
                updated_at: row.get(5)?,
            })
        })?;
        Ok(rows.filter_map(|row| row.ok()).collect())
    }

    pub fn rename_session(&self, session_id: &str, name: &str) -> Result<()> {
        let name = name.trim();
        anyhow::ensure!(!name.is_empty(), "Conversation title cannot be empty");
        anyhow::ensure!(
            name.chars().count() <= 120,
            "Conversation title is too long"
        );
        let changed = self.conn.execute(
            "UPDATE sessions SET name = ?2, automatic_name = 0, updated_at = datetime('now') WHERE id = ?1",
            rusqlite::params![session_id, name],
        )?;
        anyhow::ensure!(changed == 1, "Conversation not found");
        Ok(())
    }

    pub fn set_session_pinned(&self, session_id: &str, pinned: bool) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE sessions SET pinned = ?2, updated_at = datetime('now') WHERE id = ?1",
            rusqlite::params![session_id, i64::from(pinned)],
        )?;
        anyhow::ensure!(changed == 1, "Conversation not found");
        Ok(())
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

    pub fn update_message_presentation(
        &self,
        message_id: i64,
        display_content: Option<&str>,
        reasoning_content: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE messages SET display_content = ?2, reasoning_content = ?3 WHERE id = ?1",
            rusqlite::params![message_id, display_content, reasoning_content],
        )?;
        Ok(())
    }

    pub fn get_messages(&self, session_id: &str) -> Result<Vec<MessageInfo>> {
        let mut messages = {
            let mut stmt = self.conn.prepare(
                "SELECT id, role, content, display_content, reasoning_content, image_base64, token_count, tokens_evaluated, tokens_predicted, created_at FROM messages WHERE session_id = ?1 ORDER BY id",
            )?;
            let rows = stmt.query_map(rusqlite::params![session_id], |row| {
                Ok(MessageInfo {
                    id: row.get(0)?,
                    role: row.get(1)?,
                    content: row.get(2)?,
                    display_content: row.get(3)?,
                    reasoning_content: row.get(4)?,
                    image_base64: row.get(5)?,
                    token_count: row.get(6)?,
                    tokens_evaluated: row.get(7)?,
                    tokens_predicted: row.get(8)?,
                    created_at: row.get(9)?,
                    tool_calls: Vec::new(),
                })
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };

        let mut tool_stmt = self.conn.prepare(
            "SELECT id, call_id, name, arguments, result FROM tool_calls WHERE message_id = ?1 ORDER BY id",
        )?;
        for message in &mut messages {
            let calls = tool_stmt.query_map(rusqlite::params![message.id], |row| {
                Ok(ToolCallInfo {
                    id: row.get(0)?,
                    call_id: row.get(1)?,
                    name: row.get(2)?,
                    arguments: row.get(3)?,
                    result: row.get(4)?,
                })
            })?;
            message.tool_calls = calls.collect::<rusqlite::Result<Vec<_>>>()?;
        }
        Ok(messages)
    }

    pub fn add_tool_call(
        &self,
        message_id: i64,
        call_id: &str,
        name: &str,
        arguments: &str,
        result: Option<&str>,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO tool_calls (message_id, call_id, name, arguments, result) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![message_id, call_id, name, arguments, result],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn add_context_snapshot(
        &self,
        session_id: &str,
        snapshot: &str,
        kv_tokens: u32,
    ) -> Result<i64> {
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
        let transaction = self.conn.unchecked_transaction()?;
        transaction.execute(
            "DELETE FROM tool_calls WHERE message_id IN (SELECT id FROM messages WHERE session_id = ?1)",
            rusqlite::params![session_id],
        )?;
        transaction.execute(
            "DELETE FROM context_snapshots WHERE session_id = ?1",
            rusqlite::params![session_id],
        )?;
        transaction.execute(
            "DELETE FROM messages WHERE session_id = ?1",
            rusqlite::params![session_id],
        )?;
        transaction.execute(
            "DELETE FROM sessions WHERE id = ?1",
            rusqlite::params![session_id],
        )?;
        Self::renumber_automatic_sessions_in(&transaction)?;
        transaction.commit()?;
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub name: Option<String>,
    pub model_id: Option<String>,
    pub pinned: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MessageInfo {
    pub id: i64,
    pub role: String,
    pub content: Option<String>,
    pub display_content: Option<String>,
    pub reasoning_content: Option<String>,
    pub image_base64: Option<String>,
    pub token_count: Option<u32>,
    pub tokens_evaluated: Option<u32>,
    pub tokens_predicted: Option<u32>,
    pub created_at: String,
    pub tool_calls: Vec<ToolCallInfo>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolCallInfo {
    pub id: i64,
    pub call_id: Option<String>,
    pub name: String,
    pub arguments: Option<String>,
    pub result: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextSnapshotInfo {
    pub id: i64,
    pub session_id: String,
    pub snapshot: String,
    pub kv_tokens: u32,
    pub created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_memory_db() -> SessionDb {
        let db = SessionDb {
            conn: Connection::open_in_memory().expect("open in-memory session database"),
        };
        db.init_schema().expect("initialize session schema");
        db
    }

    fn session_name(db: &SessionDb, id: &str) -> String {
        db.conn
            .query_row(
                "SELECT name FROM sessions WHERE id = ?1",
                rusqlite::params![id],
                |row| row.get(0),
            )
            .expect("session name")
    }

    #[test]
    fn deleting_an_automatic_chat_compacts_names_and_reuses_the_count() {
        let db = in_memory_db();
        let first = db.create_automatic_session(None).expect("create Chat 1");
        let middle = db.create_automatic_session(None).expect("create Chat 2");
        let last = db.create_automatic_session(None).expect("create Chat 3");

        db.delete_session(&middle).expect("delete middle chat");

        assert_eq!(session_name(&db, &first), "Chat 1");
        assert_eq!(session_name(&db, &last), "Chat 2");

        let next = db.create_automatic_session(None).expect("create next chat");
        assert_eq!(session_name(&db, &next), "Chat 3");
    }

    #[test]
    fn compaction_preserves_custom_titles() {
        let db = in_memory_db();
        let first = db.create_automatic_session(None).expect("create Chat 1");
        let custom = db
            .create_session("Release notes", None)
            .expect("create custom session");
        let last = db.create_automatic_session(None).expect("create Chat 2");

        db.delete_session(&first).expect("delete automatic chat");

        assert_eq!(session_name(&db, &custom), "Release notes");
        assert_eq!(session_name(&db, &last), "Chat 1");
    }

    #[test]
    fn explicit_custom_chat_number_is_not_treated_as_automatic() {
        let db = in_memory_db();
        let custom = db
            .create_session("Chat 99", None)
            .expect("create explicitly named session");
        let automatic = db.create_automatic_session(None).expect("create Chat 1");

        db.renumber_automatic_sessions().expect("compact names");

        assert_eq!(session_name(&db, &custom), "Chat 99");
        assert_eq!(session_name(&db, &automatic), "Chat 1");
    }

    #[test]
    fn recent_activity_does_not_change_creation_order_numbering() {
        let db = in_memory_db();
        let first = db.create_automatic_session(None).expect("create Chat 1");
        let middle = db.create_automatic_session(None).expect("create Chat 2");
        let last = db.create_automatic_session(None).expect("create Chat 3");
        db.conn
            .execute(
                "UPDATE sessions SET updated_at = '2099-01-01 00:00:00' WHERE id = ?1",
                rusqlite::params![first],
            )
            .expect("mark oldest session as most recent");

        db.delete_session(&middle).expect("delete middle chat");

        assert_eq!(session_name(&db, &first), "Chat 1");
        assert_eq!(session_name(&db, &last), "Chat 2");
        assert_eq!(db.list_sessions().expect("list sessions")[0].id, first);
    }

    #[test]
    fn rename_and_pin_are_persisted_and_pinned_sessions_sort_first() {
        let db = in_memory_db();
        let first = db
            .create_automatic_session(None)
            .expect("create first chat");
        let second = db
            .create_automatic_session(None)
            .expect("create second chat");

        db.rename_session(&first, "  Tool research  ")
            .expect("rename session");
        db.set_session_pinned(&first, true).expect("pin session");

        let sessions = db.list_sessions().expect("list sessions");
        assert_eq!(sessions[0].id, first);
        assert_eq!(sessions[0].name.as_deref(), Some("Tool research"));
        assert!(sessions[0].pinned);
        assert_eq!(sessions[1].id, second);
        assert!(!sessions[1].pinned);
    }

    #[test]
    fn schema_upgrade_marks_and_compacts_legacy_automatic_titles() {
        let conn = Connection::open_in_memory().expect("open legacy database");
        conn.execute_batch(
            "
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                model_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO sessions (id, name) VALUES ('first', 'Chat 1');
            INSERT INTO sessions (id, name) VALUES ('custom', 'Project notes');
            INSERT INTO sessions (id, name) VALUES ('last', 'Chat 5');
            ",
        )
        .expect("create legacy rows");
        let db = SessionDb { conn };

        db.init_schema().expect("upgrade schema");

        assert_eq!(session_name(&db, "first"), "Chat 1");
        assert_eq!(session_name(&db, "custom"), "Project notes");
        assert_eq!(session_name(&db, "last"), "Chat 2");
        let custom_marker: i64 = db
            .conn
            .query_row(
                "SELECT automatic_name FROM sessions WHERE id = 'custom'",
                [],
                |row| row.get(0),
            )
            .expect("custom marker");
        assert_eq!(custom_marker, 0);
    }
}
