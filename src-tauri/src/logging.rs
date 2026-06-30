use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use chrono::Utc;
use serde::Serialize;
use tracing::field::Field;
use tracing::{Event, Subscriber};
use tracing_subscriber::field::Visit;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{EnvFilter, Registry};

const LOG_CAPACITY: usize = 2000;
const LOG_FILE_MAX_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Clone, Serialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub target: String,
    pub message: String,
}

struct LogBuffer {
    entries: Mutex<VecDeque<LogEntry>>,
    file: Mutex<Option<File>>,
}

static LOG_BUFFER: OnceLock<LogBuffer> = OnceLock::new();

fn buffer() -> &'static LogBuffer {
    LOG_BUFFER.get_or_init(|| LogBuffer {
        entries: Mutex::new(VecDeque::with_capacity(LOG_CAPACITY)),
        file: Mutex::new(open_log_file()),
    })
}

pub fn log_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("InferenceBridge")
        .join("logs")
}

pub fn log_file_path() -> PathBuf {
    log_dir().join("inference-bridge.log")
}

pub fn crash_report_path() -> PathBuf {
    log_dir().join("last-llama-crash.log")
}

fn open_log_file() -> Option<File> {
    let dir = log_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = log_file_path();
    if path
        .metadata()
        .map(|metadata| metadata.len() > LOG_FILE_MAX_BYTES)
        .unwrap_or(false)
    {
        let rotated = dir.join("inference-bridge.previous.log");
        let _ = std::fs::rename(&path, rotated);
    }
    OpenOptions::new().create(true).append(true).open(path).ok()
}

pub fn write_llama_crash_report(report: &str) -> Option<PathBuf> {
    let path = crash_report_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&path, report) {
        Ok(_) => Some(path),
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "Failed to write llama crash report");
            None
        }
    }
}

pub fn list(limit: usize) -> Vec<LogEntry> {
    let log_buffer = buffer();
    let guard = match log_buffer.entries.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let limit = limit.min(LOG_CAPACITY);
    let skip = guard.len().saturating_sub(limit);
    guard.iter().skip(skip).cloned().collect()
}

pub fn clear() {
    let log_buffer = buffer();
    let mut guard = match log_buffer.entries.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.clear();
}

pub fn init(default_filter: &str, stderr_only: bool) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| default_filter.into());
    let memory_layer = MemoryLogLayer;

    if stderr_only {
        Registry::default()
            .with(filter)
            .with(memory_layer)
            .with(fmt::layer().with_writer(std::io::stderr))
            .init();
        return;
    }

    Registry::default()
        .with(filter)
        .with(memory_layer)
        .with(fmt::layer())
        .init();
}

struct MemoryLogLayer;

impl<S> Layer<S> for MemoryLogLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let entry = LogEntry {
            timestamp: Utc::now().format("%Y-%m-%d %H:%M:%S%.3fZ").to_string(),
            level: metadata.level().to_string(),
            target: metadata.target().to_string(),
            message: visitor.finish(),
        };

        let log_buffer = buffer();
        let mut guard = match log_buffer.entries.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        if guard
            .back()
            .map(|last| {
                last.level == entry.level
                    && last.target == entry.target
                    && last.message == entry.message
            })
            .unwrap_or(false)
        {
            return;
        }

        if guard.len() >= LOG_CAPACITY {
            guard.pop_front();
        }
        guard.push_back(entry);

        if let Ok(mut file_guard) = log_buffer.file.lock() {
            if file_guard.is_none() {
                *file_guard = open_log_file();
            }
            if let Some(file) = file_guard.as_mut() {
                let _ = writeln!(
                    file,
                    "{} [{}] {} {}",
                    guard.back().map(|e| e.timestamp.as_str()).unwrap_or(""),
                    guard.back().map(|e| e.level.as_str()).unwrap_or(""),
                    guard.back().map(|e| e.target.as_str()).unwrap_or(""),
                    guard.back().map(|e| e.message.as_str()).unwrap_or("")
                );
            }
        }
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: Option<String>,
    fields: Vec<String>,
}

impl MessageVisitor {
    fn finish(self) -> String {
        match (self.message, self.fields.is_empty()) {
            (Some(message), true) => message,
            (Some(message), false) => format!("{message} | {}", self.fields.join(" ")),
            (None, false) => self.fields.join(" "),
            (None, true) => String::new(),
        }
    }
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        } else {
            self.fields.push(format!("{}={value}", field.name()));
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let rendered = format!("{value:?}");
        if field.name() == "message" {
            self.message = Some(rendered);
        } else {
            self.fields.push(format!("{}={rendered}", field.name()));
        }
    }
}
