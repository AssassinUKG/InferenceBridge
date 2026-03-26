use std::collections::VecDeque;
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

#[derive(Clone, Serialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub target: String,
    pub message: String,
}

struct LogBuffer {
    entries: Mutex<VecDeque<LogEntry>>,
}

static LOG_BUFFER: OnceLock<LogBuffer> = OnceLock::new();

fn buffer() -> &'static LogBuffer {
    LOG_BUFFER.get_or_init(|| LogBuffer {
        entries: Mutex::new(VecDeque::with_capacity(LOG_CAPACITY)),
    })
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

        if guard.len() >= LOG_CAPACITY {
            guard.pop_front();
        }
        guard.push_back(entry);
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
