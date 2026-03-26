# InferenceBridge — Architecture

## Project Structure

```
InferenceBridge/
├── src-tauri/                    # Rust backend (Tauri)
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── src/
│   │   ├── main.rs               # Tauri entry point
│   │   ├── lib.rs                # Module re-exports
│   │   ├── state.rs              # AppState (shared across commands)
│   │   ├── commands/             # Tauri #[command] handlers
│   │   │   ├── mod.rs
│   │   │   ├── chat.rs           # send_message, stop_generation, get_history
│   │   │   ├── model.rs          # load_model, unload_model, list_models, scan_models
│   │   │   ├── session.rs        # create_session, list_sessions, delete_session
│   │   │   ├── context.rs        # get_context_status, rebuild_context, compact
│   │   │   └── debug.rs          # get_raw_prompt, get_parse_trace, get_process_log
│   │   ├── engine/               # Core inference engine
│   │   │   ├── mod.rs
│   │   │   ├── process.rs        # LlamaProcess: spawn, kill, health-check, restart
│   │   │   ├── client.rs         # HTTP client for llama-server /completion endpoint
│   │   │   └── streaming.rs      # SSE stream consumer + event emitter
│   │   ├── models/               # Model management
│   │   │   ├── mod.rs
│   │   │   ├── profiles.rs       # ModelProfile, ModelFamily (ported from HelixClaw)
│   │   │   ├── scanner.rs        # Scan directories for .gguf files, extract metadata
│   │   │   └── registry.rs       # Known models DB (SQLite table)
│   │   ├── templates/            # Chat template subsystem
│   │   │   ├── mod.rs
│   │   │   ├── engine.rs         # TemplateEngine: render messages → raw prompt string
│   │   │   ├── builtin.rs        # Built-in templates (ChatML, Qwen, Llama3, etc.)
│   │   │   └── patches.rs        # Model-specific template patches (Qwen think suppression, etc.)
│   │   ├── normalize/            # Output normalization pipeline
│   │   │   ├── mod.rs            # Pipeline orchestrator
│   │   │   ├── think_strip.rs    # Strip <think>/<|think|> blocks
│   │   │   ├── qwen_parser.rs    # QwenStreamParser state machine (ported)
│   │   │   ├── json_repair.rs    # JSON repair pipeline (ported)
│   │   │   ├── tool_extract.rs   # Extract tool calls → structured ToolCall objects
│   │   │   └── validate.rs       # Final validation + retry logic
│   │   ├── context/              # Context & KV management
│   │   │   ├── mod.rs
│   │   │   ├── tracker.rs        # Track token counts, KV usage via /slots
│   │   │   ├── strategy.rs       # Layered context: pinned, rolling, compressed, rebuild
│   │   │   └── compressor.rs     # Summarize older messages to free context space
│   │   ├── session/              # Session persistence
│   │   │   ├── mod.rs
│   │   │   ├── db.rs             # SQLite schema + queries (sessions, messages, tool_calls)
│   │   │   └── export.rs         # Export to JSONL / import
│   │   ├── api/                  # HelixClaw-compatible REST API
│   │   │   ├── mod.rs
│   │   │   ├── server.rs         # Axum HTTP server (runs alongside Tauri)
│   │   │   ├── completions.rs    # POST /v1/chat/completions
│   │   │   ├── models.rs         # GET /v1/models
│   │   │   └── extensions.rs     # /v1/context/status, /v1/sessions, /v1/debug
│   │   └── config.rs             # InferenceBridge config (model dirs, port, defaults)
│   └── templates/                # Jinja-style template files (optional overrides)
│       ├── chatml.txt
│       ├── qwen3.txt
│       └── llama3.txt
├── src/                          # React/TypeScript frontend
│   ├── main.tsx                  # Entry point
│   ├── App.tsx                   # Root layout with tab navigation
│   ├── components/
│   │   ├── Chat/
│   │   │   ├── ChatPanel.tsx     # Message list + input
│   │   │   ├── MessageBubble.tsx # Single message (supports markdown, code blocks)
│   │   │   ├── ToolCallCard.tsx  # Rendered tool call display
│   │   │   └── StreamingText.tsx # Live streaming text display
│   │   ├── Model/
│   │   │   ├── ModelSelector.tsx # Dropdown + load/unload controls
│   │   │   ├── ModelCard.tsx     # Model info (family, quant, params, context)
│   │   │   └── ProcessStatus.tsx # llama-server process health indicator
│   │   ├── Context/
│   │   │   ├── ContextPanel.tsx  # KV usage bar, token counts, layer breakdown
│   │   │   └── ContextActions.tsx # Rebuild / compact / clear buttons
│   │   ├── Debug/
│   │   │   ├── DebugInspector.tsx # Raw prompt viewer, parse trace, process log
│   │   │   └── TemplatePreview.tsx # Live template rendering preview
│   │   └── common/
│   │       ├── StatusBar.tsx     # Bottom status bar (model, tokens, process health)
│   │       └── Sidebar.tsx       # Session list sidebar
│   ├── hooks/
│   │   ├── useChat.ts            # Chat state management
│   │   ├── useModel.ts           # Model loading state
│   │   ├── useStream.ts          # SSE stream consumption from Tauri events
│   │   └── useContext.ts         # Context status polling
│   ├── lib/
│   │   ├── tauri.ts              # Typed Tauri invoke wrappers
│   │   └── types.ts              # Shared TypeScript types
│   └── styles/
│       └── globals.css           # Tailwind + custom styles
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── docs/
    ├── 01-migration-design-note.md
    ├── 02-architecture.md
    └── 03-implementation-plan.md
```

## Data Flow

```
User (UI or HelixClaw API)
        │
        ▼
┌─────────────────────┐
│   Tauri Commands     │  ← or Axum REST endpoints
│   (chat, model, etc) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Template Engine    │  Render messages → raw prompt using model-specific template
│   + Patches          │  (Qwen think suppression, tool format, etc.)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Context Strategy   │  Decide what fits: pinned system prompt, rolling recent,
│   + Tracker          │  compressed older messages, or trigger rebuild
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   LlamaProcess      │  POST /completion to llama-server (raw prompt, no chat template)
│   + HTTP Client      │  Stream SSE tokens back
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Normalize Pipeline │  think-strip → qwen-parse → json-repair → tool-extract → validate
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Session DB         │  Persist message + tool calls to SQLite
│   (SQLite)           │
└─────────┬───────────┘
          │
          ▼
   Response to caller
   (Tauri event / HTTP SSE / JSON)
```

## Key Architectural Choices

### 1. Dual interface: Tauri commands + Axum HTTP server

The Tauri commands serve the built-in UI. The Axum server (started on a configurable
port, default 8800) serves the OpenAI-compatible API that HelixClaw connects to.
Both share the same `AppState` (behind `Arc<RwLock<_>>`).

### 2. llama-server process lifecycle

```
                    ┌──────────┐
          load() ──►│ Starting │──► health OK ──► Running
                    └──────────┘                     │
                         ▲                           │ unload() / switch model
                         │                           ▼
                    ┌──────────┐              ┌───────────┐
                    │  Idle    │◄── exited ──│  Stopping  │
                    └──────────┘              └───────────┘
                         ▲                           │
                         │       crash               │
                         └────── (auto-restart) ─────┘
```

- `LlamaProcess` wraps `tokio::process::Child`.
- Health check: poll `GET /health` every 2s while Starting, every 10s while Running.
- On crash: auto-restart up to 3 times with exponential backoff, then enter Idle + emit error event.
- On model switch: send `/shutdown` → wait for exit (5s timeout) → kill if needed → spawn new.

### 3. Template engine

Templates are Jinja2-like strings stored as text files. The engine:
1. Selects template by `ModelProfile.renderer_type` (ChatML, QwenChat, Llama3Chat, etc.)
2. Renders conversation turns into a single raw prompt string
3. Applies model-specific patches (think suppression suffix, tool format hints)
4. Sends raw prompt to llama-server's `/completion` endpoint (NOT `/v1/chat/completions`)

This bypasses llama-server's built-in template handling entirely, giving us full control.

### 4. Context strategy layers

| Layer | Description | Eviction |
|-------|-------------|----------|
| **Pinned** | System prompt + tool definitions | Never evicted |
| **Rolling** | Recent N messages (configurable) | Oldest first |
| **Compressed** | Summarized older messages | Replaced on re-summarize |
| **Rebuild** | Full context reconstruction from session DB | Triggered when KV diverges |

The context tracker monitors `/slots` to know actual KV cache usage.
When usage exceeds 80%, it triggers compression of the oldest rolling messages.
When KV cache is fully invalidated (model switch, crash), it triggers a rebuild.

### 5. SQLite schema (core tables)

```sql
CREATE TABLE sessions (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    model_id    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    role        TEXT NOT NULL,  -- system, user, assistant, tool
    content     TEXT,
    token_count INTEGER,
    created_at  TEXT NOT NULL
);

CREATE TABLE tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER NOT NULL REFERENCES messages(id),
    call_id     TEXT,
    name        TEXT NOT NULL,
    arguments   TEXT,  -- JSON string
    result      TEXT   -- JSON string (filled when tool response arrives)
);

CREATE TABLE context_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    snapshot    TEXT NOT NULL,  -- JSON: which message IDs are pinned/rolling/compressed
    kv_tokens   INTEGER,
    created_at  TEXT NOT NULL
);
```

### 6. Config

```toml
# inference-bridge.toml
[server]
port = 8800
host = "127.0.0.1"

[models]
scan_dirs = ["C:/Users/richa/.cache/lm-studio/models"]
default_context = 8192

[process]
llama_server_path = ""  # auto-detect or explicit path
gpu_layers = -1         # -1 = all layers on GPU
threads = 0             # 0 = auto-detect

[ui]
theme = "dark"
```
