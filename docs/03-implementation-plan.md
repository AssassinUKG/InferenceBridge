# InferenceBridge — Implementation Plan

## Phase 1: Foundation & Scaffold (Current)
**Deliverables**: Migration note, architecture doc, this plan, scaffolded Tauri project.

- [x] Migration/design note
- [x] Architecture document
- [x] Implementation plan
- [ ] `cargo create-tauri-app` scaffold
- [ ] Rust workspace with `src-tauri/` structure
- [ ] React + Vite + TypeScript + Tailwind frontend scaffold
- [ ] Basic Tauri window opens with placeholder UI

---

## Phase 2: Core Backend — Model Profiles, Process Manager, Template Subsystem

### 2A: Model Profiles (port from HelixClaw)
- Port `ModelFamily` enum and `ModelProfile` struct
- Port family detection logic (`detect_family()`)
- Port profile flags: `use_qwen_parser`, `split_tool_calling`, `disable_thinking_for_tools`, etc.
- Port context tier constants
- Port think-suppression suffix generation
- Add GGUF metadata reader (extract model family from filename + GGUF header)
- Unit tests for family detection

### 2B: llama-server Process Manager
- `LlamaProcess` struct: spawn, kill, health-check, restart
- Process state machine: Idle → Starting → Running → Stopping → Idle
- Auto-detect llama-server binary (check PATH, common install locations)
- Configurable launch args: `--model`, `--ctx-size`, `--n-gpu-layers`, `--port`, `--no-chat-template`
- Health check loop: poll `/health` endpoint
- Crash detection + auto-restart (3 attempts, exponential backoff)
- Model switch: graceful shutdown → respawn with new model
- Tauri events: `process-status-changed`, `model-loaded`, `model-unloaded`

### 2C: Model Scanner
- Scan configured directories for `.gguf` files
- Extract model info from filename (family, quant level, parameter count)
- Optional: read GGUF metadata for context length, architecture
- Build model registry (in-memory, persisted to SQLite)

### 2D: Template Subsystem
- `TemplateEngine` trait with `render(messages, profile) → String`
- Built-in templates: ChatML, QwenChat, Llama3Chat
- Model-specific patches (think suppression, tool format hints)
- Template selection based on `ModelProfile.renderer_type`
- Template preview command (for debug inspector)
- Unit tests with known good outputs

---

## Phase 3: Session Persistence + Context/KV Awareness

### 3A: SQLite Session Store
- Schema: sessions, messages, tool_calls, context_snapshots
- CRUD operations for sessions and messages
- Full-text search on message content
- Export to JSONL / import from JSONL (HelixClaw compat)

### 3B: Context Tracker
- Poll llama-server `/slots` endpoint for KV cache usage
- Track token counts per message (estimate via tiktoken-rs or character heuristic)
- Expose context status: used/total tokens, KV fill percentage
- Tauri events: `context-status-updated`

### 3C: Context Strategy
- Layered context builder: pinned → rolling → compressed → rebuild
- Compression: summarize N oldest rolling messages into a single compressed block
- Rebuild: reconstruct full context from session DB when KV is invalidated
- Configurable rolling window size
- Auto-trigger: compress at 80% KV, rebuild on model switch/crash

---

## Phase 4: Output Normalization + Tool Parsing Pipeline

### 4A: Normalization Pipeline
- Pipeline orchestrator: raw → think-strip → model-parser → json-repair → tool-extract → validate
- Port `strip_think_tags()` (handles `<think>` and `<|think|>`)
- Port `QwenStreamParser` state machine
- Port `repair_json()` 5-step pipeline
- Tool call extraction: structured `ToolCall` objects from parsed output
- Validation: schema check, retry hint generation

### 4B: Streaming Integration
- SSE consumer from llama-server `/completion` endpoint
- Token-by-token accumulation with incremental normalization
- Emit Tauri events: `token`, `tool-call-started`, `tool-call-complete`, `generation-done`
- Backpressure handling (pause stream if UI isn't consuming)

### 4C: Split Tool Calling
- Two-pass approach (ported from HelixClaw): select tool → generate arguments
- Only activated for models where `profile.split_tool_calling == true`
- Transparent to callers — same API, pipeline handles the split internally

---

## Phase 5: UI — Chat, Model Control, Context Panel, Debug Inspector

### 5A: Chat Interface
- Message list with streaming text display
- Markdown rendering (code blocks, inline code, lists)
- Tool call cards (collapsible, show name + arguments + result)
- Input area with send button and stop button
- Session sidebar: list, create, delete, rename sessions

### 5B: Model Control
- Model selector dropdown (grouped by family)
- Load / unload buttons with progress indicator
- Model info card: family, quant, parameters, context size, profile flags
- Process status indicator (Idle/Starting/Running/Stopping/Error)

### 5C: Context Panel
- KV usage progress bar (used / total tokens)
- Layer breakdown: pinned | rolling | compressed counts
- Action buttons: rebuild, compact, clear context
- Auto-refresh on context changes

### 5D: Debug Inspector
- Raw prompt viewer (what was actually sent to llama-server)
- Parse trace (normalization pipeline steps and intermediate results)
- Process log (llama-server stdout/stderr)
- Template preview (live render of current template with current messages)

### 5E: Status Bar
- Current model name + quant
- Token usage (session total)
- Process health indicator
- Generation speed (tokens/sec during streaming)

---

## Phase 6: HelixClaw-Compatible API Adapter

### 6A: Axum HTTP Server
- Start alongside Tauri on configurable port (default 8800)
- Shared `AppState` with Tauri commands

### 6B: OpenAI-Compatible Endpoints
- `POST /v1/chat/completions` — accepts OpenAI-format messages, returns OpenAI-format response
  - Supports `stream: true` (SSE) and `stream: false`
  - Supports `tools` array for tool definitions
  - Handles `chat_template_kwargs` (for HelixClaw compat)
  - Returns proper `tool_calls` in assistant messages
- `GET /v1/models` — returns loaded model info

### 6C: Extension Endpoints
- `GET /v1/context/status` — KV cache usage, layer breakdown
- `GET /v1/sessions` — list active sessions
- `POST /v1/context/rebuild` — trigger context rebuild
- `POST /v1/context/compact` — trigger compression
- `GET /v1/debug/prompt` — last raw prompt sent to llama-server
- `GET /v1/debug/parse-trace` — last normalization pipeline trace

### 6D: HelixClaw Integration Testing
- Point HelixClaw's `openai.rs` at InferenceBridge
- Verify: model loading, chat completion, streaming, tool calls, think-tag handling
- Document any HelixClaw config changes needed

---

## Dependency Order

```
Phase 1 (scaffold)
    │
    ▼
Phase 2A (profiles) ──► Phase 2D (templates)
    │                        │
    ▼                        ▼
Phase 2B (process) ──► Phase 2C (scanner)
    │
    ▼
Phase 3A (SQLite) ──► Phase 3B (context tracker) ──► Phase 3C (strategy)
    │
    ▼
Phase 4A (normalize) ──► Phase 4B (streaming) ──► Phase 4C (split tool)
    │
    ├──► Phase 5 (UI — can start after 4B for basic chat)
    │
    └──► Phase 6 (API adapter — can start after 4B)
```

## Acceptance Criteria (Overall)

1. InferenceBridge launches, loads a GGUF model, and serves chat completions
2. Qwen3.5 models work with correct template, no think-tag leakage, tool calls parse correctly
3. HelixClaw can connect to InferenceBridge's `/v1/chat/completions` and complete a full CEO→Worker pipeline
4. Context doesn't silently overflow — compression/rebuild triggers automatically
5. Model switch works without orphaned processes or VRAM leaks
6. Session history persists across app restarts
7. Debug inspector shows raw prompt and parse trace for any generation
