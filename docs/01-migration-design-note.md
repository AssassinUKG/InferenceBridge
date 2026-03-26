# InferenceBridge — Migration & Design Note

## Purpose

InferenceBridge replaces LM Studio as the local inference layer for HelixClaw.
It is a Tauri desktop application (Rust backend + React/TypeScript frontend) that
manages a single `llama-server` (llama.cpp) process, provides an OpenAI-compatible
API, and adds model-aware template patching, context/KV management, tool-call
normalization, and session persistence — all things HelixClaw currently works
around via heuristics in its LLM provider layer.

---

## What to Reuse from HelixClaw

| Component | Source | Lines | Reuse strategy |
|-----------|--------|-------|----------------|
| **ModelProfile / ModelFamily** | `llm/model_profiles.rs` | ~516 | **Port directly** — family detection, profile flags (`use_qwen_parser`, `split_tool_calling`, `disable_thinking_for_tools`, etc.), context tiers, temp/penalty defaults. This becomes InferenceBridge's `model_profiles` crate. |
| **QwenStreamParser** | `llm/qwen_parser.rs` | ~779 | **Port directly** — state machine for `<tool_call>`, `<function=...>`, think-block closure. Core of the output normalization pipeline. |
| **QwenChatRenderer** | `llm/qwen_renderer.rs` | ~147 | **Port directly** — history turn rendering for Qwen chat format. |
| **Think-tag stripping** | `llm.rs` `strip_think_tags()` | ~40 | **Port directly** — handles `<think>` and `<|think|>` with salvage. |
| **JSON repair pipeline** | `llm.rs` `repair_json()` | ~80 | **Port directly** — 5-step repair (fast parse → trailing commas → unclosed strings → rebalance braces → salvage). |
| **LLM types** | `llm.rs` | ~200 | **Adapt** — `LlmMessage`, `LlmToolCall`, `LlmToolDefinition`, `LlmResponse`, `LlmStreamEvent`. Simplify: remove multi-provider abstraction (InferenceBridge only talks to its own llama-server). |
| **Context tiering constants** | `llm.rs` `SUPPORTED_CONTEXT_TIERS` | ~10 | **Port directly** — tier breakpoints for auto-context sizing. |
| **VRAM management logic** | `llm/openai.rs` | ~200 | **Adapt heavily** — `lms_ensure_vram_available()`, model load/unload. Replace LM Studio REST calls with direct llama-server process control. |
| **Think-suppression suffix** | `llm/model_profiles.rs` | ~15 | **Port** — `lmstudio_think_suppression_suffix()` becomes a general method (no longer LM Studio-specific). |
| **Retry/backoff** | `llm.rs` `retry_with_backoff()` | ~30 | **Port directly** — generic async retry helper. |

### What NOT to reuse

| Component | Reason |
|-----------|--------|
| OpenAI provider HTTP client (`openai.rs` ~6000 lines) | Too coupled to LM Studio's REST API quirks. InferenceBridge talks to its own llama-server via a thin HTTP client — much simpler. |
| Anthropic / Ollama / OpenAI Responses providers | Out of scope — InferenceBridge is local-only. |
| Agent actor loop, supervisor, CEO logic | Stays in HelixClaw. InferenceBridge is the inference layer only. |
| Session JSONL format | InferenceBridge uses SQLite for persistence (richer queries, atomic writes). |
| Config loader / types.rs | HelixClaw config is agent-focused. InferenceBridge has its own simpler config. |

---

## Key Design Decisions

### 1. Single llama-server process (Option B)
- At any time, exactly one model is loaded.
- Model switch = graceful shutdown of current process → start new process.
- Eliminates VRAM contention, simplifies KV cache management.
- HelixClaw's VRAM manager heuristics become unnecessary.

### 2. Template subsystem owns chat formatting
- InferenceBridge applies chat templates BEFORE sending to llama-server.
- llama-server runs with `--no-chat-template` (raw completion mode).
- This gives us full control over Qwen patches, think-tag injection, tool-call formatting.
- Eliminates the current pain point where LM Studio's template handling differs from expectations.

### 3. SQLite for sessions (not JSONL)
- Richer queries (search messages, filter by model, find tool calls).
- Atomic writes (no partial-line corruption on crash).
- Easy export to JSONL if needed.

### 4. Output normalization pipeline
- Runs AFTER llama-server returns tokens, BEFORE exposing to API consumers.
- Pipeline: raw tokens → think-tag strip → Qwen parser (if applicable) → JSON repair → tool-call extraction → validation.
- Same pipeline for streaming and non-streaming (streaming accumulates then normalizes per-turn).

### 5. HelixClaw-compatible API
- Exposes `/v1/chat/completions` and `/v1/models` endpoints.
- HelixClaw's `openai.rs` can point at InferenceBridge instead of LM Studio with zero code changes.
- InferenceBridge also adds richer endpoints (`/v1/context/status`, `/v1/sessions`, etc.) that HelixClaw can optionally adopt.

---

## Risk Areas

| Risk | Mitigation |
|------|-----------|
| llama-server process crashes | Health check loop + auto-restart with backoff. Expose crash count in UI. |
| KV cache state opacity | llama-server exposes `/health` and `/slots` — poll these for context usage. |
| Qwen template drift (new model versions) | Template subsystem is pluggable — new templates added without code changes. |
| GGUF model format variations | Use llama.cpp's own model detection; InferenceBridge just needs the path. |
| Large context rebuilds are slow | Layered context strategy (pinned/rolling/compressed) minimizes full rebuilds. |
