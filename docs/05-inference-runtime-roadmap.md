# Inference Runtime Roadmap

This document is the next implementation plan for the parts of InferenceBridge that most affect model quality, responsiveness, and low-level control.

It focuses on four areas:

1. streaming and async execution
2. model-aware configs and family-specific fixes
3. better context and memory handling
4. tighter `llama.cpp`-style execution control

## Goals

- make generation feel immediate and stable under load
- keep model-specific behavior explicit instead of hidden in ad hoc conditionals
- preserve quality over long sessions without silently overrunning context
- expose more of the useful `llama.cpp` runtime knobs in a safe, app-managed way

## Phase A: Streaming and Async Execution

### Why

The app already streams, but this needs to become a more deliberate runtime pipeline rather than a UI feature layered over a request loop.

### Work items

- Introduce a request-scoped generation task model with stable request IDs.
- Separate model load, unload, completion, and stats polling into explicit async jobs.
- Add cancellation tokens for chat requests so stop and unload behave cleanly.
- Stream structured events instead of only raw text deltas:
  - token delta
  - tool-call delta
  - reasoning delta
  - completion finished
  - completion failed
- Make the GUI and `/v1/chat/completions` use the same streaming event pipeline.
- Add backpressure-safe buffering between `llama-server` output and UI/API consumers.
- Persist enough metadata to resume or inspect interrupted requests in Debug.

### Likely code areas

- `src-tauri/src/engine/process.rs`
- `src-tauri/src/api/completions.rs`
- `src-tauri/src/commands/chat.rs`
- `src-tauri/src/state.rs`
- `src/hooks/useChat.ts`
- `src/components/Chat/*`

### Acceptance criteria

- `Stop` interrupts generation without leaving the runtime in a bad state.
- Unload during generation cancels cleanly and returns a predictable error.
- GUI and API streaming produce equivalent outputs for the same request.
- Debug can show request lifecycle events, not just final text.

## Phase B: Model-Aware Configs and Family Fixes

### Why

Different GGUF families need different defaults, template handling, parser behavior, and execution flags. This should be driven by structured profiles rather than spread-out special cases.

### Work items

- Formalize a `ModelProfile` system as the single source of truth for:
  - family detection
  - chat template style
  - tool-call style
  - reasoning / think-tag behavior
  - stop tokens
  - sampling defaults
  - vision requirements
  - recommended runtime args
- Add per-family patches for:
  - Qwen / Qwen3.5 reasoning and tool-call behavior
  - DeepSeek reasoning formats
  - Llama-family stop sequences and chat formatting
  - Gemma / Mistral family quirks where needed
- Add user-overridable profile fields in config and GUI:
  - temperature
  - top-p / top-k
  - reasoning toggle default
  - tool-call mode
  - context default
  - custom stop sequences
- Add a model profile inspection view in Debug so users can see the effective runtime config.
- Persist model overrides separately from auto-detected defaults.

### Likely code areas

- `src-tauri/src/models/*`
- `src-tauri/src/config.rs`
- `src-tauri/src/engine/process.rs`
- `src-tauri/src/normalize/*`
- `src/components/Model/*`
- `src/components/Debug/DebugInspector.tsx`

### Acceptance criteria

- Loading a Qwen-family model automatically applies the right defaults and parser path.
- User overrides survive restart and do not get lost during rescan.
- Debug clearly shows detected family, effective template, and active overrides.

## Phase C: Better Context and Memory Handling

### Why

Long-running local sessions degrade quickly if context budgeting is vague. The app needs explicit token accounting, compaction, and recovery behavior.

### Work items

- Track prompt, completion, and cached context token usage separately.
- Split context into explicit layers:
  - pinned system / instructions
  - recent rolling turns
  - tool outputs
  - compressed history
  - optional recalled memory
- Add automatic compaction policies based on thresholds instead of single blunt rebuild logic.
- Add session summary blocks that can replace older turns when pressure rises.
- Add memory recall hooks for future lightweight long-term memory:
  - recent session summary
  - pinned facts / notes
  - optional retrieval later
- Improve visibility in the Context tab:
  - token budget by layer
  - why compaction happened
  - what got summarized
  - when a full rebuild was required
- Expose context stats and compaction actions through the API in a stable format.

### Likely code areas

- `src-tauri/src/context/*`
- `src-tauri/src/session/*`
- `src-tauri/src/api/extensions.rs`
- `src/components/Context/*`
- `src/hooks/useContext.ts`

### Acceptance criteria

- Long chats no longer fail silently from hidden context overflow.
- Users can see what part of the conversation is consuming space.
- Rebuild and compaction behavior is explainable in the UI and Debug logs.

## Phase D: Tighter llama.cpp-Style Execution Control

### Why

Users running local models often want sharper control over the same knobs they know from `llama.cpp`. The app should expose the useful ones while still staying safe and coherent.

### Work items

- Move backend launch settings into a structured runtime config object.
- Support more execution controls per model or per profile:
  - `ctx-size`
  - `n-gpu-layers`
  - `threads`
  - `parallel`
  - `batch-size`
  - `ubatch-size`
  - `flash-attn`
  - `rope` / scaling options where supported
  - KV cache type / quantization where supported
  - continuous batching related flags where appropriate
- Add explicit launch previews in Debug:
  - full command args
  - resolved model path
  - resolved `mmproj` path
  - source of each effective option
- Add health and slot polling around the running backend:
  - startup timing
  - slots / concurrency
  - live model stats
  - restart count
- Add safer failure modes:
  - reject invalid flag combinations before launch
  - preserve previous known-good config on failed restart
  - make restart reasons visible in logs and status

### Likely code areas

- `src-tauri/src/engine/process.rs`
- `src-tauri/src/commands/settings.rs`
- `src-tauri/src/api/models.rs`
- `src-tauri/src/api/extensions.rs`
- `src/components/Model/SettingsPanel.tsx`
- `src/components/Debug/DebugInspector.tsx`

### Acceptance criteria

- A user can understand exactly how the running backend was started.
- Runtime settings are consistent between GUI launch, API-triggered load, and restart recovery.
- Invalid configs are caught before the backend is spawned.

## Cross-Cutting API Improvements

These upgrades should also improve the public API:

- richer async load / unload / generation status responses
- clearer model stats for specific named models
- better streaming parity with OpenAI-compatible clients
- more transparent runtime and context inspection endpoints

## Recommended Delivery Order

1. Phase A: streaming and async execution
2. Phase B: model-aware configs
3. Phase D: execution control
4. Phase C: context and memory handling

That order keeps the core request path stable first, then improves model behavior, then deepens low-level control, then builds more advanced long-session behavior on top.

## Release Milestones

### Milestone 1: Runtime Stability

- unified async request lifecycle
- reliable cancel / stop
- consistent streaming path for GUI and API

### Milestone 2: Smart Model Profiles

- profile-driven family behavior
- Qwen and other family-specific fixes
- persisted model overrides

### Milestone 3: Power User Runtime Controls

- expanded `llama.cpp` launch controls
- launch preview and effective config inspection
- stronger validation before spawn

### Milestone 4: Long-Session Quality

- layered context accounting
- compaction and rebuild visibility
- summary-backed memory handling

## Definition of Done

This roadmap is complete when:

- streaming, stop, unload, and restart all behave predictably under concurrency
- model-specific quirks are driven by profiles instead of scattered exceptions
- users can understand and control context pressure
- the app exposes enough useful `llama.cpp` control to satisfy advanced local users without becoming brittle
