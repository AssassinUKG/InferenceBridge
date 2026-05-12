# Local Provider Runtime Improvement Plan

Created: 2026-05-11

This plan compares the current InferenceBridge base against current llama.cpp server capabilities, OpenCode-style provider handling, and Claw Code-style local model routing. It is improvement-only: the goal is to make InferenceBridge a stronger local LLM wrapper, especially for LM Studio, Ollama, and directly managed llama.cpp.

## Current Position

InferenceBridge already has a strong core:

- Tauri + Rust process manager around a single managed `llama-server`.
- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, load/unload, runtime status, context status, sessions, and metrics.
- Transparent proxy fallback for native llama-server endpoints such as `/props`, `/slots`, `/tokenize`, `/detokenize`, and future root-level llama.cpp endpoints.
- Model profiles for Qwen, DeepSeek, Llama, Phi, Mistral, Gemma, and generic GGUF behavior.
- HF-backed load options, template selection, Jinja support, reasoning mode, launch previews, cancellation tokens, and runtime load-state reporting.
- Initial context accounting, token stats persistence, context snapshots, and idle-only model-backed summarization.

The main gap is not "can it run a model"; it is runtime truth and provider breadth. OpenCode and Claw Code put a lot of polish into provider routing, model limits, tool permissions, health checks, and machine-readable state. llama.cpp has also grown enough server-side features that InferenceBridge should track them explicitly instead of exposing them only through `extra_args`.

## External Baseline

Sources checked on 2026-05-11:

- llama.cpp server README: OpenAI-compatible chat, responses, embeddings, Anthropic Messages, function calling, parallel decoding, continuous batching, multimodal support, monitoring, JSON schema, speculative decoding, reranking, HF repo loading, KV cache controls, context checkpoints, and cache prompt support. Source: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- llama.cpp latest release page: latest visible release was `b9060`, published 2026-05-07, with binaries across Windows CUDA 12/13, Vulkan, SYCL, HIP, macOS, Linux, Android, iOS, and OpenEuler. Source: https://github.com/ggml-org/llama.cpp/releases
- OpenCode providers/config docs: OpenCode uses provider config with custom `baseURL`, explicit model `limit.context` and `limit.output`, a separate `small_model`, provider timeouts, tool toggles, agents, MCP, LSP, and many providers through Models.dev / AI SDK. Sources: https://opencode.ai/docs/providers and https://opencode.ai/docs/config/
- Claude Code docs: MCP has local, project, user, plugin, and managed scopes; hooks can inject context and intercept many lifecycle events. Sources: https://code.claude.com/docs/en/mcp and https://code.claude.com/docs/en/hooks
- Claw Code usage: local endpoints are selected through Anthropic-compatible or OpenAI-compatible base URLs, Ollama uses `http://127.0.0.1:11434/v1`, provider routing can be model-prefix driven, and diagnostic commands emit JSON. Source: https://github.com/ultraworkers/claw-code/blob/main/USAGE.md

## Comparison Matrix

| Area | InferenceBridge today | OpenCode / Claw pattern | Improvement target |
| --- | --- | --- | --- |
| Provider discovery | Strong for local GGUF and managed llama.cpp; weaker for already-running LM Studio/Ollama/llama.cpp instances | Custom base URLs, provider IDs, env routing, model-prefix routing | Add first-class provider registry and autodiscovery for LM Studio, Ollama, llama.cpp, and custom OpenAI-compatible endpoints |
| Model limits | Profiles carry max context/output hints; active context is discovered from `/props` | Explicit `limit.context` and `limit.output` drive budgeting | Store provider/model limit metadata as runtime truth, not only profile guesses |
| Load lifecycle | Load/swapping progress and last launch preview exist | Doctor/status commands expose machine-readable health | Add provider preflight/doctor endpoint and precise ready transitions for every backend |
| llama.cpp control | Many curated flags plus `extra_args` | Advanced users expect exact args and validation | Add missing first-class controls: KV K/V cache type, device/split/tensor split, rope scaling family, prompt cache, embeddings/rerank modes, speculative draft |
| Context handling | Slots polling, token columns, strategy skeleton, context snapshots | Auto compaction and context budget visibility are first-class | Complete layered context budget, compaction history, prompt cache visibility, and request-level token audit |
| API compatibility | OpenAI chat/completions/responses, LM Studio-ish load aliases, transparent proxy | Local tools rely on subtle OpenAI/Ollama/LM Studio quirks | Add compatibility test matrix and adapters per caller family |
| Tool/runtime integration | Tool parsing normalization exists; no broader MCP/hook layer | MCP, hooks, permissions, agents | Future bridge: local MCP proxy and hooks around load/request/context events |
| Updates | GitHub release scan and managed binary download | Tools surface updates and diagnostics clearly | Add release-channel policy, latest-build metadata, rollback, and backend feature probing |

## Tracked Roadmap

### P0: Runtime Truth And Provider Preflight

- [x] Add a provider registry with provider type: `managed_llamacpp`, `external_llamacpp`, `lm_studio`, `ollama`, `openai_compatible`.
- [x] Add autodiscovery probes:
  - [x] Ollama: `http://127.0.0.1:11434/v1/models` and native `/api/tags`.
  - [x] LM Studio: common OpenAI-compatible local ports and `/v1/models`.
  - [x] llama.cpp: `http://127.0.0.1:8080/props`, `/health`, `/v1/models`.
- [x] Add `/v1/runtime/doctor` returning JSON diagnostics: provider reachability, model list, context limits, endpoint support, auth state, and actionable hints.
- [x] Make Models UI show provider origin and health: managed, external, LM Studio, Ollama, stale, unavailable.
- [x] Persist provider configs separately from model overrides so machine-specific local endpoints do not pollute model metadata.

Progress:

- 2026-05-11: Added `providers` runtime doctor module, Tauri `get_runtime_doctor`, public `GET /v1/runtime/doctor`, debug direct transport support, and Debug `Doctor` tab.
- 2026-05-11: Added persisted LM Studio provider settings, Settings provider controls/test button, `/v1/models` routing for the active LM Studio provider, and `/v1/chat/completions` forwarding while keeping InferenceBridge's API server as the stable front door.
- 2026-05-11: Hardened provider routing across `/v1/chat/completions`, `/v1/completions`, and `/v1/responses` with shared upstream streaming passthrough; added provider metadata/badges to Models and external-provider row behavior.

Acceptance:

- A user can point InferenceBridge at LM Studio, Ollama, or a standalone llama-server and see usable model/status information without manual guessing.
- A failed provider is reported as a provider problem, not as a generic backend offline state.

### P1: Model Loading And Reuse Semantics

- [ ] Normalize model identity into a `RuntimeModelRef`:
  - [ ] local file path
  - [ ] HF repo/file
  - [ ] provider model ID
  - [ ] display name
  - [ ] resolved active instance ID
- [ ] Replace filename-only reuse checks with effective runtime config comparison across provider, model ref, context, template, reasoning, KV, and concurrency settings.
- [ ] Add external-provider no-op load semantics:
  - [x] LM Studio: model may already be selected in GUI; expose as `provider-routed`.
  - [ ] Ollama: load maps to pull/warmup when supported, otherwise request-time use.
  - [ ] external llama.cpp: load is unsupported unless using router/native load endpoints; return a clear capability error.
- [ ] Add unload semantics per provider:
  - [ ] managed llama.cpp kills the child.
  - [ ] external providers are detached and cannot always unload.
  - [ ] Ollama/LM Studio unload support is capability-probed, not assumed.

Acceptance:

- API-triggered loads from coding agents do not restart a healthy matching backend.
- Provider limitations are explicit in JSON responses.

### P2: llama.cpp Feature Coverage

- [ ] First-class launch controls:
  - [ ] `cache_type_k` / `cache_type_v`
  - [ ] `kv_offload`
  - [ ] `cache_prompt`
  - [ ] `ctx_checkpoints`
  - [ ] `device`
  - [ ] `split_mode`
  - [ ] `tensor_split`
  - [ ] `rope_scaling`, `rope_scale`, `rope_freq_base`, `yarn_*`
  - [ ] `threads_http`
  - [ ] `embeddings` mode
  - [ ] `reranking` mode
  - [ ] speculative draft model fields
- [ ] Add feature probing from `/props`, build info, route probes, and launch version.
- [ ] Keep `extra_args`, but label every extra arg as "unvalidated" in launch preview.
- [ ] Add validation for incompatible modes, especially embeddings/rerank vs chat runtime, vision/mmproj readiness, and unsupported flags by backend version.

Acceptance:

- Advanced users can configure modern llama.cpp without falling back to raw args for common performance features.
- Launch preview explains exactly which options are supported, ignored, or rejected.

### P3: Context, KV Cache, And Prompt Cache Handling

- [ ] Turn current context strategy into persisted context events:
  - [ ] `context_pressure`
  - [ ] `compaction_started`
  - [ ] `compaction_completed`
  - [ ] `rebuild_required`
  - [ ] `prompt_cache_hit`
  - [ ] `prompt_cache_miss`
- [ ] Store request-level token audit:
  - [ ] prompt tokens
  - [ ] cached prompt tokens when available
  - [ ] completion tokens
  - [ ] reasoning tokens
  - [ ] slot ID
  - [ ] active context window
- [ ] Split UI/API context status into layers:
  - [ ] pinned instructions
  - [ ] current request
  - [ ] rolling session turns
  - [ ] compressed summary
  - [ ] tool/result blocks
  - [ ] provider/system overhead
- [ ] Add context policy config:
  - [ ] warn threshold
  - [ ] summarize threshold
  - [ ] rebuild threshold
  - [ ] summarizer model/provider
  - [ ] max summary tokens
- [ ] Add compatibility with long-context local coding agents: honor caller-provided `contextLength`, `num_ctx`, `maxContextLength`, but never silently exceed probed/provider max.

Acceptance:

- Long chats produce visible context events and summaries instead of hidden prompt truncation.
- OpenCode/Claw-style clients can see enough model limits to budget context correctly.

### P4: API Compatibility Matrix

- [ ] Add recorded smoke tests for callers:
  - [ ] OpenAI SDK chat/completions
  - [ ] OpenAI SDK responses
  - [ ] LM Studio load/config payloads
  - [ ] Ollama OpenAI-compatible payloads
  - [ ] OpenCode custom OpenAI-compatible provider
  - [ ] Claw Code OpenAI-compatible provider
  - [ ] raw llama.cpp OpenAI-compatible calls
- [ ] Test streaming shape:
  - [ ] data chunks
  - [ ] `[DONE]`
  - [ ] reasoning deltas
  - [ ] tool call extraction
  - [ ] cancellation
- [ ] Add response compatibility flags:
  - [ ] `compat.openai_chat`
  - [ ] `compat.openai_responses`
  - [ ] `compat.lmstudio_load`
  - [ ] `compat.ollama_options`
  - [ ] `compat.anthropic_messages_proxy`

Acceptance:

- Compatibility regressions are caught by tests before packaging.
- Users can see which client families are expected to work.

### P5: Updates, Rollback, And Feature Database

- [ ] Store managed llama.cpp install metadata:
  - [ ] tag/build
  - [ ] asset name
  - [ ] backend flavor
  - [ ] SHA256 when available
  - [ ] installed_at
  - [ ] feature probe result
- [ ] Add update channels:
  - [ ] latest
  - [ ] latest compatible
  - [ ] pinned
  - [ ] previous known good
- [ ] Keep one previous managed binary for rollback.
- [ ] Add app update notes panel that separates:
  - [ ] InferenceBridge version
  - [ ] managed llama.cpp version
  - [ ] provider health
  - [ ] known compatibility notes
- [ ] Add a small local feature database keyed by llama.cpp build tag for flags/routes that cannot be probed cheaply.

Acceptance:

- Users can update llama.cpp with confidence and roll back after a bad backend build.
- The app knows when a feature is unavailable because of backend age.

### P6: Agent/Tool Surface For Later

- [ ] Add optional MCP proxy/provider mode after provider/runtime truth is stable.
- [ ] Add hook points:
  - [ ] before model load
  - [ ] after model ready
  - [ ] before request
  - [ ] after request
  - [ ] context pressure
  - [ ] backend crash
- [ ] Add permission labels for future local tools: read-only, workspace-write, command, network.
- [ ] Keep agent/tool features opt-in; default app remains an inference runtime and local provider wrapper.

Acceptance:

- InferenceBridge can grow toward agent infrastructure without making the inference runtime brittle.

## Recommended Delivery Order

1. P0 provider registry + doctor endpoint.
2. P1 runtime model refs and load/reuse cleanup.
3. P2 missing llama.cpp launch controls and feature probing.
4. P3 context/KV/prompt-cache event model.
5. P4 compatibility test matrix.
6. P5 update/rollback system.
7. P6 optional MCP/hook layer.

## Immediate Next PR

- [x] Implement `ProviderRegistry` and provider probe structs.
- [x] Add `/v1/runtime/doctor`.
- [ ] Add UI provider badges in Models and Settings.
- [ ] Add tests with mocked provider probe responses for Ollama, LM Studio, llama.cpp, and managed process.

This gives the rest of the roadmap a stable source of truth.
