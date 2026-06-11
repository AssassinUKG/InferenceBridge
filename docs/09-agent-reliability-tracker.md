# InferenceBridge Agent Reliability Tracker

Source plan: `C:\Users\richa\Desktop\Ai master plan\ai updates.md`

## Direction

InferenceBridge is the reliability gateway. It should clean, repair, validate,
measure, and route model responses so orchestrators do not have to trust raw
model text.

## Current Status

| ID | Task | Status | Notes |
| --- | --- | --- | --- |
| IB-001 | Canonical model response wrapper | Done First Pass | Added `CanonicalModelResponse` for chat/responses API replay records with raw/visible/reasoning text, tool calls, usage, timings, backend, and correlation IDs. GUI chat can adopt it next. |
| IB-002 | Think-tag cleaner | Done | `normalize::think_strip` handles standard/Qwen tags and orphan closing tags; streaming and saved-message display also hide leaks. |
| IB-003 | JSON extractor | Done | `normalize::agent_action::extract_first_json_value` extracts first object/array from noisy model output. |
| IB-004 | JSON repair pipeline | Done | `normalize::json_repair::repair_json` is reused by the new action validator. |
| IB-005 | Schema validator | Done | Added strict AgentAction schema validation in `normalize::agent_action`. |
| IB-006 | Retry formatter prompt | Done First Pass | Tool-argument repair prompt is now built by a testable formatter with deterministic sampling and strict JSON-only instructions. |
| IB-007 | Backend fallback router | Not Started | LM Studio proxy exists, but automatic fallback routing is not implemented. |
| IB-008 | True token/sec metrics | In Progress | Context tab now shows live prefill/decode/end-to-end rates, active request age, scheduler pressure, KV pressure, and GPU memory. Stream latency and perceived UI speed still need backend event timing. |
| IB-009 | Timeout and cancellation | In Progress | Streaming has first/inter-token timeouts and `/v1/inference/cancel`; non-stream request timeout policy still needs tightening. |
| IB-010 | Streaming stabiliser | Not Started | Need buffered flush control. |
| IB-011 | Replay logs | Done First Pass | Non-stream and streaming `/v1/chat/completions` plus `/v1/responses` append JSONL replay records under the user data `InferenceBridge/replay/api-replay.jsonl` path. |
| IB-012 | Model health endpoint | In Progress | `/v1/health`, `/v1/metrics`, `/v1/runtime/status`, and `/v1/runtime/doctor` exist; `/health/models` and `/health/backend` aliases remain. |
| IB-013 | Completion failure diagnostics | Done First Pass | On `/completion` send/stream failures, capture backend port/model/ctx, post-failure `/health`/`/slots`/`/props` probes, active slot state, and last llama-server stderr lines. |
| IB-014 | Same-model context reuse | Done First Pass | If the same model is already loaded with equal or larger ctx, reuse it and log the skipped lower-ctx request instead of reloading down and mutating parent agent runs. |
| IB-015 | Request correlation IDs | Done First Pass | Replay records include internal request ID plus optional client correlation IDs from `x-request-id`, `x-correlation-id`, `x-trace-id`, Mauler headers, top-level body, `metadata`, or `extra`. |
| IB-016 | Log marker assertions | Done First Pass | Added replay-level tests that flag `<|tool_call>`, `<channel|>`, `<turn|>`, and doubled native tool blobs. Next: assert UI/live log visible text exports. |
| IB-017 | Tool schema torture tests | Done First Pass | Added typed tool tests for string integer/boolean/array/object coercion plus enum and required-field validation errors. |
| IB-018 | Context pressure regression | Planned | Inject huge tool outputs before the next model turn and assert deterministic compaction or clear preflight failure. |
| IB-019 | Browser benchmark verification | Planned | Verify DOM load, console cleanliness, search, type filter, duplicate team blocking, and max-team rejection for benchmark apps. |
| IB-020 | Stop/unload lifecycle hardening | Planned | Stop model requests should settle UI/API state even when llama-server exits slowly or disappears mid-stop. |
| IB-021 | Audit #3 context/override sniffing | Done | Unknown numeric/nested fields no longer trigger context-size or load overrides; explicit top-level/options fields still work. |
| IB-022 | Audit #4 API auth bypass | Done | API key middleware now covers `/api/v1/*` and proxy fallback; only exact health routes and OPTIONS are exempt. |
| IB-023 | Audit #5 safe port eviction | Done | Startup eviction only kills recognized `llama-server` / `inference-bridge` owners and refuses unknown/self PIDs. |
| IB-024 | Audit #1/#2/#22 request cancellation lifecycle | Done First Pass | Added per-request cancellation tokens, stream-lifetime scheduler permits, disconnect drop guards, and receiver-closed stream shutdown. Follow-up: request-scoped UI/API cancel controls and multi-active UI state. |
| IB-025 | Audit #6 Windows process/RAM queries | Done First Pass | Replaced WMIC calls with PowerShell `Get-CimInstance` for managed process cleanup, llama-server process listing, and system RAM detection; process listing keeps its `tasklist` fallback. |
| IB-026 | Audit #7 `/v1/completions` streaming flag | Done First Pass | `/v1/completions` now rejects `stream: true` with a clear 400 instead of returning non-stream JSON to streaming clients. Full SSE support remains a follow-up. |
| IB-027 | Audit #8 non-stream request timeout | Done First Pass | Added a 600s overall timeout around non-stream llama `/completion` and `/v1/chat/completions` calls while leaving streaming on its dedicated timeout path. |
| IB-028 | Audit #9 finish reason length mapping | Done First Pass | Propagates llama.cpp `stopped_limit` / limit-like `stop_type` through non-stream and streaming paths and maps it to OpenAI `finish_reason: "length"` unless tool calls take precedence. |
| IB-029 | Audit #14 stream usage semantics | Done First Pass | `stream_options.include_usage` now defaults false and, when true, emits a separate usage chunk with `choices: []` before `[DONE]`. |
| IB-030 | Audit #27 unknown model status | Done First Pass | Unknown plain cloud/API model names now return OpenAI-style 404 `model_not_found`; active-model aliases still reuse the loaded model and plausible local GGUF/path requests may still attempt JIT load. |
| IB-031 | Audit #10 safer API model matching | Done First Pass | API model resolution no longer uses registry substring matching; loaded-model reuse rejects broad aliases like `qwen` or `27b` while preserving specific aliases such as `qwen3.6`. Follow-up: make implicit API JIT load opt-in and queue swaps behind active generations. |
| IB-032 | Audit #12 stream replay trace cleanup | Done First Pass | Streaming replay traces no longer duplicate reasoning deltas as synthetic `<think>...</think>` blocks because raw backend deltas are already captured. Follow-up: gate parser hiding/truncation heuristics by model profile. |
| IB-033 | Audit #11 API compaction visibility | Done First Pass | Chat completion compaction returns metadata and surfaces `x-inference-bridge-compacted`, `x-inference-bridge-compacted-messages`, and `x-inference-bridge-compaction` response headers. Follow-up: add a config switch/default policy for API compaction. |
| IB-034 | Audit #13 tool-argument repair control | Done First Pass | Tool-argument repair is controlled by `server.tool_argument_repair_enabled` (default true), remains bounded to one pass, and logs when repair is disabled or attempted. |
| IB-035 | Audit #16 default context wiring | Done First Pass | The shared backend load path now applies `server.default_ctx_size` when no explicit request context is provided, so GUI loads and API-triggered loads inherit the configured default. |
| IB-036 | Audit #17 session-only startup port fallback | Done First Pass | Startup fallback still uses a free session port when configured 8800 is unavailable, but no longer rewrites or saves `server.port` in config. |
| IB-037 | Audit #18 safe mmproj auto-pairing | Done First Pass | mmproj auto-pairing now requires a non-zero shared token score and matching detected model family when available, preventing unrelated sidecars from attaching to the wrong model. |
| IB-038 | Audit #19 UI stream render coalescing | Done First Pass | GUI chat stream deltas now buffer in refs and flush on `requestAnimationFrame`, removing per-token `flushSync` renders while flushing immediately on done/error. |
| IB-039 | Audit #20 frontend poll cadence | Done First Pass | Expensive frontend polls now run on a gentler 5s cadence for model status, context status, and GPU stats while existing model/API events still trigger immediate refreshes. |
| IB-040 | Audit #23 preserve launch model stats | Done First Pass | GUI chat no longer rewrites `model_stats` after each turn; launch-derived model context stays intact and chat speed is stored in `last_generation_metrics`. |
| IB-041 | Audit #24 deferred GUI user persistence | Done First Pass | GUI chat appends the user turn to the prompt in memory and only persists the user/assistant pair after generation succeeds, preventing failed sends from duplicating on retry. |
| IB-042 | Audit #28 health wait watches child | Done First Pass | `wait_for_healthy` now takes `&mut self` and calls `poll_exited()` inside the health loop so it returns immediately if llama-server exits before becoming healthy. |
| IB-043 | Audit #26 constant-time API-key comparison | Done First Pass | API key comparison uses the local `constant_time_eq` helper and has regression coverage. |
| IB-044 | Audit #25 transparent proxy quality | Done First Pass | Backend proxy reuses a shared reqwest client, forwards non-hop-by-hop request/response headers, raises the body cap to 128 MiB, and streams backend response bodies instead of buffering them. |
| IB-045 | Audit #21 keyed live stream status | Done First Pass | Process status exposes keyed `live_streams`, derives the current stream from the latest running/streaming snapshot, and the Logs page seeds from the full stream list instead of only the legacy singleton. |
| IB-046 | Audit #15 shared headless model load | Done First Pass | Headless `--model` now routes through `backend_load_model_with_overrides`, so launch preview, model stats, loading generation, and context tracking follow the same path as GUI/API loads. Follow-up: remove the quarantined legacy duplicate loader after smoke testing. |
| IB-047 | Audit #29 parse trace helper consolidation | Done First Pass | Parse-trace construction is centralized in `normalize::parse_trace` and reused by GUI chat, chat completions, text completions, and responses. Follow-up: continue consolidating quant/name matching and process-port parser helpers. |
| IB-048 | Audit #30 first structural split | Done First Pass | Moved parser/parse-trace policy out of `api/completions.rs` and `commands/chat.rs` into `normalize::parse_trace`; larger transport/UI splits remain follow-up work. |

## Done This Pass

- Added strict AgentAction extraction, repair, and validation module.
- Added `POST /v1/reliability/agent-action/validate`.
- Added tests for noisy output extraction, think-tag stripping, invalid text rejection, and confidence bounds.
- Upgraded the Context tab into a live runtime stats view using runtime status, context status, generation metrics, scheduler pressure, and GPU memory.

## Recent Related Work Already In Tree

- Fixed orphan `</think>` leaks in streaming and saved chat rendering.
- Improved managed-model API reuse so external apps do not reload the same GGUF on every chat/tool-call turn.

## New Endpoint

`POST /v1/reliability/agent-action/validate`

Request:

```json
{
  "text": "raw model output",
  "think_tag_style": "qwen"
}
```

Response:

```json
{
  "object": "agent_action.validation",
  "valid": true,
  "action": {
    "step_id": "uuid",
    "role": "worker",
    "goal": "current objective",
    "action": "registered_tool_name_or_final_answer",
    "arguments": {},
    "expected_outcome": "what should change",
    "success_check": "how progress is verified",
    "confidence": 0.9,
    "next_step": "continue"
  },
  "repaired_json": {},
  "visible_text": "cleaned model output",
  "errors": []
}
```

## Next Implementation Order

1. Audit #30 follow-up: split request parsing, prompt build, streaming, tool extraction, selector list, load options, and advanced flags into dedicated modules/components.
2. Audit #29 follow-up: consolidate `extract_quant`, `names_match`, launch-preview matching, and process-port parsers.
3. Audit #15 follow-up: remove `legacy_headless_load_model` after smoke testing shared headless loads.
4. Audit #12 follow-up: stream parser profile gating for hidden/control markers.
5. Audit #11 follow-up: add a config switch/default policy for API compaction.
6. Audit #10 follow-up: make implicit API JIT load opt-in and queue swaps behind active generations.
7. IB-024 follow-up: request-scoped cancel controls and multi-active UI state.
8. IB-016 follow-up: assert UI/live log visible text exports for leaked native markers and doubled tool blobs.
9. IB-018: context pressure regression around huge tool outputs.
10. IB-020: stop/unload lifecycle hardening.
