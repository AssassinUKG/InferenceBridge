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

1. IB-016 follow-up: assert UI/live log visible text exports for leaked native markers and doubled tool blobs.
2. IB-018: context pressure regression around huge tool outputs.
3. IB-020: stop/unload lifecycle hardening.
4. IB-012: add `/v1/health/models` and `/v1/health/backend` aliases with structured detail.
5. IB-007: backend fallback routing.
6. IB-017 follow-up: add one integration-style replay fixture for malformed tool args.
