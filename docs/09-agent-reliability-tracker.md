# InferenceBridge Agent Reliability Tracker

Source plan: `C:\Users\richa\Desktop\Ai master plan\ai updates.md`

## Direction

InferenceBridge is the reliability gateway. It should clean, repair, validate,
measure, and route model responses so orchestrators do not have to trust raw
model text.

## Current Status

| ID | Task | Status | Notes |
| --- | --- | --- | --- |
| IB-001 | Canonical model response wrapper | In Progress | Existing chat/responses APIs include usage and timing metadata. A single internal wrapper type is still needed. |
| IB-002 | Think-tag cleaner | Done | `normalize::think_strip` handles standard/Qwen tags and orphan closing tags; streaming and saved-message display also hide leaks. |
| IB-003 | JSON extractor | Done | `normalize::agent_action::extract_first_json_value` extracts first object/array from noisy model output. |
| IB-004 | JSON repair pipeline | Done | `normalize::json_repair::repair_json` is reused by the new action validator. |
| IB-005 | Schema validator | Done | Added strict AgentAction schema validation in `normalize::agent_action`. |
| IB-006 | Retry formatter prompt | Not Started | Need retry prompt builder using validation errors. |
| IB-007 | Backend fallback router | Not Started | LM Studio proxy exists, but automatic fallback routing is not implemented. |
| IB-008 | True token/sec metrics | In Progress | Context tab now shows live prefill/decode/end-to-end rates, active request age, scheduler pressure, KV pressure, and GPU memory. Stream latency and perceived UI speed still need backend event timing. |
| IB-009 | Timeout and cancellation | In Progress | Streaming has first/inter-token timeouts and `/v1/inference/cancel`; non-stream request timeout policy still needs tightening. |
| IB-010 | Streaming stabiliser | Not Started | Need buffered flush control. |
| IB-011 | Replay logs | Not Started | Need prompt/output/backend/validation persistence. |
| IB-012 | Model health endpoint | In Progress | `/v1/health`, `/v1/metrics`, `/v1/runtime/status`, and `/v1/runtime/doctor` exist; `/health/models` and `/health/backend` aliases remain. |

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

1. IB-006: retry formatter prompt from validation errors.
2. IB-001: internal canonical response wrapper used by chat, responses, and completions.
3. IB-011: replay log table/file for prompt, raw output, repaired output, validation result, backend, timings.
4. IB-010: stream flush stabiliser.
5. IB-012: add `/v1/health/models` and `/v1/health/backend` aliases with structured detail.
