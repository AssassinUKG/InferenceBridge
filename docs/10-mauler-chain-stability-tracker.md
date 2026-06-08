# Mauler Chain Stability Tracker

Goal: make the full local chain stable enough for autonomous coding work:

`TheMauler -> InferenceBridge OpenAI-compatible API -> llama.cpp -> model tool calls -> Mauler tools -> project edits/tests/logs`

## Active Benchmark

- Workspace: `C:\Users\richa\Documents\MaulerBench\pokedex-arena`
- App: Pokedex Arena, React + TypeScript + Vite
- Model A: `gemma-4-26B-A4B-it-QAT-Q4_0.gguf`
- Model B: `qwen3.6-27B-Q4_K_M.gguf` / profile alias `qwen3.6-think`
- Provider: InferenceBridge OpenAI-compatible endpoint
- Primary endpoint: `http://127.0.0.1:8802/v1`

## Run Protocol

1. Start InferenceBridge and load the target model/profile.
2. Set TheMauler workspace to the benchmark project.
3. Select the matching model profile.
4. Use the same prompt for each model.
5. Keep Mauler autonomous/balanced unless a confirmation gate is specifically being tested.
6. After each run, export or inspect Mauler logs and InferenceBridge Logs.
7. Record every issue below before fixing it.

## Standard Prompt

```text
Inspect this project and fix it end to end. Make the Pokedex Arena app build and test. Implement search by Pokemon name and type. Make the team builder prevent duplicates and enforce a maximum of 6 Pokemon. Improve the UI only where useful. Run verification commands. Keep changes scoped to this workspace and summarize changed files plus verification results.
```

## Scorecard

| Area | Gemma4 | Qwen3.6 | Notes |
| --- | --- | --- | --- |
| Loads correct model/context | Not run | Not run |  |
| Starts without API/tool schema errors | Not run | Not run |  |
| Uses file/search tools correctly | Not run | Not run |  |
| Handles shell/test failures | Not run | Not run |  |
| Avoids malformed tool JSON | Not run | Not run |  |
| Avoids leaked thought/tool markup | Not run | Not run |  |
| Avoids context overflow | Not run | Not run |  |
| Finishes without manual rescue | Not run | Not run |  |
| `npm run build` passes | Not run | Not run |  |
| `npm test` passes | Not run | Not run |  |
| UI quality acceptable | Not run | Not run |  |

## Issue Log

| ID | Status | Chain Area | Model/Profile | Symptom | Evidence | Fix/Notes |
| --- | --- | --- | --- | --- | --- | --- |
| MCS-001 | Fixed | InferenceBridge logs | Gemma4 | Parsed `tool_call` JSON was duplicated into Raw logs. | Raw log showed literal native tool text plus normalized JSON. | Stopped appending `tool_call` events to `raw_output` in backend/frontend. |
| MCS-002 | Fixed | InferenceBridge parser/logs | Gemma4 | Malformed native tool blobs leaked as visible text. | `<|channel>thought ... <|tool_call>callcall::...` shown to user. | Gemma native extractor now removes unparseable native tool blocks; UI scrubs dangling marker noise. |
| MCS-003 | Fixed | InferenceBridge Logs UI | Tool-only runs | Logs Text tab showed “No visible text captured” for tool-only calls. | Logs page had repeated empty Text panels. | Text falls back to buffered visible text or tool-call summaries. |
| MCS-004 | Fixed | InferenceBridge API/context | Mauler long run | llama.cpp rejected prompt over loaded context instead of preflight compaction. | `prompt tokens exceed context size` HTTP 400. | Added deterministic pre-send compaction for `/v1/chat/completions` and `/v1/responses`. |
| MCS-005 | Open | Run environment | Pre-run | InferenceBridge API was not reachable at `http://127.0.0.1:8802` during benchmark setup. | `Invoke-RestMethod /health` and `/v1/models` returned “Unable to connect to the remote server”. | Start/restart InferenceBridge before model runs; recheck with evidence collector. |
| MCS-006 | Planned | Mauler automation | All profiles | Runs still require hand-driving the UI, which makes regression testing slow and inconsistent. | Manual runbook only. | Add automated Mauler replay runner for workspace/profile/prompt/log export. |
| MCS-007 | Planned | Cross-app tracing | Mauler + InferenceBridge | Requests cannot be reliably correlated across Mauler task logs and InferenceBridge runtime logs. | Logs currently rely on time/model matching. | Propagate a per-run request/correlation ID from Mauler into InferenceBridge request metadata and log entries. |
| MCS-008 | Planned | Log assertions | Gemma4/Qwen | Regressions can reintroduce leaked `<|tool_call>` / `<channel|>` markers without failing tests. | Previous UI screenshots showed leaked markers. | Add log assertions over Raw/Text/history exports for native marker leaks and doubled-token blobs. |
| MCS-009 | Done First Pass | Tool schema validation | All local models | Integer, boolean, enum, and array args can be emitted as strings or malformed JSON. | Prior errors: `cannot unmarshal string into Go struct field ... of type int`. | Added tests for typed coercion, enum validation, required fields, and strict repair prompt formatting. |
| MCS-010 | Planned | Browser verification | Mauler benchmark | Passing build/test can miss broken or blank UI. | No screenshot assertion in current benchmark. | Add browser/screenshot verification after app builds. |
| MCS-011 | Planned | Context pressure | Long autonomous runs | Huge tool outputs may still break the next model turn or bury current instructions. | Prior context overflow in Mauler run. | Add pressure test with large tool outputs before the next model turn and assert compaction occurs. |
| MCS-012 | Open | HelixClaw model routing | HelixClaw + InferenceBridge | Existing HelixClaw `ib-q36-*` agents point at `Qwen3.6-27B-Q4_K_S.gguf`, while this benchmark request targets Qwen3.6 Q4_K_M. | `helixclaw config agents`; InferenceBridge `/v1/models`. | Add benchmark-specific InferenceBridge model profiles and project role-model routing rather than changing global defaults. |
| MCS-013 | Done First Pass | InferenceBridge visibility | All profiles | Model responses are still spread across raw text, live streams, parse traces, and API response objects. | Current debug requires stitching UI/log/time/model manually. | Added canonical response wrapper for non-stream and streaming API replay records. |
| MCS-014 | Done First Pass | Cross-app tracing | Mauler + InferenceBridge | A run cannot reliably map a Mauler agent turn to one InferenceBridge request. | Request UUID exists in InferenceBridge but is not yet exposed as a stable replay/correlation artifact. | Replay records include internal request ID plus optional client IDs from headers, body metadata, extra, or top-level fields. |
| MCS-015 | Done First Pass | Replay evidence | All profiles | Failed runs do not leave enough structured evidence to replay parser/tool issues. | Existing UI logs are useful but not a deterministic replay fixture. | Chat/responses API paths append prompt, normalized output, tool calls, usage, timing, backend, and correlation data to replay JSONL. |
| MCS-016 | Planned | Mauler client tracing | Mauler + InferenceBridge | Mauler does not yet stamp every model call with a stable turn/run correlation ID. | InferenceBridge can now accept correlation headers/body metadata, but Mauler must send them. | Add `x-correlation-id` or `metadata.correlation_id` shaped like `mauler:{projectId}:{agentId}:{turnId}` on every model call. |
| MCS-017 | Planned | Mauler tool dispatch | All local models | Mauler can still pass typed tool params as strings before InferenceBridge repairs them. | Prior malformed args included string ints such as `"max_tool_calls": "8"` and `"timeout_seconds": "180"`. | Normalize tool args against schema before dispatch: ints, booleans, enums, arrays, and nested objects. |
| MCS-018 | Planned | Mauler failure evidence | Mauler + InferenceBridge | Failed agent turns do not automatically attach the matching InferenceBridge replay evidence. | Replay JSONL now exists, but Mauler does not consume or link it. | On failure, attach or reference the replay row matching the correlation ID. |
| MCS-019 | Planned | Mauler recovery loop | Tool-calling models | Bad tool JSON can still become a hard agent failure too early. | Qwen/Gemma can emit malformed tool payloads under pressure. | Treat schema validation failure as recoverable: send one strict repair retry before failing the turn. |
| MCS-020 | Planned | Mauler context hygiene | Long autonomous runs | Huge tool outputs may still be dumped into the next model turn. | Context overflow previously broke llama.cpp calls. | Summarize, truncate, or attach large tool outputs before the next model request. |
| MCS-021 | Planned | Mauler log assertions | Gemma4/Qwen | Mauler visible logs can still accept leaked native markers unless explicitly checked. | Prior visible output showed `<|tool_call>`, `<channel|>`, `<turn|>`, and `callcall::` blobs. | Flag/fail final visible output containing control markers or doubled native tool blobs. |

## Candidate Improvements To Add

- Add an automated Mauler benchmark runner that can replay the same prompt/profile/workspace and export logs without manual UI driving.
- Add InferenceBridge `/v1/runtime/logs` or export endpoint so Mauler can attach backend evidence to task-run logs.
- Add per-run correlation IDs propagated from Mauler to InferenceBridge request metadata.
- Add a “tool-call conformance” benchmark with intentionally typed schemas: integers, booleans, enums, arrays, and nested objects.
- Add a context pressure benchmark that injects large tool outputs and verifies compaction before the next model request.
- Add a log assertion test that fails if Raw/Text contains known control markers such as `<|tool_call>`, `<channel|>`, or doubled-token native blobs.
- Add a recovery benchmark for disabled tools, malformed JSON retry, shell failure repair, and online/offline toolset switching.
- Add screenshot/browser verification to the benchmark so “build passes” is not mistaken for “app works.”
