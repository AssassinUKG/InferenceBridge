# InferenceBridge Audit — Findings for Fixing (2026-06-11)

Scope: full read of the Rust backend (`src-tauri/src`) and React frontend (`src/`).
`cargo check` and `tsc --noEmit` both pass clean — everything below is a logic,
stability, compatibility, or security issue found by code review.

Each item is self-contained so a separate session can pick it up cold.

## Progress Summary

Done first pass: 13 / 30
Remaining: 17 / 30

| ID | Status |
| --- | --- |
| 1 | Fixed first pass |
| 2 | Fixed first pass |
| 3 | Fixed first pass |
| 4 | Fixed first pass |
| 5 | Fixed first pass |
| 6 | Fixed first pass |
| 7 | Fixed first pass |
| 8 | Fixed first pass |
| 9 | Fixed first pass |
| 10 | Fixed first pass |
| 14 | Fixed first pass |
| 22 | Fixed first pass |
| 27 | Fixed first pass |

---

## P0 — Critical (breaks API usage under real load)

### 1. Concurrent API requests cancel each other (single global cancellation token)
Status: Fixed first pass (2026-06-11) — API/live generations now get per-request cancellation tokens; starting a new generation no longer cancels older request tokens. Streaming scheduler permits are moved into the SSE stream lifetime.

- `begin_api_generation` ([state.rs:204-243](src-tauri/src/state.rs)) does
  `s.generation_cancel.cancel()` and replaces the token on **every** new request.
- The request scheduler permit in `chat_completions`
  ([completions.rs:1952](src-tauri/src/api/completions.rs)) is `let _permit = scheduler.acquire().await;`
  inside the handler — for streaming responses the permit is **dropped when the
  handler returns the SSE response**, not when the stream finishes. So a second
  request is admitted immediately, calls `begin_api_generation`, and cancels the
  first request's token. `consume_sse_stream` then emits `Done` with partial text
  and `finish_reason: "stop"` — the first client gets a silently truncated response.
- This is the #1 stability issue for agent clients (Claude Code, HelixClaw, etc.)
  that fire overlapping requests.
- Fix: per-request `CancellationToken` (not a shared field); move the scheduler
  permit *into* the SSE stream closure so it is held until `[DONE]`; only the
  explicit `/v1/inference/cancel`, model unload, or app shutdown should cancel.

### 2. Client disconnect does not stop generation (GPU burns until completion)
Status: Fixed first pass (2026-06-11) — streamed API responses now install a drop guard that cancels the request token and marks the request `disconnected`; the llama SSE consumer exits when its receiver is closed.

- `consume_sse_stream` ([streaming.rs:245-465](src-tauri/src/engine/streaming.rs))
  ignores all `tx.send` errors (`let _ = tx.send(...)`). When the HTTP client
  disconnects, the axum SSE stream is dropped, `rx` is dropped, but the spawned
  consumer keeps reading the llama-server SSE stream to the very end.
- Also: when the SSE stream is dropped, `finish_api_generation_for_request` never
  runs — `active_generation` stays `"running"`/`"streaming N tokens"` forever
  (until the next request overwrites it), and the Debug UI shows a stuck generation.
- Fix: when `tx.send` returns `Err` (receiver gone), cancel the token / abort the
  reqwest stream so llama-server stops decoding; add a drop-guard around the SSE
  stream that calls `finish_api_generation_for_request(..., "disconnected")`.

### 3. Any unknown numeric JSON field can trigger a model reload (context-size sniffing)
Status: Fixed first pass (2026-06-11) — context/load override extraction now only reads known top-level fields and explicit config objects; regression tests cover unrelated numeric fields and nested unknown override objects.

- `extract_context_size_from_hash_map` / `extract_context_size_from_value`
  ([completions.rs:447-494](src-tauri/src/api/completions.rs)) fall back to
  **recursively scanning every unknown request field** and accept **any bare
  number** as the requested context size (`Value::Number` branch returns the
  number itself, not just values under known keys).
- Example: an OpenAI client sending `{"timeout": 600000}` or `{"top_logprobs": 5}`
  or any vendor extension with a large number makes `requested_context_size()`
  return that number → `context_request_matches_preview` fails → full model
  reload mid-conversation (which also cancels the active generation, see #1).
- Same pattern in `extract_runtime_load_overrides` ([completions.rs:370-420]):
  it deep-scans `extra` for generic keys like `"file"`, `"filename"`, `"quant"`,
  `"fit"` — any unknown nested object containing those keys becomes an hf_file /
  fit_mode override and forces a swap.
- Fix: only read known keys at the **top level** (and inside the explicit
  `options` object). Delete the recursive fallback over `map.values()`.

### 4. Auth bypass on `/api/v1/*` and the transparent proxy
Status: Fixed first pass (2026-06-11) — API-key middleware now covers `/api/v1/*` and proxy fallback routes when a key is configured; only exact health routes and OPTIONS are exempt; comparison is constant-time.

- `require_api_key` ([server.rs:563-607](src-tauri/src/api/server.rs)) skips auth
  for any path **not** starting with `/v1`. That leaves unauthenticated:
  - `/api/v1/models/load` and `/api/v1/models/unload` (full model control),
  - **every** transparently proxied llama-server endpoint: `/completion`
    (full inference!), `/props`, `/slots`, `/tokenize`, etc.
  - Also `path.ends_with("/health")` exempts any `/v1/anything/health`.
- Combined with `CorsLayer::permissive()` and the ability to bind `0.0.0.0`, an
  API key offers no real protection, and even on localhost any webpage can
  drive the server cross-origin (drive-by model load with `extra_args` =
  arbitrary llama-server CLI args).
- Fix: apply the API-key check to all routes except exactly `/v1/health` and
  `/api/v1/health`; require auth on the proxy fallback; consider locking CORS
  down by default; treat `extra_args`/`custom_template_path`/`draft_model_path`
  from remote requests as privileged.

### 5. Port eviction force-kills arbitrary processes
Status: Fixed first pass (2026-06-11) — startup port eviction now refuses to kill unknown/self processes and only force-kills recognized `llama-server` / `inference-bridge` owners; tests cover the killable policy.

- `evict_port_blocker` ([server.rs:167-258](src-tauri/src/api/server.rs)) runs
  `taskkill /F` on **whatever PID** is LISTENING on the configured API port,
  regardless of what it is. If the user runs anything else on 8800 (another dev
  server, etc.), InferenceBridge kills it at startup.
- `detect_api_port_owner_windows` ([commands/model.rs:1782](src-tauri/src/commands/model.rs))
  already implements the right policy (only `llama-server`/`inference-bridge`
  marked `killable`). Reuse that: only kill recognized own processes; otherwise
  fall back to the 8802–8810 port fallback and surface a clear error.

### 6. WMIC is used but is removed on current Windows 11
Status: Fixed first pass (2026-06-11) — Windows process cleanup/listing and system RAM lookup now use PowerShell `Get-CimInstance` instead of `wmic`; process listing keeps the existing `tasklist` fallback.

- `kill_all_managed_processes` ([engine/process.rs:1098](src-tauri/src/engine/process.rs)),
  `list_llama_processes` ([commands/model.rs:1877]), and `get_system_ram_mb`
  ([commands/model.rs:2009]) shell out to `wmic`, which is deprecated and
  **absent by default on Windows 11 24H2+** (this machine is 26200).
- Consequence: stale llama-server cleanup on launch/exit silently does nothing →
  orphaned GPU-resident processes, VRAM exhaustion on next load; system RAM stat
  shows 0. `list_llama_processes` has a tasklist fallback; the other two do not.
- Fix: replace with PowerShell `Get-CimInstance Win32_Process` or native APIs
  (`sysinfo` crate would cover all three cleanly).

---

## P1 — High (API correctness / OpenAI compatibility)

### 7. `/v1/completions` ignores `stream: true`
Status: Fixed first pass (2026-06-11) — `/v1/completions` now rejects `stream: true` with a clear 400 instead of returning non-stream JSON to a streaming client. Full SSE support is still a follow-up.

- `text_completions` ([completions.rs:2500-2617]) hard-codes `stream: false` in
  the backend request and always returns a JSON body. OpenAI clients that
  request streaming hang waiting for SSE or fail parsing. Implement streaming or
  reject with a clear error.

### 8. Non-streaming requests have no overall HTTP timeout
Status: Fixed first pass (2026-06-11) — non-streaming llama `/completion` and `/v1/chat/completions` client calls now have a 600s overall timeout; streaming keeps its separate first/inter-token timeout path.

- The shared client ([engine/client.rs:135-147](src-tauri/src/engine/client.rs))
  sets only `connect_timeout`. `LlamaClient::complete()` (non-stream chat,
  text completions, tool-repair calls, GUI vision path in
  [chat.rs:489](src-tauri/src/commands/chat.rs)) can hang **forever** if
  llama-server wedges. Streaming has first/inter-token timeouts; non-streaming
  has nothing. Add a generous per-request timeout (e.g. reuse
  `first_token_timeout_secs` + a max-duration cap).

### 9. `finish_reason` is never `"length"`
Status: Fixed first pass (2026-06-11) — llama.cpp `stopped_limit` / limit-like `stop_type` metadata is now carried through non-stream and stream paths and mapped to OpenAI `finish_reason: "length"` unless tool calls take precedence.

- Both chat paths and text completions report `"stop"` (or `"tool_calls"`) even
  when generation hit `n_predict`. Clients cannot detect truncation. llama-server
  reports `stop_type`/`stopped_limit` — propagate it ("length" when token-limited).

### 10. Fuzzy model matching + implicit auto-swap is too aggressive
Status: Fixed first pass (2026-06-11) — API model resolution no longer uses registry substring matching, and loaded-model reuse rejects vague single-token aliases like `qwen` or `27b` while preserving specific aliases such as `qwen3.6`. Follow-up: make implicit API JIT loading fully opt-in and queue swaps behind active generations.

- `loaded_model_matches_request` ([completions.rs:659-714]) accepts
  `loaded.contains(search)` and token-subset matches — `"qwen"` matches any
  loaded Qwen file; conversely a near-miss name triggers `resolve_loaded_model`
  → `backend_load_model_with_overrides`, which **cancels active inference**
  ([commands/model.rs:574-585]) and swaps models. Two clients configured with
  different model strings will thrash model loads continuously.
- Fix: make the match strict (exact filename / registry alias); make implicit
  JIT load-on-request opt-in via config; never cancel an in-flight generation
  for an implicit swap — queue behind it (the `model_load_mutex` already exists).

### 11. Silent conversation rewriting (`compact_messages_to_fit`)
- [completions.rs:1661-1736] silently drops middle messages and injects a
  summary system message when the estimated prompt exceeds the context. For an
  OpenAI-compatible endpoint this is surprising and corrupts agent transcripts
  (tool-call pairs can be split). Make it configurable (off by default for the
  API), and surface it (response header/field) when it happens.

### 12. Streaming parse-state corruption and false positives
- `ReasoningDelta` handling wraps **each delta** in `<think>…</think>` when
  rebuilding `raw_full_text` ([completions.rs:2226-2229]) → parse traces/replay
  contain `<think>a</think><think>b</think>…`.
- `emit_parsed_content` ([streaming.rs:67-243]) hides literal `<think>`,
  `<tool_call>`, `<div class="tool_code">` from *any* model output — if the user
  asks the model to print those strings (code samples, docs), visible content is
  swallowed; boundary tags (`<|im_end|>` etc.) hard-truncate output. These
  heuristics should be gated per model profile, not global.

### 13. Tool-call argument "repair" issues hidden extra LLM calls
- `repaired_tool_arguments` ([completions.rs:1533-1572]) silently runs a second
  completion against the loaded model when schema validation fails — extra
  latency, no cap, invisible to the client. Make it a config flag and bound it.

### 14. `stream_options.include_usage` semantics inverted vs OpenAI
Status: Fixed first pass (2026-06-11) — stream usage now defaults off; when requested it is emitted as a separate final chunk with `"choices": []` instead of being attached to the finish chunk.

- Default is `true` ([completions.rs:1891-1895]) and usage is attached to the
  final chunk that also carries `finish_reason`. OpenAI: default **false**, and
  usage arrives in a dedicated final chunk with `"choices": []`. Strict clients
  mis-parse. Match the spec.

### 15. Headless `--model` auto-load bypasses the real load path
- `headless_load_model` ([lib.rs:517-717](src-tauri/src/lib.rs)) duplicates
  launch logic and never sets `last_launch_preview` / `model_stats` /
  `loading_generation`. Consequences: first API request with an explicit ctx
  always forces a reload; `ensure_runtime_vision_ready` always rejects images;
  context-limit compaction has no limit. Route it through
  `backend_load_model_with_overrides` instead.

### 16. `server.default_ctx_size` only works in headless mode
- Config field ([config.rs:31]) is honored only by the headless auto-load
  ([lib.rs:412]). GUI loads and API loads ignore it. Wire it into
  `resolve_launch_context_size` or remove it from config to avoid confusion.

### 17. Startup port fallback silently rewrites saved config
- `reserve_startup_api_port` ([api/runtime.rs:142-177]) picks 8802–8810 when
  8800 is busy and **persists** the new port to the config file. Every client
  pointing at 8800 breaks permanently, even after the conflict disappears.
  Use the fallback for the session only, or make persistence opt-in and surface
  a prominent UI notice.

### 18. mmproj auto-pairing can attach the wrong projector
- `find_mmproj_for_model` ([engine/process.rs:71-87]) returns the
  best-token-overlap mmproj in the model's folder **even when the overlap score
  is 0**. In a mixed folder a Gemma model can get a Qwen mmproj → garbage vision
  output or crash. Require a minimum score / family match, else log + skip.

---

## P2 — Medium (app stability & polish)

### 19. `flushSync` per streamed token in the chat UI
- [useChat.ts:61,71](src/hooks/useChat.ts) forces a synchronous React re-render
  for **every token event**. At 100+ tok/s this stutters the whole window (the
  "faster than LM Studio" story dies in the UI). Buffer deltas and flush on
  `requestAnimationFrame`, or coalesce events on the Rust side.

### 20. Status polling storm
- `useModel` polls `get_process_status` every 1 s; each call takes a write lock,
  runs `check_crashed`, makes an HTTP self-probe (1.2 s timeout), and when the
  API is unreachable shells out to `netstat` + `tasklist`
  ([commands/model.rs:1497-1717]). Plus `useGpuStats` (nvidia-smi), `useContext`
  (/slots), ProcessManager and DebugInspector intervals. Push state via the
  existing Tauri events and slow background polls to 5–10 s.

### 21. Single `active_generation` / `live_stream` slot
- With `parallel_slots > 1`, concurrent requests overwrite each other's
  `active_generation` status and the status bar / Debug Inspector shows the
  wrong request. `live_streams` (vec) already exists — derive UI state from it
  and key everything by `request_id`.

### 22. GUI `stop_generation` cancels whatever is running, including API requests
Status: Fixed first pass (2026-06-11) — cancellation now goes through a request-token map. Current broad stop/cancel commands deliberately cancel all active generations; follow-up should add request-scoped UI/API stop controls.

- [chat.rs:758-768] cancels the global token and marks the live stream
  "cancelled" — clicking Stop in the chat kills an unrelated in-flight API
  request. Follows directly from #1's shared-token design.

### 23. GUI chat clobbers `model_stats`
- [chat.rs:706-717] rewrites `model_stats` after every chat turn with
  `context_size` taken from `last_context_status.total_tokens` (may be 0/stale)
  and `memory_mb: 0`. `model_stats.context_size` is what the API uses as the
  loaded-context signal — keep launch-derived ctx immutable, store chat tok/s
  elsewhere.

### 24. User message persisted before generation can start
- [chat.rs:376-381] writes the user message to the session DB before checking
  the backend can serve it. On failure the message stays, so a retry duplicates
  it. Persist after success, or delete on failure.

### 25. Transparent proxy quality (also see #4)
- `backend_proxy_fallback` ([server.rs:450-557]) buffers the whole backend
  response (`resp.bytes()`) so native streaming endpoints can't stream; caps
  request bodies at 10 MB (large embedding batches fail); drops all response
  headers except content-type; forces `Content-Type: application/json` on
  forwarded requests; builds a fresh `reqwest::Client` per call. Stream the
  body through, forward headers both ways, reuse the shared client.

### 26. API-key comparison is not constant-time
- [server.rs:595] uses `==` on strings. Low risk locally, trivial to fix with
  `subtle`-style constant-time compare.

### 27. Unknown requested model returns 503 instead of 404
Status: Fixed first pass (2026-06-11) — unknown plain model names now return OpenAI-style `model_not_found` 404 instead of attempting a doomed JIT load; loaded aliases still reuse the active model and plausible local GGUF/path requests may still attempt load.

- `resolve_loaded_model` falls back to treating an unknown name as a filename
  and surfaces "Could not load model 'gpt-4o': …" with 503. OpenAI returns 404
  `model_not_found` — clients branch on this. Return 404 when the name resolves
  to nothing in the registry and JIT load is not applicable.

### 28. `wait_for_healthy` / health loops don't watch the child
- `LlamaProcess::wait_for_healthy` ([engine/process.rs:895-912]) polls `/health`
  for the full timeout even if the child already exited (the main load path in
  `commands/model.rs` does check crash every ~2 s, but only that path). Fold a
  `poll_exited` check into `wait_for_healthy` itself.

### 29. Duplicate/diverging helper implementations
- Two `extract_quant`, two `names_match`, two `build_parse_trace`,
  `launch_preview_matches_model` duplicated in `api/completions.rs` and
  `commands/chat.rs`, two netstat/tasklist parsers (`server.rs` vs
  `commands/model.rs`). They have already started drifting (model-matching
  rules differ between API and GUI). Consolidate into shared modules.

### 30. `api/completions.rs` (3 200 lines) and `ModelSelector.tsx` (2 000 lines)
- Both files mix transport, parsing, policy, and UI concerns and are the two
  highest-churn files in the repo. Split (request parsing / prompt build /
  streaming / tool-extraction; selector list / load options / advanced flags)
  before the next feature lands.

---

## Suggested fix order

1. #1 + #2 + #22 together (one redesign: per-request tokens, permit held through
   stream, disconnect propagation) — biggest stability win for API users.
2. #3 + #10 (stop sniffing context/overrides from arbitrary fields; strict model
   match) — stops surprise model reloads.
3. #4 + #5 + #6 (auth on all routes, safe port eviction, WMIC replacement) —
   security + Windows 11 correctness.
4. #7, #8, #9, #14, #27 — OpenAI-compat batch; mostly small, all in
   `api/completions.rs` / `errors.rs`.
5. #19 + #20 — UI smoothness (the visible "polish" items).
6. Remainder in listed order.
