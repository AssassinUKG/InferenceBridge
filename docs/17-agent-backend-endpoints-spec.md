# 17 — Agent-Backend Endpoint Spec (embeddings · structured output · Anthropic messages)

**Audience:** Codex (autonomous implementer). **Status:** ready to build.
**Goal:** Close the three highest-value gaps between InferenceBridge (IB) and LM Studio / Unsloth
Studio 2026, all of which llama-server already supports so IB only has to wire them through.

Implement in this order — each is independently shippable:

1. **Structured output** — `response_format` (JSON schema) → llama-server constrained decoding. *Smallest, highest agent-reliability win.*
2. **`/v1/embeddings`** — proxy to llama-server so agents get local memory/RAG without a second server.
3. **`/v1/messages`** — Anthropic-compatible endpoint so Claude Code / HelixClaw can point straight at IB.
4. **Context compaction fix** — replace the crude hot-path truncation (System B) with model summarization (System A), and fix a UTF-8 panic + silent-drop + tool-pairing bug. *Independent of 1–3; see Feature 4.*

> **Ground rules for Codex**
> - Do **not** break the existing `/v1/chat/completions`, `/v1/completions`, `/v1/responses` paths. Add, don't rewrite.
> - Match surrounding code style (serde structs with `skip_serializing_if`, `ApiErrorResponse` for errors, `tracing` for logs).
> - Every new API struct field uses `#[serde(default)]` so old clients keep working.
> - The upstream-provider proxy short-circuit (`active_openai_provider`) at the top of each handler must be preserved — new endpoints should honor it where it makes sense (see each section).
> - After each feature: `cargo build` in `src-tauri/`, `cargo clippy`, add a unit test, and add a config/README note.

---

## Current-state map (verified against the tree)

| Thing | Location |
|---|---|
| Public route table (`/v1/*`) | `src-tauri/src/api/server.rs` → `api_routes()` @ ~L691 |
| Native route table (`/api/v1/*`) | `src-tauri/src/api/server.rs` → `native_api_routes()` @ ~L767 |
| Incoming OpenAI request struct | `src-tauri/src/api/completions.rs` → `ChatCompletionRequest` @ ~L66 |
| Request builder (prompt + sampling) | `src-tauri/src/api/completions.rs` → `build_chat_request()` @ ~L1719 |
| `grammar: None` sites to wire | `completions.rs` @ ~L1776 (chat), ~L2755 (text), ~L1611 (repair — leave as-is) |
| llama-server HTTP client | `src-tauri/src/engine/client.rs` → `LlamaClient` @ ~L155 |
| Upstream `CompletionRequest` struct | `engine/client.rs` @ ~L18 — **already has `grammar: Option<String>`** @ L50 |
| How a handler gets the client + port | `chat_completions` @ ~L2003: read `s.process.port()`, then `LlamaClient::new(llama_port)` |
| Launch config | `src-tauri/src/engine/process.rs` → `LaunchConfig` @ ~L810 |
| Example config | `inference-bridge.example.toml` |

**Key architectural facts:**
- IB runs **one** managed llama-server on an internal ephemeral port; IB's public API (port 8800) proxies to it. Internal port is `state.read().await.process.port()`.
- `LlamaClient` (`engine/client.rs`) already knows how to reach `/completion`, `/v1/chat/completions`, `/slots`, `/props`, `/health` on that internal port. Adding an endpoint = one new method there.
- `CompletionRequest` (the struct sent to llama-server) **already carries a `grammar` field** — it's just hardcoded to `None` on every build path today.

---

## Feature 1 — Structured output (`response_format` → constrained decoding)

### What clients send (OpenAI shape)
```jsonc
"response_format": { "type": "json_object" }
// or
"response_format": {
  "type": "json_schema",
  "json_schema": { "name": "weather", "strict": true, "schema": { /* JSON Schema */ } }
}
```

### What llama-server accepts on `/completion`
llama-server converts schema → GBNF internally. You do **not** need to write a schema→GBNF converter in Rust. Pass through, in priority order:
- `json_schema`: `{...}` — the raw JSON Schema object (preferred; llama-server builds the grammar).
- `grammar`: `"..."` — raw GBNF (already supported by `CompletionRequest`).
- For `{"type":"json_object"}` with no schema — send a permissive `json_schema` of `{"type":"object"}` (or set grammar to the built-in `root ::= object` JSON grammar). Simplest: pass `json_schema: {"type":"object"}`.

### Steps

1. **`engine/client.rs` — `CompletionRequest` (~L18):** add a field next to `grammar`:
   ```rust
   /// Raw JSON Schema for constrained decoding. llama-server converts it to a
   /// grammar server-side. Mutually exclusive with `grammar` (schema wins).
   #[serde(skip_serializing_if = "Option::is_none")]
   pub json_schema: Option<serde_json::Value>,
   ```
   Then fix every `CompletionRequest { ... }` literal to include `json_schema: None` (compiler will list them: chat @ ~L1752, text @ ~L2755-ish, repair @ ~L1596). There are ~3-4 sites.

2. **`api/completions.rs` — `ChatCompletionRequest` (~L66):** add
   ```rust
   #[serde(default, alias = "responseFormat")]
   pub response_format: Option<serde_json::Value>,
   ```
   (Keep it `serde_json::Value` — avoids a rigid enum and tolerates client variants. Parse it in a helper.)

3. **Add a translator** in `completions.rs`:
   ```rust
   /// Maps an OpenAI `response_format` into a llama-server json_schema payload.
   /// Returns None when no constraint is requested.
   fn response_format_to_json_schema(rf: Option<&serde_json::Value>) -> Option<serde_json::Value> {
       let rf = rf?;
       match rf.get("type").and_then(|v| v.as_str()) {
           Some("json_schema") => rf.get("json_schema")
               .and_then(|js| js.get("schema"))
               .cloned(),
           Some("json_object") => Some(serde_json::json!({ "type": "object" })),
           _ => None,
       }
   }
   ```

4. **`build_chat_request()` (~L1719):** replace `grammar: None` (~L1776) with wiring:
   ```rust
   json_schema: response_format_to_json_schema(req.response_format.as_ref()),
   grammar: None,
   ```
   `req` is already owned here, so `req.response_format.as_ref()` is available before the struct is consumed — pull it into a local at the top of the fn if borrow-check complains.

5. **Text completions path** (`TextCompletionRequest` @ ~L2606, build @ ~L2755): add the same `response_format` field + wiring for parity. Lower priority; agents use chat.

6. **Streaming:** no change needed — `stream_chat_completion` forwards the same `CompletionRequest`, so grammar/json_schema flow through automatically. Verify by grepping the streaming build path reuses `build_chat_request`.

### Acceptance
- `curl` with `response_format: {"type":"json_object"}` returns strictly-parseable JSON.
- `response_format: {"type":"json_schema", json_schema:{schema:{...required...}}}` returns output matching the schema (test with a required-enum field — llama-server will refuse to emit off-schema tokens).
- Requests **without** `response_format` are byte-identical to today (no `json_schema` key serialized — guaranteed by `skip_serializing_if`).
- Unit test: `response_format_to_json_schema` for all three arms (`json_schema`, `json_object`, `None`).

---

## Feature 2 — `/v1/embeddings`

### Design decision (read before coding)
llama-server serves embeddings at `POST /v1/embeddings` **only when its model is loaded in embedding mode** (`--embeddings`, and ideally an embedding model like `nomic-embed-text` / `bge`). A generation model launched normally will **not** reliably return embeddings.

Two implementation phases — **do Phase A first, ship it, then decide on B:**

**Phase A (minimal, this PR): transparent proxy + honest failure.**
Proxy `POST /v1/embeddings` to the currently-loaded llama-server's `/v1/embeddings`. If the loaded model isn't embedding-capable, surface llama-server's error verbatim through `ApiErrorResponse` with a hint. This unblocks users who load an embedding model and costs almost nothing.

**Phase B (follow-up, separate doc/PR): dedicated embedding instance.**
Add an optional second managed llama-server for an embedding model (config `[embeddings] model_path`, its own port in `AppState`), so chat + embeddings run concurrently. This is the LM-Studio-parity experience but needs process-management plumbing — **out of scope for this PR**, note it as a TODO.

### Phase A steps

1. **`engine/client.rs` — add a method to `LlamaClient`:**
   ```rust
   /// Proxy an OpenAI-shaped embeddings request to llama-server's /v1/embeddings.
   pub async fn embeddings(&self, body: &serde_json::Value) -> Result<serde_json::Value> {
       let url = format!("{}/v1/embeddings", self.base_url);
       let resp = shared_client().post(&url).json(body).send().await?;
       if !resp.status().is_success() {
           let status = resp.status();
           let text = resp.text().await.unwrap_or_default();
           anyhow::bail!("llama-server /v1/embeddings returned {status}: {text}");
       }
       Ok(resp.json().await?)
   }
   ```
   (Use the same shared `reqwest::Client` accessor the other methods use — see `complete()` @ ~L173 for the exact pattern; don't build a fresh client.)

2. **New handler** — put it in a new file `src-tauri/src/api/embeddings.rs` (mirror `models.rs` structure) or append to `completions.rs`. Signature mirrors `chat_completions`:
   ```rust
   pub async fn embeddings(
       State(state): State<SharedState>,
       headers: HeaderMap,
       Json(mut body): Json<serde_json::Value>,
   ) -> Result<Json<serde_json::Value>, ApiErrorResponse> {
       // 1. Honor upstream provider proxy (embeddings often want the same routing):
       if let Some(upstream) = crate::api::upstream::active_openai_provider(&state).await {
           return crate::api::upstream::proxy_json_to_openai_provider(
               state.clone(), upstream, "/embeddings", body, &headers).await;
       }
       // (check the exact proxy fn signature in api/upstream.rs and match it)

       // 2. Require a loaded model / resolve internal port:
       let llama_port = {
           let s = state.read().await;
           if !s.process.is_running() { /* return ApiErrorResponse::no_model_loaded() */ }
           s.process.port()
       };

       // 3. Proxy:
       let client = LlamaClient::new(llama_port);
       match client.embeddings(&body).await {
           Ok(v) => Ok(Json(v)),
           Err(e) => Err(ApiErrorResponse::inference_failed(&format!(
               "Embeddings failed. The loaded model may not be an embedding model \
                (load one built with --embeddings, e.g. nomic-embed-text). Underlying: {e}"))),
       }
   }
   ```
   Check `api/errors.rs` for the exact `no_model_loaded` / `inference_failed` constructors (they exist — see errors.rs L55/L62).

3. **Register routes** in `server.rs`:
   - In `api_routes()` (~L705): `.route("/embeddings", axum::routing::post(super::embeddings::embeddings))`
   - Add `pub mod embeddings;` to `api/mod.rs`.
   - The auth-exempt list in `is_api_auth_exempt` should **not** include embeddings (require API key like other endpoints).

4. **`mod.rs`:** declare the module.

### Acceptance
- Load an embedding-capable GGUF → `POST /v1/embeddings {"input":"hello","model":"..."}` returns `{"data":[{"embedding":[...],"index":0}],...}` in OpenAI shape.
- Load a normal chat model → clear, actionable error (not a 500 with a raw stack).
- No model loaded → `ApiErrorResponse::no_model_loaded()`.
- OpenAI Python SDK `client.embeddings.create(...)` against IB works.
- Document the "load an embedding model" requirement in README + example TOML, and leave a `// TODO(phase-b): dedicated embedding instance` marker.

---

## Feature 3 — `/v1/messages` (Anthropic Messages API)

This is the largest of the three. Goal: Claude Code / HelixClaw configured with `ANTHROPIC_BASE_URL=http://127.0.0.1:8800` can talk to IB natively. Strategy: **translate Anthropic ⇄ internal, reusing `build_chat_request` and the existing completion/stream engine.** Do **not** fork the generation path.

### New module: `src-tauri/src/api/messages.rs`

#### Request translation (Anthropic → internal)
Anthropic `POST /v1/messages` body highlights:
- `model`, `max_tokens` (**required** in Anthropic), `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream`.
- `system`: top-level string **or** array of text blocks (NOT a message with role=system).
- `messages[]`: each has `role` (`user`|`assistant`) and `content` that is either a string or an array of blocks: `{type:"text"}`, `{type:"image", source:{type:"base64",media_type,data}}`, `{type:"tool_use", id, name, input}`, `{type:"tool_result", tool_use_id, content}`.
- `tools[]`: `{name, description, input_schema}` (JSON Schema). Anthropic tool shape differs from OpenAI's `{type:"function", function:{...}}`.
- `tool_choice`: `{type:"auto"|"any"|"tool", name?}`.

Translate into IB's existing `ChatCompletionRequest`/`ApiMessage` shape:
- Map Anthropic `system` → a leading `ApiMessage{role:"system"}`.
- Map `content` blocks → IB `ApiContentPart`s (text + image already exist — see `ApiContentPart::ImageUrl`/`InputImage` @ completions.rs ~L551). Anthropic base64 image → reuse `normalize_image_payload`.
- Map Anthropic `tool_use` (assistant) / `tool_result` (user) blocks → IB's tool message convention (inspect how `build_chat_request` + `prepend_tool_schema_message` represent tool calls/results, ~L1730, and mirror it).
- Map `tools[].input_schema` → the OpenAI-style tool JSON IB already ingests (`{type:"function", function:{name,description,parameters}}`).
- `max_tokens` → `max_tokens`; `stop_sequences` → `stop`; `top_k`/`top_p`/`temperature` straight across.

Then call `build_chat_request(...)` and drive `client.complete()` / the streaming engine exactly like `chat_completions` does.

#### Response translation (internal → Anthropic), non-streaming
Return:
```jsonc
{
  "id": "msg_...", "type": "message", "role": "assistant",
  "model": "<resolved>",
  "content": [ {"type":"text","text":"..."} /* + {"type":"tool_use",id,name,input} per tool call */ ],
  "stop_reason": "end_turn" | "max_tokens" | "tool_use" | "stop_sequence",
  "stop_sequence": null,
  "usage": { "input_tokens": N, "output_tokens": M }
}
```
- Reuse IB's existing tool-call extraction (`normalize/tool_extract.rs`) + JSON repair — this is where IB's self-healing advantage carries into the Anthropic surface. Each extracted tool call becomes a `tool_use` block; set `stop_reason:"tool_use"` when any are present.
- Map token counts from `CompletionResponse.timings`/`tokens_*` fields (see struct @ client.rs ~L55).
- `stop_reason`: `stopped_limit==true` → `"max_tokens"`; stop marker hit → `"stop_sequence"`; tool calls present → `"tool_use"`; else `"end_turn"`.

#### Response translation — streaming SSE
Anthropic's stream is an **event-typed SSE** (different from OpenAI deltas). Emit this exact event sequence:
```
event: message_start          data: {"type":"message_start","message":{...,"usage":{"input_tokens":N,"output_tokens":0}}}
event: content_block_start     data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
event: content_block_delta     data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"..."}}   (repeat per token)
event: content_block_stop      data: {"type":"content_block_stop","index":0}
event: message_delta           data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":M}}
event: message_stop            data: {"type":"message_stop"}
```
- For tool calls, open a second content block with `content_block_start` `{"type":"tool_use",...}` and stream `input_json_delta` deltas (or, simpler for v1: buffer tool JSON and emit it as one block at close — acceptable, note it as a limitation).
- Send SSE `ping` events periodically if you want parity, optional.
- Reuse IB's SSE plumbing in `engine/streaming.rs` / `stream_chat_completion` for reading llama-server tokens; only the **outgoing** framing differs. Consider factoring the token source out of `stream_chat_completion` or writing a parallel `stream_anthropic_message` that consumes the same `SseChunk` iterator.

#### Endpoint + auth
- Register `.route("/messages", axum::routing::post(super::messages::messages))` in `api_routes()`.
- Anthropic clients send the API key in the **`x-api-key`** header (not `Authorization: Bearer`). Update the auth middleware to accept `x-api-key` as an alternative to Bearer for this route (check `is_api_auth_exempt` / the auth layer in server.rs ~L126). Also tolerate the `anthropic-version` header (ignore its value).
- Honor the upstream-provider short-circuit only if it can speak Anthropic; otherwise skip it for this route and go straight to the managed backend.

### Acceptance
- Non-streaming: `POST /v1/messages` with `{model,max_tokens,messages:[{role:"user",content:"hi"}]}` returns a valid Anthropic message object.
- Streaming: the six event types arrive in order and the official `anthropic` Python/TS SDK (`base_url=http://127.0.0.1:8800`) consumes it without error.
- Tools: a request with `tools[]` yields `tool_use` blocks and `stop_reason:"tool_use"`; feeding a `tool_result` back continues the turn.
- System prompt, `stop_sequences`, `temperature`, `top_p`, `top_k` all take effect.
- Vision: a base64 `image` block reaches a loaded vision model.
- `x-api-key` auth works; missing/invalid key is rejected like other endpoints.
- **Reality check for Codex:** the streaming translator is the risky part. Ship non-streaming first behind the same route, then add streaming. Write an integration test that captures the raw SSE bytes and asserts the event ordering.

---

## Feature 4 — Context compaction: replace System B with System A

**Why:** IB has two compaction systems and the good one isn't on the path agents use.
- **System A (good):** `summarize_messages_with_model` ([context/compressor.rs:23](../src-tauri/src/context/compressor.rs)) calls the model to produce a *semantic* summary ("preserve facts, decisions, unresolved questions, user preferences") and stores a session snapshot. Layered strategy in [context/strategy.rs](../src-tauri/src/context/strategy.rs). **But it is only wired to the GUI/session flow, not to `/v1/chat/completions`.**
- **System B (on the hot path, crude):** `compact_messages_to_fit` ([api/completions.rs:1803](../src-tauri/src/api/completions.rs)) evicts oldest messages, then calls `compress_messages` ([compressor.rs:5](../src-tauri/src/context/compressor.rs)) which is a **naive 200-char truncation per message**, then a second loop **silently deletes** more messages ([completions.rs:1862-1879](../src-tauri/src/api/completions.rs)).

**Decision:** System A is better for agent coherence — adopt it on the API path, with a safe deterministic fallback so a summarizer hiccup degrades gracefully instead of crashing. This is **independent of Features 1–3** and can ship as its own PR.

### Bugs to fix regardless of the A/B swap

- 🐞 **FIX-1 — UTF-8 panic.** [compressor.rs:13-18](../src-tauri/src/context/compressor.rs): `&trimmed[..200]` slices at byte index 200. The guard checks byte length, not a char boundary, so any non-ASCII byte at position 200 (emoji, accents, CJK, box-drawing in tool output) **panics and kills the request.** Replace with a char-safe take, e.g. `trimmed.chars().take(200).collect::<String>()`.
- ⚠️ **FIX-2 — tool-pair breakage.** Eviction is blind oldest-first, so it can drop an assistant `tool_call` while keeping its `tool_result` (or vice-versa), producing an orphaned pair that some chat templates render malformed. Evict `tool_call ↔ tool_result` as whole units.
- ⚠️ **FIX-3 — silent drops.** The second loop ([completions.rs:1862-1879](../src-tauri/src/api/completions.rs)) removes messages without recording them. Every removal must be counted in `CompactionInfo`; nothing disappears unaccounted.
- **FIX-4 (perf, optional).** The loop re-renders the *entire* prompt to re-estimate tokens every iteration → O(n²) on history length. Cache per-message token estimates or estimate incrementally.

### Target design for the hot path

Rewrite `compact_messages_to_fit` to a summarize-then-evict flow:

1. Keep the existing budget calc (`context_limit - output_reserve`, floored at `context_limit/2`).
2. **Never evict:** all leading system/pinned messages **and** the current (last) user turn plus the minimal recent context needed for coherence.
3. Select the **oldest evictable span** as whole tool-paired units (FIX-2).
4. **Summarize that span with the model** — reuse the prompt + params from `summarize_messages_with_model` (system: "Summarize… preserve facts, decisions, unresolved questions, user preferences"; `n_predict: 384`, `temp: 0.2`). Insert the result as a single `[Earlier conversation summary]` system message.
5. If still over budget, evict additional oldest whole-units and fold them into the **same** summary (incremental re-summarize) — record all removals in `CompactionInfo` (FIX-3). No silent drops.
6. **Fallback:** if the summarization call fails or times out, fall back to the *fixed* char-safe, tool-pair-aware `compress_messages` (FIX-1/FIX-2). A summarizer failure degrades to a safe deterministic trim, never a crash and never a silent delete.

### Threading changes (Codex: the compiler will guide you)

- `compact_messages_to_fit` must become **`async`** and needs a `LlamaClient` (or the internal `port`) + `profile` to call the model. `build_chat_request` ([completions.rs:1719](../src-tauri/src/api/completions.rs)) already has `profile` and is already `async`; thread the `llama_port` (or a `&LlamaClient`) into it and down into the compaction call at ~L1744.
- Update `build_chat_request`'s signature and its call sites — chat path ~L2066, plus the text/stream builders (grep for `build_chat_request(` to find all).
- **Ordering note:** in `chat_completions`, `build_chat_request` runs *before* `scheduler.acquire()`, so the summarization model call happens outside the generation permit. That means two sequential model calls when compaction fires — acceptable, since compaction only triggers at ≥ budget. Ensure the summarizer's own request can't recurse into compaction (it uses a fixed 2-message prompt, so it can't).
- Keep System A's existing guard behavior (`active_generation.is_some()` → skip) in mind; on the API path `active_generation` is set *after* build, so it won't false-skip this request.

### Acceptance

- Unit test: `compress_messages` on a string whose byte-200 lands mid-codepoint (emoji/CJK) **does not panic**.
- Eviction never leaves an orphaned `tool_call`/`tool_result`.
- `CompactionInfo.removed_messages` equals the true number removed (no silent drops).
- Integration: a long agent history compacts to a **model-generated** summary system message; forcing the summarizer to fail falls back to the safe deterministic trim (assert no panic, no silent loss).
- Latency: document that compaction now adds one model round-trip; it only fires at ≥ budget so steady-state turns are unaffected.
- Requests that fit under budget are unchanged (early return preserved).

---

## Config & docs to touch (all features)

- `inference-bridge.example.toml`: add a commented `[embeddings]` stub (Phase B placeholder) and note that `response_format` + `/v1/messages` need no config.
- `README.md`: add the three endpoints to the API surface list; add an Anthropic `base_url` example for Claude Code.
- `docs/07-lmstudio-llamacpp-tracker.md`: tick these off against the LM Studio comparison.
- Add a short "Anthropic compatibility" section to `API-ARCHITECTURE.md`.

## Suggested PR breakdown
1. PR-1: structured output (Feature 1) — smallest, land first.
2. PR-2: `/v1/embeddings` Phase A (Feature 2).
3. PR-3: `/v1/messages` non-streaming (Feature 3a).
4. PR-4: `/v1/messages` streaming (Feature 3b).
5. PR-5: context-compaction bug fixes (Feature 4 FIX-1..3) — small, land early; FIX-1 is a real crash.
6. PR-6: hot-path summarize-then-evict (Feature 4 target design) — depends on PR-5.

Keep each PR green (`cargo build && cargo clippy && cargo test` in `src-tauri/`) with a unit/integration test for the new surface.
