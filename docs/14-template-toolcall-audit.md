# Template & Tool-Call Audit

## Files reviewed

- `src-tauri/src/models/profiles.rs` — profile detection and family settings
- `src-tauri/src/templates/engine.rs` — prompt rendering per family
- `src-tauri/src/templates/patches.rs` — post-render patches
- `src-tauri/src/api/completions.rs` — tool schema injection, history rendering, stop logic
- `src-tauri/src/normalize/tool_extract.rs` — parser / extraction state machine
- `src-tauri/src/normalize/think_strip.rs` — think/channel tag handling
- `src-tauri/src/engine/process.rs` — llama-server argument construction

---

## What is working well

- **Multi-tier Qwen parser with safety-nets**: `QwenStateMachine` tries native `<function=...>` XML first, then Hermes JSON, then `[tool_name]{...}`, `[tool_call]name({...})`, bare `<tool_call>{json}`, and fenced JSON. This is robust for the wide variety of GGUF quantizations on HuggingFace.
- **Architecture fallback**: when the filename is ambiguous (renamed GGUF), `detect_with_arch` recovers the correct profile from the GGUF `general.architecture` field.
- **Gemma 4 native tool call format** parsed and rendered correctly.
- **`truncate_at_generation_boundary`** correctly truncates leaked template continuation markers before they reach the UI.
- **`coerce_tool_arg_value`** prevents type mismatches (string "5" → int 5) when models serialize numeric args as strings.
- **Stop markers** on Qwen stop at `</function>` which prevents the model generating extra tokens past the tool call.

---

## Issues found

### Issue 1 — Duplicate `think_guidance_suffix` injection (Qwen3/Qwen3.5)

**Files:** `src-tauri/src/templates/engine.rs:31-37` and `src-tauri/src/templates/patches.rs:16-31`

Both `render_chatml` (called by `render_qwen_chat`) and `patch_qwen` attempt to insert `think_guidance_suffix()` into the system message. `patches.rs` guards with `!result.contains("Output Format (STRICT)")` to avoid doubling it, but this is fragile — if the user's own system prompt contains that exact string the patch silently no-ops.

**Fix:** Remove the injection from `render_chatml` (`engine.rs:28-37`). Let `patch_qwen` be the single place where the suffix is applied. The `render_chatml` function is generic and used by non-Qwen models (Phi, Qwen2.5, Generic); it should not contain Qwen-specific logic.

In `engine.rs`, delete:
```rust
if let Some(suffix) = profile.think_guidance_suffix() {
    if let Some(pos) = prompt.rfind("<|im_start|>system\n") {
        if let Some(end) = prompt[pos..].find("<|im_end|>") {
            let insert_pos = pos + end;
            prompt.insert_str(insert_pos, suffix);
        }
    }
}
```

Keep the equivalent logic only in `patch_qwen` in `patches.rs`, which already handles the idempotency guard correctly.

---

### Issue 2 — Conflicting think guidance when tools are present (Qwen3/Qwen3.5)

**Files:** `src-tauri/src/models/profiles.rs:506-515` and `src-tauri/src/api/completions.rs:1326-1329`

`think_guidance_suffix()` (applied to every request) says:
> "You MUST produce a tool call OR a text response on EVERY turn … NEVER stop after `</think>`"

`prepend_tool_schema_message` with `disable_thinking_for_tools=true` (applied when tools are present) says:
> "When deciding whether to call a tool, do not emit `<think>` blocks."

These are contradictory. The first message trains the model to always use think blocks; the second message (injected only when tools are present and earlier in the system prompt list) says not to. Qwen3 models in tool mode may output `<think>...</think>` before the tool call and then get confused about whether to stop.

**Fix:** When `disable_thinking_for_tools` is true and the request contains tools, skip appending `think_guidance_suffix()` in `patch_qwen`. Add a `has_tools: bool` parameter to `apply_patches` and thread it through from `prepend_tool_schema_message` call site, or check the rendered text for the tool schema message presence:

```rust
// patches.rs
pub fn apply_patches(prompt: &str, profile: &ModelProfile, has_tools: bool) -> String {
    match profile.family {
        ModelFamily::Qwen3_5 | ModelFamily::Qwen3 => patch_qwen(prompt, profile, has_tools),
        _ => prompt.to_string(),
    }
}

fn patch_qwen(prompt: &str, profile: &ModelProfile, has_tools: bool) -> String {
    let mut result = prompt.to_string();
    let skip_think_guidance = has_tools && profile.disable_thinking_for_tools;
    if !skip_think_guidance {
        if let Some(suffix) = profile.think_guidance_suffix() {
            if !result.contains("Output Format (STRICT)") {
                if let Some(sys_end) = result.find("<|im_end|>") {
                    result.insert_str(sys_end, suffix);
                }
            }
        }
    }
    result
}
```

Update `render_prompt` in `engine.rs` to accept `has_tools: bool` and pass it to `apply_patches`.

---

### Issue 3 — `split_tool_calling` field is defined but never consumed

**Files:** `src-tauri/src/models/profiles.rs:71`, `src-tauri/src/models/overrides.rs:106-107`

`split_tool_calling` is set to `true` for Qwen3, Qwen3.5, and Phi profiles, and can be overridden via model overrides. But there is no production code path that reads `profile.split_tool_calling` outside of the overrides setter. It has no effect.

**Action required:** Either:
- (a) Implement the split-tool-calling path: when `split_tool_calling=true` and a response contains both text and a tool call, split the response so the tool call is returned separately from the preceding text, preventing the model's preamble text from being shown to the user when a tool is being dispatched.
- (b) Remove the field if it was abandoned — keeping it creates false confidence that the behavior is active.

If implementing (a), the split point in `tool_extract.rs` already has the `tool_remaining_text` return value from `extract_tool_calls_for_profile`, which captures post-tool text. The "pre-tool text" (prose before the first tool call) is what needs to be suppressed.

---

### Issue 4 — Qwen `tool` role messages rendered with wrong role label

**File:** `src-tauri/src/api/completions.rs:1134-1140`

When a `tool` role message arrives in history, it is rendered with `render_tool_response_history` to produce the correct `<tool_response>...</tool_response>` wrapper, but then pushed to `normalized` with `role: "tool"`. The `render_chatml` renderer emits:
```
<|im_start|>tool
<tool_response>result</tool_response>
<|im_end|>
```

Qwen3's official chat template expects tool results in the `tool` role, which is correct. **But** when Jinja is disabled and llama.cpp uses the built-in ChatML template (no tool-call awareness), the `tool` role becomes a literal `<|im_start|>tool` turn which most models' internal token mappings don't have. In this case the role should be `user` so it gets consumed as a normal user turn.

**Fix:** When the template mode is not Jinja (i.e., the custom renderer is in use), remap `tool` role to `user` before normalizing. Since `render_prompt` in `engine.rs` doesn't know the template mode, the remapping should happen at the normalization step in `completions.rs`:

```rust
// In the normalize loop, after building content_parts:
let effective_role = if message.role == "tool" && !jinja_enabled {
    "user".to_string()
} else {
    message.role.clone()
};
normalized.push(ChatMessage { role: effective_role, content: content_parts.join("\n") });
```

The `jinja_enabled` flag is already available in the completions handler context.

---

### Issue 5 — Qwen2.5 tool calling is NativeApi but tools still injected as system message

**File:** `src-tauri/src/api/completions.rs:1316-1356`

`prepend_tool_schema_message` always inserts a system message with the tool schema, regardless of parser type. For Qwen2.5 (`ParserType::NativeApi`, `ToolCallFormat::NativeApi`), this means tools are described in a system message AND llama-server's native `/v1/chat/completions` endpoint may also receive the `tools` field separately. If both happen, Qwen2.5 gets the schema twice: once in the system prompt and once from the API native path, which can confuse the model.

**Fix:** Check whether the request path uses the native llama-server `/v1/chat/completions` API with tools forwarded, and if so skip `prepend_tool_schema_message` for `NativeApi` parser profiles. Alternatively, only prepend the schema message for `QwenStateMachine`, `HermesFallback`, and `Gemma4StateMachine` parsers.

```rust
fn prepend_tool_schema_message(...) {
    // Don't inject for NativeApi — llama-server handles tool schemas itself
    // when tools are forwarded via the API tools field.
    if profile.parser_type == ParserType::NativeApi && !profile.allow_fallback_extraction {
        return;
    }
    // ... rest of existing logic
}
```

---

### Issue 6 — Qwen3 think_tag_style is `Standard` but should handle `<|think|>` too

**File:** `src-tauri/src/models/profiles.rs:233`

`Qwen3` uses `ThinkTagStyle::Standard` which only strips `<think>...</think>` and `<|think|>...</|think|>` tags (both are handled in `strip_think_tags`). `Qwen3_5` uses `ThinkTagStyle::Qwen` which additionally handles the channel reasoning format (`<|channel>thought...`).

Looking at Qwen3's actual output in recent llama.cpp builds: Qwen3 models do emit `<think>` tags (Standard covers this). Qwen3.5 adds channel-style markers in some configurations. This appears correct **as long as** Qwen3 GGUFs never emit channel markers. Verify this against current Unsloth/Bartowski Qwen3 GGUFs. If Qwen3 also emits channel markers in some configs, upgrade its `ThinkTagStyle` to `Qwen`.

**Action:** Test with a Qwen3 GGUF under `--jinja` mode and verify no `<|channel>` output leaks through. Low risk but worth confirming.

---

### Issue 7 — `render_qwen_tool_call` (history) and format guidance (inference) use different outer wrappers

**File:** `src-tauri/src/api/completions.rs:1220-1239`

`render_qwen_tool_call` emits:
```
<tool_call>
<function=NAME>
<parameter=PARAM>VALUE</parameter>
</function>
</tool_call>
```

The format guidance shown to the model in `prepend_tool_schema_message` (line 1333) shows:
```
<tool_call>
<function=TOOL_NAME>
<parameter=PARAM_NAME>VALUE</parameter>
</function>
</tool_call>
```

These match — good. The stop marker `</function>` stops generation after the function block. The outer `</tool_call>` is not emitted by the model but is present in history replay. The parser regex at `tool_extract.rs:17` handles both `</function>` and end-of-string. This is consistent and correct.

---

### Issue 8 — Missing `Qwen2.5` vision profile detection for `qwen2.5-vl` GGUFs

**File:** `src-tauri/src/models/profiles.rs:108-110`

The `detect_by_architecture` path maps `qwen2` architecture to `Self::qwen2_5()`. The `qwen2_5()` profile has `supports_vision: false`. But Qwen2.5-VL GGUFs have the same architecture string (`qwen2` in llama.cpp's GGUF header). Vision support is rescued via `infer_vision_support` checking for `-vl` in the filename, and `detect_by_name` runs first, so `qwen2.5-vl` in the filename correctly sets vision via `supports_vision = profile.supports_vision || supports_vision`. This is correct.

**No change needed** — just documenting that the path is safe.

---

## Summary table

| # | Severity | File | Issue | Action |
|---|---|---|---|---|
| 1 | Medium | `engine.rs`, `patches.rs` | Duplicate `think_guidance_suffix` injection | Remove from `render_chatml`, keep only in `patch_qwen` |
| 2 | High | `profiles.rs`, `completions.rs` | Conflicting think guidance when tools present | Skip `think_guidance_suffix` when `has_tools && disable_thinking_for_tools` |
| 3 | Medium | `profiles.rs` | `split_tool_calling` defined but unused | Implement or remove |
| 4 | Medium | `completions.rs` | `tool` role message sent with `role:"tool"` in non-Jinja mode | Remap to `"user"` when Jinja is disabled |
| 5 | Low | `completions.rs` | Qwen2.5 NativeApi gets tool schema twice | Guard `prepend_tool_schema_message` for NativeApi profiles |
| 6 | Low | `profiles.rs` | Qwen3 `ThinkTagStyle::Standard` may miss channel markers | Verify with real GGUF; upgrade to `Qwen` style if needed |
| 7 | None | `completions.rs` | Qwen history render vs guidance mismatch (check) | No change needed |
| 8 | None | `profiles.rs` | Qwen2.5-VL vision detection via `detect_by_name` | No change needed |

## Priority order for Codex

1. **Issue 2** — the conflicting `disable_thinking_for_tools` + `think_guidance_suffix` is the most likely cause of Qwen3 tool-call failures where the model thinks and then produces no tool call or empty output
2. **Issue 1** — remove duplicate suffix injection; clean precondition for Issue 2 fix
3. **Issue 4** — `tool` role in non-Jinja mode causes llama-server to see unrecognised `<|im_start|>tool` which may corrupt conversation history
4. **Issue 3** — implement `split_tool_calling` or remove it to avoid dead code confusion
5. **Issue 5** — Qwen2.5 double-schema is low priority since Qwen2.5 tool calling is relatively uncommon; most users run Qwen3+
6. **Issue 6** — verify Qwen3 channel marker output; low risk
