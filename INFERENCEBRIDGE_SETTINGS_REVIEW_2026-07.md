# InferenceBridge — Settings Review vs Unsloth Qwen3.6 / tool-calling guidance

**Date:** 2026-07-06
**Reviewed against:** Unsloth docs (Qwen3.6 inference settings, tool-calling guide for local LLMs,
chat-templates, troubleshooting) and the two agent clients on this backend (TheMauler, HelixClaw).
**Live config reviewed:** `inference-bridge.toml`.

## Current live config (relevant keys)

```toml
default_ctx_size = 40960
gpu_layers       = -1        # all layers offloaded
flash_attn       = true
main_gpu         = 0
use_jinja        = true
reasoning_mode   = ""        # <-- empty
cache_type_k     = "q8_0"
cache_type_v     = "q8_0"
ctx_shift        = false
spec_type        = ""
```

## What is already correct ✅

- **`use_jinja = true`** — the single most important flag for tool calling. Unsloth: without `--jinja`
  Qwen emits `<tool_call>` XML and `</think>` as plain text instead of an OpenAI `tool_calls` array.
  Correct.
- **`flash_attn = true`** — Unsloth recommends this on CUDA/RTX (~15% decode speedup, lower KV memory).
  Correct.
- **`gpu_layers = -1`** — full offload; correct for a 24 GB card with a Q4 model that fits.
- **Per-request `chat_template_kwargs` / `use_jinja` overrides are honored** (`api/completions.rs`),
  so clients can toggle `enable_thinking` per turn. Good — TheMauler relies on this.

## Issues / recommended updates

### 1. `reasoning_mode = ""` → set to `"deepseek"` (HIGH)
Qwen3.6 (and the Qwen3.5/3 family) use the DeepSeek-R1 `<think>…</think>` delimiters. Without
`--reasoning-format deepseek`, thinking output is not reliably split into `reasoning_content`, so on
thinking turns the raw `<think>` block can leak into `content` — which then corrupts tool-call parsing
and pollutes the context the agent stores. This is exactly what TheMauler's Doctor flags as a
root-cause item.
**Action:** default `reasoning_mode = "deepseek"` for Qwen3-class models (gate by model family if you
also serve non-Qwen models that want a different reasoning format).

### 2. `cache_type_k/v = "q8_0"` → watch for Qwen3.6 gibberish (MEDIUM)
Quantized KV cache saves VRAM, but Unsloth's Qwen3.6 troubleshooting explicitly says: if you get
gibberish at lower context, switch to `--cache-type-k f16 --cache-type-v f16` (or bf16). q8_0 KV is
usually fine, but Qwen3.6 is more sensitive than most. On a 24 GB card the model (Q4, ~17–20 GB) plus
40 K context at f16 KV is generally affordable.
**Action:** if any Qwen3.6 gibberish/instability is observed, flip KV to `f16` first (before blaming
the model/quant). Consider making f16 the default for Qwen3.6 profiles and reserving q8_0 KV for when
context must be pushed past ~48 K.

### 3. No explicit `repeat_penalty` / sampler floor surfaced (LOW)
Unsloth + Qwen official: Qwen3.6 wants `repetition_penalty = 1.0` (disabled) and `min_p ≈ 0.0`
(0.01 is fine for tool calling). If IB does not pass `--repeat-penalty`, llama-server's default
applies, which on some builds is > 1.0 and can subtly distort tool JSON / code.
**Action:** pass `repeat_penalty = 1.0` explicitly for Qwen3.6 so the value is not left to the
server default, and expose it in the config for other families.

### 4. Speculative decoding off — fine, but document the interaction (INFO)
`spec_type = ""` (off). Note for when it is enabled: draft-model / MTP speculative decoding on Qwen3
thinking models can spike EOS probability at `</think>`, causing early termination — the failure both
agent clients defend against. Keep spec decoding off for **thinking** profiles, or only enable MTP on
the native-MTP-preserved GGUFs and only in non-thinking mode. Unsloth's MTP flags for reference:
`--spec-type draft-mtp --spec-draft-n-max 2`.

## Quant note (for whoever picks GGUFs)
Both agent clients currently point at `Q4_K_S` / `Q4_K_M` heretic/uncensored Qwen3.6 finetunes.
Unsloth's recommended 24 GB quant is **`UD-Q4_K_XL`** (the Unsloth Dynamic quant, ~17 GB + ~4 GB KV,
higher fidelity than plain `Q4_K_S` at similar size). If tool-call reliability is marginal on
`Q4_K_S`, moving to `UD-Q4_K_XL` is the cheapest quality lever. Not an IB config change — a model-file
choice — but IB's model picker is where it lands.

## Summary
| Key | Now | Recommend | Severity |
|---|---|---|---|
| `reasoning_mode` | `""` | `"deepseek"` (Qwen3) | High |
| `cache_type_k/v` | `q8_0` | `f16` for Qwen3.6 (watch gibberish) | Medium |
| `repeat_penalty` | (server default) | `1.0` explicit | Low |
| `use_jinja` | `true` | keep | ✅ |
| `flash_attn` | `true` | keep | ✅ |
| spec decoding | off | keep off for thinking profiles | ✅/info |
