# InferenceBridge Tess-4-27B / Qwen3.6 Settings Decision

**Updated:** 2026-07-16

**Status:** Implemented and verified

The previous review in this file described an older Qwen setup and contained
recommendations that no longer match current llama.cpp. In particular,
`reasoning_mode = "deepseek"`, a 40K default context, and launch-time
`enable_thinking` template kwargs are not the approved Tess-4-27B settings.

The canonical settings, tool format, verification procedure, and regression
matrix now live in
[docs/20-tess-4-27b-runtime-guide.md](docs/20-tess-4-27b-runtime-guide.md).

## Approved decisions

| Area | Decision |
| --- | --- |
| Model identity | Treat Tess-4-27B and GGUF architecture `qwen35` as InferenceBridge family `Qwen3_5` |
| Default context | `32768` on a single 24 GB RTX 3090 |
| Default workload | Tools/direct answers |
| Thinking control | Explicit llama.cpp `--reasoning on` or `--reasoning off` |
| Tool format | Tess embedded Jinja plus native Qwen XML `<tool_call>` format |
| Tool scheduling | One parallel slot by default |
| Vision | Requires a matching Tess `mmproj` attached at launch |
| MTP | Disabled unless a real, compatible draft GGUF is explicitly selected and verified; names containing `MTP` do not qualify |
| Empty think tags | Normal disabled-thinking template boundary; strip it without dropping the answer |
| Runtime verification | Check `/v1/debug/profile` and `/v1/runtime/status` launch preview before diagnosing model quality |

## Approved sampler presets

| Preset | Thinking | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repetition penalty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| General reasoning | On | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Precise coding | On | `0.6` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Tools/direct answers | Off | `0.7` | `0.80` | `20` | `0.0` | `1.5` | `1.0` |

The Tools/direct preset is the InferenceBridge default for Tess. Profile
defaults must remain overridable by explicit per-request sampler values.
