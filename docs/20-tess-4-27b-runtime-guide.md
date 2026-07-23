# Tess-4-27B Runtime Guide

This is the InferenceBridge source of truth for running `migtissera/Tess-4-27B`
with managed llama.cpp. Tess-4-27B is built on Qwen3.6-27B and its GGUF reports
the `qwen35` architecture. InferenceBridge must therefore resolve Tess files to
its `Qwen3_5` profile even when `qwen` is absent from the filename.

Primary references:

- [Tess-4-27B model card](https://huggingface.co/migtissera/Tess-4-27B)
- [Tess-4-27B official GGUF repository](https://huggingface.co/migtissera/Tess-4-27B-GGUF)
- [Tess-4-27B embedded chat template](https://huggingface.co/migtissera/Tess-4-27B/blob/main/chat_template.jinja)
- [Qwen3.6-27B generation guidance](https://huggingface.co/Qwen/Qwen3.6-27B)
- [llama.cpp function-calling guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)

## Recommended load profile

The following is the stable default for a Q4_K_M main model on one 24 GB RTX
3090. The advertised 262,144-token window is a model maximum, not a sensible
default allocation for this card.

| Setting | Recommended value | Reason |
| --- | --- | --- |
| Context length | `32768` | Practical quality and KV-memory balance on a 24 GB GPU |
| GPU offload | `-1` | Offload every compatible layer when the Q4 main model fits |
| Parallel slots | `1` | Most reliable ordering and tool-call parsing |
| Flash attention | On | Reduces KV pressure and improves supported CUDA runtimes |
| Unified KV cache | On | Appropriate for the single-slot managed runtime |
| KV cache K/V | `q8_0` | Good memory/quality balance; use `f16` only to diagnose output corruption |
| CPU thread pool | `0` (auto) | Let llama.cpp select the host-appropriate value |
| Evaluation batch | `0` (llama.cpp default) | Avoid a machine-specific forced value |
| Physical batch | `0` (llama.cpp default) | Avoid a machine-specific forced value |
| Template source | Repository/embedded | Keeps Tess's own Jinja template authoritative |
| Jinja | On | Required for native template and structured tools |
| Reasoning | `off` for the default tool preset | Avoids reasoning prose interfering with tool dispatch |
| Preserve reasoning | Off | Avoids retaining unnecessary reasoning history |
| Speculative decoding | Off | Safe default; a main GGUF is not an MTP draft |

InferenceBridge exposes the three workload presets below. Per-request sampler
values override the active preset, but they do not switch the running model's
reasoning mode. Selecting a full preset in the model-load dialog reloads the
runtime when its Thinking value changes; the chat generation presets are
sampler-only and leave the loaded `--reasoning` mode unchanged.

| Preset | Thinking | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repetition penalty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| General reasoning | On | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Precise coding | On | `0.6` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Tools/direct answers | Off | `0.7` | `0.80` | `20` | `0.0` | `1.5` | `1.0` |

`Tools/direct answers` is the Tess default in InferenceBridge. Use General
reasoning for open-ended analysis and Precise coding for implementation or
debugging where lower variance is useful.

The equivalent public API load is:

```bash
curl -X POST http://127.0.0.1:8800/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Tess-4-27B-Q4_K_M.gguf",
    "context_size": 32768,
    "gpu_layers": -1,
    "parallel_slots": 1,
    "flash_attn": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "kv_unified": true,
    "cont_batching": true,
    "use_jinja": true,
    "reasoning_mode": "off",
    "reasoning_preserve": false,
    "template_mode": "builtin",
    "chat_template_kwargs_json": "{}",
    "attach_mmproj": true,
    "extra_args": [
      "--temp", "0.7", "--top-p", "0.8", "--top-k", "20",
      "--min-p", "0", "--presence-penalty", "1.5",
      "--repeat-penalty", "1"
    ],
    "echo_load_config": true,
    "force_reload": true
  }'
```

`attach_mmproj` resolves the matching projector when present. It does not make
an unrelated projector valid; inspect `last_launch_preview.mmproj_path` before
sending images.

## Thinking on and off

Thinking mode is an explicit llama-server launch choice:

- Thinking on emits `--reasoning on`.
- Thinking off emits `--reasoning off`.

Do not use `enable_thinking` inside `--chat-template-kwargs` as the canonical
mode switch. Current llama.cpp deprecates that launch-time route. The launch
preview must contain exactly one explicit reasoning mode, and changing it
requires a managed-runtime reload because it changes the llama-server process
configuration.

`--reasoning-preserve` remains off by default. It is a separate history
preservation option and does not mean "enable thinking."

Tess's template deliberately emits an empty block when thinking is disabled:

```text
<think>

</think>
```

This is a template boundary, not a failed reasoning attempt. InferenceBridge
must strip an empty block from user-facing content while retaining the final
answer or structured tool call that follows it.

## Tool calling

Pass tools through the normal OpenAI-compatible `tools` field and preserve
`role: "tool"` result turns. Do not replace Tess's embedded template with
generic Hermes JSON instructions.

For Tess/Qwen3.5-family chat semantics, InferenceBridge forwards structured
messages, tools, tool-result turns, and image parts to llama-server's native
`/v1/chat/completions` endpoint. This is required so the selected embedded
Jinja template and launch-time `--reasoning` mode are actually applied. The raw
`/v1/completions` endpoint remains a raw-prompt API by definition.

Structured tool definitions and tool results are supported by the public Chat
Completions, Responses, and Messages compatibility APIs. Desktop Chat uses the
same native Jinja, reasoning, vision, and structured-history transport, but it
does not currently expose a desktop tool registry or execute tool calls itself.

Tess's native assistant tool-call syntax is:

```text
<tool_call>
<function=TOOL_NAME>
<parameter=PARAMETER_NAME>VALUE</parameter>
</function>
</tool_call>
```

Multiple parameters are represented by additional `<parameter=...>` elements
inside the same function. Tool results use the model template's tool-result
turn and `<tool_response>...</tool_response>` wrapper. InferenceBridge must
parse the XML into the OpenAI-compatible `tool_calls` array and reconstruct the
same native format when replaying assistant/tool history.

For the most reliable agent loop:

1. Load the Tools/direct preset (`--reasoning off`).
2. Keep one parallel slot and do not request parallel tool calls unless the
   complete client loop has been verified with them.
3. Send a structured tool definition rather than placing an informal schema in
   the system prompt.
4. Return the result as a `role: "tool"` message associated with the call ID.
5. Allow Tess to produce the final natural-language answer after the result.

## Vision

The main Tess GGUF does not contain the vision projector. Image input requires
the matching `mmproj-Tess-4-27B-F16.gguf` from the official Tess GGUF repository
to be attached at launch.

The runtime is vision-ready only when the launch preview contains a real
`mmproj_path`. Filename capability detection alone is insufficient. A text-only
load remains valid, but image paste/API requests must fail clearly rather than
silently sending an image to a runtime without its projector.

The readiness check applies to the complete conversation history, not only the
newest turn. A later text-only follow-up still requires the projector when an
earlier retained message contains an image.

## MTP and speculative decoding

Leave speculative decoding disabled by default. A folder or model name that
contains `MTP` is not proof that the main GGUF contains MTP prediction layers.
Enabling self-MTP on an ordinary main model causes llama.cpp errors such as:

```text
context type MTP requested but model doesn't contain MTP layers
failed to create MTP context
```

Enable `draft-mtp` only when a real llama.cpp-compatible draft GGUF has been
selected and the launch preview shows its path as the draft model. The Tess
model card's separate EAGLE3 acceleration instructions target SGLang/vLLM and
are not evidence that the main Tess GGUF contains self-MTP tensors. Do not
auto-enable MTP from the repository name, model name, or an `MTP` directory
component.

## Verify the effective profile

Before diagnosing model quality, verify that InferenceBridge is not using the
Generic profile:

```bash
curl "http://127.0.0.1:8800/v1/debug/profile?model=Tess-4-27B-Q4_K_M.gguf"
```

The effective profile should report:

- `family`: `Qwen3_5`
- `tool_call_format`: `QwenXml`
- `parser_type`: `QwenStateMachine`
- `renderer_type`: `QwenChat`
- `think_tag_style`: `Qwen`
- reasoning, tools, and vision capability enabled by the resolved model data

Then inspect the exact managed-runtime launch:

```bash
curl "http://127.0.0.1:8800/v1/runtime/status"
```

In `last_launch_preview`, verify all of the following:

- `context_size` is `32768`.
- `use_jinja` is `true` and the template source is repository/embedded.
- `args` contains `--reasoning off` or `--reasoning on`, matching the selected
  preset.
- `reasoning_preserve` is `false` unless deliberately enabled.
- `parallel_slots` is `1` for the reliable tool preset.
- `spec_type` and `draft_model_path` are empty unless a matching MTP draft was
  deliberately selected.
- `mmproj_path` points to the matching Tess projector before sending images.

The in-app equivalent is **API > Doctor/Profile/Launch Preview**. The raw prompt
and parse trace are the next checks when the effective profile and launch are
correct but a tool call is not returned as structured data.

## Troubleshooting

| Symptom | Check and corrective action |
| --- | --- |
| Raw `<tool_call>` appears in assistant text | Confirm `QwenXml`, `QwenStateMachine`, embedded Jinja, and a current llama.cpp runtime |
| Model discusses a tool but does not call it | Use Tools/direct, reasoning off, one slot, and structured OpenAI tool definitions |
| Empty `<think></think>` is visible | Treat it as disabled-thinking template output and verify UI/API normalization strips it |
| Thinking is unexpectedly on/off | Inspect launch preview for explicit `--reasoning on` or `--reasoning off`; do not rely on template kwargs |
| Image is pasted but not understood | Confirm the matching Tess `mmproj` is attached and `mmproj_path` is non-empty |
| `failed to create MTP context` | Disable speculation; only re-enable it with a verified llama.cpp-compatible draft GGUF |
| Output becomes corrupt or nonsensical | Retest with `f16` K/V cache before changing the template or quant |
| Effective family is Generic or Qwen3 | Rescan metadata/update the build; Tess must resolve from `qwen35` to `Qwen3_5` |

## Regression expectations

Changes to model detection, templates, chat normalization, load options, or API
translation must retain automated coverage for:

1. Tess filenames and GGUF architecture `qwen35` resolving to `Qwen3_5`, never
   Generic or the older Qwen3 profile.
2. Tess profile defaults matching Tools/direct sampling and a 32K recommended
   load context.
3. Launch previews emitting Jinja plus exactly one explicit `--reasoning on` or
   `--reasoning off`, with no deprecated thinking template kwarg.
4. Exact Qwen XML tool calls parsing into structured calls, including multiple
   parameters and assistant/tool history round-trips.
5. Thinking-on content being separated correctly and empty thinking-off blocks
   being removed without dropping the final answer.
6. Native Jinja/reasoning/history behavior through desktop Chat, Chat
   Completions, Responses, and Messages; structured tool-call translation
   through the three public compatibility APIs.
7. Vision requests being rejected without a matching projector and succeeding
   when the Tess `mmproj` is attached.
8. MTP remaining disabled for an ordinary main GGUF even when a parent path
   contains `MTP`, plus valid launch construction when a matching draft is
   explicitly supplied.
9. Request sampler values overriding profile defaults without silently changing
   the loaded context or reasoning mode.
