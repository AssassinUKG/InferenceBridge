# llama.cpp b9842 Runtime Polish Plan

Status: in progress
Date: 2026-06-29

## Context

Current managed llama-server observed in `%LOCALAPPDATA%/InferenceBridge/bin`:

- Installed: `b9804`
- Latest checked: `b9842` published 2026-06-29

The current downloader still matches the latest Windows CUDA asset layout:

- `llama-b9842-bin-win-cuda-12.4-x64.zip`
- `llama-b9842-bin-win-cuda-13.3-x64.zip`
- `cudart-llama-bin-win-cuda-12.4-x64.zip`
- `cudart-llama-bin-win-cuda-13.3-x64.zip`

## Goals

- [ ] Keep the managed llama.cpp update flow compatible with current release assets.
- [ ] Make speculative decoding clearer in Settings.
- [ ] Expose self-MTP and DFlash as first-class spec presets.
- [ ] Add an explicit `--reasoning-preserve` setting, off by default.
- [ ] Keep quality-first defaults for Qwen on RTX 3090: CUDA, flash attention on, continuous batching on, parallel slots 1, q8_0 KV.
- [ ] Verify launch preview args and frontend type/build checks.

## Findings

- `b9842` includes `/v1/models` duplicate-entry cleanup, useful for model pickers and OpenAI-compatible clients.
- Recent CUDA/scheduler fixes after `b9804` are worth taking as quiet stability/performance updates.
- DFlash speculative decoding landed upstream and uses `--spec-type draft-dflash` with a compatible draft model.
- `--reasoning-preserve` landed upstream; useful for debugging/thinking preservation, but it should stay disabled by default because it can bloat prompts/context.
- InferenceBridge already emits `--spec-type` without `-md`, so self-MTP support is already correct.

## Implementation Steps

1. [x] Inspect existing InferenceBridge llama.cpp settings, launch preview, and downloader paths.
2. [x] Add `reasoning_preserve` to config, settings bindings, launch config, launch preview, and llama-server args.
3. [x] Add Settings UI presets:
   - Disabled: empty `spec_type`, empty draft tokens.
   - Self MTP: `spec_type=draft-mtp`, blank draft model path, `spec_draft_n_max=2`.
   - DFlash: `spec_type=draft-dflash`, keep draft model path user-controlled, `spec_draft_n_max=8`.
   - Custom: raw fields remain editable.
4. [x] Update TypeScript settings/preview types.
5. [x] Add focused launch-arg tests for `--reasoning-preserve` and DFlash.
6. [x] Run verification:
   - `cargo check` passes.
   - `npm run build` passes.
   - `cargo test process` compiled, then the Windows test binary failed to start with `STATUS_ENTRYPOINT_NOT_FOUND`; this appears to be a local DLL/runtime loading issue rather than a compile failure.

## Runtime Recommendation

For Qwen3.6 27B on RTX 3090:

- Keep CUDA 12.4 unless the newer 13.3 stack is deliberately tested.
- Keep `flash_attn=true`.
- Keep `cont_batching=true`.
- Keep `parallel_slots=1` for agent stability.
- Keep `cache_type_k=q8_0` and `cache_type_v=q8_0` for quality.
- Use self-MTP first: `spec_type=draft-mtp`, `spec_draft_n_max=2`, blank draft model path.
- Test DFlash only with a matching Qwen DFlash draft GGUF: `spec_type=draft-dflash`, `spec_draft_n_max=8-15`, draft model path set.

## Notes

Do not enable `--reasoning-preserve` by default. It is helpful when inspecting model reasoning behaviour, but not a general speed/quality improvement.
