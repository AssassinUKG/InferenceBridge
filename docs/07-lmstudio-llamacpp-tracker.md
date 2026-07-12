# InferenceBridge LM Studio-Style llama.cpp Tracker

This tracker covers the current multi-tranche push to make InferenceBridge behave like a polished LM Studio-style frontend while still exposing the llama.cpp controls advanced users need.

## Status

- [x] T1 Launch config foundation
  - [x] T1.1 Add shared effective launch config type
  - [x] T1.2 Add curated llama.cpp fields
  - [x] T1.3 Add reload-on-effective-config-change logic
  - [x] T1.4 Add `extra_args` fallback
- [x] T2 Template system
  - [x] T2.1 Add template source modes
  - [x] T2.2 Add repo template resolution
  - [x] T2.3 Add custom template override
  - [x] T2.4 Add `chat_template_kwargs`
  - [x] T2.5 Add launch/debug template preview
- [ ] T3 Hugging Face first-class source
  - [x] T3.1 Persist HF source metadata
  - [x] T3.2 Support HF repo/file as native load source
  - [ ] T3.3 Surface HF metadata in Models/Browse
- [x] T4 Vision reliability
  - [x] T4.1 Add shared runtime vision readiness evaluation
  - [x] T4.2 Unify image normalization across desktop/API
  - [x] T4.3 Add runtime vision badges and reasons
  - [x] T4.4 Block non-ready image requests clearly
- [ ] T5 Advanced UI
  - [x] T5.1 Settings advanced runtime panel
  - [x] T5.2 Per-model overrides panel
  - [ ] T5.3 API/Debug launch and template inspection
  - [ ] T5.4 Live runtime state cleanup in Models/Status
- [ ] T6 API compatibility
  - [x] T6.1 Extend load/config request aliases
  - [x] T6.2 Add richer runtime/model response fields
  - [x] T6.3 Add structured output, embeddings proxy, and Anthropic Messages endpoint
  - [ ] T6.4 Add explicit runtime-ready transitions

## Last Update

- Added end-to-end runtime load options plumbing across Tauri load/swap, frontend wrappers, and API autoload/reload.
- Added per-model advanced load controls in the Models panel for template mode, reasoning mode, fit mode, Jinja, template kwargs, and extra args.
- Tightened OpenAI-compatible autoload so context and launch-affecting runtime hints now flow through the same effective-config reuse/reload path as native loads.
- Shared Claw-side stability work also landed:
  - staged LM Studio prepare/load lifecycle reporting is now in the Claw API/runtime path
  - `.claw/settings.local.json` is again a real local override layer for machine-specific local-runtime behavior
  - the known unrelated Windows test noise in Claw plugin/MCP, TUI, and slash-command coverage was removed so this shared track can move forward on a clean baseline

## Next Focus

This tracker now aligns with the shared Claw + InferenceBridge load-system work:

1. Finish runtime-truth and ready-state cleanup in InferenceBridge before adding more provider-specific features.
2. Keep LM Studio and other OpenAI-compatible callers on the same effective-config path.
3. Prefer stability and truthful lifecycle reporting over expanding settings surface area further.

The broader provider/runtime plan is now tracked in [docs/08-local-provider-runtime-improvement-plan.md](docs/08-local-provider-runtime-improvement-plan.md).

## Next Concrete Steps

- `T7` Provider registry and runtime doctor
  - [x] Add first-class provider kinds for managed llama.cpp, external llama.cpp, LM Studio, Ollama, and generic OpenAI-compatible endpoints.
  - [x] Probe common local endpoints and expose provider/model/limit/runtime capability truth through a `/v1/runtime/doctor` endpoint.
  - [x] Surface the runtime doctor in Debug.
  - [x] Add Settings provider controls for LM Studio with a saved base URL and health test.
  - [x] Add Models provider badges.
  - [x] Keep machine-specific provider config separate from model overrides.
  - [x] Route `/v1/models` and `/v1/chat/completions` through the active LM Studio provider for testing.
  - [x] Route `/v1/completions` and `/v1/responses` through the active LM Studio provider with shared upstream streaming passthrough.
  - [x] Add OpenAI `response_format` constrained decoding passthrough to llama-server.
  - [x] Add `/v1/embeddings` Phase A proxy to the loaded llama-server runtime.
  - [x] Add `/v1/messages` Anthropic-compatible request/response translation for Claude-style clients.
- `T5.3` API/Debug launch and template inspection
  - Surface the richer launch preview and effective profile data already being produced in:
    - `src-tauri/src/commands/model.rs`
    - `src-tauri/src/api/models.rs`
    - `src/lib/tauri.ts`
  - Make the Models/Debug surface show the resolved template source, context, HF source, and final launch args clearly.
- `T5.4` Live runtime state cleanup in Models/Status
  - Tighten `src/hooks/useModel.ts` so ready transitions clear stale loading UI immediately.
  - Normalize displayed labels in `src/components/Model/ModelSelector.tsx` and `src/components/Model/ProcessStatus.tsx` so `loading`, `warming`, `ready`, and `error` map to real runtime state.
- `T6.4` Explicit runtime-ready transitions
  - Keep offline/error banners suppressed during healthy swap/load windows.
  - Publish and consume a clear post-health-check ready state from `src-tauri/src/commands/model.rs`.
- `T3.3` Surface HF metadata in Models/Browse
  - Finish showing HF repo/file/template origin in the model list and browse surfaces so OpenAI-compatible and HF-backed loads are explainable.

## Shared Stability Direction

- Treat LM Studio as the main current target, but keep the runtime path generic enough for other OpenAI-compatible local providers.
- Do not fork separate load logic for GUI loads versus API-triggered loads.
- Reuse the loaded runtime whenever the effective model + load config truly match.
- Only report the backend as offline after transition completion plus a failing health probe.

## Notes

- Template precedence should be:
  1. custom template override
  2. repo/HF template
  3. built-in bridge fallback
- Vision support must be based on runtime readiness, not filename guesses alone.
- This tranche is for image understanding/input only, not text-to-image generation.
