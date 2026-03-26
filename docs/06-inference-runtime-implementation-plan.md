# Inference Runtime Execution Plan

## Summary

Save this execution plan at the start of implementation, then execute it in seven PR-sized tranches. Use the Safe Debug Core sequence first, and keep GitHub packaged builds on the default branch plus manual dispatch only.

This plan keeps the roadmap order `A -> B -> D -> C` but corrects three repo-specific issues:

- `C-1` requires a DB migration because `messages` does not currently store `tokens_evaluated` / `tokens_predicted`.
- `A-3` should ship `ReasoningDelta` now and defer incremental `ToolCallDelta` until the parser can emit structured mid-stream tool state safely.
- The GitHub Actions problem should be treated as workflow hardening, not primarily as an app-path bug.

## Key Changes

### 1. Phase 0: plan persistence + GitHub Actions hardening

- Write this plan to `docs/06-inference-runtime-implementation-plan.md`.
- Keep workflow split:
  - `CI`: `npm run build` + `cargo check` on push and PR
  - `Build`: package artifacts on default-branch pushes and `workflow_dispatch`
  - `Release`: tag-only draft releases
- In `Build`, upload only `src-tauri/target/${target}/release/bundle/**`.
- Ensure workflow files are present on the GitHub default branch so manual dispatch appears.
- Treat remaining Actions failures as one of:
  - workflow not on default branch
  - Actions disabled / insufficient permissions
  - build job failure before artifact upload

### 2. Tranche 1: safe debug/runtime introspection

- Add shared-state debug fields:
  - `last_prompt`
  - `last_parse_trace`
  - `last_launch_preview`
  - `active_generation` metadata
- Implement real debug commands for:
  - raw prompt
  - parse trace
  - launch preview
  - effective profile
- Add Debug UI tabs/panels for:
  - `Launch`
  - `Profile`
- Add `GET /v1/debug/profile`.
- Keep this tranche introspection-only; do not change streaming behavior yet.

### 3. Phase A: streaming and async execution

- Replace the shared `AtomicBool` stop flag with `tokio_util::sync::CancellationToken`.
- Add a request-scoped `GenerationRequest` in shared state with UUID, session, model, start time, and status.
- Make GUI chat and `/v1/chat/completions` use the same SSE consumer pipeline from `engine/streaming.rs`.
- Extend stream events with:
  - `Token`
  - `ReasoningDelta`
  - `Done`
  - `Error`
- Detect reasoning spans during stream consumption and emit them separately to the frontend.
- Do not implement `ToolCallDelta` in this phase; continue extracting tool calls from final normalized output only.
- Persist prompt, trace, and generation metadata at the end of every request.

### 4. Phase B: model-aware configs

- Make `ModelProfile` the single source of truth for:
  - family
  - parser
  - renderer
  - tool style
  - think behavior
  - vision support
  - default sampling/runtime hints
- Remove duplicated vision detection from chat/API/model list codepaths and derive it only from effective profile.
- Add `Gemma` support with:
  - family enum entry
  - renderer entry
  - template rendering path
  - defaults
- Add per-model override persistence in app support data as `model-overrides.json`, keyed by model filename.
- Merge overrides after auto-detected profile resolution in both GUI and API paths.

### 5. Phase D: tighter llama.cpp-style execution control

- Add `build_args_preview()` / launch snapshot generation before process spawn.
- Add `validate_launch_config()` and fail fast on invalid runtime combinations.
- Store `last_known_good_config` only after a healthy backend start.
- Add runtime status reporting for:
  - startup duration
  - crash count
  - backend
  - resolved launch args
  - slot/concurrency state when available
- Expose runtime status through:
  - `GET /v1/runtime/status`
  - Tauri debug/runtime command
  - expanded process status UI

### 6. Phase C, subphase 1: context accounting and pressure

- Migrate `messages` to add nullable:
  - `tokens_evaluated`
  - `tokens_predicted`
- Populate those columns from the final stream/completion result for assistant messages.
- Expand context status payloads to include:
  - total tokens
  - used tokens
  - fill ratio
  - pinned / rolling / compressed breakdown
  - last compaction action
- Wire `context::strategy::decide_action()` into the post-response path.
- Emit a `context-pressure` frontend event when thresholds are crossed.

### 7. Phase C, subphase 2: idle-only summarization

- Replace the heuristic compressor with a model-backed summarizer only when there is no active request.
- Never run summarization re-entrantly during an active generation.
- Write summaries to `context_snapshots`.
- Surface compaction/rebuild reasons in Context UI and Debug logs.

## Public/API/Type Changes

- Add `GET /v1/debug/profile`.
- Add `GET /v1/runtime/status`.
- Expand debug Tauri commands to return real prompt, parse trace, launch preview, and effective profile data.
- Expand context status types to return per-layer breakdown.
- Expand session/message schema with `tokens_evaluated` and `tokens_predicted`.
- Expand process/runtime frontend types to include startup duration and runtime status details.

## Test Plan

- Workflow
  - push to a feature branch runs `CI`
  - push to the default branch runs `CI` + `Build`
  - manual `Build` is available in GitHub
  - tag push creates a draft release
- Streaming/runtime
  - GUI and API streaming produce matching final content
  - `Stop` cancels without hanging
  - unload during generation fails predictably and leaves runtime healthy
  - raw prompt / parse trace / launch preview update after requests and loads
- Model/profile
  - Qwen models use the correct parser and reasoning behavior
  - Gemma models resolve to the new renderer path
  - vision support is derived from effective profile only
  - model overrides survive restart and rescan
- Context
  - DB migration preserves old sessions
  - assistant rows store `tokens_evaluated` and `tokens_predicted`
  - context-pressure emits at threshold crossings
  - summarization never runs while `active_generation` is set
- Runtime control
  - invalid launch configs are rejected before spawn
  - healthy launches update `last_known_good_config`
  - runtime status endpoint reports startup duration and crash count correctly

## Assumptions and defaults

- Keep the current single active backend process model.
- Store per-model overrides in a dedicated JSON file, not in SQLite and not in the main TOML config.
- Defer true incremental `ToolCallDelta` until the parser/normalizer can emit structured mid-stream tool state.
- Keep packaged GitHub builds on default-branch plus manual dispatch only.
- First implementation PR:
  - prompt / trace / launch preview persistence
  - debug command wiring
  - Debug `Launch` + `Profile` UI
- Recommended PR order:
  1. Actions hardening + save plan file
  2. Debug introspection tranche
  3. Cancellation + unified streaming
  4. Profile unification + overrides + Gemma
  5. Launch validation + runtime status
  6. Context accounting + migration + pressure events
  7. Idle-only summarization
