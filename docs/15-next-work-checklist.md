# InferenceBridge Next Work Checklist

## 1. Clean and stabilize

- [x] Create this checklist as the working tracker.
- [x] Remove or ignore generated build debris from the working tree.
- [x] Confirm repo/default configs do not contain secrets.
- [ ] Decide whether active user config should keep or clear saved HF/API tokens.
- [x] Review untracked source files and keep only intentional additions.
- [x] Run `npm run build`.
- [x] Run `cargo check`.
- [x] Run `build.ps1` when source verification is clean.

Triage note: current source changes group into benchmark UI, HF sidecar/auth support, runtime pack/update support, normalized parser events, API stop feedback, and config hygiene. `src-tauri/src/normalize/events.rs` and `src/components/Benchmark/BenchmarkPanel.tsx` are intentional source additions.

## 2. Benchmark polish

- [x] Persist benchmark history across restarts.
- [x] Add saved benchmark presets.
- [x] Add a current-loaded-model shortcut.
- [x] Strengthen cancellation while a benchmark generation is active.
- [ ] Add a smoke check for 8k, 16k, and 32k context sequencing.
- [x] Add recent runs summary once history exists.
- [ ] Add chart views once enough history exists.

## 3. Stop and unload lifecycle

- [ ] Give API stop/start, model stop, and unload consistent pending UI states.
- [ ] Emit explicit backend progress events for slow stop/unload paths.
- [ ] Treat already-exited processes as clean stop success.
- [ ] Add regression coverage for slow shutdown, port release, and ghost process cases.

## 4. Config hygiene

- [x] Show the active config file path in Settings.
- [x] Add an Open Config Folder action.
- [x] Document canonical config path in README and example config.
- [ ] Decide whether to migrate or delete stale legacy Roaming config files.
- [x] Keep example/default config secret-free.

## 5. Agent reliability tests

- [ ] Add a context-pressure benchmark with huge tool outputs.
- [ ] Add malformed tool-argument repair tests.
- [ ] Add prompt-injection-from-retrieved-text tests.
- [ ] Add secret-redaction tests.
- [ ] Add destructive-operation restraint tests.
- [ ] Add request/correlation ID trace-linking tests.

## 6. Model and profile accuracy

- [ ] Use cached HF sidecars for richer model/profile hints.
- [ ] Introduce a normalized `RuntimeModelRef`.
- [ ] Compare loaded runtime config beyond filename/context.
- [ ] Make implicit API model swapping opt-in or queued behind active work.
