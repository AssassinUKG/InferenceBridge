# Benchmark Section Plan

## Goal

Build a best-in-class Benchmark section for InferenceBridge that can compare one or many local models under repeatable test profiles, report performance and quality signals, and produce clear tuning advice for faster local inference.

## Product Shape

- [x] Add a top-level `Benchmark` tab next to Models, Browse, Context, Logs, API, and Settings.

The first screen should be the usable benchmark workspace, not an explanation page:

- [x] Left rail: model selection.
- [ ] Saved benchmark presets.
- [x] Main pane: test suite configuration and run controls.
- [x] Right pane: live run status, current metrics, and advice.
- [x] Bottom/results area: comparison table.
- [ ] Chart views.

## Model Selection

The Benchmark tab should support:

- [x] Single model benchmark.
- [x] Multi-model comparison.
- [ ] Current loaded model only shortcut.
- [x] Sequential auto-load of selected models.
- [ ] Optional comparison of different launch settings for the same model.

Each selectable model row should show:

- [x] Model name.
- [x] Quant.
- [x] Size.
- [x] Architecture.
- [ ] Context window.
- [ ] Capability badges: reasoning, tools, vision, coding, MoE.
- [x] Current load state.
- [ ] Estimated fit status.

## Benchmark Test Types

### Chat Speed

Basic assistant response test using a short normal chat prompt.

Metrics:

- Time to first token.
- Prompt tokens/sec.
- Decode tokens/sec.
- End-to-end tokens/sec.
- Completion tokens.
- Total latency.

### Prompt Eval

Long prompt prefill test with fixed prompt sizes.

Profiles:

- 512 tokens.
- [ ] 2K tokens.
- [x] 8K tokens.
- [x] 32K tokens.
- [x] 64K tokens.
- [x] Custom editable context lengths.

Metrics:

- Prompt tokens/sec.
- Prefill latency.
- Context pressure.
- KV cache estimate.

### Tool Calling

Tests whether the model can produce valid tool calls in the expected local format.

Scenarios:

- [x] Single function call.
- [ ] Multiple function calls.
- [x] Required argument extraction.
- [x] No-tool-needed refusal.
- [x] First-step agent sequencing.
- [x] Mauler-style agent delegation.

Metrics:

- Valid structure rate.
- Correct tool selection.
- Argument correctness.
- Parse repair needed.
- Latency and token speed.

### Reasoning

Small deterministic reasoning set.

Scenarios:

- Arithmetic.
- Multi-step logic.
- Instruction following.
- Short answer extraction.

Metrics:

- Pass/fail.
- Output length.
- Reasoning mode used.
- Speed.

### Coding

Small coding tests that are cheap to run locally.

Scenarios:

- Write a small function.
- Fix a small bug.
- Explain an error.
- Produce JSON test cases.

Metrics:

- Syntax validity.
- Test pass where runnable.
- Format compliance.
- Speed.

### Long Context

Needle-in-context style test at selectable context sizes.

Metrics:

- [x] Recall correctness.
- [x] Prompt eval speed.
- [x] Context size reached.
- Failure mode: refused, forgot, malformed, timed out.

### Vision

Only shown for vision-capable models.

Scenarios:

- Image description.
- OCR-like extraction.
- Visual question answering.

Metrics:

- Completion validity.
- Latency.
- Token speed.
- MMProj attached status.

### Stress

Server and scheduling test.

Profiles:

- 1 request.
- 2 concurrent.
- 4 concurrent.
- Custom.

Metrics:

- Queue time.
- Median latency.
- P95 latency.
- Tokens/sec under load.
- Failure/crash rate.

## Results Table

Columns:

- Model.
- Quant.
- Size.
- Test suite.
- Load time.
- TTFT.
- Prompt tok/s.
- Decode tok/s.
- End-to-end tok/s.
- Total latency.
- Score.
- Errors.
- Advice.

Interactions:

- Sort by any metric.
- Pin baseline.
- Compare against baseline percentage.
- Filter by suite.
- Export JSON.
- Export CSV.
- Copy summary.

## Live Run UI

During a run, show:

- Current model.
- Current test.
- Step progress.
- Load progress.
- Streaming token speed.
- Cancel benchmark button.
- Error panel with recovery hint.

The benchmark runner should be cancellable between tests and during active generation.

## Advice Engine

Advice should be generated from measured data and launch configuration, not generic text.

Examples:

- Low decode tok/s and large quant: suggest `Q4_K_M` or `Q5_K_M`.
- High prompt latency: suggest lower context, higher batch/ubatch, or flash attention if available.
- Context causes RAM pressure: suggest smaller context or fit mode.
- Tool test fails: suggest Jinja/template/profile settings.
- Vision test fails: suggest attaching MMProj.
- Long TTFT but good decode: suggest prompt length reduction or context cache strategy.
- Server instability: suggest fewer parallel slots or smaller context.

## Backend Commands

Add new commands:

```rust
run_benchmark_suite(request: BenchmarkRunRequest) -> Result<BenchmarkRunStarted, String>
cancel_benchmark(run_id: String) -> Result<(), String>
list_benchmark_runs() -> Result<Vec<BenchmarkRunSummary>, String>
get_benchmark_run(run_id: String) -> Result<BenchmarkRunDetails, String>
delete_benchmark_run(run_id: String) -> Result<(), String>
```

Events:

```rust
benchmark-run-progress
benchmark-run-result
benchmark-run-complete
```

The existing `run_model_test` and `engine::benchmark::test_model` should become the seed for the new suite runner rather than being duplicated.

## Data Model

Benchmark run:

- `run_id`
- `created_at`
- `models`
- `suite`
- `settings`
- `status`
- `results`
- `advice`
- `errors`

Benchmark result:

- `model_name`
- `model_path`
- `quant`
- `size_gb`
- `test_id`
- `test_name`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `ttft_ms`
- `prefill_ms`
- `decode_ms`
- `total_ms`
- `prompt_tokens_per_second`
- `decode_tokens_per_second`
- `end_to_end_tokens_per_second`
- `score`
- `passed`
- `error`

Persist results under app support storage so history survives restarts.

## Rollout

### Phase 1: MVP

- [x] Add Benchmark tab.
- [x] Multi-model selector.
- [x] Chat speed test.
- [x] Prompt eval test.
- [x] Tool calling test.
- [x] Parsed tool-call extraction in benchmark results.
- [x] Agentic tool workflow tests.
- [x] Tool restraint/no-tool-needed test.
- [x] First-step sequencing test for file-edit workflows.
- [x] Mauler-style delegation test.
- [x] Sequential runner.
- [x] Live progress.
- [x] Results table.
- [x] Basic advice.
- [x] JSON/CSV export.
- [x] Load time metric.
- [x] Wait for llama-server readiness before each test request.
- [x] Reuse already-loaded benchmark model between tests.
- [x] Benchmark loads use the normal model loader/template resolution path.
- [x] Expand result rows to inspect prompt, output, errors, and timing details.
- [x] Add High/Medium/Low/Failed quality bands for partial-credit interpretation.
- [x] Add best-for/settings recommendations to benchmark advice.
- [x] Rename speed columns to Prompt eval tok/s and Generation tok/s.
- [x] Correct MVP reasoning test expected answer.
- [x] Add longer Decode Speed test for steadier tok/s comparison.
- [x] Add three editable context lengths defaulted to 8K, 32K, and 64K.
- [x] Stop context inputs snapping back to 512 while editing.
- [x] Add large-report context recall test.
- [x] Size context recall prompts near the requested context instead of tripling it.
- [x] Run context matrix largest-first so smaller contexts can reuse the largest load.
- [x] Show actual prompt tokens from llama-server beside requested context.
- [x] Hide TTFT from the UI until true streaming TTFT is implemented.
- [x] Seed a default model after model scan populates.
- [x] Add first-pass per-model composite ranking in run summary.
- [x] TTFT result field reserved in backend.
- [ ] True streaming TTFT measurement.
- [ ] Backend event stream for benchmark progress.
- [ ] Persist benchmark history across restarts.

### Phase 2: Quality Suites

- Reasoning suite.
- Coding suite.
- Long-context suite.
- Vision suite.
- Score normalization.
- Baseline comparison.

### Phase 3: Tuning Sweeps

- Compare context sizes.
- Compare fit modes.
- Compare batch/ubatch.
- Compare flash attention on/off.
- Compare draft model/speculative settings.

### Phase 4: History And Reports

- Saved run history.
- Trend charts.
- Markdown report export.
- “Best model for this machine” recommendation.
- “Fastest acceptable quant” recommendation.

## Acceptance Criteria

- [x] User can select multiple local models and run one benchmark suite.
- [x] Results include prompt tok/s, decode tok/s, end-to-end tok/s, load time, and errors.
- [ ] Results include true measured TTFT.
- [x] Tool tests report validity, not just speed.
- [x] Failed runs do not wedge the UI.
- [x] Advice is specific to measured bottlenecks.
- [x] Results are exportable.
- [ ] Benchmark runs are cancellable during active generation.
