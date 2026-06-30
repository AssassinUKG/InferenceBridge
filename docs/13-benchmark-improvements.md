# Benchmark Improvements Plan

## Context

`src/components/Benchmark/BenchmarkPanel.tsx` defines all benchmark cases and scoring inline (the `TESTS` array, ~132–376).  
`src-tauri/src/engine/benchmark.rs` runs a single completion and returns `ModelTestStats`.  
`src-tauri/src/commands/benchmark.rs` is the Tauri command layer that loads the model and calls the engine.

The benchmarks are structured well but have the following concrete problems that need fixing.

---

## Fix 1 — Enforce seed and temperature=0 for all benchmark runs

**File:** `src/components/Benchmark/BenchmarkPanel.tsx`

The UI exposes `temperature` (default 0.2) and `seed` (default 42) but lets the user change them. Benchmarks are meaningless if they are non-deterministic. The same model will produce different scores on different runs.

**Change:**
- In the `runBenchmarks` function (~line 659), always pass `temperature: 0` and `seed: 42` when calling `api.runModelTest`, ignoring the user-controlled sliders for the actual API call.
- Keep the UI sliders but grey them out with a tooltip note: *"Benchmarks run at temperature=0, seed=42 for reproducibility. These settings apply to manual test runs only."*
- Alternatively, remove the temperature/seed controls from the benchmark UI entirely and document the fixed values in the description text.

---

## Fix 2 — TTFT measurement via streaming

**File:** `src-tauri/src/engine/benchmark.rs`

`ttft_ms` is always `None` because the engine uses non-streaming completions. TTFT (time to first token) is the most user-perceptible latency metric for a chat application.

**Change in `src-tauri/src/engine/benchmark.rs`:**

Switch the benchmark request to streaming mode. Record `Instant::now()` before sending, then timestamp the arrival of the first token chunk. Return that delta as `ttft_ms`.

Skeleton:
```rust
let start = std::time::Instant::now();
let mut stream = client.complete_stream(&request).await?;
let mut first_token_ms: Option<u128> = None;
let mut full_response = String::new();
while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    if first_token_ms.is_none() && !chunk.content.is_empty() {
        first_token_ms = Some(start.elapsed().as_millis());
    }
    full_response.push_str(&chunk.content);
    if chunk.stop { break; }
}
```

Add `ttft_ms: first_token_ms` to the returned `ModelTestStats`. The timings (tok/s) will still come from the llama.cpp `/timings` response in the stop chunk.

**File:** `src/components/Benchmark/BenchmarkPanel.tsx`

Surface TTFT in the results row and CSV export. Already has `fmtMs` helper. Add it next to decode tok/s in the results table and the summary.

---

## Fix 3 — Replace Chat and Decode fake quality scores with honest throughput labels

**File:** `src/components/Benchmark/BenchmarkPanel.tsx`, `TESTS` array lines ~133–157

Currently the `chat` and `decode` tests score `passed: response.length > 120` and `passed: response.length > 600`. These are not quality checks — they are throughput warmup runs that happen to check the model doesn't truncate early. The scores are misleading.

**Change:**
- Rename the tests to make their purpose explicit: `"Decode Speed (short)"` and `"Decode Speed (long)"`.
- Change the score function to always return `passed: true` if the model generated any tokens, with `score` set to a normalized tok/s value relative to thresholds (e.g. `score = Math.min(1, decode_tokens_per_second / 60)` for interactive-speed target of 60 tok/s).
- Update `note` to say `"Decode at X tok/s"` rather than `"Produced a complete chat response."`.
- These tests are still valuable — they give the decode baseline. They just shouldn't pretend to be quality checks.

---

## Fix 4 — Replace single-question Reasoning test with a small fixed GSM8K-style set

**File:** `src/components/Benchmark/BenchmarkPanel.tsx`, `TESTS` array, `reasoning` entry (~line 249)

One question ("42") has no statistical meaning. A wrong answer could be a decoding artifact rather than a reasoning failure.

**Change:**
Add a fixed set of 5 grade-school math word problems with deterministic integer answers. Run all 5 in a single prompt (or as 5 sequential calls if the backend supports it), score based on how many are correct.

Suggested fixed question set (all answers are deterministic integers):
```
Q1: A baker makes 12 loaves a day for 5 days, then gives away 18. How many remain? → 42
Q2: A machine makes 7 parts every 3 minutes. It rests 6 min after every 21 parts. How many parts after 30 min? → 42
Q3: A train travels 60 km/h for 2 hours then 80 km/h for 1 hour. Total km? → 200
Q4: 15 workers finish a job in 8 days. How many days for 10 workers at the same rate? → 12
Q5: A shop sells 3 items at $4, 5 items at $7, and 2 items at $15. Total revenue? → $77
```

Score: `passed_count / 5`, `passed: score >= 0.6`.

Each question should be in its own prompt call (reuse the existing per-test run loop), or packed into a single numbered prompt if you want to keep one API call. Either approach is fine — just make the expected answers a constant array checked by exact regex match.

Simplest approach — keep as one test but multi-question prompt:
```
prompt: "Answer each question with only the number.\n1. A baker makes 12 loaves/day for 5 days then gives away 18. Remaining?\n2. 60 km/h for 2h then 80 km/h for 1h. Total km?\n3. 15 workers finish in 8 days. 10 workers at same rate? How many days?\n4. 3×$4 + 5×$7 + 2×$15. Total?"
```
```
score: checks [/\b42\b/, /\b200\b/, /\b12\b/, /\b77\b/], passAt: 0.67
```

---

## Fix 5 — Replace single-function Coding test with a small deterministic set

**File:** `src/components/Benchmark/BenchmarkPanel.tsx`, `TESTS` array, `coding` entry (~line 263)

The `clamp` check passes on any code that mentions `Math.min` and `Math.max`. That's not a code quality signal.

**Change:**
Use a multi-function prompt with 3 short problems, each with a verifiable structural signature:

```
prompt: "Write these three TypeScript functions. Return only code, no prose.\n1. clamp(value: number, min: number, max: number): number — bound value to [min, max]\n2. sum(arr: number[]): number — return the sum of an array\n3. capitalize(s: string): string — uppercase the first character"
```

Score checks:
```ts
const checks = [
  /function\s+clamp|const\s+clamp/.test(r) && /Math\.min/.test(r) && /Math\.max/.test(r),
  /function\s+sum|const\s+sum/.test(r) && (/reduce/.test(r) || /forEach/.test(r) || /for\s*\(/.test(r)),
  /function\s+capitalize|const\s+capitalize/.test(r) && (/\[0\]|charAt|slice|substring/.test(r)),
];
```

`passAt: 0.67` (needs 2 of 3 correct).

---

## Fix 6 — Add IFEval-style instruction following test

**File:** `src/components/Benchmark/BenchmarkPanel.tsx`, `TESTS` array

Add a new `Core` group test `"instruction_follow"` that tests whether the model can obey verifiable constraints.

```ts
{
  id: "instruction_follow",
  group: "Core",
  name: "Instruction Following",
  description: "Obeys multiple explicit format constraints simultaneously.",
  prompt: "Respond with exactly 3 bullet points (lines starting with -). Each bullet must be under 15 words. The second bullet must include the word 'latency'. Topic: what slows down local LLM inference.",
  maxTokens: 160,
  score: (stats) => {
    const lines = stats.response.split('\n').filter(l => l.trim().startsWith('-'));
    const checks = [
      lines.length === 3,
      lines.every(l => l.trim().split(/\s+/).length <= 20), // generous word count
      /latency/i.test(lines[1] ?? ''),
    ];
    const result = scoreFromChecks(checks, 0.67);
    return { ...result, note: result.passed ? "Followed all format constraints." : `Followed ${checks.filter(Boolean).length}/3 constraints (3 bullets, word limit, 'latency' in bullet 2).` };
  },
},
```

---

## Fix 7 — Wire BenchmarkPanel into the app

**File:** `src/App.tsx`

`BenchmarkPanel` exists at `src/components/Benchmark/BenchmarkPanel.tsx` but is not imported or rendered anywhere.

**Change:**
1. Import `BenchmarkPanel` in `App.tsx`.
2. Add a `"benchmark"` entry to whatever tab/navigation system is used (same pattern as the existing Model, Settings, Debug tabs).
3. Pass the required props: `models` (already available from model registry state) and `processStatus` (already available from process state).

Look at how `DebugInspector` or `ModelBrowser` are wired in `App.tsx` for the exact pattern to follow.

---

## Fix 8 — Add `ttft_ms` and normalized scores to CSV export

**File:** `src/components/Benchmark/BenchmarkPanel.tsx`, `toCsv` function (~line 465)

Once Fix 2 is implemented, add `ttft_ms` to the CSV header and rows so exported results include the full picture for external analysis.

```ts
const header = [..., "ttft_ms", ...];
// in rows:
result.stats?.ttft_ms ?? "",
```

---

## Summary of priority order

| Priority | Fix | Effort |
|---|---|---|
| 1 | Fix 7 — Wire panel into app | Low — import + nav entry |
| 2 | Fix 1 — Force seed=42, temp=0 | Low — 2 line change in runBenchmarks |
| 3 | Fix 3 — Honest throughput labels | Low — rename + rescore 2 tests |
| 4 | Fix 6 — Instruction following test | Low — add one test case |
| 5 | Fix 5 — Multi-function coding test | Low — replace prompt + score fn |
| 6 | Fix 4 — Multi-question reasoning | Medium — new prompts + expected answers |
| 7 | Fix 2 — TTFT via streaming | Medium — Rust streaming client change |
| 8 | Fix 8 — CSV ttft_ms column | Low — depends on Fix 2 |
