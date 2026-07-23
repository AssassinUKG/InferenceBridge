import assert from "node:assert/strict";
import test from "node:test";

import {
  applyBenchmarkDeletion,
  retainExpandedBenchmarkRows,
} from "../src/lib/benchmarkData.ts";

const state = () => ({
  results: [{ id: "result-1" }, { id: "result-2" }],
  history: [{ id: "run-1" }, { id: "run-2" }],
});

test("deletes one current result without changing saved run history", () => {
  assert.deepEqual(
    applyBenchmarkDeletion(state(), { scope: "result", id: "result-1" }),
    {
      results: [{ id: "result-2" }],
      history: [{ id: "run-1" }, { id: "run-2" }],
    },
  );
});

test("clears current results without clearing history", () => {
  const next = applyBenchmarkDeletion(state(), { scope: "current-results" });
  assert.deepEqual(next.results, []);
  assert.deepEqual(next.history, [{ id: "run-1" }, { id: "run-2" }]);
});

test("deletes one saved run without changing current results", () => {
  assert.deepEqual(
    applyBenchmarkDeletion(state(), { scope: "history-run", id: "run-2" }),
    {
      results: [{ id: "result-1" }, { id: "result-2" }],
      history: [{ id: "run-1" }],
    },
  );
});

test("clears history independently", () => {
  const next = applyBenchmarkDeletion(state(), { scope: "history" });
  assert.deepEqual(next.results, [{ id: "result-1" }, { id: "result-2" }]);
  assert.deepEqual(next.history, []);
});

test("clears all benchmark records while leaving unrelated state out of scope", () => {
  assert.deepEqual(applyBenchmarkDeletion(state(), { scope: "all" }), {
    results: [],
    history: [],
  });
});

test("missing record deletion is idempotent", () => {
  assert.deepEqual(
    applyBenchmarkDeletion(state(), { scope: "result", id: "missing" }),
    state(),
  );
});

test("expanded row state is pruned after result deletion", () => {
  assert.deepEqual(
    retainExpandedBenchmarkRows(
      { "result-1": true, "result-2": true, collapsed: false },
      new Set(["result-2"]),
    ),
    { "result-2": true },
  );
});
