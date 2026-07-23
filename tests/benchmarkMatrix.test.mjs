import assert from "node:assert/strict";
import test from "node:test";

import {
  benchmarkCombinationCount,
  selectedRuntimeCandidates,
} from "../src/lib/benchmarkMatrix.ts";

test("MTP runtime candidates are offered only to MTP models", () => {
  assert.deepEqual(
    selectedRuntimeCandidates(["auto", "mtp-1", "mtp-2", "mtp-3"], "Qwen3.6-27B-MTP-Q4_K_XL.gguf").map((item) => item.id),
    ["auto", "mtp-1", "mtp-2", "mtp-3"],
  );
  assert.deepEqual(
    selectedRuntimeCandidates(["auto", "mtp-1", "mtp-2", "mtp-3"], "Qwen3.6-27B-Q4_K_XL.gguf").map((item) => item.id),
    ["auto"],
  );
  assert.deepEqual(
    selectedRuntimeCandidates(["auto", "mtp-3"], "Qwen3.6 27B MTP Q4_K_XL.gguf").map((item) => item.id),
    ["auto", "mtp-3"],
  );
});

test("combination count reflects model-specific runtime candidates", () => {
  assert.equal(
    benchmarkCombinationCount(
      ["Qwen3.6-27B-MTP-Q4_K_XL.gguf", "Gemma-12B-Q4_K_M.gguf"],
      2,
      3,
      ["deterministic", "coding"],
      ["auto", "mtp-1", "mtp-2", "mtp-3"],
    ),
    60,
  );
});
