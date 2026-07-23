import assert from "node:assert/strict";
import test from "node:test";

import {
  defaultRecommendedLoadPreset,
  describePromptRendering,
  isTessQwenModel,
  readSamplingSettings,
  recommendedContextForModel,
  recommendedLoadPresets,
  replaceSamplingArgs,
  samplingArgsMatch,
  stripStaleThinkingKwargs,
} from "../src/lib/modelLoadProfiles.ts";

const tess = (extra = {}) => ({
  filename: "Tess-4-27B-Q4_K_M.gguf",
  family: "Generic",
  gguf_architecture: "qwen35",
  context_window: 262_144,
  max_context_window: 262_144,
  supports_reasoning: true,
  supports_tools: true,
  supports_parallel_tools: true,
  supports_vision: true,
  has_chat_template: true,
  hf_repo: "migtissera/Tess-4-27B-GGUF",
  ...extra,
});

test("Tess and qwen35 metadata activate the mode-aware profile", () => {
  assert.equal(isTessQwenModel(tess({ gguf_architecture: null })), true);
  assert.equal(isTessQwenModel(tess({ filename: "renamed.gguf" })), true);
  assert.equal(isTessQwenModel(tess({ filename: "llama.gguf", family: "Llama", gguf_architecture: "llama" })), false);
});

test("fresh Tess defaults to the approved serial Tools / Direct profile at 32K", () => {
  const profile = defaultRecommendedLoadPreset(tess());
  assert.ok(profile);
  assert.equal(profile.id, "tools-direct");
  assert.equal(profile.reasoningMode, "off");
  assert.equal(profile.contextSize, 32_768);
  assert.equal(profile.parallelSlots, 1);
  assert.deepEqual(profile.sampling, {
    temperature: 0.7,
    topP: 0.8,
    topK: 20,
    minP: 0,
    presencePenalty: 1.5,
    repeatPenalty: 1,
  });
  assert.equal(recommendedContextForModel(tess(), 262_144), 32_768);
});

test("all three Tess sampler modes retain their agreed values", () => {
  const profiles = Object.fromEntries(recommendedLoadPresets(tess()).map((item) => [item.id, item]));
  assert.deepEqual(profiles["general-thinking"].sampling, {
    temperature: 1,
    topP: 0.95,
    topK: 20,
    minP: 0,
    presencePenalty: 0,
    repeatPenalty: 1,
  });
  assert.equal(profiles["general-thinking"].reasoningMode, "on");
  assert.deepEqual(profiles["precise-coding"].sampling, {
    temperature: 0.6,
    topP: 0.95,
    topK: 20,
    minP: 0,
    presencePenalty: 0,
    repeatPenalty: 1,
  });
  assert.equal(profiles["precise-coding"].reasoningMode, "on");
});

test("applying a sampler replaces stale sampler flags and preserves unrelated launch args", () => {
  const profile = defaultRecommendedLoadPreset(tess());
  assert.ok(profile);
  const args = replaceSamplingArgs(
    ["--flash-attn", "--temp", "0.2", "--top-p=0.4", "--repeat-penalty", "1.2"],
    profile.sampling,
  );

  assert.equal(args.includes("--flash-attn"), true);
  assert.equal(args.includes("0.2"), false);
  assert.equal(samplingArgsMatch(args, profile.sampling), true);
  assert.deepEqual(readSamplingSettings(args), {
    temperature: 0.7,
    topP: 0.8,
    topK: 20,
    minP: 0,
    presencePenalty: 1.5,
    repeatPenalty: 1,
  });
});

test("legacy thinking kwargs are removed without discarding unrelated template options", () => {
  assert.deepEqual(
    stripStaleThinkingKwargs('{"enable_thinking":true,"preserve_thinking":false,"custom":"ok"}'),
    { value: '{"preserve_thinking":false,"custom":"ok"}', removed: true },
  );
  assert.deepEqual(stripStaleThinkingKwargs('{"enable_thinking":false}'), { value: "", removed: true });
  assert.deepEqual(stripStaleThinkingKwargs("{invalid"), { value: "{invalid", removed: false });
});

test("effective prompt rendering identifies the embedded GGUF source truthfully", () => {
  assert.deepEqual(
    describePromptRendering(tess(), {
      templateMode: "builtin",
      templateName: "",
      customTemplatePath: "",
      useJinja: true,
    }),
    { label: "Embedded GGUF Jinja", source: "gguf:embedded-jinja", effective: true },
  );
  assert.equal(
    describePromptRendering(tess(), {
      templateMode: "builtin",
      templateName: "",
      customTemplatePath: "",
      useJinja: false,
    }).effective,
    false,
  );
});
