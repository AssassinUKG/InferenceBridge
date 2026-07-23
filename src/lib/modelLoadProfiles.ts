export const TESS_QWEN_RECOMMENDED_CONTEXT = 32_768;

export interface LoadProfileModel {
  filename: string;
  family: string;
  gguf_architecture: string | null;
  context_window: number | null;
  max_context_window: number | null;
  supports_reasoning: boolean;
  supports_tools: boolean;
  supports_parallel_tools?: boolean;
  supports_vision: boolean;
  has_chat_template: boolean;
  hf_repo: string | null;
}

export interface SamplingSettings {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  presencePenalty: number;
  repeatPenalty: number;
}

export interface RecommendedLoadPreset {
  id: "general-thinking" | "precise-coding" | "tools-direct";
  name: string;
  description: string;
  reasoningMode: "on" | "off";
  contextSize: number;
  sampling: SamplingSettings;
  parallelSlots: number | null;
}

const TESS_QWEN_PRESETS: readonly RecommendedLoadPreset[] = [
  {
    id: "general-thinking",
    name: "General Thinking",
    description: "Adaptive reasoning for broad questions and multi-step work.",
    reasoningMode: "on",
    contextSize: TESS_QWEN_RECOMMENDED_CONTEXT,
    sampling: {
      temperature: 1,
      topP: 0.95,
      topK: 20,
      minP: 0,
      presencePenalty: 0,
      repeatPenalty: 1,
    },
    parallelSlots: null,
  },
  {
    id: "precise-coding",
    name: "Precise Coding",
    description: "Lower-variance reasoning for code and exact technical work.",
    reasoningMode: "on",
    contextSize: TESS_QWEN_RECOMMENDED_CONTEXT,
    sampling: {
      temperature: 0.6,
      topP: 0.95,
      topK: 20,
      minP: 0,
      presencePenalty: 0,
      repeatPenalty: 1,
    },
    parallelSlots: null,
  },
  {
    id: "tools-direct",
    name: "Tools / Direct",
    description: "Direct answers and reliable serial tool calls with thinking disabled.",
    reasoningMode: "off",
    contextSize: TESS_QWEN_RECOMMENDED_CONTEXT,
    sampling: {
      temperature: 0.7,
      topP: 0.8,
      topK: 20,
      minP: 0,
      presencePenalty: 1.5,
      repeatPenalty: 1,
    },
    parallelSlots: 1,
  },
];

export function isTessQwenModel(model: Pick<LoadProfileModel, "filename" | "family" | "gguf_architecture">) {
  const filename = model.filename.toLowerCase();
  const family = model.family.toLowerCase().replace(/[._-]/g, "");
  const architecture = (model.gguf_architecture ?? "").toLowerCase().replace(/[._-]/g, "");

  return (
    filename.includes("tess-4") ||
    filename.includes("tess_4") ||
    filename.includes("qwen3.6") ||
    filename.includes("qwen3_6") ||
    filename.includes("qwen3-6") ||
    family.includes("qwen35") ||
    family.includes("qwen36") ||
    architecture.includes("qwen35") ||
    architecture.includes("qwen36")
  );
}

export function recommendedLoadPresets(model: LoadProfileModel): readonly RecommendedLoadPreset[] {
  return isTessQwenModel(model) ? TESS_QWEN_PRESETS : [];
}

export function defaultRecommendedLoadPreset(model: LoadProfileModel) {
  return recommendedLoadPresets(model).find((preset) => preset.id === "tools-direct") ?? null;
}

export function recommendedContextForModel(model: LoadProfileModel, advertisedContext: number) {
  if (!isTessQwenModel(model)) return null;
  return Math.min(TESS_QWEN_RECOMMENDED_CONTEXT, advertisedContext);
}

function formatNumber(value: number) {
  return Number.isInteger(value)
    ? value.toString()
    : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

const SAMPLING_VALUE_FLAGS = new Set([
  "--temp",
  "--temperature",
  "--top-p",
  "--top-k",
  "--min-p",
  "--presence-penalty",
  "--repeat-penalty",
]);

export function samplingArgs(settings: SamplingSettings) {
  return [
    "--temp",
    formatNumber(settings.temperature),
    "--top-p",
    formatNumber(settings.topP),
    "--top-k",
    formatNumber(settings.topK),
    "--min-p",
    formatNumber(settings.minP),
    "--presence-penalty",
    formatNumber(settings.presencePenalty),
    "--repeat-penalty",
    formatNumber(settings.repeatPenalty),
  ];
}

export function replaceSamplingArgs(args: readonly string[], settings: SamplingSettings) {
  const preserved: string[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    const flag = arg.split("=", 1)[0];
    if (SAMPLING_VALUE_FLAGS.has(flag)) {
      if (!arg.includes("=") && index + 1 < args.length) index += 1;
      continue;
    }
    preserved.push(arg);
  }
  return [...preserved, ...samplingArgs(settings)];
}

function samplingValue(args: readonly string[], wantedFlag: string) {
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    const [flag, inlineValue] = arg.split("=", 2);
    if (flag !== wantedFlag && !(wantedFlag === "--temp" && flag === "--temperature")) continue;
    const raw = inlineValue ?? args[index + 1];
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }
  return null;
}

export function readSamplingSettings(args: readonly string[]) {
  return {
    temperature: samplingValue(args, "--temp"),
    topP: samplingValue(args, "--top-p"),
    topK: samplingValue(args, "--top-k"),
    minP: samplingValue(args, "--min-p"),
    presencePenalty: samplingValue(args, "--presence-penalty"),
    repeatPenalty: samplingValue(args, "--repeat-penalty"),
  };
}

export function samplingArgsMatch(args: readonly string[], settings: SamplingSettings) {
  const expected: [string, number][] = [
    ["--temp", settings.temperature],
    ["--top-p", settings.topP],
    ["--top-k", settings.topK],
    ["--min-p", settings.minP],
    ["--presence-penalty", settings.presencePenalty],
    ["--repeat-penalty", settings.repeatPenalty],
  ];
  return expected.every(([flag, value]) => samplingValue(args, flag) === value);
}

const STALE_THINKING_KEYS = new Set([
  "enable_thinking",
  "enableThinking",
  "reasoning",
  "thinking",
]);

export function stripStaleThinkingKwargs(value: string) {
  const trimmed = value.trim();
  if (!trimmed) return { value: "", removed: false };

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
      return { value, removed: false };
    }

    const entries = Object.entries(parsed as Record<string, unknown>);
    const filtered = entries.filter(([key]) => !STALE_THINKING_KEYS.has(key));
    if (filtered.length === entries.length) return { value, removed: false };
    return {
      value: filtered.length > 0 ? JSON.stringify(Object.fromEntries(filtered)) : "",
      removed: true,
    };
  } catch {
    return { value, removed: false };
  }
}

export interface PromptRenderingSelection {
  templateMode: string;
  templateName: string;
  customTemplatePath: string;
  useJinja: boolean;
}

export function describePromptRendering(model: LoadProfileModel, selection: PromptRenderingSelection) {
  if (selection.templateMode === "custom") {
    return {
      label: "Custom Jinja file",
      source: selection.customTemplatePath.trim() || "No file selected",
      effective: !!selection.customTemplatePath.trim(),
    };
  }

  if (selection.templateMode === "repo") {
    return {
      label: "Hugging Face repo Jinja",
      source: model.hf_repo ? `${model.hf_repo}/chat_template.jinja` : "Repo metadata unavailable",
      effective: !!model.hf_repo,
    };
  }

  if (selection.templateName.trim()) {
    return {
      label: "Named llama.cpp template",
      source: `builtin:${selection.templateName.trim()}`,
      effective: true,
    };
  }

  if (selection.useJinja && model.has_chat_template) {
    return {
      label: "Embedded GGUF Jinja",
      source: "gguf:embedded-jinja",
      effective: true,
    };
  }

  return {
    label: "llama.cpp fallback",
    source: selection.useJinja ? "No embedded template detected" : "Jinja disabled",
    effective: false,
  };
}
