import type { SamplingParams } from "./tauri";

export interface Preset {
  label: string;
  icon: string;
  description: string;
  sampling: SamplingParams;
  suggestThinking: boolean | null;
}

export const PRESETS: Record<string, Preset> = {
  coding: {
    label: "Coding",
    icon: "</>",
    description:
      "Near-greedy. Deterministic, low hallucination. Enable Thinking for best results on Qwen3/R1.",
    sampling: {
      temperature: 0.1,
      top_p: 0.95,
      top_k: 20,
    },
    suggestThinking: true,
  },

  chat: {
    label: "Chat",
    icon: "...",
    description: "Balanced creativity and coherence. Good for conversation and Q&A.",
    sampling: {
      temperature: 0.7,
      top_p: 0.8,
      top_k: 40,
    },
    suggestThinking: false,
  },

  creative: {
    label: "Creative",
    icon: "*",
    description: "Higher temperature for diverse, imaginative, varied responses.",
    sampling: {
      temperature: 0.9,
      top_p: 0.95,
      top_k: 50,
    },
    suggestThinking: false,
  },

  precise: {
    label: "Precise",
    icon: "=",
    description: "Greedy and deterministic. Best for structured output, JSON, and classification.",
    sampling: {
      temperature: 0.0,
      top_p: 1.0,
      top_k: -1,
    },
    suggestThinking: true,
  },

  reasoning: {
    label: "Reasoning",
    icon: "?",
    description: "Tuned for math, logic, and multi-step reasoning. Thinking mode on.",
    sampling: {
      temperature: 0.6,
      top_p: 0.95,
      top_k: -1,
    },
    suggestThinking: true,
  },
} as const;

export type PresetKey = keyof typeof PRESETS;

export const PRESET_ORDER: PresetKey[] = [
  "coding",
  "chat",
  "creative",
  "precise",
  "reasoning",
];

export function modelSupportsThinking(modelName: string | null | undefined): boolean {
  if (!modelName) return false;
  const m = modelName.toLowerCase();
  return (
    m.includes("qwen3") ||
    m.includes("qwen3.5") ||
    (m.includes("deepseek") && (m.includes("r1") || m.includes("reasoning")))
  );
}
