/**
 * Sampling presets for InferenceBridge.
 *
 * Each preset maps a use-case to a recommended set of sampling parameters.
 * Parameters are merged on top of model-profile defaults, so you can rely on
 * the model knowing its own good baseline and only override what matters for
 * the use-case.
 *
 * ──────────────────────────────────────────────────────────────────────────────
 * WHY THESE VALUES?
 * ──────────────────────────────────────────────────────────────────────────────
 *
 * Temperature   — controls randomness. 0 = greedy (always picks most likely
 *                 token), 1+ = increasingly creative/random.
 *
 * Top-P         — nucleus sampling. Only consider tokens whose cumulative
 *                 probability ≤ top_p. 0.9 = drop the long tail. 1.0 = disabled.
 *
 * Top-K         — hard cap on candidate tokens. -1 = disabled. 20–40 = good
 *                 for focused tasks; higher = more diverse.
 *
 * Min-P         — minimum probability threshold relative to the most likely
 *                 token. Cuts low-quality tokens even when top-p doesn't.
 *                 0.01–0.05 works well for coding; 0 disables.
 *
 * ──────────────────────────────────────────────────────────────────────────────
 * QWEN 3 / 3.5 SPECIFIC NOTES
 * ──────────────────────────────────────────────────────────────────────────────
 * Qwen3/3.5 have a thinking mode toggled by /think and /no_think tokens.
 * The "Thinking" toggle in the chat toolbar controls this automatically.
 *
 *   Thinking ON  → model reasons step-by-step inside <think>…</think>
 *                  before answering. Strongly recommended for coding, math,
 *                  multi-step reasoning, and agentic tasks.
 *
 *   Thinking OFF → model answers directly (no reasoning trace).
 *                  Better for fast chat, creative writing, simple Q&A.
 *
 * Alibaba recommended sampling for Qwen3/3.5:
 *   Non-thinking: temperature=0.7, top_p=0.8, repetition_penalty=1.05
 *   Thinking:     temperature=0.6, top_p=0.95, min_p=0.0
 *
 * ──────────────────────────────────────────────────────────────────────────────
 * DEEPSEEK R1 SPECIFIC NOTES
 * ──────────────────────────────────────────────────────────────────────────────
 * DeepSeek-R1 always reasons (thinks before answering). The model works best
 * with low temperature. Avoid top_k limits — they hurt long reasoning chains.
 *
 * ──────────────────────────────────────────────────────────────────────────────
 * LLAMA 3 / MISTRAL SPECIFIC NOTES
 * ──────────────────────────────────────────────────────────────────────────────
 * These models don't have thinking mode. Temperature 0.6–0.7 for chat,
 * 0.1–0.2 for coding. Presence penalty 1.05–1.1 helps avoid repetition.
 */

import type { SamplingParams } from "./tauri";

export interface Preset {
  /** Display label */
  label: string;
  /** Short emoji/icon */
  icon: string;
  /** One-line description shown in tooltip */
  description: string;
  /** Sampling parameters to apply */
  sampling: SamplingParams;
  /**
   * Whether to suggest enabling "Thinking" for models that support it
   * (Qwen3, Qwen3.5, DeepSeek-R1).
   * null = don't change the current thinking state.
   */
  suggestThinking: boolean | null;
}

export const PRESETS: Record<string, Preset> = {
  // ── Coding ───────────────────────────────────────────────────────────────
  // Near-greedy. Minimises hallucinated syntax/APIs. Best for:
  //   - Writing functions, classes, algorithms
  //   - Code review / explaining code
  //   - Debugging / error tracing
  // Thinking ON is strongly recommended for Qwen3/3.5 and R1 here.
  coding: {
    label: "Coding",
    icon: "⌨",
    description: "Near-greedy. Deterministic, low hallucination. Enable Thinking for best results on Qwen3/R1.",
    sampling: {
      temperature: 0.1,
      top_p: 0.95,
      top_k: 20,
      // min_p not in SamplingParams yet but leave room
    },
    suggestThinking: true,
  },

  // ── Chat ─────────────────────────────────────────────────────────────────
  // Balanced. Friendly and natural without being repetitive or random.
  // Alibaba's recommended non-thinking config for Qwen3/3.5.
  // Thinking OFF is recommended here for speed and naturalness.
  chat: {
    label: "Chat",
    icon: "💬",
    description: "Balanced creativity and coherence. Good for conversation and Q&A.",
    sampling: {
      temperature: 0.7,
      top_p: 0.8,
      top_k: 40,
    },
    suggestThinking: false,
  },

  // ── Creative ─────────────────────────────────────────────────────────────
  // Higher entropy. Good for:
  //   - Brainstorming, story writing, roleplay
  //   - Marketing copy, emails, social posts
  //   - Diverse idea generation
  creative: {
    label: "Creative",
    icon: "✨",
    description: "Higher temperature for diverse, imaginative, varied responses.",
    sampling: {
      temperature: 0.9,
      top_p: 0.95,
      top_k: 50,
    },
    suggestThinking: false,
  },

  // ── Precise ──────────────────────────────────────────────────────────────
  // Greedy decoding. Maximum determinism.
  // Best for:
  //   - Structured output (JSON, YAML, XML)
  //   - Classification / single correct answer tasks
  //   - Agentic tool calls where exact format matters
  //   - Repeatable results (e.g. tests, evaluations)
  // Thinking ON strongly recommended for Qwen3/3.5 and R1 on complex tasks.
  precise: {
    label: "Precise",
    icon: "🎯",
    description: "Greedy / fully deterministic. Best for structured output, JSON, classification.",
    sampling: {
      temperature: 0.0,
      top_p: 1.0,
      top_k: -1,
    },
    suggestThinking: true,
  },

  // ── Reasoning ────────────────────────────────────────────────────────────
  // Tuned for step-by-step reasoning tasks:
  //   - Math / proofs
  //   - Logic puzzles
  //   - Multi-step planning
  // Uses Alibaba's thinking-mode recommended values.
  // Thinking ON is required.
  reasoning: {
    label: "Reasoning",
    icon: "🧠",
    description: "Tuned for math, logic, and multi-step reasoning. Thinking mode ON.",
    sampling: {
      temperature: 0.6,
      top_p: 0.95,
      top_k: -1, // disable top_k for long reasoning chains
    },
    suggestThinking: true,
  },
} as const;

export type PresetKey = keyof typeof PRESETS;

/** Ordered list for display */
export const PRESET_ORDER: PresetKey[] = [
  "coding",
  "chat",
  "creative",
  "precise",
  "reasoning",
];

/**
 * Returns true if the model name suggests it has a thinking/reasoning mode
 * (Qwen3, Qwen3.5, DeepSeek R1, etc.).
 */
export function modelSupportsThinking(modelName: string | null | undefined): boolean {
  if (!modelName) return false;
  const m = modelName.toLowerCase();
  return (
    m.includes("qwen3") ||
    m.includes("qwen3.5") ||
    (m.includes("deepseek") && (m.includes("r1") || m.includes("reasoning")))
  );
}
