import {
  BrainCircuit,
  Eye,
  Infinity as InfinityIcon,
  Sparkles,
  Waves,
  Wrench,
} from "lucide-react";
import type { ModelInfo } from "../../lib/types";

export function modelPresentationKey(model: ModelInfo) {
  return `${model.provider_type}:${model.provider_base_url || model.provider_name}:${model.path || model.filename}`;
}

export function modelDisplayName(model: ModelInfo) {
  const source = model.hf_repo ?? model.filename.replace(/\.gguf$/i, "");
  const name = source.includes("/") ? source.split("/").pop() || source : source;
  return name.replace(/[-_]?GGUF$/i, "");
}

export function modelPublisher(model: ModelInfo) {
  if (model.hf_repo?.includes("/")) return model.hf_repo.split("/")[0];
  if (model.provider_name) return model.provider_name;
  return "Local";
}

export function modelParameterLabel(model: ModelInfo) {
  const text = `${model.filename} ${model.hf_repo ?? ""}`.toLowerCase();
  const match = text.match(/(?:^|[-_\s])(\d+(?:\.\d+)?)\s*(b|m)(?:[-_\s]|$)/);
  return match ? `${match[1]}${match[2].toUpperCase()}` : null;
}

export function modelContextLabel(model: ModelInfo) {
  const candidates = [model.context_window, model.max_context_window].filter(
    (value): value is number => value != null && value > 0,
  );
  const value = candidates.length > 0 ? Math.max(...candidates) : null;
  if (!value) return null;
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(value % 1_000_000 === 0 ? 0 : 1)}M`;
  }
  if (value >= 1024) {
    return `${(value / 1024).toFixed(value % 1024 === 0 ? 0 : 1)}K`;
  }
  return value.toLocaleString();
}

export function modelSummary(model: ModelInfo) {
  const family = model.family || model.gguf_architecture || "AI";
  const capabilities = [
    model.supports_vision ? "image input" : null,
    model.supports_reasoning ? "reasoning" : null,
    model.supports_tools ? "tool calling" : null,
  ].filter(Boolean);

  const capabilityText = capabilities.length > 0
    ? ` It advertises ${capabilities.join(", ")}.`
    : "";
  const runtimeText = model.provider_managed
    ? "It is stored locally and runs through the managed llama.cpp runtime."
    : `It is exposed by ${model.provider_name || "an external provider"}.`;

  return `${family} model. ${runtimeText}${capabilityText}`;
}

type ArtworkFamily = "gemma" | "qwen" | "llama" | "mistral" | "phi" | "deepseek" | "nvidia" | "generic";

function artworkFamily(model: ModelInfo): ArtworkFamily {
  const providerIdentity = model.provider_managed ? "" : model.provider_name;
  const identity = `${model.family} ${model.gguf_architecture ?? ""} ${model.filename} ${model.hf_repo ?? ""} ${providerIdentity}`.toLowerCase();
  if (identity.includes("gemma") || identity.includes("google")) return "gemma";
  if (identity.includes("qwen") || identity.includes("alibaba")) return "qwen";
  if (identity.includes("llama") || identity.includes("meta-llama")) return "llama";
  if (identity.includes("mistral") || identity.includes("mixtral")) return "mistral";
  if (identity.includes("phi") || identity.includes("microsoft")) return "phi";
  if (identity.includes("deepseek")) return "deepseek";
  if (identity.includes("nvidia") || identity.includes("nemotron")) return "nvidia";
  return "generic";
}

const ARTWORK: Record<ArtworkFamily, { label: string; foreground: string; background: string; border: string }> = {
  gemma: {
    label: "G",
    foreground: "#dbeafe",
    background: "linear-gradient(145deg, rgba(66,133,244,0.34), rgba(52,168,83,0.16) 46%, rgba(234,67,53,0.22))",
    border: "rgba(96,165,250,0.38)",
  },
  qwen: {
    label: "Q",
    foreground: "#e0e7ff",
    background: "linear-gradient(145deg, rgba(99,102,241,0.42), rgba(139,92,246,0.18))",
    border: "rgba(129,140,248,0.42)",
  },
  llama: {
    label: "L",
    foreground: "#dbeafe",
    background: "linear-gradient(145deg, rgba(37,99,235,0.38), rgba(14,165,233,0.16))",
    border: "rgba(56,189,248,0.38)",
  },
  mistral: {
    label: "M",
    foreground: "#ffedd5",
    background: "linear-gradient(145deg, rgba(249,115,22,0.42), rgba(234,179,8,0.16))",
    border: "rgba(251,146,60,0.42)",
  },
  phi: {
    label: "Φ",
    foreground: "#dbeafe",
    background: "linear-gradient(145deg, rgba(14,165,233,0.4), rgba(37,99,235,0.16))",
    border: "rgba(56,189,248,0.4)",
  },
  deepseek: {
    label: "D",
    foreground: "#dbeafe",
    background: "linear-gradient(145deg, rgba(59,130,246,0.4), rgba(34,211,238,0.14))",
    border: "rgba(96,165,250,0.42)",
  },
  nvidia: {
    label: "N",
    foreground: "#dcfce7",
    background: "linear-gradient(145deg, rgba(34,197,94,0.38), rgba(132,204,22,0.14))",
    border: "rgba(74,222,128,0.4)",
  },
  generic: {
    label: "AI",
    foreground: "#e5e7eb",
    background: "linear-gradient(145deg, rgba(148,163,184,0.24), rgba(71,85,105,0.14))",
    border: "rgba(148,163,184,0.3)",
  },
};

export function ModelArtwork({
  model,
  size = "md",
  className = "",
}: {
  model: ModelInfo;
  size?: "xs" | "sm" | "md" | "lg";
  className?: string;
}) {
  const family = artworkFamily(model);
  const theme = ARTWORK[family];
  const sizeClass = {
    xs: "h-6 w-6 rounded-md text-[9px]",
    sm: "h-8 w-8 rounded-lg text-[10px]",
    md: "h-11 w-11 rounded-xl text-xs",
    lg: "h-14 w-14 rounded-2xl text-base",
  }[size];
  const iconSize = size === "lg" ? 25 : size === "md" ? 20 : size === "sm" ? 16 : 13;

  return (
    <span
      aria-hidden="true"
      className={`relative flex shrink-0 items-center justify-center overflow-hidden font-bold tracking-[-0.04em] shadow-sm ${sizeClass} ${className}`}
      style={{
        background: theme.background,
        border: `1px solid ${theme.border}`,
        color: theme.foreground,
      }}
    >
      <span className="absolute inset-x-1 top-0 h-px bg-white/35" />
      {family === "qwen" ? (
        <Sparkles size={iconSize} strokeWidth={1.9} />
      ) : family === "llama" ? (
        <InfinityIcon size={iconSize} strokeWidth={2} />
      ) : family === "deepseek" ? (
        <Waves size={iconSize} strokeWidth={2} />
      ) : family === "nvidia" ? (
        <Eye size={iconSize} strokeWidth={2} />
      ) : (
        theme.label
      )}
    </span>
  );
}

export function CapabilityBadge({
  capability,
  ready = false,
  compact = false,
}: {
  capability: "vision" | "tools" | "reasoning";
  ready?: boolean;
  compact?: boolean;
}) {
  const spec = capability === "vision"
    ? { Icon: Eye, label: ready ? "Vision ready" : "Vision", color: ready ? "#6ee7b7" : "#fcd34d", background: ready ? "rgba(52,211,153,0.1)" : "rgba(245,158,11,0.1)" }
    : capability === "tools"
      ? { Icon: Wrench, label: "Tools", color: "#93c5fd", background: "rgba(59,130,246,0.1)" }
      : { Icon: BrainCircuit, label: "Reasoning", color: "#c4b5fd", background: "rgba(139,92,246,0.1)" };

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full font-medium ${compact ? "h-5 px-1.5 text-[9px]" : "h-6 px-2 text-[10px]"}`}
      style={{ color: spec.color, background: spec.background, border: `1px solid ${spec.color}33` }}
      title={spec.label}
    >
      <spec.Icon size={compact ? 10 : 11} />
      {!compact && spec.label}
    </span>
  );
}
