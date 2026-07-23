import { useEffect, useState, type ReactNode } from "react";
import {
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  Copy,
  Eye,
  FolderSync,
  HardDrive,
  Play,
  RefreshCw,
  Search,
  Settings2,
  Wrench,
  X,
} from "lucide-react";
import { useRef } from "react";
import type {
  AppSettings,
  LoadProgress,
  ModelInfo,
  ProcessStatusInfo,
} from "../../lib/types";
import { useGpuStats } from "../../hooks/useGpuStats";
import * as api from "../../lib/tauri";
import { parseCliArgs } from "../../lib/args";
import {
  describePromptRendering,
  defaultRecommendedLoadPreset,
  recommendedContextForModel,
  recommendedLoadPresets,
  replaceSamplingArgs,
  samplingArgs,
  samplingArgsMatch,
  stripStaleThinkingKwargs,
  type RecommendedLoadPreset,
} from "../../lib/modelLoadProfiles";
import type { LoadModelOptions } from "../../lib/tauri";
import { Button } from "../ui/Controls";
import { ModelArtwork } from "./modelPresentation";

interface Props {
  models: ModelInfo[];
  loadedModel: string | null;
  previousModel: string | null;
  processStatus: ProcessStatusInfo | null;
  settings: AppSettings | null;
  error: string | null;
  isLoading: boolean;
  loadProgress: LoadProgress | null;
  onUnload: () => void;
  onSwap: (modelName?: string, options?: LoadModelOptions) => void;
  onScan: () => void;
  onOpenSettings: () => void;
  onConfigureLoad: (model: ModelInfo, returnFocus: HTMLElement | null) => void;
}

const FILTERS = ["all", "reasoning", "tools", "vision", "loaded"] as const;
type FilterKey = (typeof FILTERS)[number];
export type LoadDialogMode = "load" | "swap" | "reload";

function buildServerUrl(settings: AppSettings | null) {
  if (!settings) return "http://127.0.0.1:8800/v1";
  const host = settings.server_host === "0.0.0.0" ? "127.0.0.1" : settings.server_host;
  return `http://${host}:${settings.server_port}/v1`;
}

function formatContext(contextWindow: number | null, fallback?: number | null) {
  const v = contextWindow ?? fallback ?? null;
  if (!v) return null; // hide entirely rather than show "?K"
  if (v >= 1024)
    return `${(v / 1024).toFixed(v % 1024 === 0 ? 0 : 1)}K ctx`;
  return `${v} ctx`;
}

function fmtToolFormat(f: string) {
  return f.replace("NativeApi", "Native API").replace("Xml", " XML");
}

function fmtNum(v: number | null | undefined) {
  if (v == null) return "n/a";
  return v.toLocaleString();
}

function advertisedContextLimit(model: ModelInfo) {
  const detected = [model.context_window, model.max_context_window].filter(
    (value): value is number => value != null && value > 0,
  );
  return detected.length > 0 ? Math.max(...detected) : 8192;
}

function safeDefaultContext(model: ModelInfo) {
  const advertised = advertisedContextLimit(model);
  if (!model.provider_managed) return advertised;

  const recommended = recommendedContextForModel(model, advertised);
  if (recommended != null) return recommended;

  const sizeGb = model.size_gb ?? 0;
  const safeCap = sizeGb <= 0 || sizeGb >= 18 ? 8192 : 16384;
  return Math.min(advertised, safeCap);
}

function safeContextPresets(model: ModelInfo) {
  const advertised = advertisedContextLimit(model);
  const presets = [8192, 16384, recommendedContextForModel(model, advertised)]
    .filter((value): value is number => value != null && value <= advertised)
    .filter((value, index, values) => values.indexOf(value) === index);
  return presets.length > 0 ? presets : [advertised];
}

function fmtSamplingValue(v: number | null | undefined) {
  if (v == null) return "n/a";
  return Number.isInteger(v) ? v.toString() : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function parseEditableNumber(value: string) {
  const trimmed = value.trim().toLowerCase();
  if (!trimmed || trimmed === "n/a" || trimmed === "-") return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function profileDefaultSummary(model: ModelInfo) {
  const parts = [
    `temp ${fmtSamplingValue(model.default_temperature)}`,
    `top-p ${fmtSamplingValue(model.default_top_p)}`,
    `top-k ${fmtSamplingValue(model.default_top_k)}`,
  ];
  if (model.default_min_p != null) parts.push(`min-p ${fmtSamplingValue(model.default_min_p)}`);
  if (model.default_presence_penalty != null) {
    parts.push(`presence ${fmtSamplingValue(model.default_presence_penalty)}`);
  }
  return parts.join(" | ");
}

interface SavedModelConfig {
  name: string;
  contextSize: number;
  fitMode: string;
  useJinja: boolean;
  reasoningMode: string;
  templateMode: string;
  chatTemplateKwargsJson: string;
  extraArgs: string;
}

interface RecommendedModelConfig {
  name: string;
  source: string;
  reasoningMode: string;
  templateMode?: string;
  useJinja?: boolean;
  chatTemplateKwargsJson?: string;
  extraArgs: string;
}

function buildArgs(values: {
  temp?: number | null;
  topP?: number | null;
  topK?: number | null;
  minP?: number | null;
  repeatPenalty?: number | null;
  presencePenalty?: number | null;
}) {
  const args: string[] = [];
  if (values.temp != null) args.push("--temp", fmtSamplingValue(values.temp));
  if (values.topP != null) args.push("--top-p", fmtSamplingValue(values.topP));
  if (values.topK != null) args.push("--top-k", fmtSamplingValue(values.topK));
  if (values.minP != null) args.push("--min-p", fmtSamplingValue(values.minP));
  if (values.repeatPenalty != null) {
    args.push("--repeat-penalty", fmtSamplingValue(values.repeatPenalty));
  }
  if (values.presencePenalty != null) {
    args.push("--presence-penalty", fmtSamplingValue(values.presencePenalty));
  }
  return args.join(" ");
}

function recommendedProfilesForModel(model: ModelInfo): RecommendedModelConfig[] {
  const family = model.family.toLowerCase();
  const name = model.filename.toLowerCase();
  const tessQwenProfiles = recommendedLoadPresets(model);

  if (tessQwenProfiles.length > 0) {
    return tessQwenProfiles.map((profile) => ({
      name: profile.name,
      source: "Tess / Qwen3.6 recommended settings",
      reasoningMode: profile.reasoningMode,
      templateMode: model.template_mode === "repo" ? "repo" : model.has_chat_template ? "builtin" : "repo",
      useJinja: true,
      chatTemplateKwargsJson: "",
      extraArgs: formatEditableArgs(samplingArgs(profile.sampling)),
    }));
  }

  if (family.includes("qwen") || name.includes("qwen")) {
    const isCoder = name.includes("coder");
    const codingArgs = isCoder
      ? buildArgs({ temp: 0.7, topP: 0.8, topK: 20, minP: 0, repeatPenalty: 1.05 })
      : buildArgs({ temp: 0.7, topP: 0.8, topK: 20, minP: 0, repeatPenalty: 1.0, presencePenalty: 0 });

    return [
      {
        name: "General",
        source: "Qwen/Unsloth non-thinking recommendation",
        reasoningMode: "off",
        templateMode: "repo",
        useJinja: true,
        chatTemplateKwargsJson: '{"preserve_thinking": false}',
        extraArgs: buildArgs({ temp: 0.7, topP: 0.8, topK: 20, minP: 0, repeatPenalty: 1.0, presencePenalty: 0 }),
      },
      {
        name: "Coding",
        source: isCoder ? "Qwen3-Coder recommendation" : "Qwen non-thinking conservative coding preset",
        reasoningMode: "off",
        templateMode: "repo",
        useJinja: true,
        chatTemplateKwargsJson: '{"preserve_thinking": false}',
        extraArgs: codingArgs,
      },
      {
        name: "Tools",
        source: "Strict tool/research preset",
        reasoningMode: "off",
        templateMode: "repo",
        useJinja: true,
        chatTemplateKwargsJson: '{"preserve_thinking": false}',
        extraArgs: buildArgs({ temp: 0.2, topP: 0.8, topK: 20, minP: 0, repeatPenalty: 1.0, presencePenalty: 0 }),
      },
      {
        name: "Thinking",
        source: "Qwen thinking recommendation",
        reasoningMode: "on",
        templateMode: "repo",
        useJinja: true,
        chatTemplateKwargsJson: '{"preserve_thinking": true}',
        extraArgs: buildArgs({ temp: 0.6, topP: 0.95, topK: 20, minP: 0, repeatPenalty: 1.0, presencePenalty: 0 }),
      },
    ];
  }

  const generalArgs = buildArgs({
    temp: model.default_temperature ?? 0.7,
    topP: model.default_top_p ?? 0.9,
    topK: model.default_top_k ?? -1,
    minP: model.default_min_p,
    presencePenalty: model.default_presence_penalty,
  });

  const profiles: RecommendedModelConfig[] = [
    {
      name: "General",
      source: "Detected model profile defaults",
      reasoningMode: model.supports_reasoning ? "auto" : "off",
      extraArgs: generalArgs,
    },
    {
      name: "Coding",
      source: "Conservative deterministic preset",
      reasoningMode: model.supports_reasoning ? "off" : "off",
      extraArgs: buildArgs({ temp: 0.2, topP: 0.9, topK: model.default_top_k ?? -1, minP: model.default_min_p }),
    },
    {
      name: "Tools",
      source: "Strict tool/research preset",
      reasoningMode: "off",
      extraArgs: buildArgs({ temp: 0.2, topP: 0.8, topK: model.default_top_k ?? -1, minP: model.default_min_p }),
    },
  ];

  if (model.supports_reasoning) {
    profiles.push({
      name: "Thinking",
      source: "Detected reasoning-capable model profile",
      reasoningMode: "on",
      extraArgs: generalArgs,
    });
  }

  return profiles;
}

function savedConfigKey(modelName: string) {
  return `inference-bridge:model-configs:${modelName}`;
}

// Panel wrapper

function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={className}
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
        borderRadius: "10px",
        overflow: "hidden",
      }}
    >
      {children}
    </div>
  );
}

function Divider() {
  return <div style={{ height: "1px", background: "var(--border)" }} />;
}

// VRAM bar
// Full bar = dedicated VRAM + system RAM (spill zone).
// Green fill = used dedicated VRAM. Amber zone = system RAM overflow area.
// Divider marks the boundary between dedicated (fast) and spill (slow) memory.

function VramBar({
  usedMb,
  dedicatedMb,
  systemRamMb,
  mode = "current",
}: {
  usedMb: number;
  dedicatedMb: number;
  systemRamMb: number;
  mode?: "current" | "estimate";
}) {
  const spillMb = Math.min(systemRamMb, dedicatedMb * 4); // cap spill zone at 4x VRAM
  const showSpill = mode === "estimate";
  const totalMb = showSpill ? dedicatedMb + spillMb : dedicatedMb;
  const usedPct = totalMb > 0 ? Math.min((usedMb / totalMb) * 100, 100) : 0;
  const dedicatedPct = showSpill && totalMb > 0 ? (dedicatedMb / totalMb) * 100 : 100;

  const usedGb = (usedMb / 1024).toFixed(1);
  const dedicatedGb = (dedicatedMb / 1024).toFixed(1);
  const spillGb = (spillMb / 1024).toFixed(0);

  // Fill colour: green while in dedicated zone, amber if spilling
  const fillColor = usedMb < dedicatedMb * 0.9 ? "#34d399" : "#f59e0b";
  const label = mode === "estimate" ? "Predicted" : "Current";
  const title =
    mode === "estimate"
      ? `Predicted model + KV memory: ${usedGb}/${dedicatedGb}GB dedicated | +${spillGb}GB RAM overflow zone`
      : `Current live GPU VRAM from nvidia-smi: ${usedGb}/${dedicatedGb}GB dedicated`;

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] uppercase tracking-widest whitespace-nowrap" style={{ color: "var(--text-2)" }}>
        {label}
      </span>
      <div
        className="relative rounded-full overflow-hidden"
        style={{ width: "110px", height: "6px", background: "var(--surface-3)" }}
        title={title}
      >
        {/* Spill zone (right portion = system RAM), always amber tint */}
        {showSpill && (
          <div
            style={{
              position: "absolute",
              left: `${dedicatedPct}%`,
              top: 0,
              height: "100%",
              width: `${100 - dedicatedPct}%`,
              background: "rgba(245,158,11,0.18)",
            }}
          />
        )}
        {/* Used VRAM fill */}
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            height: "100%",
            width: `${usedPct}%`,
            background: fillColor,
            transition: "width 0.4s ease",
          }}
        />
        {/* Divider at dedicated boundary */}
        {showSpill && (
          <div
            style={{
              position: "absolute",
              left: `${dedicatedPct}%`,
              top: 0,
              width: "1px",
              height: "100%",
              background: "rgba(255,255,255,0.3)",
            }}
          />
        )}
      </div>
      <span className="text-[10px] whitespace-nowrap tabular-nums" style={{ color: "var(--text-1)" }}>
        {usedGb}/{dedicatedGb}GB
      </span>
    </div>
  );
}

function PredictedVramBar({
  predictedMb,
  dedicatedMb,
}: {
  predictedMb: number;
  dedicatedMb: number;
}) {
  const ratio = dedicatedMb > 0 ? predictedMb / dedicatedMb : 0;
  const state = ratio > 1 ? "over" : ratio >= 0.85 ? "warning" : "safe";
  const color = state === "over" ? "#f87171" : state === "warning" ? "#f59e0b" : "#34d399";
  const label = state === "over" ? "Over VRAM" : state === "warning" ? "Near limit" : "Safe";
  const predictedGb = predictedMb / 1024;
  const dedicatedGb = dedicatedMb / 1024;
  const headroomGb = dedicatedGb - predictedGb;

  return (
    <div>
      <div className="flex items-center justify-between gap-3 text-[11px]">
        <span className="font-semibold" style={{ color: "var(--text-0)" }}>Predicted VRAM</span>
        <span className="tabular-nums" style={{ color }}>
          {predictedGb.toFixed(1)} / {dedicatedGb.toFixed(1)} GB · {label}
        </span>
      </div>
      <div
        className="mt-2 overflow-hidden rounded"
        style={{ height: 8, background: "var(--surface-3)", border: "1px solid var(--border)" }}
        title={`Estimated model, graph, and KV allocation: ${predictedGb.toFixed(1)} GB of ${dedicatedGb.toFixed(1)} GB dedicated VRAM`}
      >
        <div
          style={{
            width: `${Math.min(Math.max(ratio * 100, 0), 100)}%`,
            height: "100%",
            background: color,
            transition: "width 0.25s ease, background 0.25s ease",
          }}
        />
      </div>
      <div className="mt-1.5 text-[10px] leading-4" style={{ color: state === "safe" ? "var(--text-2)" : color }}>
        {state === "over"
          ? `${Math.abs(headroomGb).toFixed(1)} GB over dedicated VRAM; system-RAM spill or load failure is likely.`
          : `${headroomGb.toFixed(1)} GB predicted headroom before other GPU allocations.`}
      </div>
    </div>
  );
}

// Main component

export function ModelSelector({
  models,
  loadedModel,
  previousModel,
  processStatus,
  settings,
  error,
  isLoading,
  loadProgress,
  onUnload,
  onSwap,
  onScan,
  onOpenSettings,
  onConfigureLoad,
}: Props) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterKey>("all");
  const [copied, setCopied] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [mobileInspectorOpen, setMobileInspectorOpen] = useState(false);
  const [sidecarSyncing, setSidecarSyncing] = useState(false);
  const [sidecarSyncingModel, setSidecarSyncingModel] = useState<string | null>(null);
  const [sidecarSyncMessage, setSidecarSyncMessage] = useState<string | null>(null);
  const [sidecarStatuses, setSidecarStatuses] = useState<Record<string, api.HfSidecarCacheStatus>>({});
  const serverUrl = buildServerUrl(settings);
  const gpuStats = useGpuStats();
  const modelFingerprint = models.map((model) => model.filename).join("\n");

  useEffect(() => {
    if (!copied) return;
    const t = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(t);
  }, [copied]);

  useEffect(() => {
    if (!mobileInspectorOpen) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileInspectorOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [mobileInspectorOpen]);

  useEffect(() => {
    let cancelled = false;
    if (models.length === 0) {
      setSidecarStatuses({});
      return;
    }
    void api.getHfSidecarCacheStatus()
      .then((statuses) => {
        if (cancelled) return;
        setSidecarStatuses(Object.fromEntries(statuses.map((status) => [status.filename, status])));
      })
      .catch(() => {
        if (!cancelled) setSidecarStatuses({});
      });
    return () => {
      cancelled = true;
    };
  }, [models.length, modelFingerprint]);

  const filteredModels = models.filter((m) => {
    const q = query.trim().toLowerCase();
    const matchQ =
      !q ||
      m.filename.toLowerCase().includes(q) ||
      m.family.toLowerCase().includes(q) ||
      (m.quant ?? "").toLowerCase().includes(q) ||
      (m.hf_repo ?? "").toLowerCase().includes(q) ||
      m.provider_name.toLowerCase().includes(q);
    if (!matchQ) return false;
    if (filter === "reasoning") return m.supports_reasoning;
    if (filter === "tools") return m.supports_tools;
    if (filter === "vision") return m.supports_vision;
    if (filter === "loaded") return m.filename === loadedModel;
    return true;
  });

  useEffect(() => {
    if (selectedKey && !filteredModels.some((model) => modelKey(model) === selectedKey)) {
      setSelectedKey(null);
    }
  }, [filter, modelFingerprint, query, selectedKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const activeModel = loadedModel
    ? (models.find((m) => m.filename === loadedModel) ?? {
        filename: loadedModel,
        path: "",
        size_gb: 0,
        family: "Loaded via API",
        supports_tools: false,
        supports_parallel_tools: false,
        supports_reasoning: false,
        supports_vision: false,
        context_window: null,
        max_context_window: null,
        max_output_tokens: null,
        default_temperature: null,
        default_top_p: null,
        default_top_k: null,
        default_min_p: null,
        default_presence_penalty: null,
        quant: null,
        tool_call_format: "NativeApi",
        think_tag_style: "None",
        hf_repo: null,
        hf_file: null,
        template_mode: null,
        template_source: null,
        vision_runtime_ready: false,
        vision_status: "Unknown",
        mmproj_available: false,
        mmproj_candidate_path: null,
        provider_type: "managed_llamacpp",
        provider_name: "Managed llama.cpp",
        provider_base_url: null,
        provider_managed: true,
        n_layers: null,
        n_kv_heads: null,
        head_dim: null,
        gguf_architecture: null,
        has_chat_template: false,
      })
    : null;
  const selectedModel =
    filteredModels.find((m) => modelKey(m) === selectedKey) ??
    activeModel ??
    filteredModels[0] ??
    null;
  const localDiskGb = models
    .filter((m) => m.provider_managed)
    .reduce((sum, m) => sum + (m.size_gb || 0), 0);
  const availableHfUpdateCount = models.filter((model) => sidecarStatuses[model.filename]?.update_available).length;

  const handleCopyUrl = async () => {
    try {
      await navigator.clipboard.writeText(serverUrl);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const handleSyncSidecars = async () => {
    if (sidecarSyncing) return;
    setSidecarSyncing(true);
    setSidecarSyncMessage(null);
    try {
      const localModelNames = models
        .filter((model) => model.provider_managed && model.hf_repo)
        .map((model) => model.filename);
      const modelsWithUpdates = localModelNames.filter((name) => sidecarStatuses[name]?.update_available);
      const applying = modelsWithUpdates.length > 0;
      const summary = applying
        ? await api.syncHfSidecarCache(modelsWithUpdates)
        : await api.checkHfSidecarUpdates(localModelNames);
      const tokenHint = summary.hf_token_configured ? "HF token used" : "public access";
      setSidecarSyncMessage(
        applying
          ? `Updated ${summary.repos_updated} model source${summary.repos_updated === 1 ? "" : "s"}: ${summary.files_updated} small files changed, ${summary.files_failed} failed (${tokenHint}). Reload an active model to use a new template.`
          : summary.repos_with_updates > 0
            ? `${summary.repos_with_updates} model source update${summary.repos_with_updates === 1 ? " is" : "s are"} ready. Review the model status, then choose Update HF files to apply (${tokenHint}).`
            : `Templates and model metadata are current across ${summary.repos_checked} source${summary.repos_checked === 1 ? "" : "s"} (${tokenHint}).`
      );
      const statuses = await api.getHfSidecarCacheStatus();
      setSidecarStatuses(Object.fromEntries(statuses.map((status) => [status.filename, status])));
    } catch (error) {
      setSidecarSyncMessage(`HF sidecar sync failed: ${String(error)}`);
    } finally {
      setSidecarSyncing(false);
    }
  };

  const handleSyncModelSidecars = async (model: ModelInfo) => {
    if (sidecarSyncingModel || !model.hf_repo) return;
    setSidecarSyncingModel(model.filename);
    setSidecarSyncMessage(null);
    try {
      const applying = !!sidecarStatuses[model.filename]?.update_available;
      const summary = applying
        ? await api.syncHfSidecarCache([model.filename])
        : await api.checkHfSidecarUpdates([model.filename]);
      setSidecarSyncMessage(
        applying
          ? `${shortModelName(model)} updated from ${summary.repos_updated} HF source: ${summary.files_updated} small files changed${summary.files_failed ? `, ${summary.files_failed} failed` : ""}. Reload the model to use a changed template.`
          : summary.repos_with_updates > 0
            ? `${shortModelName(model)} has a template or metadata update ready. Choose Update HF files to apply it.`
            : `${shortModelName(model)} templates and metadata are current.`
      );
      const statuses = await api.getHfSidecarCacheStatus();
      setSidecarStatuses(Object.fromEntries(statuses.map((status) => [status.filename, status])));
    } catch (error) {
      setSidecarSyncMessage(`HF sidecar sync failed for ${shortModelName(model)}: ${String(error)}`);
    } finally {
      setSidecarSyncingModel(null);
    }
  };

  const handleRollbackModelSidecars = async (model: ModelInfo) => {
    if (sidecarSyncingModel || !sidecarStatuses[model.filename]?.rollback_available) return;
    if (!window.confirm(`Restore the previous template and metadata snapshot for ${modelDisplayName(model)}?`)) return;
    setSidecarSyncingModel(model.filename);
    setSidecarSyncMessage(null);
    try {
      const summary = await api.rollbackHfSidecarCache(model.filename);
      setSidecarSyncMessage(
        `${shortModelName(model)} restored its previous HF snapshot (${summary.files_restored} metadata files${summary.template_restored ? " and chat template" : ""}). Reload the model to use it.`
      );
      const statuses = await api.getHfSidecarCacheStatus();
      setSidecarStatuses(Object.fromEntries(statuses.map((status) => [status.filename, status])));
    } catch (error) {
      setSidecarSyncMessage(`HF rollback failed for ${shortModelName(model)}: ${String(error)}`);
    } finally {
      setSidecarSyncingModel(null);
    }
  };

  const state = processStatus?.state ?? "Idle";
  const apiState = processStatus?.api_state ?? "Idle";
  const apiRunning = apiState === "Running" || !!processStatus?.api_reachable;

  const openLoadDialog = (model: ModelInfo) => {
    const activeElement = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const persistentModelRow = mobileInspectorOpen
      ? Array.from(document.querySelectorAll<HTMLElement>("[data-model-focus-key]")).find(
          (element) => element.dataset.modelFocusKey === modelKey(model),
        ) ?? null
      : null;
    setSelectedKey(modelKey(model));
    setMobileInspectorOpen(false);
    onConfigureLoad(
      model,
      persistentModelRow ?? activeElement,
    );
  };

  return (
    <div className="relative flex h-full min-h-0 overflow-hidden" style={{ background: "var(--bg)" }}>
      <aside className="hidden w-[180px] shrink-0 border-r 2xl:block" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
        <div className="px-4 py-4">
          <div className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>My Models</div>
          <div className="mt-4 space-y-1">
            {FILTERS.map((key) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                aria-pressed={filter === key}
                className="flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm transition"
                style={{
                  background: filter === key ? "rgba(255,255,255,0.10)" : "transparent",
                  color: filter === key ? "var(--text-0)" : "var(--text-1)",
                  fontWeight: filter === key ? 600 : 400,
                  boxShadow: "none",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                <span>{key === "all" ? "View All" : key[0].toUpperCase() + key.slice(1)}</span>
                <span className="text-[10px]" style={{ opacity: 0.72 }}>
                  {key === "all"
                    ? models.length
                    : key === "loaded"
                      ? activeModel ? 1 : 0
                      : models.filter((m) =>
                          key === "reasoning" ? m.supports_reasoning :
                          key === "tools" ? m.supports_tools :
                          key === "vision" ? m.supports_vision : true
                        ).length}
                </span>
              </button>
            ))}
          </div>
        </div>
      </aside>

      <section className="ib-model-table flex min-h-0 min-w-0 flex-1 flex-col border-r" style={{ borderColor: "var(--border)" }}>
        <div className="flex min-h-12 shrink-0 items-center gap-2 border-b px-4 py-2" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>Local models</h2>
            <p className="text-[10px]" style={{ color: "var(--text-2)" }}>{models.length} discovered</p>
          </div>
          <div className="ml-auto flex min-w-0 items-center gap-2">
            <label className="relative w-[360px] max-w-[45vw]">
              <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: "var(--text-2)" }}>
                <Search size={14} />
              </span>
              <input
                type="text"
                aria-label="Filter local models"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter local models"
                className="w-full rounded-md py-1.5 pl-8 pr-3 text-sm outline-none transition"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border-mid)", color: "var(--text-0)" }}
              />
            </label>
            <ToolBtn onClick={onOpenSettings} icon={<Settings2 size={14} />} label="Model settings" />
            <button
              onClick={() => void handleSyncSidecars()}
              disabled={sidecarSyncing || models.length === 0}
              title={availableHfUpdateCount > 0 ? `Apply ${availableHfUpdateCount} checked HF update${availableHfUpdateCount === 1 ? "" : "s"}` : "Check for small template and metadata updates"}
              className="hidden items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 lg:flex"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
            >
              <FolderSync size={13} className={sidecarSyncing ? "animate-pulse" : ""} />
              {sidecarSyncing ? "Working" : availableHfUpdateCount > 0 ? `Update ${availableHfUpdateCount}` : "Check HF"}
            </button>
            <button
              onClick={onScan}
              disabled={isLoading}
              className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-50"
              style={{ background: "#f4f4f4", color: "#171717", border: "none" }}
            >
              <RefreshCw size={13} className={isLoading && !loadProgress ? "animate-spin" : ""} />
              {isLoading && !loadProgress ? "Scanning..." : "Scan"}
            </button>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1 overflow-x-auto border-b px-3 py-1.5 2xl:hidden" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
          {FILTERS.map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => setFilter(key)}
              aria-pressed={filter === key}
              className="shrink-0 rounded-md px-2.5 py-1 text-[11px] font-medium transition"
              style={{
                background: filter === key ? "var(--surface-3)" : "transparent",
                border: filter === key ? "1px solid var(--border-mid)" : "1px solid transparent",
                color: filter === key ? "var(--text-0)" : "var(--text-2)",
              }}
            >
              {key === "all" ? "All" : key === "loaded" ? "Active" : key[0].toUpperCase() + key.slice(1)}
            </button>
          ))}
        </div>

        {(error || loadProgress) && (
          <div className="shrink-0 border-b" style={{ borderColor: "var(--border)" }}>
            {error && error !== loadProgress?.error && (
              <>
                <div className="px-4 py-2 text-sm" style={{ background: "rgba(239,68,68,0.08)", color: "#fca5a5" }}>{error}</div>
                <LoadErrorHint message={error} />
              </>
            )}
            {loadProgress && !loadProgress.done && <LoadingBar progress={loadProgress} />}
            {loadProgress?.error && (
              <>
                <div className="px-4 py-2 text-sm" style={{ background: "rgba(239,68,68,0.08)", color: "#fca5a5" }}>{loadProgress.error}</div>
                <LoadErrorHint message={loadProgress.error} />
              </>
            )}
          </div>
        )}
        {sidecarSyncMessage && (
          <div className="shrink-0 border-b px-4 py-2 text-xs" style={{ borderColor: "var(--border)", background: "rgba(255,255,255,0.05)", color: "var(--text-1)" }}>
            {sidecarSyncMessage} Only small allowlisted template/config files are fetched; model weights are blocked.
          </div>
        )}

        <div className="ib-model-grid h-9 shrink-0 items-center border-b px-4 text-[10px] font-semibold uppercase tracking-[0.12em]" style={{ borderColor: "var(--border)", color: "var(--text-2)", background: "var(--surface-1)" }}>
          <span>Model</span>
          <span className="ib-model-col-arch">Arch</span>
          <span className="ib-model-col-params">Params</span>
          <span className="ib-model-col-publisher">Publisher</span>
          <span>Quant</span>
          <span className="ib-model-col-size">Size</span>
          <span className="text-right">Actions</span>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {models.length === 0 ? (
            <EmptyMsg fill title="No models discovered yet" body="Set model directories in Settings then scan to populate the library." />
          ) : filter === "loaded" && !activeModel ? (
            <EmptyMsg fill title="No model loaded" body="Load a model to see it here." />
          ) : filteredModels.length === 0 ? (
            <EmptyMsg fill title="No matches" body="Clear the search or change the capability filter." />
          ) : (
            filteredModels.map((m) => (
              <DenseModelRow
                key={modelKey(m)}
                model={m}
                selected={selectedModel ? modelKey(selectedModel) === modelKey(m) : false}
                loaded={m.filename === loadedModel}
                isLoading={isLoading}
                showSwap={!!loadedModel && m.filename !== loadedModel}
                sidecarStatus={sidecarStatuses[m.filename]}
                sidecarSyncing={sidecarSyncingModel === m.filename}
                onSelect={() => {
                  setSelectedKey(modelKey(m));
                  setMobileInspectorOpen(true);
                }}
                onLoad={() => openLoadDialog(m)}
                onSwap={() => openLoadDialog(m)}
                onSyncSidecars={() => void handleSyncModelSidecars(m)}
              />
            ))
          )}
        </div>

        <div className="flex h-9 shrink-0 items-center gap-2 border-t px-4 text-[11px]" style={{ borderColor: "var(--border)", color: "var(--text-2)", background: "var(--surface-1)" }}>
          <HardDrive size={13} />
          <span>{models.filter((entry) => entry.provider_managed).length} local · {localDiskGb.toFixed(2)} GB</span>
          <span className="ml-auto hidden items-center gap-1.5 truncate font-mono lg:flex" title={serverUrl}>
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: apiRunning ? "#34d399" : "var(--text-3)" }} />
            {serverUrl}
          </span>
        </div>
      </section>

      <aside className="hidden min-h-0 w-[300px] shrink-0 flex-col lg:flex xl:w-[320px] 2xl:w-[360px]" style={{ background: "var(--surface-1)" }}>
        <div className="border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-2">
            <div className="min-w-0 flex-1">
              <div className="text-xs font-semibold" style={{ color: "var(--text-0)" }}>Model details</div>
              <div className="mt-0.5 flex items-center gap-1.5 text-[10px]" style={{ color: "var(--text-2)" }}>
                <StatusPill state={selectedModel?.filename === loadedModel ? state : "Idle"} />
                <span>{selectedModel?.filename === loadedModel ? "Active runtime" : selectedModel ? "Ready to load" : "Select a model"}</span>
              </div>
            </div>
            <ToolBtn onClick={handleCopyUrl} icon={<Copy size={14} />} label={copied ? "Copied endpoint" : "Copy endpoint"} />
          </div>
          {gpuStats && (
            <div className="mt-3">
              <VramBar usedMb={gpuStats.used_mb} dedicatedMb={gpuStats.dedicated_mb} systemRamMb={gpuStats.system_ram_mb} mode="current" />
            </div>
          )}
        </div>

        <ModelInspector
          model={selectedModel}
          loadedModel={loadedModel}
          previousModel={previousModel}
          processStatus={processStatus}
          sidecarStatus={selectedModel ? sidecarStatuses[selectedModel.filename] : undefined}
          sidecarSyncing={selectedModel ? sidecarSyncingModel === selectedModel.filename : false}
          isLoading={isLoading}
          onConfigureLoad={openLoadDialog}
          onUnload={onUnload}
          onSwapBack={() => {
            const previous = models.find((model) => model.filename === previousModel);
            if (previous) openLoadDialog(previous);
            else onSwap();
          }}
          onSyncSidecars={(model) => void handleSyncModelSidecars(model)}
          onRollbackSidecars={(model) => void handleRollbackModelSidecars(model)}
        />
      </aside>

      {mobileInspectorOpen && selectedModel && (
        <div className="absolute inset-0 z-30 flex justify-end lg:hidden" role="dialog" aria-modal="true" aria-label="Model details">
          <button
            type="button"
            aria-label="Close model details"
            className="absolute inset-0 bg-black/60"
            onClick={() => setMobileInspectorOpen(false)}
          />
          <aside className="relative z-10 flex h-full w-[min(92vw,380px)] min-h-0 flex-col shadow-2xl" style={{ background: "var(--surface-1)", borderLeft: "1px solid var(--border)" }}>
            <div className="flex h-11 shrink-0 items-center border-b px-3" style={{ borderColor: "var(--border)" }}>
              <span className="text-xs font-semibold" style={{ color: "var(--text-0)" }}>Model details</span>
              <button
                type="button"
                onClick={() => setMobileInspectorOpen(false)}
                className="ml-auto flex h-8 w-8 items-center justify-center rounded-lg transition hover:bg-white/5"
                aria-label="Close model details"
                style={{ color: "var(--text-1)" }}
              >
                <X size={15} />
              </button>
            </div>
            <ModelInspector
              model={selectedModel}
              loadedModel={loadedModel}
              previousModel={previousModel}
              processStatus={processStatus}
              sidecarStatus={sidecarStatuses[selectedModel.filename]}
              sidecarSyncing={sidecarSyncingModel === selectedModel.filename}
              isLoading={isLoading}
              onConfigureLoad={openLoadDialog}
              onUnload={onUnload}
              onSwapBack={() => {
                const previous = models.find((model) => model.filename === previousModel);
                if (previous) openLoadDialog(previous);
                else onSwap();
              }}
              onSyncSidecars={(model) => void handleSyncModelSidecars(model)}
              onRollbackSidecars={(model) => void handleRollbackModelSidecars(model)}
            />
          </aside>
        </div>
      )}

    </div>
  );

}

// Status pill

function modelKey(model: ModelInfo) {
  return `${model.provider_type}:${model.provider_base_url || model.provider_name}:${model.path || model.filename}`;
}

function modelParamsLabel(model: ModelInfo) {
  const text = `${model.filename} ${model.hf_repo ?? ""}`.toLowerCase();
  const match = text.match(/(\d+(?:\.\d+)?)\s*(b|m)\b/);
  return match ? `${match[1]}${match[2].toUpperCase()}` : "-";
}

function modelPublisher(model: ModelInfo) {
  if (model.hf_repo?.includes("/")) return model.hf_repo.split("/")[0];
  if (model.provider_name) return model.provider_name;
  return "local";
}

function shortModelName(model: ModelInfo) {
  return model.hf_repo ?? model.filename.replace(/\.gguf$/i, "");
}

function modelDisplayName(model: ModelInfo) {
  const source = shortModelName(model);
  const name = source.includes("/") ? source.split("/").pop() || source : source;
  return name.replace(/[-_]?GGUF$/i, "");
}

function hfCacheSummary(status: api.HfSidecarCacheStatus | undefined) {
  if (!status?.repo_id) return "No HF repo";
  const template = status.template_cached ? "template cached" : "template missing";
  const update = status.update_available ? "update ready" : status.last_checked_at ? "current" : "not checked";
  return `${update}, ${template}, ${status.sidecar_cached_count}/${status.sidecar_expected_count} sidecars`;
}

function shortHfRevision(revision: string | null | undefined) {
  if (!revision) return "-";
  if (revision.startsWith("local-cache")) return "Local backup";
  return revision.slice(0, 12);
}

function DenseModelRow({
  model,
  selected,
  loaded,
  isLoading,
  showSwap,
  sidecarStatus,
  sidecarSyncing,
  onSelect,
  onLoad,
  onSwap,
  onSyncSidecars,
}: {
  model: ModelInfo;
  selected: boolean;
  loaded: boolean;
  isLoading: boolean;
  showSwap: boolean;
  sidecarStatus?: api.HfSidecarCacheStatus;
  sidecarSyncing: boolean;
  onSelect: () => void;
  onLoad: () => void;
  onSwap: () => void;
  onSyncSidecars: () => void;
}) {
  const contextLabel = formatContext(model.context_window, model.max_context_window);

  return (
    <div
      onClick={onSelect}
      className="ib-model-grid ib-model-row min-h-[58px] items-center border-b px-4 text-xs transition"
      style={{
        borderColor: "var(--border)",
        background: selected ? "rgba(255,255,255,0.07)" : loaded ? "rgba(52,211,153,0.045)" : "transparent",
        color: "var(--text-1)",
        boxShadow: selected ? "inset 2px 0 0 rgba(255,255,255,0.42)" : loaded ? "inset 2px 0 0 rgba(52,211,153,0.5)" : "none",
        cursor: "pointer",
      }}
    >
      <button
        type="button"
        data-model-focus-key={modelKey(model)}
        onClick={(event) => {
          event.stopPropagation();
          onSelect();
        }}
        aria-pressed={selected}
        className="flex min-w-0 items-center gap-2.5 text-left"
        style={{ background: "transparent", border: 0 }}
      >
        <ModelArtwork model={model} size="sm" />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <span className="truncate text-xs font-semibold" style={{ color: "var(--text-0)" }} title={shortModelName(model)}>
              {modelDisplayName(model)}
            </span>
            {loaded && <CheckCircle2 size={13} className="shrink-0 text-emerald-400" aria-label="Loaded" />}
          </div>
          <div className="mt-1 flex min-w-0 items-center gap-1.5 text-[10px]" style={{ color: "var(--text-2)" }}>
            <span className="truncate" title={model.filename}>{model.filename}</span>
            {contextLabel && <span className="shrink-0">· {contextLabel}</span>}
            {model.supports_reasoning && <BrainCircuit size={11} className="shrink-0" aria-label="Reasoning" />}
            {model.supports_tools && <Wrench size={11} className="shrink-0" aria-label="Tools" />}
            {model.supports_vision && <Eye size={11} className="shrink-0" aria-label="Vision" />}
          </div>
        </div>
      </button>
      <span className="ib-model-col-arch truncate font-mono text-[11px]" title={model.family || model.gguf_architecture || "GGUF"}>{model.family || model.gguf_architecture || "GGUF"}</span>
      <span className="ib-model-col-params font-mono text-[11px]">{modelParamsLabel(model)}</span>
      <span className="ib-model-col-publisher truncate text-[11px]" title={modelPublisher(model)}>{modelPublisher(model)}</span>
      <span className="justify-self-start rounded px-1.5 py-0.5 font-mono text-[10px]" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}>{model.quant ?? "-"}</span>
      <span className="ib-model-col-size tabular-nums text-[11px]" style={{ color: "var(--text-1)" }}>{model.size_gb > 0 ? `${model.size_gb.toFixed(1)} GB` : "—"}</span>
      <div className="flex justify-end gap-1.5">
        {model.hf_repo && (
          <button
            onClick={(e) => { e.stopPropagation(); onSyncSidecars(); }}
            disabled={isLoading || sidecarSyncing}
            className="flex h-7 w-7 items-center justify-center rounded-md disabled:opacity-45"
            aria-label={sidecarStatus?.update_available ? "Apply Hugging Face template and metadata update" : "Check Hugging Face template and metadata updates"}
            title={hfCacheSummary(sidecarStatus)}
            style={{ background: sidecarStatus?.update_available ? "rgba(245,158,11,0.12)" : "var(--surface-2)", color: sidecarStatus?.update_available ? "#fbbf24" : "var(--text-1)", border: sidecarStatus?.update_available ? "1px solid rgba(245,158,11,0.32)" : "1px solid var(--border)" }}
          >
            <FolderSync size={13} className={sidecarSyncing ? "animate-pulse" : ""} />
          </button>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); showSwap ? onSwap() : onLoad(); }}
          disabled={isLoading || !model.provider_managed || loaded}
          className="flex h-7 items-center gap-1.5 rounded-md px-2 text-[11px] font-semibold disabled:opacity-45"
          aria-label={!model.provider_managed ? "This model is routed by an external provider" : loaded ? "Model is active" : showSwap ? `Open switch options for ${modelDisplayName(model)}` : `Open load options for ${modelDisplayName(model)}`}
          title={!model.provider_managed ? "This model is routed by an external provider" : loaded ? "Model is active" : showSwap ? "Open switch options" : "Open load options"}
          style={{ background: loaded || !model.provider_managed ? "var(--surface-2)" : "#f4f4f4", color: loaded || !model.provider_managed ? "var(--text-2)" : "#171717", border: loaded || !model.provider_managed ? "1px solid var(--border)" : "none" }}
        >
          {loaded ? <CheckCircle2 size={12} /> : showSwap ? <RefreshCw size={12} /> : <Play size={12} />}
          <span className="ib-model-action-label">{loaded ? "Active" : !model.provider_managed ? "Routed" : showSwap ? "Switch…" : "Load…"}</span>
        </button>
      </div>
    </div>
  );
}

interface ModelLoadConfig {
  contextSize: number;
  gpuLayers: number;
  threads: number;
  threadsBatch: number;
  batchSize: number;
  ubatchSize: number;
  parallelSlots: number;
  flashAttn: boolean;
  useMmap: boolean;
  useMlock: boolean;
  contBatching: boolean;
  kvUnified: boolean;
  noWarmup: boolean;
  ctxShift: boolean;
  mainGpu: number;
  defragThold: number;
  ropeFreqScale: number;
  cacheTypeK: string;
  cacheTypeV: string;
  fitMode: string;
  cacheRamMb: number;
  ctxcp: number;
  useJinja: boolean;
  reasoningMode: string;
  reasoningPreserve: boolean;
  templateMode: string;
  templateName: string;
  customTemplatePath: string;
  chatTemplateKwargsJson: string;
  attachMmproj: boolean;
  speculativeEnabled: boolean;
  draftModelPath: string;
  specType: string;
  specDraftNMax: number;
  draftMaxTokens: number;
  draftMinTokens: number;
  draftPMin: number;
  extraArgs: string;
}

interface StoredModelLoadConfig {
  version: 1;
  config: ModelLoadConfig;
}

const CONTROLLED_LOAD_FLAGS = new Set([
  "-m", "--model", "-hf", "--port", "--host", "-c", "--ctx-size", "-np", "--parallel", "--slots",
  "-ngl", "--n-gpu-layers", "-t", "--threads", "-tb", "--threads-batch", "-b", "--batch-size", "-ub", "--ubatch-size",
  "-fa", "--flash-attn", "--no-mmap", "--mlock", "-cb", "--cont-batching", "--main-gpu",
  "--defrag-thold", "--rope-freq-scale", "--cache-type-k", "--cache-type-v",
  "--kv-unified", "--no-warmup", "--ctx-shift", "--fit", "--cache-ram", "-ctxcp",
  "--jinja", "--reasoning", "--reasoning-preserve", "--chat-template",
  "--chat-template-file", "--chat-template-kwargs", "--mmproj", "-md", "--spec-type",
  "--spec-draft-n-max", "--draft-max", "--draft-min", "--draft-p-min",
]);

const CONTROLLED_VALUE_FLAGS = new Set([
  "-m", "--model", "-hf", "--port", "--host", "-c", "--ctx-size", "-np", "--parallel",
  "-ngl", "--n-gpu-layers", "-t", "--threads", "-tb", "--threads-batch", "-b", "--batch-size", "-ub", "--ubatch-size",
  "-fa", "--flash-attn", "--main-gpu", "--defrag-thold", "--rope-freq-scale",
  "--cache-type-k", "--cache-type-v", "--fit", "--cache-ram", "-ctxcp",
  "--reasoning", "--chat-template", "--chat-template-file", "--chat-template-kwargs",
  "--mmproj", "-md", "--spec-type", "--spec-draft-n-max", "--draft-max",
  "--draft-min", "--draft-p-min",
]);

function loadDialogStorageKey(model: ModelInfo) {
  return `inference-bridge:model-load:${modelKey(model)}`;
}

function readArgNumber(args: string[], flag: string) {
  const index = args.indexOf(flag);
  if (index < 0 || index + 1 >= args.length) return null;
  const value = Number(args[index + 1]);
  return Number.isFinite(value) ? value : null;
}

function readArgString(args: string[], flag: string) {
  const index = args.indexOf(flag);
  return index >= 0 && index + 1 < args.length ? args[index + 1] : null;
}

function formatEditableArgs(args: string[]) {
  return args.map((value) => /[\s,]/.test(value) ? JSON.stringify(value) : value).join(" ");
}

function extractPreviewExtraArgs(args: string[]) {
  const extra: string[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    const flag = arg.split("=")[0];
    if (CONTROLLED_LOAD_FLAGS.has(flag)) {
      if (!arg.includes("=") && CONTROLLED_VALUE_FLAGS.has(flag)) index += 1;
      continue;
    }
    extra.push(arg);
  }
  return extra;
}

function estimateModelLoadMemory(model: ModelInfo, contextSize: number, gpuLayers: number) {
  const modelMb = Math.max(0, model.size_gb || 0) * 1024;
  const residentModelMb = modelMb * 0.96;
  const graphMb = Math.max(256, Math.min(1024, modelMb * 0.03));
  const kvBytesPerToken =
    model.n_layers != null && model.n_kv_heads != null && model.head_dim != null
      ? model.n_layers * 2 * model.n_kv_heads * model.head_dim
      : null;
  const kvMb = kvBytesPerToken != null
    ? (contextSize * kvBytesPerToken * 1.15) / (1024 * 1024)
    : contextSize * (modelMb > 10 * 1024 ? 0.035 : 0.025);
  const maxLayers = Math.max(1, model.n_layers ?? 80);
  const offloadRatio = gpuLayers < 0 ? 1 : Math.max(0, Math.min(1, gpuLayers / maxLayers));
  const totalMb = modelMb + graphMb + kvMb;
  const gpuMb = offloadRatio > 0 ? residentModelMb * offloadRatio + graphMb + kvMb : 0;
  return { gpuMb, totalMb };
}

function defaultModelLoadConfig(
  model: ModelInfo,
  settings: AppSettings | null,
  processStatus: ProcessStatusInfo | null,
): ModelLoadConfig {
  const preview = processStatus?.model === model.filename ? processStatus.last_launch_preview : null;
  const args = preview?.args ?? [];
  const gpuArg = readArgNumber(args, "--n-gpu-layers");
  const hasPreview = !!preview;
  const recommendedPreset = defaultRecommendedLoadPreset(model);
  const configuredExtraArgs = preview
    ? extractPreviewExtraArgs(args)
    : settings?.extra_args ?? [];
  const effectiveExtraArgs = !preview && recommendedPreset
    ? replaceSamplingArgs(configuredExtraArgs, recommendedPreset.sampling)
    : configuredExtraArgs;
  const configuredTemplateKwargs =
    preview?.chat_template_kwargs_json ?? settings?.chat_template_kwargs_json ?? "";
  const templateKwargs = stripStaleThinkingKwargs(configuredTemplateKwargs).value;
  const getPreviewNumber = (flag: string, fallback: number) =>
    hasPreview ? (readArgNumber(args, flag) ?? 0) : fallback;

  return {
    contextSize: preview?.context_size ?? safeDefaultContext(model),
    gpuLayers: gpuArg === 999 ? -1 : gpuArg ?? settings?.gpu_layers ?? -1,
    threads: getPreviewNumber("--threads", settings?.threads ?? 0),
    threadsBatch: getPreviewNumber("--threads-batch", settings?.threads_batch ?? 0),
    batchSize: getPreviewNumber("--batch-size", settings?.batch_size ?? 0),
    ubatchSize: getPreviewNumber("--ubatch-size", settings?.ubatch_size ?? 0),
    parallelSlots: Math.max(1, preview?.parallel_slots ?? recommendedPreset?.parallelSlots ?? settings?.parallel_slots ?? 1),
    flashAttn: hasPreview ? args.includes("--flash-attn") : settings?.flash_attn ?? true,
    useMmap: hasPreview ? !args.includes("--no-mmap") : settings?.use_mmap ?? true,
    useMlock: hasPreview ? args.includes("--mlock") : settings?.use_mlock ?? false,
    contBatching: hasPreview ? args.includes("--cont-batching") : settings?.cont_batching ?? true,
    kvUnified: hasPreview ? args.includes("--kv-unified") : true,
    noWarmup: hasPreview ? args.includes("--no-warmup") : false,
    ctxShift: hasPreview ? args.includes("--ctx-shift") : false,
    mainGpu: getPreviewNumber("--main-gpu", settings?.main_gpu ?? 0),
    defragThold: getPreviewNumber("--defrag-thold", settings?.defrag_thold ?? 0.1),
    ropeFreqScale: getPreviewNumber("--rope-freq-scale", settings?.rope_freq_scale ?? 0),
    cacheTypeK: hasPreview ? (readArgString(args, "--cache-type-k") ?? "q8_0") : "q8_0",
    cacheTypeV: hasPreview ? (readArgString(args, "--cache-type-v") ?? "q8_0") : "q8_0",
    fitMode: preview?.fit_mode ?? settings?.fit_mode ?? "",
    cacheRamMb: preview?.cache_ram_mb ?? settings?.cache_ram_mb ?? 0,
    ctxcp: preview?.ctxcp ?? settings?.ctxcp ?? 0,
    useJinja: preview?.use_jinja ?? (recommendedPreset ? true : settings?.use_jinja ?? model.has_chat_template),
    reasoningMode: preview?.reasoning_mode ?? recommendedPreset?.reasoningMode ?? (model.supports_reasoning ? settings?.reasoning_mode || "auto" : "off"),
    reasoningPreserve: preview?.reasoning_preserve ?? (recommendedPreset ? false : settings?.reasoning_preserve ?? false),
    templateMode: preview?.template_mode ?? (model.template_mode === "repo" ? "repo" : recommendedPreset && model.has_chat_template ? "builtin" : model.template_mode ?? settings?.template_mode ?? "repo"),
    templateName: preview?.template_name ?? settings?.template_name ?? "",
    customTemplatePath: preview?.template_mode === "custom" ? preview.template_path ?? settings?.custom_template_path ?? "" : settings?.custom_template_path ?? "",
    chatTemplateKwargsJson: templateKwargs,
    attachMmproj: preview ? !!preview.mmproj_path : model.supports_vision,
    speculativeEnabled: !!(preview?.spec_type || preview?.draft_model_path || settings?.spec_type || settings?.draft_model_path),
    draftModelPath: preview?.draft_model_path ?? settings?.draft_model_path ?? "",
    specType: preview?.spec_type ?? settings?.spec_type ?? "draft-mtp",
    specDraftNMax: preview?.spec_draft_n_max ?? settings?.spec_draft_n_max ?? 0,
    draftMaxTokens: preview?.draft_max_tokens ?? settings?.draft_max_tokens ?? 0,
    draftMinTokens: preview?.draft_min_tokens ?? settings?.draft_min_tokens ?? 0,
    draftPMin: preview?.draft_p_min ?? settings?.draft_p_min ?? 0,
    extraArgs: formatEditableArgs(effectiveExtraArgs),
  };
}

function readStoredModelLoadConfig(model: ModelInfo, fallback: ModelLoadConfig) {
  try {
    const raw = window.localStorage.getItem(loadDialogStorageKey(model));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredModelLoadConfig>;
    if (parsed.version !== 1 || !parsed.config || typeof parsed.config !== "object") return null;
    const candidate = parsed.config as unknown as Record<string, unknown>;
    const merged = { ...fallback } as unknown as Record<string, unknown>;
    for (const key of Object.keys(fallback)) {
      if (typeof candidate[key] === typeof merged[key]) merged[key] = candidate[key];
    }
    const config = merged as unknown as ModelLoadConfig;
    config.chatTemplateKwargsJson = stripStaleThinkingKwargs(config.chatTemplateKwargsJson).value;
    return config;
  } catch {
    return null;
  }
}

export function ModelLoadDialog({
  model,
  mode,
  loadedModel,
  processStatus,
  settings,
  isLoading,
  returnFocus,
  onClose,
  onSubmit,
}: {
  model: ModelInfo;
  mode: LoadDialogMode;
  loadedModel: string | null;
  processStatus: ProcessStatusInfo | null;
  settings: AppSettings | null;
  isLoading: boolean;
  returnFocus: HTMLElement | null;
  onClose: () => void;
  onSubmit: (options: LoadModelOptions) => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const advancedRef = useRef<HTMLElement>(null);
  const baseDefaults = defaultModelLoadConfig(model, settings, processStatus);
  const storedConfig = readStoredModelLoadConfig(model, baseDefaults);
  const [config, setConfig] = useState<ModelLoadConfig>(storedConfig ?? baseDefaults);
  const [remember, setRemember] = useState(!!storedConfig);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const gpuStats = useGpuStats();
  const maxContext = advertisedContextLimit(model);
  const recommendedPresets = recommendedLoadPresets(model);
  const maxGpuLayers = Math.max(1, model.n_layers ?? 128);
  const memory = estimateModelLoadMemory(model, config.contextSize, config.gpuLayers);
  const gpuHeadroomMb = gpuStats ? gpuStats.dedicated_mb - memory.gpuMb : null;
  const memoryState = gpuHeadroomMb == null ? "unknown" : gpuHeadroomMb < 0 ? "over" : gpuHeadroomMb < 2048 ? "near" : "safe";
  const primaryLabel = mode === "reload" ? "Reload model" : mode === "swap" ? "Switch & load" : "Load model";

  function updateConfig<K extends keyof ModelLoadConfig>(key: K, value: ModelLoadConfig[K]) {
    setConfig((current) => {
      const next = { ...current, [key]: value };
      if (key === "reasoningMode") {
        next.chatTemplateKwargsJson = stripStaleThinkingKwargs(next.chatTemplateKwargsJson).value;
      }
      return next;
    });
  }

  function applyRecommendedPreset(preset: RecommendedLoadPreset) {
    setConfig((current) => ({
      ...current,
      contextSize: Math.min(preset.contextSize, maxContext),
      parallelSlots: preset.parallelSlots ?? current.parallelSlots,
      useJinja: true,
      reasoningMode: preset.reasoningMode,
      reasoningPreserve: false,
      templateMode: model.template_mode === "repo" ? "repo" : model.has_chat_template ? "builtin" : current.templateMode,
      templateName: model.template_mode === "repo" || model.has_chat_template ? "" : current.templateName,
      customTemplatePath: model.template_mode === "repo" || model.has_chat_template ? "" : current.customTemplatePath,
      chatTemplateKwargsJson: stripStaleThinkingKwargs(current.chatTemplateKwargsJson).value,
      extraArgs: formatEditableArgs(
        replaceSamplingArgs(parseCliArgs(current.extraArgs), preset.sampling),
      ),
    }));
  }

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (!dialog.open) dialog.showModal();
    return () => {
      if (dialog.open) dialog.close();
      returnFocus?.focus();
    };
  }, [returnFocus]);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || isLoading) return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", closeOnEscape, true);
    return () => window.removeEventListener("keydown", closeOnEscape, true);
  }, [isLoading, onClose]);

  useEffect(() => {
    if (!advancedOpen) return;
    window.requestAnimationFrame(() => advancedRef.current?.scrollIntoView({ block: "start" }));
  }, [advancedOpen]);

  const parsedExtraArgs = parseCliArgs(config.extraArgs);
  const activeRecommendedPreset = recommendedPresets.find(
    (preset) =>
      preset.reasoningMode === config.reasoningMode &&
      samplingArgsMatch(parsedExtraArgs, preset.sampling),
  ) ?? null;
  const promptRendering = describePromptRendering(model, config);
  const loadedPreview = processStatus?.model === model.filename
    ? processStatus.last_launch_preview
    : null;
  const mmprojPath = loadedPreview?.mmproj_path ?? model.mmproj_candidate_path ?? null;
  const visionStatus = !config.attachMmproj
    ? "Disabled for next load"
    : mmprojPath
      ? `Matching projector: ${mmprojPath}`
      : model.mmproj_available === false
        ? "No matching projector found"
        : model.mmproj_available
          ? "Matching projector found; it will be attached automatically"
          : "Matching projector will be auto-discovered during load";
  const conflictingArg = parsedExtraArgs.find((arg) => CONTROLLED_LOAD_FLAGS.has(arg.split("=")[0]));
  let validationError: string | null = null;
  if (config.contextSize < 512 || config.contextSize > maxContext) {
    validationError = `Context must be between 512 and ${fmtNum(maxContext)} tokens.`;
  } else if (config.ubatchSize > 0 && config.batchSize > 0 && config.ubatchSize > config.batchSize) {
    validationError = "Physical batch size cannot exceed evaluation batch size.";
  } else if (config.chatTemplateKwargsJson.trim()) {
    try {
      const parsed = JSON.parse(config.chatTemplateKwargsJson);
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
        validationError = "Template kwargs must be a JSON object.";
      } else if (stripStaleThinkingKwargs(config.chatTemplateKwargsJson).removed) {
        validationError = "Thinking is controlled by Reasoning mode. Remove enable_thinking/reasoning from Template kwargs.";
      }
    } catch {
      validationError = "Template kwargs contain invalid JSON.";
    }
  }
  if (!validationError && config.templateMode === "custom" && !config.customTemplatePath.trim()) {
    validationError = "Choose a custom template file or change the template source.";
  }
  if (!validationError && config.speculativeEnabled && !config.specType.trim()) {
    validationError = "Choose a speculative decoding type.";
  }
  if (!validationError && config.draftMaxTokens > 0 && config.draftMinTokens > config.draftMaxTokens) {
    validationError = "Draft minimum cannot exceed draft maximum.";
  }
  if (!validationError && conflictingArg) {
    validationError = `${conflictingArg.split("=")[0]} already has a dedicated control above; remove it from Extra arguments.`;
  }

  const submit = () => {
    if (validationError || isLoading) return;
    if (remember) {
      const stored: StoredModelLoadConfig = { version: 1, config };
      window.localStorage.setItem(loadDialogStorageKey(model), JSON.stringify(stored));
    } else {
      window.localStorage.removeItem(loadDialogStorageKey(model));
    }
    onSubmit({
      contextSize: config.contextSize,
      gpuLayers: config.gpuLayers,
      threads: config.threads,
      threadsBatch: config.threadsBatch,
      batchSize: config.batchSize,
      ubatchSize: config.ubatchSize,
      flashAttn: config.flashAttn,
      useMmap: config.useMmap,
      useMlock: config.useMlock,
      contBatching: config.contBatching,
      parallelSlots: config.parallelSlots,
      mainGpu: config.mainGpu,
      defragThold: config.defragThold,
      ropeFreqScale: config.ropeFreqScale,
      cacheTypeK: config.cacheTypeK,
      cacheTypeV: config.cacheTypeV,
      kvUnified: config.kvUnified,
      noWarmup: config.noWarmup,
      ctxShift: config.ctxShift,
      fitMode: config.fitMode,
      cacheRamMb: config.cacheRamMb > 0 ? config.cacheRamMb : undefined,
      ctxcp: config.ctxcp > 0 ? config.ctxcp : undefined,
      useJinja: config.useJinja,
      reasoningMode: config.reasoningMode,
      reasoningPreserve: config.reasoningPreserve,
      templateMode: config.templateMode,
      templateName: config.templateName,
      customTemplatePath: config.customTemplatePath,
      chatTemplateKwargsJson: config.chatTemplateKwargsJson,
      attachMmproj: config.attachMmproj,
      draftModelPath: config.speculativeEnabled ? config.draftModelPath : "",
      specType: config.speculativeEnabled ? config.specType : "",
      specDraftNMax: config.speculativeEnabled ? config.specDraftNMax : 0,
      draftMaxTokens: config.speculativeEnabled ? config.draftMaxTokens : 0,
      draftMinTokens: config.speculativeEnabled ? config.draftMinTokens : 0,
      draftPMin: config.speculativeEnabled ? config.draftPMin : 0,
      forceReload: mode === "reload",
      extraArgs:
        (!!config.extraArgs.trim() || !!settings || processStatus?.model === model.filename)
          ? parsedExtraArgs
          : undefined,
    });
  };

  return (
    <dialog
      ref={dialogRef}
      className="ib-model-load-dialog m-auto h-[min(740px,calc(100vh-24px))] w-[min(900px,calc(100vw-24px))] max-w-none overflow-hidden p-0"
      aria-labelledby="model-load-dialog-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!isLoading) onClose();
      }}
      onMouseDown={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        const outside = event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom;
        if (outside && !isLoading) onClose();
      }}
    >
      <form
        className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)_auto]"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <header className="flex min-h-[68px] items-center gap-3 border-b px-4 py-3 sm:px-5" style={{ borderColor: "var(--border)" }}>
          <button type="button" onClick={onClose} disabled={isLoading} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition hover:bg-white/5 disabled:opacity-50" aria-label="Close load settings" style={{ color: "var(--text-1)" }}>
            <ArrowLeft size={19} />
          </button>
          <div className="min-w-0 flex-1">
            <h2 id="model-load-dialog-title" className="truncate text-base font-semibold" style={{ color: "var(--text-0)" }}>{modelDisplayName(model)}</h2>
            <p className="mt-0.5 truncate text-xs" style={{ color: "var(--text-2)" }}>
              {modelPublisher(model)} · {model.quant ?? "Unquantized"} · {model.size_gb ? `${model.size_gb.toFixed(1)} GB` : "Size unknown"}
            </p>
          </div>
        </header>

        <div className="min-h-0 overflow-y-auto px-4 py-4 sm:px-5">
          {mode === "swap" && loadedModel && (
            <div className="mb-4 rounded-lg px-3 py-2 text-xs" style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.24)", color: "#fcd34d" }}>
              Loading this model will replace <span className="font-mono">{loadedModel}</span> for chat, API, and connected clients.
            </div>
          )}

          <section className="rounded-xl" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
            <div className="flex flex-wrap items-center gap-3 border-b px-3 py-3 sm:px-4" style={{ borderColor: "var(--border)" }}>
              <div className="mr-auto">
                <div className="flex items-center gap-2 text-xs font-semibold" style={{ color: "var(--text-0)" }}>
                  Estimated memory usage
                  <span className="rounded px-1.5 py-0.5 text-[9px] uppercase" style={{ background: "var(--surface-3)", color: "var(--text-2)" }}>Beta</span>
                </div>
                <div className="mt-1 text-[10px]" style={{ color: "var(--text-2)" }}>Model, graph, and estimated KV cache for this context.</div>
              </div>
              <MemoryMetric label="GPU" value={`${(memory.gpuMb / 1024).toFixed(2)} GB`} />
              <MemoryMetric label="Total" value={`${(memory.totalMb / 1024).toFixed(2)} GB`} />
            </div>
            {gpuStats && (
              <div className="px-3 py-3 sm:px-4" aria-live="polite">
                <PredictedVramBar predictedMb={memory.gpuMb} dedicatedMb={gpuStats.dedicated_mb} />
              </div>
            )}
          </section>

          <section className="mt-4 grid gap-2 sm:grid-cols-2" aria-label="Effective next-load configuration">
            <LoadStateTile
              label="Prompt template"
              value={promptRendering.label}
              detail={`Next load: ${promptRendering.source}${model.template_source ? ` · Current: ${model.template_source}` : ""}`}
              ready={promptRendering.effective}
            />
            <LoadStateTile
              label="Reasoning"
              value={model.supports_reasoning ? (config.reasoningMode === "auto" ? "Auto · model profile" : `${config.reasoningMode === "on" ? "On" : "Off"} · --reasoning ${config.reasoningMode}`) : "Not detected"}
              detail={config.reasoningPreserve ? "Reasoning preservation is enabled" : "Reasoning preservation is off; no deprecated thinking kwargs"}
              ready={!model.supports_reasoning || config.reasoningMode !== "auto"}
            />
            <LoadStateTile
              label="Tool calling"
              value={model.supports_tools ? fmtToolFormat(model.tool_call_format) : "Not detected"}
              detail={model.supports_parallel_tools ? "Parallel format supported; Tools / Direct uses one serial slot for reliability" : "Serial tool calls recommended"}
              ready={model.supports_tools && config.useJinja}
            />
            <LoadStateTile
              label="Vision"
              value={model.supports_vision ? (config.attachMmproj ? "Projector enabled" : "Text-only load") : "Not detected"}
              detail={model.supports_vision ? visionStatus : "No image projector required"}
              ready={!model.supports_vision || (!!config.attachMmproj && model.mmproj_available !== false) || !config.attachMmproj}
            />
          </section>

          <section className="mt-4 space-y-5">
            {recommendedPresets.length > 0 && (
              <LoadField label="Recommended profile" hint="Tess / Qwen3.6 settings. A profile applies its sampler, reasoning mode, 32K context, and embedded Jinja selection together.">
                <div className="space-y-2">
                  <div className="grid gap-1.5 sm:grid-cols-3" role="group" aria-label="Tess and Qwen recommended profiles">
                    {recommendedPresets.map((preset) => {
                      const active = activeRecommendedPreset?.id === preset.id;
                      return (
                        <button
                          key={preset.id}
                          type="button"
                          onClick={() => applyRecommendedPreset(preset)}
                          aria-pressed={active}
                          className="rounded-lg px-2.5 py-2 text-left transition"
                          style={{
                            background: active ? "rgba(59,130,246,0.14)" : "var(--surface-1)",
                            border: active ? "1px solid rgba(96,165,250,0.7)" : "1px solid var(--border)",
                            color: "var(--text-0)",
                          }}
                        >
                          <span className="block text-[11px] font-semibold">{preset.name}</span>
                          <span className="mt-1 block text-[9px] leading-3.5" style={{ color: "var(--text-2)" }}>{preset.description}</span>
                        </button>
                      );
                    })}
                  </div>
                  {activeRecommendedPreset && (
                    <div className="rounded-md px-2.5 py-1.5 font-mono text-[10px]" style={{ background: "var(--surface-1)", color: "var(--text-1)", border: "1px solid var(--border)" }}>
                      temp {fmtSamplingValue(activeRecommendedPreset.sampling.temperature)} · top-p {fmtSamplingValue(activeRecommendedPreset.sampling.topP)} · top-k {activeRecommendedPreset.sampling.topK} · min-p {fmtSamplingValue(activeRecommendedPreset.sampling.minP)} · presence {fmtSamplingValue(activeRecommendedPreset.sampling.presencePenalty)} · repeat {fmtSamplingValue(activeRecommendedPreset.sampling.repeatPenalty)}
                    </div>
                  )}
                </div>
              </LoadField>
            )}

            <LoadField label="Context length" hint={`Model advertises up to ${fmtNum(maxContext)} tokens. Choose a smaller context to reduce KV memory.`}>
              <div className="space-y-2">
                <div className="flex flex-wrap gap-1.5" role="group" aria-label="Recommended context sizes">
                  {safeContextPresets(model).map((value) => (
                    <button key={value} type="button" onClick={() => updateConfig("contextSize", value)} aria-pressed={config.contextSize === value} className="rounded-md px-2.5 py-1.5 text-xs font-semibold" style={{ background: config.contextSize === value ? "var(--surface-3)" : "var(--surface-1)", border: config.contextSize === value ? "1px solid var(--text-1)" : "1px solid var(--border)", color: "var(--text-0)" }}>
                      {value / 1024}K{value === safeDefaultContext(model) ? " · Recommended" : ""}
                    </button>
                  ))}
                </div>
                <div className="flex items-center gap-3">
                  <input aria-label="Context length" type="range" min={512} max={maxContext} step={1} value={config.contextSize} onChange={(event) => updateConfig("contextSize", Number(event.target.value))} className="min-w-0 flex-1" />
                  <input aria-label="Context length value" type="number" min={512} max={maxContext} step={1} value={config.contextSize} onChange={(event) => updateConfig("contextSize", Math.max(512, Math.min(maxContext, Number(event.target.value) || 512)))} className="ib-field h-8 w-28 px-2 text-right font-mono text-xs" />
                </div>
              </div>
            </LoadField>

            <LoadField label="GPU offload" hint="-1 lets llama.cpp offload every compatible layer. Use 0 for CPU-only loading.">
              <div className="flex items-center gap-3">
                <input aria-label="GPU layers" type="range" min={0} max={maxGpuLayers} step={1} value={config.gpuLayers < 0 ? maxGpuLayers : Math.min(maxGpuLayers, config.gpuLayers)} onChange={(event) => updateConfig("gpuLayers", Number(event.target.value))} className="min-w-0 flex-1" />
                <button type="button" onClick={() => updateConfig("gpuLayers", -1)} aria-pressed={config.gpuLayers < 0} className="h-8 rounded-md px-2 text-[11px] font-semibold" style={{ background: config.gpuLayers < 0 ? "var(--surface-3)" : "var(--surface-1)", border: "1px solid var(--border)", color: "var(--text-0)" }}>Auto</button>
                <input aria-label="GPU layer count" type="number" min={-1} max={maxGpuLayers} value={config.gpuLayers} onChange={(event) => updateConfig("gpuLayers", Math.max(-1, Math.min(maxGpuLayers, Number(event.target.value) || 0)))} className="ib-field h-8 w-20 px-2 text-right font-mono text-xs" />
              </div>
            </LoadField>

            <div className="grid gap-4 sm:grid-cols-2">
              <LoadNumberField label="CPU thread pool" hint="0 lets llama.cpp choose." value={config.threads} min={0} max={256} onChange={(value) => updateConfig("threads", value)} />
              <LoadNumberField label="Concurrent requests" hint="Inference slots exposed by the server. This is separate from parallel tool calls." value={config.parallelSlots} min={1} max={32} onChange={(value) => updateConfig("parallelSlots", value)} />
              <LoadNumberField label="Evaluation batch" hint="0 uses llama.cpp's default (usually 2048)." value={config.batchSize} min={0} max={8192} step={128} onChange={(value) => updateConfig("batchSize", value)} />
              <LoadNumberField label="Physical batch" hint="0 uses llama.cpp's default (usually 512)." value={config.ubatchSize} min={0} max={4096} step={64} onChange={(value) => updateConfig("ubatchSize", value)} />
            </div>

            {model.supports_reasoning && (
              <LoadField label="Reasoning mode" hint="Applied when this model loads or reloads. On/off maps to llama-server --reasoning on/off; Auto follows the detected profile.">
                <div className="grid grid-cols-3 gap-1 rounded-lg p-1" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }} role="group" aria-label="Reasoning mode">
                  {["auto", "on", "off"].map((value) => (
                    <button key={value} type="button" onClick={() => updateConfig("reasoningMode", value)} aria-pressed={config.reasoningMode === value} className="rounded-md px-2 py-1.5 text-xs font-semibold capitalize" style={{ background: config.reasoningMode === value ? "var(--surface-3)" : "transparent", color: config.reasoningMode === value ? "var(--text-0)" : "var(--text-2)", border: "none" }}>{value}</button>
                  ))}
                </div>
              </LoadField>
            )}

            {model.supports_vision && (
              <LoadField label="Vision projector" hint={visionStatus}>
                <LoadToggle checked={config.attachMmproj} onChange={(value) => updateConfig("attachMmproj", value)} label={config.attachMmproj ? "Attach matching mmproj" : "Text-only load"} hint={mmprojPath ? mmprojPath : "InferenceBridge will use only a matching projector; it will not guess a client-side path."} />
              </LoadField>
            )}

            <div className="grid gap-2 sm:grid-cols-2">
              <LoadToggle checked={config.flashAttn} onChange={(value) => updateConfig("flashAttn", value)} label="Flash attention" hint="Faster attention with lower memory use on supported GPUs." />
              <LoadToggle checked={config.kvUnified} onChange={(value) => updateConfig("kvUnified", value)} label="Unified KV cache" hint="Shares one contiguous KV buffer across parallel slots." />
              <LoadToggle checked={config.contBatching} onChange={(value) => updateConfig("contBatching", value)} label="Continuous batching" hint="Allows requests to join an active batch." />
              <LoadToggle checked={config.useMlock} onChange={(value) => updateConfig("useMlock", value)} label="Lock model in memory" hint="Prevents loaded model pages being swapped to disk." />
            </div>
          </section>

          {advancedOpen && (
            <section ref={advancedRef} className="mt-5 border-t pt-4" style={{ borderColor: "var(--border)" }}>
              <div className="mb-3 flex items-center gap-2 px-1">
                <ChevronDown size={16} />
                <h3 className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>Advanced settings</h3>
                <span className="ml-auto text-[10px]" style={{ color: "var(--text-2)" }}>Runtime, templates, KV cache, speculative decoding</span>
              </div>
              <div id="model-load-advanced" className="space-y-6 rounded-xl p-3 sm:p-4" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
                <AdvancedLoadSettings config={config} model={model} updateConfig={updateConfig} />
              </div>
            </section>
          )}

          {validationError && <div className="mt-4 rounded-lg px-3 py-2 text-xs" role="alert" style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.24)", color: "#fca5a5" }}>{validationError}</div>}
        </div>

        <footer className="flex min-h-[68px] flex-wrap items-center gap-3 border-t px-4 py-3 sm:px-5" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
          <div className="flex flex-col gap-1.5">
            <label className="flex cursor-pointer items-center gap-2 text-xs" style={{ color: "var(--text-1)" }}>
              <input type="checkbox" checked={remember} onChange={(event) => setRemember(event.target.checked)} />
              Remember settings for this model
            </label>
            <button type="button" role="switch" aria-checked={advancedOpen} aria-expanded={advancedOpen} aria-controls="model-load-advanced" onClick={() => setAdvancedOpen((open) => !open)} className="flex items-center gap-2 text-left text-xs" style={{ color: "var(--text-1)" }}>
              <span className="relative h-4 w-7 shrink-0 rounded-full transition" style={{ background: advancedOpen ? "#3b82f6" : "var(--surface-3)", border: "1px solid var(--border-mid)" }}><span className="absolute top-[2px] h-2.5 w-2.5 rounded-full bg-white transition" style={{ left: advancedOpen ? 14 : 2 }} /></span>
              Show advanced settings
            </button>
          </div>
          <div className="hidden text-[10px] lg:block" aria-live="polite" style={{ color: memoryState === "over" ? "#fca5a5" : memoryState === "near" ? "#fcd34d" : "var(--text-2)" }}>
            {(memory.gpuMb / 1024).toFixed(1)} GB GPU · {memoryState === "over" ? "Over dedicated VRAM" : memoryState === "near" ? "Near VRAM limit" : memoryState === "safe" ? "Estimated safe" : "Estimate ready"}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Button type="button" size="sm" variant="ghost" onClick={() => setConfig(baseDefaults)} disabled={isLoading}>Reset</Button>
            <Button type="button" size="sm" variant="secondary" onClick={onClose} disabled={isLoading}>Cancel</Button>
            <Button type="submit" size="sm" variant="primary" icon={<Play size={13} />} disabled={isLoading || !!validationError}>{isLoading ? "Starting…" : primaryLabel}</Button>
          </div>
        </footer>
      </form>
    </dialog>
  );
}

function MemoryMetric({ label, value }: { label: string; value: string }) {
  return <span className="flex items-center gap-1.5 text-xs" style={{ color: "var(--text-2)" }}><strong style={{ color: "var(--text-1)" }}>{label}</strong><span className="rounded-md px-2 py-1 font-mono" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}>{value}</span></span>;
}

function LoadStateTile({ label, value, detail, ready }: { label: string; value: string; detail: string; ready: boolean }) {
  return (
    <div className="min-w-0 rounded-lg px-3 py-2.5" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
      <div className="flex items-center gap-2">
        <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: ready ? "#34d399" : "#fbbf24" }} />
        <span className="text-[9px] font-bold uppercase tracking-[0.12em]" style={{ color: "var(--text-2)" }}>{label}</span>
      </div>
      <div className="mt-1 truncate text-xs font-semibold" style={{ color: "var(--text-0)" }} title={value}>{value}</div>
      <div className="mt-1 break-all text-[9px] leading-3.5" style={{ color: "var(--text-2)" }}>{detail}</div>
    </div>
  );
}

function LoadField({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <div className="grid gap-2 sm:grid-cols-[minmax(180px,0.8fr)_minmax(260px,1.2fr)] sm:gap-5">
      <div><div className="text-xs font-semibold" style={{ color: "var(--text-0)" }}>{label}</div><p className="mt-1 text-[10px] leading-4" style={{ color: "var(--text-2)" }}>{hint}</p></div>
      <div className="min-w-0 self-center">{children}</div>
    </div>
  );
}

function LoadNumberField({ label, hint, value, min, max, step = 1, onChange }: { label: string; hint: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void }) {
  return (
    <label className="rounded-lg p-3" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
      <span className="flex items-center justify-between gap-3"><span className="text-xs font-semibold" style={{ color: "var(--text-0)" }}>{label}</span><input type="number" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Math.max(min, Math.min(max, Number(event.target.value) || 0)))} className="ib-field h-8 w-24 px-2 text-right font-mono text-xs" /></span>
      <span className="mt-1.5 block text-[10px] leading-4" style={{ color: "var(--text-2)" }}>{hint}</span>
    </label>
  );
}

function LoadToggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (value: boolean) => void; label: string; hint: string }) {
  return (
    <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)} className="flex items-start gap-3 rounded-lg p-3 text-left transition hover:bg-white/5" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
      <span className="relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition" style={{ background: checked ? "#3b82f6" : "var(--surface-3)", border: "1px solid var(--border-mid)" }}><span className="absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white transition" style={{ left: checked ? 18 : 2 }} /></span>
      <span><span className="block text-xs font-semibold" style={{ color: "var(--text-0)" }}>{label}</span><span className="mt-1 block text-[10px] leading-4" style={{ color: "var(--text-2)" }}>{hint}</span></span>
    </button>
  );
}

function AdvancedLoadSettings({ config, model, updateConfig }: { config: ModelLoadConfig; model: ModelInfo; updateConfig: <K extends keyof ModelLoadConfig>(key: K, value: ModelLoadConfig[K]) => void }) {
  const cacheTypes = ["f32", "f16", "q8_0", "q4_0", "q4_1"];
  return (
    <>
      <div>
        <h3 className="text-[10px] font-bold uppercase tracking-[0.14em]" style={{ color: "var(--text-2)" }}>Runtime</h3>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <AdvancedField label="Fit mode"><select className="ib-field w-full" value={config.fitMode} onChange={(event) => updateConfig("fitMode", event.target.value)}><option value="">Unset</option><option value="auto">Auto</option><option value="off">Off</option><option value="on">On</option></select></AdvancedField>
          <AdvancedField label="Batch threads"><input className="ib-field w-full" type="number" min={0} max={256} value={config.threadsBatch} onChange={(event) => updateConfig("threadsBatch", Math.max(0, Number(event.target.value) || 0))} /></AdvancedField>
          <AdvancedField label="Main GPU"><input className="ib-field w-full" type="number" min={0} max={16} value={config.mainGpu} onChange={(event) => updateConfig("mainGpu", Math.max(0, Number(event.target.value) || 0))} /></AdvancedField>
          <AdvancedField label="RoPE scale"><input className="ib-field w-full" type="number" min={0} step={0.05} value={config.ropeFreqScale} onChange={(event) => updateConfig("ropeFreqScale", Math.max(0, Number(event.target.value) || 0))} /></AdvancedField>
          <AdvancedField label="Cache RAM override (MiB; 0 = inherit)"><input className="ib-field w-full" type="number" min={0} value={config.cacheRamMb} onChange={(event) => updateConfig("cacheRamMb", Math.max(0, Number(event.target.value) || 0))} /></AdvancedField>
          <AdvancedField label="Context checkpoint (0 = inherit)"><input className="ib-field w-full" type="number" min={0} value={config.ctxcp} onChange={(event) => updateConfig("ctxcp", Math.max(0, Number(event.target.value) || 0))} /></AdvancedField>
          <AdvancedField label="KV key type"><select className="ib-field w-full" value={config.cacheTypeK} onChange={(event) => updateConfig("cacheTypeK", event.target.value)}>{cacheTypes.map((value) => <option key={value} value={value}>{value}</option>)}</select></AdvancedField>
          <AdvancedField label="KV value type"><select className="ib-field w-full" value={config.cacheTypeV} onChange={(event) => updateConfig("cacheTypeV", event.target.value)}>{cacheTypes.map((value) => <option key={value} value={value}>{value}</option>)}</select></AdvancedField>
        </div>
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          <LoadToggle checked={config.useMmap} onChange={(value) => updateConfig("useMmap", value)} label="Memory map model" hint="Load model pages through the operating-system file cache." />
          <LoadToggle checked={config.noWarmup} onChange={(value) => updateConfig("noWarmup", value)} label="Skip warmup" hint="Loads faster; the first request can be slower." />
          <LoadToggle checked={config.ctxShift} onChange={(value) => updateConfig("ctxShift", value)} label="Context shift" hint="Discard oldest tokens when the context fills." />
        </div>
      </div>

      <div className="border-t pt-5" style={{ borderColor: "var(--border)" }}>
        <h3 className="text-[10px] font-bold uppercase tracking-[0.14em]" style={{ color: "var(--text-2)" }}>Prompt rendering</h3>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <AdvancedField label="Template source"><select className="ib-field w-full" value={config.templateMode} onChange={(event) => updateConfig("templateMode", event.target.value)}><option value="builtin">{model.has_chat_template ? "Embedded GGUF Jinja" : "llama.cpp built-in fallback"}</option><option value="repo">Hugging Face repo override</option><option value="custom">Custom file override</option></select></AdvancedField>
          {config.templateMode === "builtin" && <AdvancedField label="Template name"><input className="ib-field w-full" value={config.templateName} onChange={(event) => updateConfig("templateName", event.target.value)} placeholder="Auto-detect" /></AdvancedField>}
          {config.templateMode === "custom" && <AdvancedField label="Template file"><input className="ib-field w-full" value={config.customTemplatePath} onChange={(event) => updateConfig("customTemplatePath", event.target.value)} placeholder="C:\\templates\\chat.jinja" /></AdvancedField>}
        </div>
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          <LoadToggle checked={config.useJinja} onChange={(value) => updateConfig("useJinja", value)} label="Jinja templates" hint="Use llama.cpp's Jinja renderer for embedded and repo templates." />
          <LoadToggle checked={config.reasoningPreserve} onChange={(value) => updateConfig("reasoningPreserve", value)} label="Preserve reasoning" hint="Keep reasoning blocks in rendered assistant output." />
        </div>
        <label className="mt-3 block"><span className="mb-1.5 block text-xs font-semibold" style={{ color: "var(--text-0)" }}>Template kwargs JSON</span><span className="mb-2 block text-[10px] leading-4" style={{ color: "var(--text-2)" }}>Reasoning is controlled above. Deprecated enable_thinking/reasoning keys are rejected here so the visible mode stays authoritative.</span><textarea className="settings-input min-h-20 resize-y font-mono text-xs" value={config.chatTemplateKwargsJson} onChange={(event) => updateConfig("chatTemplateKwargsJson", event.target.value)} placeholder='{"custom_template_option": true}' /></label>
      </div>

      <div className="border-t pt-5" style={{ borderColor: "var(--border)" }}>
        <LoadToggle checked={config.speculativeEnabled} onChange={(value) => updateConfig("speculativeEnabled", value)} label="Speculative decoding" hint="Use self-MTP or a compatible draft GGUF to accelerate decoding." />
        {config.speculativeEnabled && <div className="mt-3 grid gap-3 sm:grid-cols-2"><AdvancedField label="Spec type"><input className="ib-field w-full" value={config.specType} onChange={(event) => updateConfig("specType", event.target.value)} placeholder="draft-mtp" /></AdvancedField><AdvancedField label="Draft GGUF (optional for self-MTP)"><input className="ib-field w-full" value={config.draftModelPath} onChange={(event) => updateConfig("draftModelPath", event.target.value)} placeholder="C:\\models\\draft.gguf" /></AdvancedField><AdvancedField label="Draft N"><input className="ib-field w-full" type="number" min={0} max={64} value={config.specDraftNMax} onChange={(event) => updateConfig("specDraftNMax", Math.max(0, Number(event.target.value) || 0))} /></AdvancedField><AdvancedField label="Draft maximum"><input className="ib-field w-full" type="number" min={0} max={128} value={config.draftMaxTokens} onChange={(event) => updateConfig("draftMaxTokens", Math.max(0, Number(event.target.value) || 0))} /></AdvancedField><AdvancedField label="Draft minimum"><input className="ib-field w-full" type="number" min={0} max={128} value={config.draftMinTokens} onChange={(event) => updateConfig("draftMinTokens", Math.max(0, Number(event.target.value) || 0))} /></AdvancedField><AdvancedField label="Acceptance probability"><input className="ib-field w-full" type="number" min={0} max={1} step={0.05} value={config.draftPMin} onChange={(event) => updateConfig("draftPMin", Math.max(0, Math.min(1, Number(event.target.value) || 0)))} /></AdvancedField></div>}
      </div>

      <div className="border-t pt-5" style={{ borderColor: "var(--border)" }}>
        <label className="block"><span className="block text-xs font-semibold" style={{ color: "var(--text-0)" }}>Extra llama-server arguments</span><span className="mt-1 block text-[10px] leading-4" style={{ color: "var(--text-2)" }}>Only arguments without a dedicated control are accepted here. Quoted values are supported.</span><textarea className="settings-input mt-2 min-h-20 resize-y font-mono text-xs" value={config.extraArgs} onChange={(event) => updateConfig("extraArgs", event.target.value)} placeholder="--temp 0.7 --top-p 0.9" /></label>
      </div>
    </>
  );
}

function AdvancedField({ label, children }: { label: string; children: ReactNode }) {
  return <label><span className="mb-1.5 block text-[10px] font-semibold" style={{ color: "var(--text-1)" }}>{label}</span>{children}</label>;
}

function ModelInspector({
  model,
  loadedModel,
  previousModel,
  processStatus,
  sidecarStatus,
  sidecarSyncing,
  isLoading,
  onConfigureLoad,
  onUnload,
  onSwapBack,
  onSyncSidecars,
  onRollbackSidecars,
}: {
  model: ModelInfo | null;
  loadedModel: string | null;
  previousModel: string | null;
  processStatus: ProcessStatusInfo | null;
  sidecarStatus?: api.HfSidecarCacheStatus;
  sidecarSyncing: boolean;
  isLoading: boolean;
  onConfigureLoad: (model: ModelInfo) => void;
  onUnload: () => void;
  onSwapBack: () => void;
  onSyncSidecars: (model: ModelInfo) => void;
  onRollbackSidecars: (model: ModelInfo) => void;
}) {
  if (!model) {
    return <div className="min-h-0 flex-1"><EmptyMsg fill title="No model selected" body="Select a model to inspect its details and loading options." /></div>;
  }

  const loaded = model.filename === loadedModel;
  const launchPreview = processStatus?.model === model.filename ? processStatus.last_launch_preview : null;
  const lastMetrics = processStatus?.model === model.filename ? processStatus.last_generation_metrics : null;
  const canOpenHfCache = !!(sidecarStatus?.sidecar_cache_dir || sidecarStatus?.template_cache_path);
  const primaryLabel = loaded ? "Reload options" : loadedModel ? "Switch options" : "Load options";
  const hfActionLabel = sidecarSyncing
    ? "Working…"
    : sidecarStatus?.update_available
      ? "Update HF files"
      : sidecarStatus?.last_checked_at
        ? "Check HF updates"
        : "Check HF files";

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="border-b px-4 py-4" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-start gap-3">
          <ModelArtwork model={model} size="md" />
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-sm font-semibold" style={{ color: "var(--text-0)" }}>{modelDisplayName(model)}</h3>
            <p className="mt-0.5 truncate text-[11px]" style={{ color: "var(--text-2)" }}>{modelPublisher(model)} · {model.quant ?? "Unquantized"} · {model.size_gb ? `${model.size_gb.toFixed(1)} GB` : "Size unknown"}</p>
          </div>
          {loaded && <span className="rounded px-2 py-0.5 text-[10px] font-bold uppercase" style={{ background: "rgba(52,211,153,0.14)", color: "#6ee7b7", border: "1px solid rgba(52,211,153,0.28)" }}>Loaded</span>}
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2">
          <ActionBtn label={primaryLabel} disabled={isLoading || !model.provider_managed} variant="primary" onClick={() => onConfigureLoad(model)} />
          {loaded ? (
            <ActionBtn label="Unload model" disabled={isLoading || !model.provider_managed} variant="danger" onClick={onUnload} />
          ) : (
            <ActionBtn label={hfActionLabel} disabled={isLoading || sidecarSyncing || !model.hf_repo} variant="ghost" onClick={() => onSyncSidecars(model)} />
          )}
          {loaded && <ActionBtn label={hfActionLabel} disabled={isLoading || sidecarSyncing || !model.hf_repo} variant="ghost" onClick={() => onSyncSidecars(model)} />}
          {sidecarStatus?.rollback_available && <ActionBtn label="Restore previous" disabled={isLoading || sidecarSyncing} variant="ghost" onClick={() => onRollbackSidecars(model)} />}
          <ActionBtn label="Open HF cache" disabled={!canOpenHfCache} variant="ghost" onClick={() => {
            const path = sidecarStatus?.sidecar_cache_dir || sidecarStatus?.template_cache_path;
            if (path) void api.showInFolder(path);
          }} />
          {previousModel && previousModel !== model.filename && (
            <button onClick={onSwapBack} disabled={isLoading} className="col-span-2 rounded-md px-3 py-1.5 text-xs font-semibold" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
              Configure previous model
            </button>
          )}
        </div>
      </div>

      <section className="px-4 py-4">
        <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-0)" }}>Model information</h4>
        <InfoRow label="Model" value={shortModelName(model)} />
        <InfoRow label="File" value={model.filename} />
        <InfoRow label="Format" value={model.provider_managed ? "GGUF" : "OpenAI-compatible"} />
        <InfoRow label="Quantization" value={model.quant ?? "-"} />
        <InfoRow label="Architecture" value={model.family || model.gguf_architecture || "-"} />
        <InfoRow label="Parameters" value={modelParamsLabel(model)} />
        <InfoRow label="Capabilities" value={[
          model.supports_vision ? "Vision" : null,
          model.supports_tools ? "Tool use" : null,
          model.supports_reasoning ? "Reasoning" : null,
        ].filter(Boolean).join(", ") || "Chat"} />
        <InfoRow label="Context" value={launchPreview?.context_size ? `${fmtNum(launchPreview.context_size)} live` : `${fmtNum(advertisedContextLimit(model))} max`} />
        <InfoRow label="Size on disk" value={model.size_gb ? `${model.size_gb.toFixed(2)} GB` : "-"} />
        <InfoRow label="Provider" value={model.provider_name} />
        <InfoRow label="HF repo" value={sidecarStatus?.repo_id ?? model.hf_repo ?? "-"} />
        <InfoRow label="Update source" value={sidecarStatus?.source_repo_id ?? sidecarStatus?.repo_id ?? model.hf_repo ?? "-"} />
        <InfoRow label="HF cache" value={hfCacheSummary(sidecarStatus)} />
        <InfoRow label="Template source" value={sidecarStatus?.template_source ?? (model.has_chat_template ? "Embedded in GGUF" : "-")} />
        <InfoRow label="Active HF revision" value={shortHfRevision(sidecarStatus?.active_revision)} />
        <InfoRow label="Latest checked revision" value={shortHfRevision(sidecarStatus?.remote_revision)} />
        <InfoRow label="Update state" value={sidecarStatus?.update_available ? "Update ready" : sidecarStatus?.last_checked_at ? "Current at last check" : "Not checked yet"} />
      </section>

      {loaded && (
        <section className="border-t px-4 py-4" style={{ borderColor: "var(--border)" }}>
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-0)" }}>Live runtime</h4>
          <InfoRow label="State" value={processStatus?.state ?? "Running"} />
          <InfoRow label="Template" value={launchPreview?.template_source ?? launchPreview?.template_mode ?? "-"} />
          <InfoRow label="Parallel" value={launchPreview ? `${launchPreview.parallel_slots} slot${launchPreview.parallel_slots === 1 ? "" : "s"}` : "-"} />
          <InfoRow label="Vision" value={launchPreview?.mmproj_path ? "Projector attached" : model.supports_vision ? "No projector" : "Not required"} />
          <InfoRow label="Speculation" value={launchPreview?.spec_type || "Disabled"} />
          <InfoRow label="Last speed" value={lastMetrics?.decode_tokens_per_second != null ? `${lastMetrics.decode_tokens_per_second.toFixed(2)} tok/s` : "-"} />
        </section>
      )}
    </div>
  );
}

function LegacyModelInspectorPane({
  model,
  loadedModel,
  previousModel,
  processStatus,
  sidecarStatus,
  sidecarSyncing,
  isLoading,
  onLoad,
  onSwap,
  onUnload,
  onSwapBack,
  onSyncSidecars,
}: {
  model: ModelInfo | null;
  loadedModel: string | null;
  previousModel: string | null;
  processStatus: ProcessStatusInfo | null;
  sidecarStatus?: api.HfSidecarCacheStatus;
  sidecarSyncing: boolean;
  isLoading: boolean;
  onLoad: (modelName: string, options?: LoadModelOptions) => void;
  onSwap: (modelName: string, options?: LoadModelOptions) => void;
  onUnload: () => void;
  onSwapBack: () => void;
  onSyncSidecars: (model: ModelInfo) => void;
}) {
  const [activeInspectorTab, setActiveInspectorTab] = useState<"info" | "load" | "inference">("info");
  const defaultContext = model ? safeDefaultContext(model) : 8192;
  const maxContext = model ? advertisedContextLimit(model) : defaultContext;
  const minContext = 0;
  const [contextSize, setContextSize] = useState(defaultContext);
  const [fitMode, setFitMode] = useState("off");
  const [useJinja, setUseJinja] = useState(model?.template_mode === "repo");
  const [reasoningMode, setReasoningMode] = useState(model?.supports_reasoning ? "auto" : "off");
  const [templateMode, setTemplateMode] = useState(model?.template_mode ?? "repo");
  const [chatTemplateKwargsJson, setChatTemplateKwargsJson] = useState("");
  const [draftModelPath, setDraftModelPath] = useState("");
  const [specType, setSpecType] = useState("");
  const [specDraftNMax, setSpecDraftNMax] = useState(0);
  const [extraArgs, setExtraArgs] = useState("");
  const [temperature, setTemperature] = useState(fmtSamplingValue(model?.default_temperature));
  const [topP, setTopP] = useState(fmtSamplingValue(model?.default_top_p));
  const [topK, setTopK] = useState(fmtSamplingValue(model?.default_top_k));
  const [minP, setMinP] = useState(fmtSamplingValue(model?.default_min_p));
  const [presencePenalty, setPresencePenalty] = useState(fmtSamplingValue(model?.default_presence_penalty));
  const gpuStats = useGpuStats();

  useEffect(() => {
    const nextContext = model ? safeDefaultContext(model) : 8192;
    setContextSize(nextContext);
    setFitMode("off");
    setUseJinja(model?.template_mode === "repo");
    setReasoningMode(model?.supports_reasoning ? "auto" : "off");
    setTemplateMode(model?.template_mode ?? "repo");
    setChatTemplateKwargsJson("");
    setDraftModelPath("");
    setSpecType("");
    setSpecDraftNMax(0);
    setExtraArgs("");
    setTemperature(fmtSamplingValue(model?.default_temperature));
    setTopP(fmtSamplingValue(model?.default_top_p));
    setTopK(fmtSamplingValue(model?.default_top_k));
    setMinP(fmtSamplingValue(model?.default_min_p));
    setPresencePenalty(fmtSamplingValue(model?.default_presence_penalty));
  }, [model?.filename, model?.context_window, model?.max_context_window, model?.provider_managed, model?.size_gb, model?.template_mode, model?.supports_reasoning, model?.default_temperature, model?.default_top_p, model?.default_top_k, model?.default_min_p, model?.default_presence_penalty]);

  const setClampedContextSize = (value: number) => {
    if (!Number.isFinite(value)) {
      setContextSize(defaultContext);
      setFitMode("off");
      return;
    }
    const rounded = Math.round(value);
    setContextSize(rounded <= 0 ? 0 : Math.max(512, Math.min(maxContext, rounded)));
    setFitMode("off");
  };

  if (!model) {
    return (
      <div className="min-h-0 flex-1">
        <EmptyMsg fill title="No model selected" body="Select a model to inspect details and launch controls." />
      </div>
    );
  }

  const kvBytesPerToken =
    model.n_layers != null && model.n_kv_heads != null && model.head_dim != null
      ? model.n_layers * 2 * model.n_kv_heads * model.head_dim
      : null;
  const estimateContextVRAM = (tokens: number) => {
    const modelMb = (model.size_gb || 0) * 1024;
    const graphOverheadMb = Math.max(512, Math.min(2048, modelMb * 0.08));
    if (kvBytesPerToken != null) {
      const kvMb = (tokens * kvBytesPerToken) / (1024 * 1024);
      return modelMb + graphOverheadMb + kvMb * 1.15;
    }
    const name = `${model.filename} ${model.family ?? ""}`.toLowerCase();
    const fallbackKvMbPerToken =
      name.includes("gemma-4-26b") || name.includes("a4b")
        ? 0.04
        : modelMb > 10 * 1024
          ? 0.035
          : 0.025;
    return modelMb + graphOverheadMb + tokens * fallbackKvMbPerToken;
  };

  const loaded = model.filename === loadedModel;
  const contextPresets = safeContextPresets(model);
  const customContextSelected = !contextPresets.includes(contextSize);
  const advertisedContext = advertisedContextLimit(model);
  const predictionContext = contextSize === 0 ? advertisedContext : contextSize;
  const liveContext =
    processStatus?.model === model.filename ? processStatus.last_launch_preview?.context_size ?? null : null;
  const launchPreview =
    processStatus?.model === model.filename ? processStatus.last_launch_preview ?? null : null;
  const lastMetrics = processStatus?.last_generation_metrics ?? null;
  const samplingExtraArgs = buildArgs({
    temp: parseEditableNumber(temperature),
    topP: parseEditableNumber(topP),
    topK: parseEditableNumber(topK),
    minP: parseEditableNumber(minP),
    presencePenalty: parseEditableNumber(presencePenalty),
  });
  const mergedExtraArgs = [...parseCliArgs(samplingExtraArgs), ...parseCliArgs(extraArgs)];
  const loadOptions: LoadModelOptions = {
    contextSize,
    fitMode,
    useJinja,
    reasoningMode,
    templateMode,
    chatTemplateKwargsJson: chatTemplateKwargsJson.trim() || null,
    draftModelPath: draftModelPath.trim() || null,
    specType: specType.trim() || null,
    specDraftNMax,
    extraArgs: mergedExtraArgs,
  };
  const canOpenHfCache = !!(sidecarStatus?.sidecar_cache_dir || sidecarStatus?.template_cache_path);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="border-b px-4 py-4" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-start gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-[11px] font-bold uppercase" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
            {(model.family || "AI").slice(0, 2)}
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-sm font-semibold" style={{ color: "var(--text-0)" }}>{modelDisplayName(model)}</h3>
            <p className="mt-0.5 truncate text-[11px]" style={{ color: "var(--text-2)" }}>{modelPublisher(model)} · {model.quant ?? "Unquantized"} · {model.size_gb ? `${model.size_gb.toFixed(1)} GB` : "Size unknown"}</p>
          </div>
          {loaded && <span className="rounded px-2 py-0.5 text-[10px] font-bold uppercase" style={{ background: "rgba(52,211,153,0.14)", color: "#6ee7b7", border: "1px solid rgba(52,211,153,0.28)" }}>Loaded</span>}
        </div>
        {model.provider_managed && (
          <div className="hidden" aria-hidden="true" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-xs font-semibold" style={{ color: "var(--text-0)" }}>Load context</div>
                <div className="mt-0.5 text-[10px]" style={{ color: "var(--text-2)" }}>Safe local allocation</div>
              </div>
              <span className="font-mono text-sm font-semibold tabular-nums" style={{ color: "var(--text-0)" }}>
                {contextSize === 0 ? "Auto" : fmtNum(contextSize)}
              </span>
            </div>
            <input
              aria-label="Load context size"
              type="range"
              min={0}
              max={maxContext}
              step={1}
              value={contextSize}
              onChange={(event) => setClampedContextSize(Number(event.target.value))}
              className="mt-3 w-full"
            />
            <div className="mt-1 flex items-center justify-between text-[10px] tabular-nums" style={{ color: "var(--text-2)" }}>
              <span>0 · Auto</span>
              <span>{fmtNum(maxContext)} max</span>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-1.5">
              {contextPresets.map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setClampedContextSize(value)}
                  className="rounded-md px-2 py-1.5 text-xs font-semibold"
                  style={{
                    background: contextSize === value ? "var(--surface-3)" : "transparent",
                    border: contextSize === value ? "1px solid var(--text-1)" : "1px solid var(--border)",
                    color: contextSize === value ? "var(--text-0)" : "var(--text-1)",
                  }}
                >
                  {value >= 1024 ? `${value / 1024}K` : fmtNum(value)}
                  {value === defaultContext ? " · Recommended" : ""}
                </button>
              ))}
            </div>
            {contextSize === 0 ? (
              <div className="mt-2 text-[10px] font-medium leading-4" style={{ color: "#f59e0b" }}>
                Auto uses the model default; prediction assumes {fmtNum(advertisedContext)} tokens.
              </div>
            ) : customContextSelected && (
              <div className="mt-2 text-[10px] font-medium" style={{ color: "#f59e0b" }}>
                Custom advanced context selected: {fmtNum(contextSize)} tokens
              </div>
            )}
            {gpuStats && (
              <div className="mt-3">
                <PredictedVramBar predictedMb={estimateContextVRAM(predictionContext)} dedicatedMb={gpuStats.dedicated_mb} />
              </div>
            )}
            <div className="mt-2 text-[10px] leading-4" style={{ color: "var(--text-2)" }}>
              Safe presets stay available while the slider exposes the full model range.
            </div>
          </div>
        )}
        {loaded && (
          <div className="mt-3 rounded-md px-3 py-2" style={{ background: "rgba(248,113,113,0.075)", border: "1px solid rgba(248,113,113,0.22)" }}>
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="text-[10px] font-bold uppercase tracking-[0.16em]" style={{ color: "#fca5a5" }}>Global runtime</div>
                <div className="mt-0.5 text-xs" style={{ color: "var(--text-1)" }}>Unload stops the active llama-server model for API, chat, and all clients.</div>
              </div>
              <ActionBtn label="Unload Active Model" disabled={isLoading || !model.provider_managed} variant="danger" onClick={onUnload} />
            </div>
          </div>
        )}
        <div className="mt-3 grid grid-cols-2 gap-2">
          <ActionBtn label={loaded ? "Reload Options" : "Load Model"} disabled={isLoading || !model.provider_managed} variant="primary" onClick={() => onLoad(model.filename, loadOptions)} />
          <ActionBtn label="Swap In" disabled={isLoading || loaded || !loadedModel || !model.provider_managed} variant="indigo" onClick={() => onSwap(model.filename, loadOptions)} />
          <ActionBtn label={sidecarSyncing ? "Syncing HF..." : "Sync HF Files"} disabled={isLoading || sidecarSyncing || !model.hf_repo} variant="ghost" onClick={() => onSyncSidecars(model)} />
          <ActionBtn label="Open HF Cache" disabled={!canOpenHfCache} variant="ghost" onClick={() => {
            const path = sidecarStatus?.sidecar_cache_dir || sidecarStatus?.template_cache_path;
            if (path) void api.showInFolder(path);
          }} />
          {previousModel && previousModel !== model.filename && (
            <button
              onClick={onSwapBack}
              disabled={isLoading}
              className="col-span-2 rounded-md px-3 py-1.5 text-xs font-semibold"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}
            >
              Swap back to {previousModel}
            </button>
          )}
        </div>
      </div>

      <div className="border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
        <div className="flex rounded-md p-0.5" style={{ background: "var(--surface-2)" }}>
          {([
            ["info", "Info"],
            ["load", "Advanced"],
            ["inference", "Inference"],
          ] as const).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setActiveInspectorTab(key)}
              className="flex-1 rounded px-2 py-1.5 text-xs font-semibold"
              style={{
                background: activeInspectorTab === key ? "var(--surface-3)" : "transparent",
                color: activeInspectorTab === key ? "var(--text-0)" : "var(--text-1)",
                border: "none",
                cursor: "pointer",
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {activeInspectorTab === "info" && (
        <section className="px-4 py-4">
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-0)" }}>Model Information</h4>
          <InfoRow label="Model" value={shortModelName(model)} />
          <InfoRow label="File" value={model.filename} />
          <InfoRow label="Format" value={model.provider_managed ? "GGUF" : "OpenAI-compatible"} />
          <InfoRow label="Quantization" value={model.quant ?? "-"} />
          <InfoRow label="Arch" value={model.family || model.gguf_architecture || "-"} />
          <InfoRow label="Params" value={modelParamsLabel(model)} />
          <InfoRow label="Capabilities" value={[
            model.supports_vision ? "Vision" : null,
            model.supports_tools ? "Tool use" : null,
            model.supports_reasoning ? "Reasoning" : null,
          ].filter(Boolean).join(", ") || "Chat"} />
          <InfoRow label="Context" value={liveContext ? `${fmtNum(liveContext)} live` : `${fmtNum(advertisedContext)} tokens`} />
          <InfoRow label="Size on disk" value={model.size_gb ? `${model.size_gb.toFixed(2)} GB` : "-"} />
          <InfoRow label="Provider" value={model.provider_name} />
          <InfoRow label="HF Repo" value={sidecarStatus?.repo_id ?? model.hf_repo ?? "-"} />
          <InfoRow label="HF Cache" value={hfCacheSummary(sidecarStatus)} />
        </section>
      )}

      {activeInspectorTab === "load" && (
        <section className="px-4 py-4">
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-0)" }}>Advanced Load Configuration</h4>
          <InfoRow label="State" value={loaded ? "Loaded" : "Not loaded"} />
          <InfoRow label="Context" value={launchPreview?.context_size ? `${fmtNum(launchPreview.context_size)} live` : contextSize === 0 ? "Auto next load" : `${fmtNum(contextSize)} next load`} />
          <InfoRow label="Template" value={launchPreview?.template_source ?? model.template_source ?? model.template_mode ?? "-"} />
          <InfoRow label="Chat Template" value={model.has_chat_template ? "Embedded (uses --jinja)" : "Built-in fallback"} />
          <InfoRow label="Template Cache" value={sidecarStatus?.template_cached ? "Cached locally" : sidecarStatus?.repo_id ? "Missing locally" : "No HF repo"} />
          <InfoRow label="HF Sidecars" value={sidecarStatus?.repo_id ? `${sidecarStatus.sidecar_cached_count}/${sidecarStatus.sidecar_expected_count} cached` : "-"} />
          <InfoRow label="MMProj" value={launchPreview?.mmproj_path ? "Attached" : model.supports_vision ? "Not attached" : "Not required"} />
          <InfoRow label="Draft" value={launchPreview?.spec_type ? launchPreview.draft_model_path ? "Draft GGUF" : "Self-MTP" : "Disabled"} />
          {launchPreview?.spec_type && (
            <>
              <InfoRow label="Spec Type" value={launchPreview.spec_type || "-"} />
              <InfoRow label="Draft N" value={launchPreview.spec_draft_n_max ? String(launchPreview.spec_draft_n_max) : "-"} />
              {launchPreview.draft_model_path && <InfoRow label="Draft File" value={launchPreview.draft_model_path} />}
            </>
          )}
          <div className="mt-4 space-y-2">
            <EditableRow label="Context">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={minContext}
                    max={maxContext}
                    step={1}
                    value={contextSize}
                    onChange={(e) => setClampedContextSize(Number(e.target.value))}
                    className="min-w-0 flex-1"
                  />
                  <input
                    type="number"
                    min={minContext}
                    max={maxContext}
                    step={1}
                    value={contextSize}
                    onChange={(e) => setClampedContextSize(Number(e.target.value))}
                    style={{ ...editInputStyle(), width: 112 }}
                  />
                </div>
                <div className="flex justify-between text-[10px]" style={{ color: "var(--text-2)" }}>
                  <span>{fmtNum(minContext)}</span>
                  <button
                    type="button"
                    onClick={() => setClampedContextSize(defaultContext)}
                    className="rounded px-1.5 py-0.5"
                    style={{
                      background: "var(--surface-2)",
                      border: "1px solid var(--border)",
                      color: "var(--text-1)",
                    }}
                  >
                    Reset {fmtNum(defaultContext)}
                  </button>
                  <span>{fmtNum(maxContext)}</span>
                </div>
                {gpuStats && (
                  <PredictedVramBar predictedMb={estimateContextVRAM(predictionContext)} dedicatedMb={gpuStats.dedicated_mb} />
                )}
                {(contextSize === 0 || contextSize > defaultContext) && (
                  <div className="text-[10px] leading-4" style={{ color: "#f59e0b" }}>
                    {contextSize === 0
                      ? `Auto may use the ${fmtNum(advertisedContext)}-token model limit. Check the VRAM estimate before reloading.`
                      : `This exceeds the recommended ${fmtNum(defaultContext)}-token local default. Check the VRAM estimate before reloading.`}
                  </div>
                )}
              </div>
            </EditableRow>
            <EditableRow label="Fit">
              <input value={fitMode} onChange={(e) => setFitMode(e.target.value)} style={editInputStyle()} placeholder="on / off / auto" />
            </EditableRow>
            <EditableRow label="Jinja">
              <input type="checkbox" checked={useJinja} onChange={(e) => setUseJinja(e.target.checked)} />
            </EditableRow>
            <EditableRow label="Reasoning">
              <select value={reasoningMode} onChange={(e) => setReasoningMode(e.target.value)} style={editInputStyle()}>
                <option value="auto">auto</option>
                <option value="on">on</option>
                <option value="off">off</option>
              </select>
            </EditableRow>
            <EditableRow label="Template">
              <select value={templateMode} onChange={(e) => setTemplateMode(e.target.value)} style={editInputStyle()}>
                <option value="repo">repo</option>
                <option value="builtin">builtin</option>
                <option value="custom">custom</option>
              </select>
            </EditableRow>
            <EditableRow label="Kwargs JSON">
              <input value={chatTemplateKwargsJson} onChange={(e) => setChatTemplateKwargsJson(e.target.value)} style={editInputStyle()} placeholder='{"enable_thinking": true}' />
            </EditableRow>
            <EditableRow label="Draft GGUF">
              <input value={draftModelPath} onChange={(e) => setDraftModelPath(e.target.value)} style={editInputStyle()} placeholder="Only for matching draft-capable models" />
            </EditableRow>
            <EditableRow label="Spec Type">
              <input value={specType} onChange={(e) => setSpecType(e.target.value)} style={editInputStyle()} placeholder="draft-mtp" />
            </EditableRow>
            <EditableRow label="Draft N">
              <input type="number" min={0} max={16} value={specDraftNMax} onChange={(e) => setSpecDraftNMax(Math.max(0, Number(e.target.value) || 0))} style={editInputStyle()} />
            </EditableRow>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <ActionBtn label={loaded ? "Reload Options" : "Load Model"} disabled={isLoading || !model.provider_managed} variant="primary" onClick={() => onLoad(model.filename, loadOptions)} />
            <ActionBtn label="Swap In" disabled={isLoading || loaded || !loadedModel || !model.provider_managed} variant="indigo" onClick={() => onSwap(model.filename, loadOptions)} />
          </div>
        </section>
      )}

      {activeInspectorTab === "inference" && (
        <section className="px-4 py-4">
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-0)" }}>Inference</h4>
          <div className="space-y-2">
            <EditableRow label="Temperature"><input value={temperature} onChange={(e) => setTemperature(e.target.value)} style={editInputStyle()} /></EditableRow>
            <EditableRow label="Top P"><input value={topP} onChange={(e) => setTopP(e.target.value)} style={editInputStyle()} /></EditableRow>
            <EditableRow label="Top K"><input value={topK} onChange={(e) => setTopK(e.target.value)} style={editInputStyle()} /></EditableRow>
            <EditableRow label="Min P"><input value={minP} onChange={(e) => setMinP(e.target.value)} style={editInputStyle()} /></EditableRow>
            <EditableRow label="Presence"><input value={presencePenalty} onChange={(e) => setPresencePenalty(e.target.value)} style={editInputStyle()} /></EditableRow>
            <EditableRow label="Extra Args"><input value={extraArgs} onChange={(e) => setExtraArgs(e.target.value)} style={editInputStyle()} placeholder="Raw llama-server args" /></EditableRow>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <ActionBtn label={loaded ? "Reload Options" : "Load Model"} disabled={isLoading || !model.provider_managed} variant="primary" onClick={() => onLoad(model.filename, loadOptions)} />
            <ActionBtn label="Swap In" disabled={isLoading || loaded || !loadedModel || !model.provider_managed} variant="indigo" onClick={() => onSwap(model.filename, loadOptions)} />
          </div>
          <div className="mt-4">
            <InfoRow label="Launch Args" value={mergedExtraArgs.join(" ") || "-"} />
          </div>
          <InfoRow label="Last TPS" value={lastMetrics?.decode_tokens_per_second != null ? `${lastMetrics.decode_tokens_per_second.toFixed(2)} tok/s` : "-"} />
          <InfoRow label="Last Tokens" value={lastMetrics?.total_tokens != null ? `${lastMetrics.total_tokens}` : "-"} />
        </section>
      )}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-3 border-b py-2 text-xs last:border-b-0" style={{ borderColor: "var(--border)" }}>
      <span className="w-24 shrink-0" style={{ color: "var(--text-0)" }}>{label}</span>
      <span className="min-w-0 truncate rounded-full px-2 py-0.5 font-mono text-[11px]" title={value} style={{ background: "var(--surface-3)", color: "var(--text-1)" }}>{value}</span>
    </div>
  );
}

function EditableRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex items-center gap-3 text-xs">
      <span className="w-24 shrink-0" style={{ color: "var(--text-0)" }}>{label}</span>
      <span className="min-w-0 flex-1">{children}</span>
    </label>
  );
}

function editInputStyle() {
  return {
    width: "100%",
    minWidth: 0,
    borderRadius: 6,
    border: "1px solid var(--border)",
    background: "var(--surface-2)",
    color: "var(--text-0)",
    padding: "6px 8px",
    outline: "none",
  } as const;
}

function StatusPill({ state }: { state: string }) {
  const isRunning = state === "Running";
  const isStarting = state === "Starting";
  const isCrashed = state === "Crashed";

  const dotColor = isRunning
    ? "#34d399"
    : isStarting
      ? "#fbbf24"
      : isCrashed
        ? "#f87171"
        : "#6b7280";

  const textColor = isRunning
    ? "#34d399"
    : isStarting
      ? "#fbbf24"
      : isCrashed
        ? "#f87171"
        : "var(--text-1)";

  return (
    <div
      className="flex items-center gap-1.5 rounded px-2.5 py-1 text-xs font-medium"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        color: textColor,
      }}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${isStarting ? "animate-pulse" : ""}`}
        style={{ background: dotColor }}
      />
      {state}
    </div>
  );
}

// Tool button

function ToolBtn({
  label,
  onClick,
  icon,
}: {
  label: string;
  onClick: () => void;
  icon: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1.5 rounded px-2.5 py-1 text-xs transition"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        color: "var(--text-1)",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLButtonElement).style.color = "var(--text-0)";
        (e.currentTarget as HTMLButtonElement).style.borderColor =
          "var(--border-mid)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLButtonElement).style.color = "var(--text-1)";
        (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border)";
      }}
    >
      {icon}
      {label}
    </button>
  );
}

// Loading bar

function LoadingBar({ progress }: { progress: LoadProgress }) {
  const pct = Math.round(progress.progress * 100);
  const stageLabel =
    progress.stage === "resolving"
      ? "Resolving"
      : progress.stage === "downloading"
        ? "Downloading"
        : progress.stage === "launching"
          ? "Launching"
          : progress.stage === "starting"
            ? "Starting"
            : progress.stage === "loading"
              ? "Loading model"
              : progress.stage === "ready"
                ? "Ready"
                : progress.stage;

  return (
    <div className="px-3 py-2.5">
      <div className="mb-1.5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            className="h-3.5 w-3.5 animate-spin"
            style={{ color: "#22d3ee" }}
          >
            <circle cx="12" cy="12" r="9" strokeOpacity="0.2" stroke="currentColor" strokeWidth="3" />
            <path d="M21 12A9 9 0 0 0 12 3" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
          </svg>
          <span className="text-xs font-medium" style={{ color: "#22d3ee" }}>
            {stageLabel}
          </span>
          <span className="text-xs" style={{ color: "var(--text-1)" }}>
            {progress.message}
          </span>
        </div>
        <span className="text-xs font-semibold" style={{ color: "var(--text-0)" }}>
          {pct}%
        </span>
      </div>
      <div
        className="h-1 w-full overflow-hidden rounded-full"
        style={{ background: "rgba(255,255,255,0.08)" }}
      >
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${pct}%`,
            background: "#8ab4f8",
          }}
        />
      </div>
    </div>
  );
}

function LoadErrorHint({ message }: { message: string }) {
  const lower = message.toLowerCase();
  let hint: string | null = null;
  if (
    lower.includes("self-mtp speculative decoding") ||
    lower.includes("context type mtp requested") ||
    lower.includes("does not contain mtp prediction-head") ||
    lower.includes("doesn't contain mtp layers") ||
    lower.includes("does not contain mtp layers")
  ) {
    hint = "Speculative MTP decoding was enabled for a GGUF without MTP layers. Turn off Speculative decoding in Load options, or choose an MTP-capable model.";
  } else if (lower.includes("chat_template.jinja") && lower.includes("404")) {
    hint = "That repo does not expose a standalone chat_template.jinja. InferenceBridge will fall back to the embedded GGUF template when available.";
  } else if (
    lower.includes("huggingface.co") &&
    (lower.includes("401") || lower.includes("403") || lower.includes("unauthorized") || lower.includes("forbidden"))
  ) {
    hint = "This Hugging Face repo may be gated or private. Add an HF access token in Settings > Hugging Face, then retry.";
  } else if (lower.includes("template")) {
    hint = "Try Template: builtin with Jinja enabled, or add an HF token if the template lives in a gated repo.";
  }

  if (!hint) return null;
  return (
    <div className="px-4 pb-2 text-xs" style={{ background: "rgba(239,68,68,0.08)", color: "#fecaca" }}>
      {hint}
    </div>
  );
}

// Loaded model row

function LoadedModelRow({
  model,
  previousModel,
  processStatus,
  onUnload,
  onSwapBack,
  isLoading,
}: {
  model: ModelInfo;
  previousModel: string | null;
  processStatus: ProcessStatusInfo | null;
  onUnload: () => void;
  onSwapBack: () => void;
  isLoading: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const liveContextSize =
    processStatus?.model === model.filename
      ? processStatus.last_launch_preview?.context_size ?? null
      : null;
  const liveContextLabel =
    liveContextSize != null
      ? liveContextSize >= 1024
        ? `${(liveContextSize / 1024).toFixed(
            liveContextSize % 1024 === 0 ? 0 : 1
          )}K ctx`
        : `${liveContextSize} ctx`
      : null;
  const profileContextLabel = formatContext(
    model.context_window,
    model.max_context_window
  );

  return (
    <div style={{ borderLeft: "2px solid rgba(34,211,238,0.4)" }}>
      {/* Main row */}
      <div
        className="flex flex-wrap items-center gap-3 px-3 py-3"
        style={{ background: "rgba(34,211,238,0.04)" }}
      >
        {/* Ready badge */}
        <span
          className="shrink-0 rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest"
          style={{
            background: "rgba(52,211,153,0.12)",
            border: "1px solid rgba(52,211,153,0.25)",
            color: "#34d399",
          }}
        >
          {processStatus?.state?.toUpperCase() ?? "READY"}
        </span>

        {/* Name */}
        <span
          className="min-w-0 flex-1 truncate text-sm font-medium"
          style={{ color: "var(--text-0)" }}
        >
          {model.filename}
        </span>

        {/* Meta */}
        <div className="flex flex-wrap items-center gap-3 text-xs" style={{ color: "var(--text-1)" }}>
          <ProviderBadge providerName={model.provider_name} managed={model.provider_managed} />
          {model.family && <span>{model.family}</span>}
          {model.quant && (
            <span style={{ color: "#fbbf24" }}>{model.quant}</span>
          )}
          <span>{model.size_gb.toFixed(2)} GB</span>
          {liveContextLabel ? (
            <span style={{ color: "#22d3ee" }}>{liveContextLabel}</span>
          ) : profileContextLabel ? (
            <span>{profileContextLabel}</span>
          ) : null}
          {processStatus?.backend && (
            <span style={{ color: "#22d3ee" }}>{processStatus.backend}</span>
          )}
          {model.supports_reasoning && <CapBadge label="Reasoning" tone="amber" />}
          {model.supports_tools && <CapBadge label="Tools" tone="emerald" />}
          {model.supports_vision && (
            <CapBadge
              label={model.vision_runtime_ready ? "Vision Ready" : model.vision_status}
              tone={model.vision_runtime_ready ? "rose" : "slate"}
            />
          )}
          {model.think_tag_style !== "None" && (
            <CapBadge label={`Think ${model.think_tag_style}`} tone="violet" />
          )}
          {model.tool_call_format !== "NativeApi" && (
            <CapBadge label={fmtToolFormat(model.tool_call_format)} tone="cyan" />
          )}
          {model.template_mode && (
            <CapBadge label={`Template ${model.template_mode}`} tone="slate" />
          )}
          {model.has_chat_template && <CapBadge label="Embedded Template" tone="emerald" />}
          {model.gguf_architecture && <CapBadge label={model.gguf_architecture} tone="slate" />}
        </div>

        {/* Actions */}
        <div className="flex shrink-0 items-center gap-1.5">
          <button
            onClick={() => setExpanded((v) => !v)}
            className="rounded px-2 py-1 text-xs transition"
            style={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text-1)",
              cursor: "pointer",
            }}
          >
            {expanded ? "^" : "v"}
          </button>
          {previousModel && previousModel !== model.filename && (
            <ActionBtn
              onClick={onSwapBack}
              disabled={isLoading}
              label="Swap Back"
              variant="indigo"
            />
          )}
          <ActionBtn
            onClick={onUnload}
            disabled={isLoading}
            label="Eject"
            variant="ghost"
          />
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div
          className="px-4 pb-3 pt-2"
          style={{ background: "rgba(34,211,238,0.02)", borderTop: "1px solid var(--border)" }}
        >
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile label="Size" value={`${model.size_gb.toFixed(2)} GB`} />
            <StatTile
              label="Live Context"
              value={liveContextSize ? `${fmtNum(liveContextSize)} tokens` : "Unknown"}
            />
            <StatTile label="Max Output" value={`${fmtNum(model.max_output_tokens)} tokens`} />
            <StatTile label="Tool Format" value={fmtToolFormat(model.tool_call_format)} />
            <StatTile label="Profile Defaults" value={profileDefaultSummary(model)} />
            <StatTile label="Reasoning" value={model.think_tag_style === "None" ? "Off" : model.think_tag_style} />
          </div>
        </div>
      )}
    </div>
  );
}

// Model row

function ModelRow({
  model,
  isLoading,
  showSwap,
  isLast,
  onLoad,
  onSwap,
}: {
  model: ModelInfo;
  isLoading: boolean;
  showSwap: boolean;
  isLast: boolean;
  onLoad: (options?: LoadModelOptions) => void;
  onSwap: (options?: LoadModelOptions) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [hovered, setHovered] = useState(false);
  const defaultCtx = safeDefaultContext(model);
  const detectedCtx = advertisedContextLimit(model);
  // Use the model's true training context ceiling from the profile, falling back to a conservative multiple
  const maxCtx = model.max_context_window ?? Math.max(defaultCtx * 4, 32768);
  const minCtx = 512;
  const [contextSize, setContextSize] = useState(defaultCtx);
  const [fitMode, setFitMode] = useState("off");
  const [useJinja, setUseJinja] = useState(model.template_mode === "repo");
  const [reasoningMode, setReasoningMode] = useState("on");
  const [templateMode, setTemplateMode] = useState(model.template_mode ?? "builtin");
  const [chatTemplateKwargsJson, setChatTemplateKwargsJson] = useState("");
  const [extraArgs, setExtraArgs] = useState("");
  const [configName, setConfigName] = useState("");
  const [savedConfigs, setSavedConfigs] = useState<SavedModelConfig[]>(() => {
    try {
      const parsed = JSON.parse(window.localStorage.getItem(savedConfigKey(model.filename)) ?? "[]");
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });
  const [supportsVision, setSupportsVision] = useState(model.supports_vision);
  const [visionSaving, setVisionSaving] = useState(false);
  const isExternalProvider = !model.provider_managed;
  const recommendedProfiles = recommendedProfilesForModel(model);

  function persistSavedConfigs(configs: SavedModelConfig[]) {
    setSavedConfigs(configs);
    window.localStorage.setItem(savedConfigKey(model.filename), JSON.stringify(configs));
  }

  function applySavedConfig(config: SavedModelConfig) {
    setContextSize(config.contextSize);
    setFitMode(config.fitMode);
    setUseJinja(config.useJinja);
    setReasoningMode(config.reasoningMode);
    setTemplateMode(config.templateMode);
    setChatTemplateKwargsJson(config.chatTemplateKwargsJson);
    setExtraArgs(config.extraArgs);
  }

  function setManualContextSize(value: number) {
    setContextSize(value);
    setFitMode("off");
  }

  function applyRecommendedConfig(config: RecommendedModelConfig) {
    setReasoningMode(config.reasoningMode);
    if (config.useJinja != null) setUseJinja(config.useJinja);
    if (config.templateMode) setTemplateMode(config.templateMode);
    setChatTemplateKwargsJson(config.chatTemplateKwargsJson ?? "");
    setExtraArgs(config.extraArgs);
  }

  function saveCurrentConfig() {
    const name = configName.trim() || "Custom";
    const config: SavedModelConfig = {
      name,
      contextSize,
      fitMode,
      useJinja,
      reasoningMode,
      templateMode,
      chatTemplateKwargsJson,
      extraArgs,
    };
    persistSavedConfigs([...savedConfigs.filter((item) => item.name !== name), config]);
    setConfigName("");
  }

  function removeSavedConfig(name: string) {
    persistSavedConfigs(savedConfigs.filter((item) => item.name !== name));
  }

  async function toggleVisionOverride() {
    const next = !supportsVision;
    setSupportsVision(next);
    setVisionSaving(true);
    try {
      await api.setModelVisionOverride(model.filename, next);
    } catch {
      setSupportsVision(!next);
    } finally {
      setVisionSaving(false);
    }
  }

  const loadOptions: LoadModelOptions = {
    contextSize,
    fitMode,
    useJinja,
    reasoningMode,
    templateMode,
    chatTemplateKwargsJson: chatTemplateKwargsJson.trim() || undefined,
    extraArgs: parseCliArgs(extraArgs),
  };
  // Use GPU stats hook for live VRAM/overflow info
  const gpuStats = useGpuStats();

  // KV-cache bytes per token using architecture metadata from GGUF.
  // Default cache type is q8_0 (1 byte/element), matching ProcessConfig default.
  // Formula: n_layers x 2 (K+V) x n_kv_heads x head_dim x bytes_per_element
  const KV_BPE = 1.0; // q8_0
  const kvBytesPerToken: number | null =
    model.n_layers != null && model.n_kv_heads != null && model.head_dim != null
      ? model.n_layers * 2 * model.n_kv_heads * model.head_dim * KV_BPE
      : null;

  function estimateContextVRAM(tokens: number): number {
    const modelMb = (model.size_gb || 0) * 1024;
    const graphOverheadMb = Math.max(512, Math.min(2048, modelMb * 0.08));
    if (kvBytesPerToken != null) {
      const kvMb = (tokens * kvBytesPerToken) / (1024 * 1024);
      return modelMb + graphOverheadMb + kvMb * 1.15;
    }
    // Fallback when GGUF metadata is unavailable. llama.cpp still allocates
    // sizeable KV/cache/graph buffers; using a tiny bytes-per-token fallback
    // under-reports large-context Gemma/Qwen loads by several GB.
    const name = `${model.filename} ${model.family ?? ""}`.toLowerCase();
    const fallbackKvMbPerToken =
      name.includes("gemma-4-26b") || name.includes("a4b")
        ? 0.04
        : modelMb > 10 * 1024
          ? 0.035
          : 0.025;
    return modelMb + graphOverheadMb + tokens * fallbackKvMbPerToken;
  }


  return (
    <div
      style={{
        borderBottom: isLast ? "none" : "1px solid var(--border)",
      }}
    >
      {/* Main row */}
      <div
        className="flex flex-wrap items-center gap-3 px-3 py-2.5 transition"
        style={{
          background: hovered ? "rgba(255,255,255,0.03)" : "transparent",
          cursor: "default",
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {/* Family chip */}
        {model.family && (
          <span
            className="shrink-0 rounded px-2 py-0.5 text-[10px] uppercase tracking-wider"
            style={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text-1)",
            }}
          >
            {model.family}
          </span>
        )}

        {/* Name */}
        <button
          onClick={() => setExpanded((v) => !v)}
          className="min-w-0 flex-1 truncate text-left text-sm font-medium transition"
          style={{
            color: "var(--text-0)",
            background: "none",
            border: "none",
            cursor: "pointer",
            padding: 0,
          }}
        >
          {model.filename}
        </button>

        {/* Meta */}
        <div
          className="flex flex-wrap items-center gap-3 text-xs"
          style={{ color: "var(--text-1)" }}
        >
          <ProviderBadge providerName={model.provider_name} managed={model.provider_managed} />
          {model.quant && <span style={{ color: "#fbbf24" }}>{model.quant}</span>}
          {model.size_gb > 0 && <span>{model.size_gb.toFixed(2)} GB</span>}
          {formatContext(model.context_window, model.max_context_window) && (
            <span>{formatContext(model.context_window, model.max_context_window)}</span>
          )}
          {contextSize !== defaultCtx && (
            <span style={{ color: "#22d3ee" }}>
              {contextSize.toLocaleString()} ctx
            </span>
          )}
          {model.supports_reasoning && <CapBadge label="Reasoning" tone="amber" />}
          {model.supports_tools && <CapBadge label="Tools" tone="emerald" />}
          {model.supports_vision && (
            <CapBadge
              label={model.vision_runtime_ready ? "Vision Ready" : model.vision_status}
              tone={model.vision_runtime_ready ? "rose" : "slate"}
            />
          )}
          {model.think_tag_style !== "None" && (
            <CapBadge label={`Think ${model.think_tag_style}`} tone="violet" />
          )}
          {model.tool_call_format !== "NativeApi" && (
            <CapBadge label={fmtToolFormat(model.tool_call_format)} tone="cyan" />
          )}
          {model.template_mode && (
            <CapBadge label={`Template ${model.template_mode}`} tone="slate" />
          )}
          {model.has_chat_template && <CapBadge label="Embedded Template" tone="emerald" />}
          {model.gguf_architecture && <CapBadge label={model.gguf_architecture} tone="slate" />}
        </div>

        {/* Actions */}
        <div className="flex shrink-0 items-center gap-1.5">
          <button
            onClick={() => setExpanded((v) => !v)}
            className="rounded px-2 py-1 text-xs transition"
            style={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text-1)",
              cursor: "pointer",
            }}
          >
            {expanded ? "^" : "v"}
          </button>
          {isExternalProvider ? (
            <ActionBtn
              onClick={() => undefined}
              disabled
              label="Provider Routed"
              variant="ghost"
            />
          ) : showSwap ? (
            <ActionBtn
              onClick={() => onSwap(loadOptions)}
              disabled={isLoading}
              label="Swap In"
              variant="indigo"
            />
          ) : (
            <ActionBtn
              onClick={() => onLoad(loadOptions)}
              disabled={isLoading}
              label="Load"
              variant="primary"
            />
          )}
        </div>
      </div>

      {/* Expanded section */}
      {expanded && (
        <div
          className="px-4 pb-3 pt-2"
          style={{
            background: "var(--surface-2)",
            borderTop: "1px solid var(--border)",
          }}
        >
          {/* Context slider with live VRAM/overflow monitor */}
          {!isExternalProvider && <div className="mb-3">
            <div className="mb-2 flex items-center justify-between text-xs">
              <span style={{ color: "var(--text-1)" }}>Context size</span>
              <div className="flex items-center gap-2">
                <span className="font-mono font-semibold" style={{ color: "#22d3ee" }}>
                  {contextSize.toLocaleString()}
                </span>
                <button
                  onClick={() => setManualContextSize(defaultCtx)}
                  className="rounded px-1.5 py-0.5 text-[10px] transition"
                  style={{
                    background: "var(--surface-3)",
                    border: "1px solid var(--border)",
                    color: "var(--text-1)",
                    cursor: "pointer",
                  }}
                >
                  Reset
                </button>
              </div>
            </div>
            <input
              type="range"
              min={minCtx}
              max={maxCtx}
              step={1}
              value={contextSize}
              onChange={(e) => setManualContextSize(Number(e.target.value))}
            />
            <div className="mt-1 flex justify-between text-[10px]" style={{ color: "var(--text-2)" }}>
              <span>{minCtx.toLocaleString()}</span>
              <span>Safe default: {defaultCtx.toLocaleString()}</span>
              <span>{maxCtx.toLocaleString()}</span>
            </div>
            {detectedCtx > defaultCtx && (
              <div className="mt-1 text-[10px]" style={{ color: "#fbbf24" }}>
                Model metadata advertises {detectedCtx.toLocaleString()} ctx; manual loads default lower to avoid oversized KV allocation.
              </div>
            )}
            {/* VRAM/overflow monitor as slider bar */}
            {gpuStats && (
              <div className="mt-2">
                <VramBar
                  usedMb={estimateContextVRAM(contextSize)}
                  dedicatedMb={gpuStats.dedicated_mb}
                  systemRamMb={gpuStats.system_ram_mb}
                  mode="estimate"
                />
              </div>
            )}
          </div>}

          {/* Stats grid */}
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile label="File Size" value={model.size_gb && model.size_gb > 0 ? `${model.size_gb.toFixed(2)} GB` : 'N/A'} />
            <StatTile label="Provider" value={model.provider_name} />
            <StatTile label="Default Context" value={model.context_window ? `${fmtNum(model.context_window)} tokens` : "Unknown"} />
            <StatTile label="Max Context" value={model.max_context_window ? `${fmtNum(model.max_context_window)} tokens` : "Unknown"} />
            <StatTile label="Tool Format" value={fmtToolFormat(model.tool_call_format)} />
          </div>
          {(!isExternalProvider && (!model.size_gb || model.size_gb === 0)) && (
            <div style={{ color: '#fbbf24', fontSize: 12, marginTop: 4 }}>
              File size missing? Try <b>Rescan</b> in the toolbar above.
            </div>
          )}

          {/* Model path + open folder */}
          <div
            className="mt-2 flex items-center gap-2 rounded px-2.5 py-1.5"
            style={{ background: "var(--surface-3)", border: "1px solid var(--border)" }}
          >
            <span className="text-[10px] uppercase tracking-widest shrink-0" style={{ color: "var(--text-2)" }}>
              {isExternalProvider ? "Base URL" : "Path"}
            </span>
            <span
              className="flex-1 min-w-0 truncate font-mono text-[11px]"
              style={{ color: "var(--text-1)" }}
              title={isExternalProvider ? model.provider_base_url ?? model.path : model.path}
            >
              {isExternalProvider ? model.provider_base_url ?? model.path : model.path}
            </span>
            {!isExternalProvider && <button
              onClick={() => api.showInFolder(model.path)}
              className="shrink-0 rounded px-2 py-0.5 text-[10px] font-medium transition"
              style={{
                background: "var(--surface-2)",
                border: "1px solid var(--border)",
                color: "var(--text-1)",
                cursor: "pointer",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLButtonElement).style.color = "var(--text-0)";
                (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(34,211,238,0.3)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLButtonElement).style.color = "var(--text-1)";
                (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border)";
              }}
            >
              Open Folder
            </button>}
          </div>

          {!isExternalProvider && (
            <div
              className="mt-3 rounded px-3 py-2"
              style={{ background: "var(--surface-3)", border: "1px solid var(--border)" }}
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[10px] uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
                  Model configs
                </span>
                {recommendedProfiles.map((profile) => (
                  <button
                    key={profile.name}
                    onClick={() => applyRecommendedConfig(profile)}
                    className="rounded px-2 py-1 text-[10px] font-semibold transition"
                    style={{
                      background: "rgba(34,211,238,0.1)",
                      border: "1px solid rgba(34,211,238,0.24)",
                      color: "#67e8f9",
                    }}
                    title={`${profile.source}: ${profile.extraArgs}`}
                  >
                    {profile.name}
                  </button>
                ))}
                {savedConfigs.map((config) => (
                  <span key={config.name} className="inline-flex items-center gap-1">
                    <button
                      onClick={() => applySavedConfig(config)}
                      className="rounded px-2 py-1 text-[10px] font-semibold transition"
                      style={{
                        background: "var(--surface-2)",
                        border: "1px solid var(--border)",
                        color: "var(--text-0)",
                      }}
                    >
                      {config.name}
                    </button>
                    <button
                      onClick={() => removeSavedConfig(config.name)}
                      className="rounded px-1.5 py-1 text-[10px] transition"
                      style={{
                        background: "transparent",
                        border: "1px solid var(--border)",
                        color: "var(--text-2)",
                      }}
                      title={`Remove ${config.name}`}
                    >
                      x
                    </button>
                  </span>
                ))}
                <div className="ml-auto flex min-w-[220px] items-center gap-1">
                  <input
                    type="text"
                    value={configName}
                    onChange={(e) => setConfigName(e.target.value)}
                    placeholder="Config name"
                    className="min-w-0 flex-1 rounded px-2 py-1 text-xs"
                    style={{ background: "var(--surface-1)", border: "1px solid var(--border)", color: "var(--text-0)" }}
                  />
                  <button
                    onClick={saveCurrentConfig}
                    className="shrink-0 rounded px-2 py-1 text-[10px] font-semibold transition"
                    style={{ background: "#22d3ee", border: "none", color: "#0a0a0a" }}
                  >
                    Save
                  </button>
                </div>
              </div>
            </div>
          )}

          {!isExternalProvider && <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
                Template mode
              </div>
              <select
                value={templateMode}
                onChange={(e) => setTemplateMode(e.target.value)}
                className="w-full rounded px-2 py-1.5 text-xs"
                style={{ background: "var(--surface-1)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              >
                <option value="builtin">Bridge fallback</option>
                <option value="repo">Repo template</option>
                <option value="custom">Custom template</option>
              </select>
            </div>
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
                Reasoning mode
              </div>
              <select
                value={reasoningMode}
                onChange={(e) => setReasoningMode(e.target.value)}
                className="w-full rounded px-2 py-1.5 text-xs"
                style={{ background: "var(--surface-1)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              >
                <option value="on">On</option>
                <option value="off">Off</option>
                <option value="auto">Auto</option>
              </select>
            </div>
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
                Fit mode
              </div>
              <select
                value={fitMode}
                onChange={(e) => setFitMode(e.target.value)}
                className="w-full rounded px-2 py-1.5 text-xs"
                style={{ background: "var(--surface-1)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              >
                <option value="on">On</option>
                <option value="off">Off</option>
              </select>
            </div>
            <label className="flex items-center gap-2 text-xs" style={{ color: "var(--text-1)" }}>
              <input type="checkbox" checked={useJinja} onChange={(e) => setUseJinja(e.target.checked)} />
              Use Jinja rendering
            </label>
            <label
              className="flex items-center gap-2 text-xs"
              style={{ color: supportsVision ? "#f472b6" : "var(--text-1)", opacity: visionSaving ? 0.6 : 1 }}
              title="Marks this model as vision-capable so InferenceBridge looks for a matching mmproj sidecar on next load"
            >
              <input
                type="checkbox"
                checked={supportsVision}
                disabled={visionSaving}
                onChange={toggleVisionOverride}
              />
              Supports vision (override)
            </label>
            <div className="sm:col-span-2">
              <div className="mb-1 text-[10px] uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
                Template kwargs JSON
              </div>
              <input
                type="text"
                value={chatTemplateKwargsJson}
                onChange={(e) => setChatTemplateKwargsJson(e.target.value)}
                placeholder='{"preserve_thinking": true}'
                className="w-full rounded px-2 py-1.5 text-xs"
                style={{ background: "var(--surface-1)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              />
            </div>
            <div className="sm:col-span-2">
              <div className="mb-1 text-[10px] uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
                Extra args
              </div>
              <textarea
                value={extraArgs}
                onChange={(e) => setExtraArgs(e.target.value)}
                placeholder="--temp 0.6 --top-p 0.95 --cache-type-k q8_0"
                rows={2}
                className="w-full rounded px-2 py-1.5 text-xs"
                style={{ background: "var(--surface-1)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              />
            </div>
          </div>}
        </div>
      )}
    </div>
  );
}

// Empty state

function EmptyMsg({ title, body, fill = false }: { title: string; body: string; fill?: boolean }) {
  return (
    <div className={fill ? "flex h-full min-h-0 flex-col items-center justify-center px-4 py-8 text-center" : "px-4 py-12 text-center"}>
      <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
        {title}
      </p>
      <p className="mx-auto mt-1 max-w-sm text-xs" style={{ color: "var(--text-2)" }}>
        {body}
      </p>
    </div>
  );
}

// Stat tile

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="rounded px-3 py-2"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
      }}
    >
      <p className="text-[10px] uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
        {label}
      </p>
      <p className="mt-0.5 text-xs font-semibold" style={{ color: "var(--text-0)" }}>
        {value}
      </p>
    </div>
  );
}

// Capability badge

function CapBadge({ label, tone }: { label: string; tone: "amber" | "emerald" | "rose" | "cyan" | "violet" | "slate" }) {
  const colors: Record<string, [string, string]> = {
    amber: ["rgba(251,191,36,0.1)", "rgba(251,191,36,0.25)"],
    emerald: ["rgba(52,211,153,0.1)", "rgba(52,211,153,0.25)"],
    rose: ["rgba(248,113,113,0.1)", "rgba(248,113,113,0.25)"],
    cyan: ["rgba(34,211,238,0.1)", "rgba(34,211,238,0.25)"],
    violet: ["rgba(167,139,250,0.1)", "rgba(167,139,250,0.25)"],
    slate: ["rgba(148,163,184,0.08)", "rgba(148,163,184,0.22)"],
  };
  const textColors: Record<string, string> = {
    amber: "#fcd34d",
    emerald: "#6ee7b7",
    rose: "#fca5a5",
    cyan: "#67e8f9",
    violet: "#c4b5fd",
    slate: "#cbd5e1",
  };
  const [bg, border] = colors[tone];
  return (
    <span
      className="rounded px-1.5 py-0.5 text-[10px] font-medium"
      style={{ background: bg, border: `1px solid ${border}`, color: textColors[tone] }}
    >
      {label}
    </span>
  );
}

// Action button

function ProviderBadge({ providerName, managed }: { providerName: string; managed: boolean }) {
  return (
    <span
      className="shrink-0 rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider"
      style={{
        background: managed ? "rgba(34,211,238,0.10)" : "rgba(52,211,153,0.10)",
        border: managed ? "1px solid rgba(34,211,238,0.22)" : "1px solid rgba(52,211,153,0.22)",
        color: managed ? "#22d3ee" : "#34d399",
      }}
      title={managed ? "Managed by InferenceBridge" : "External provider routed through InferenceBridge"}
    >
      {providerName}
    </span>
  );
}

function ActionBtn({
  label,
  onClick,
  disabled,
  variant,
}: {
  label: string;
  onClick: () => void;
  disabled: boolean;
  variant: "primary" | "ghost" | "indigo" | "danger";
}) {
  const styles: Record<string, { bg: string; border: string; color: string }> = {
    primary: {
      bg: "#f4f4f4",
      border: "transparent",
      color: "#171717",
    },
    ghost: {
      bg: "var(--surface-2)",
      border: "var(--border)",
      color: "var(--text-1)",
    },
    indigo: {
      bg: "rgba(255,255,255,0.08)",
      border: "rgba(255,255,255,0.14)",
      color: "var(--text-0)",
    },
    danger: {
      bg: "rgba(248,113,113,0.16)",
      border: "rgba(248,113,113,0.35)",
      color: "#fca5a5",
    },
  };
  const s = styles[variant];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded px-3 py-1 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-50"
      style={{
        background: s.bg,
        border: `1px solid ${s.border}`,
        color: s.color,
        cursor: disabled ? "not-allowed" : "pointer",
      }}
      onMouseEnter={(e) => {
        if (!disabled && variant === "primary") {
          (e.currentTarget as HTMLButtonElement).style.filter = "brightness(1.08)";
        }
      }}
      onMouseLeave={(e) => {
        if (variant === "primary") {
          (e.currentTarget as HTMLButtonElement).style.filter = "";
        }
      }}
    >
      {label}
    </button>
  );
}

// Icons

function GearIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
      <path d="M12 8.75A3.25 3.25 0 1 0 12 15.25A3.25 3.25 0 1 0 12 8.75Z" stroke="currentColor" strokeWidth="1.8" />
      <path d="M19.4 13.5C19.47 13.01 19.5 12.51 19.5 12C19.5 11.49 19.47 10.99 19.4 10.5L21.35 8.98L19.52 5.82L17.17 6.65C16.39 6.02 15.49 5.53 14.5 5.21L14.14 2.75H9.86L9.5 5.21C8.51 5.53 7.61 6.02 6.83 6.65L4.48 5.82L2.65 8.98L4.6 10.5C4.53 10.99 4.5 11.49 4.5 12C4.5 12.51 4.53 13.01 4.6 13.5L2.65 15.02L4.48 18.18L6.83 17.35C7.61 17.98 8.51 18.47 9.5 18.79L9.86 21.25H14.14L14.5 18.79C15.49 18.47 16.39 17.98 17.17 17.35L19.52 18.18L21.35 15.02L19.4 13.5Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
      <path d="M9 9.75A1.75 1.75 0 0 1 10.75 8H18A2 2 0 0 1 20 10V18A2 2 0 0 1 18 20H10.75A1.75 1.75 0 0 1 9 18.25V9.75Z" stroke="currentColor" strokeWidth="1.7" />
      <path d="M15 8V6A2 2 0 0 0 13 4H6A2 2 0 0 0 4 6V13A2 2 0 0 0 6 15H9" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
      <path d="M12 5V19M5 12H19" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
      <path d="M11 19A8 8 0 1 0 11 3A8 8 0 1 0 11 19Z" stroke="currentColor" strokeWidth="1.8" />
      <path d="M20 20L16.65 16.65" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

void [Panel, Divider, LegacyModelInspectorPane, LoadedModelRow, ModelRow, GearIcon, CopyIcon, PlusIcon, SearchIcon];
