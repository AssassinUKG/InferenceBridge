import { useEffect, useState, type ReactNode } from "react";
import type {
  ApiServerAction,
  AppSettings,
  LoadProgress,
  ModelInfo,
  ProcessStatusInfo,
} from "../../lib/types";
import { useGpuStats } from "../../hooks/useGpuStats";
import * as api from "../../lib/tauri";
import { parseCliArgs } from "../../lib/args";
import type { LoadModelOptions } from "../../lib/tauri";

interface Props {
  models: ModelInfo[];
  loadedModel: string | null;
  previousModel: string | null;
  processStatus: ProcessStatusInfo | null;
  settings: AppSettings | null;
  error: string | null;
  isLoading: boolean;
  loadProgress: LoadProgress | null;
  onLoad: (modelName: string, options?: LoadModelOptions) => void;
  onUnload: () => void;
  onSwap: (modelName?: string, options?: LoadModelOptions) => void;
  onSetApiServerRunning: (running: boolean) => void;
  apiAction?: ApiServerAction;
  onScan: () => void;
  onOpenSettings: () => void;
}

const FILTERS = ["all", "reasoning", "tools", "vision", "loaded"] as const;
type FilterKey = (typeof FILTERS)[number];

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

function safeDefaultContext(model: ModelInfo) {
  const detected = model.context_window ?? 8192;
  const name = `${model.filename} ${model.family ?? ""}`.toLowerCase();
  const largeLocalModel =
    model.provider_managed &&
    ((model.size_gb ?? 0) >= 12 ||
      name.includes("qwen") ||
      name.includes("gemma") ||
      name.includes("27b") ||
      name.includes("26b"));
  return largeLocalModel ? Math.min(detected, 16384) : detected;
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
  onLoad,
  onUnload,
  onSwap,
  onSetApiServerRunning,
  apiAction = null,
  onScan,
  onOpenSettings,
}: Props) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterKey>("all");
  const [archFilter, setArchFilter] = useState("");
  const [paramsFilter, setParamsFilter] = useState("");
  const [llmFilter, setLlmFilter] = useState("");
  const [providerFilter, setProviderFilter] = useState("");
  const [quantFilter, setQuantFilter] = useState("");
  const [copied, setCopied] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
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
    if (archFilter && !(m.family || m.gguf_architecture || "").toLowerCase().includes(archFilter.toLowerCase())) return false;
    if (paramsFilter && !modelParamsLabel(m).toLowerCase().includes(paramsFilter.toLowerCase())) return false;
    if (llmFilter && !shortModelName(m).toLowerCase().includes(llmFilter.toLowerCase())) return false;
    if (providerFilter && !modelPublisher(m).toLowerCase().includes(providerFilter.toLowerCase())) return false;
    if (quantFilter && !(m.quant ?? "").toLowerCase().includes(quantFilter.toLowerCase())) return false;
    if (filter === "reasoning") return m.supports_reasoning;
    if (filter === "tools") return m.supports_tools;
    if (filter === "vision") return m.supports_vision;
    if (filter === "loaded") return m.filename === loadedModel;
    return true;
  });

  const activeModel = loadedModel
    ? (models.find((m) => m.filename === loadedModel) ?? {
        filename: loadedModel,
        path: "",
        size_gb: 0,
        family: "Loaded via API",
        supports_tools: false,
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
    filteredModels.find((m) => m.path === selectedPath) ??
    activeModel ??
    filteredModels[0] ??
    null;
  const localDiskGb = models
    .filter((m) => m.provider_managed)
    .reduce((sum, m) => sum + (m.size_gb || 0), 0);

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
      const summary = await api.syncHfSidecarCache(localModelNames);
      const tokenHint = summary.hf_token_configured ? "HF token used" : "public access";
      setSidecarSyncMessage(
        `HF sidecars synced: ${summary.files_cached} cached, ${summary.files_skipped} skipped, ${summary.files_failed} failed across ${summary.repos_checked} repos (${tokenHint}).`
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
      const summary = await api.syncHfSidecarCache([model.filename]);
      const failed = summary.files_failed > 0 ? `, ${summary.files_failed} failed` : "";
      setSidecarSyncMessage(
        `${shortModelName(model)} HF files synced: ${summary.files_cached} cached, ${summary.files_skipped} skipped${failed}.`
      );
      const statuses = await api.getHfSidecarCacheStatus();
      setSidecarStatuses(Object.fromEntries(statuses.map((status) => [status.filename, status])));
    } catch (error) {
      setSidecarSyncMessage(`HF sidecar sync failed for ${shortModelName(model)}: ${String(error)}`);
    } finally {
      setSidecarSyncingModel(null);
    }
  };

  const state = processStatus?.state ?? "Idle";
  const apiState = processStatus?.api_state ?? "Idle";
  const apiStopping = apiAction === "stopping" || apiState === "Stopping";
  const apiStarting = apiAction === "starting" || apiState === "Starting";
  const apiRunning = apiState === "Running" || apiStarting || apiStopping || !!processStatus?.api_reachable;
  const apiBusy = apiStarting || apiStopping;
  const apiButtonState = apiStopping ? "Stopping..." : apiStarting ? "Starting..." : apiRunning ? apiState === "Running" ? "Running" : "On" : "Off";

  return (
    <div className="flex h-full min-h-0 overflow-hidden" style={{ background: "var(--bg)" }}>
      <aside className="hidden w-[172px] shrink-0 border-r md:block" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
        <div className="px-4 py-4">
          <div className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>My Models</div>
          <div className="mt-4 space-y-1">
            {FILTERS.map((key) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                className="flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm transition"
                style={{
                  background: filter === key ? "rgba(99,102,241,0.16)" : "transparent",
                  color: filter === key ? "var(--text-0)" : "var(--text-1)",
                  fontWeight: filter === key ? 600 : 400,
                  boxShadow: filter === key ? "inset 2px 0 0 #6366f1" : "none",
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

      <section className="flex min-h-0 min-w-0 flex-1 flex-col border-r" style={{ borderColor: "var(--border)" }}>
        <div className="flex h-11 shrink-0 items-center gap-2 border-b px-4" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
          <h2 className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>My Models</h2>
          <div className="ml-auto flex min-w-0 items-center gap-2">
            <label className="relative w-[360px] max-w-[45vw]">
              <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: "var(--text-2)" }}>
                <SearchIcon />
              </span>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter models... (Ctrl + F)"
                className="w-full rounded-md py-1.5 pl-8 pr-3 text-sm outline-none transition"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border-mid)", color: "var(--text-0)" }}
              />
            </label>
            <ToolBtn onClick={onOpenSettings} icon={<GearIcon />} label="Settings" />
            <button
              onClick={() => void handleSyncSidecars()}
              disabled={sidecarSyncing || models.length === 0}
              className="rounded-md px-3 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-50"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "#22d3ee" }}
            >
              {sidecarSyncing ? "Syncing..." : "Sync HF files"}
            </button>
            <button
              onClick={onScan}
              disabled={isLoading}
              className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-50"
              style={{ background: "#22d3ee", color: "#0a0a0a", border: "none" }}
            >
              <PlusIcon />
              {isLoading && !loadProgress ? "Scanning..." : "Scan"}
            </button>
          </div>
        </div>

        {(error || loadProgress) && (
          <div className="shrink-0 border-b" style={{ borderColor: "var(--border)" }}>
            {error && (
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
          <div className="shrink-0 border-b px-4 py-2 text-xs" style={{ borderColor: "var(--border)", background: "rgba(34,211,238,0.07)", color: "#67e8f9" }}>
            {sidecarSyncMessage} Only small allowlisted template/config files are fetched; model weights are blocked.
          </div>
        )}

        <div className="grid h-9 shrink-0 grid-cols-[120px_110px_minmax(220px,1fr)_130px_88px_116px] items-center border-b px-5 text-[11px] font-semibold" style={{ borderColor: "var(--border)", color: "var(--text-1)", background: "var(--surface-1)" }}>
          <span>Arch</span>
          <span>Params</span>
          <span>LLM</span>
          <span>Provider</span>
          <span>Quant</span>
          <span className="text-right">Actions</span>
        </div>
        <div className="grid h-10 shrink-0 grid-cols-[120px_110px_minmax(220px,1fr)_130px_88px_116px] items-center gap-2 border-b px-5" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
          <ColumnFilter value={archFilter} onChange={setArchFilter} placeholder="Arch" />
          <ColumnFilter value={paramsFilter} onChange={setParamsFilter} placeholder="Params" />
          <ColumnFilter value={llmFilter} onChange={setLlmFilter} placeholder="LLM" />
          <ColumnFilter value={providerFilter} onChange={setProviderFilter} placeholder="Provider" />
          <ColumnFilter value={quantFilter} onChange={setQuantFilter} placeholder="Quant" />
          <button
            onClick={() => {
              setQuery("");
              setArchFilter("");
              setParamsFilter("");
              setLlmFilter("");
              setProviderFilter("");
              setQuantFilter("");
            }}
            className="justify-self-end rounded px-2 py-1 text-[11px]"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)", cursor: "pointer" }}
          >
            Clear
          </button>
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
                key={m.path || m.filename}
                model={m}
                selected={selectedModel?.path === m.path}
                loaded={m.filename === loadedModel}
                isLoading={isLoading}
                showSwap={!!loadedModel && m.filename !== loadedModel}
                sidecarStatus={sidecarStatuses[m.filename]}
                sidecarSyncing={sidecarSyncingModel === m.filename}
                onSelect={() => setSelectedPath(m.path)}
                onLoad={() => onLoad(m.filename)}
                onSwap={() => onSwap(m.filename)}
                onSyncSidecars={() => void handleSyncModelSidecars(m)}
              />
            ))
          )}
        </div>

        <div className="flex h-9 shrink-0 items-center border-t px-5 text-xs" style={{ borderColor: "var(--border)", color: "var(--text-1)", background: "var(--surface-1)" }}>
          You have {models.length} local models, taking up {localDiskGb.toFixed(2)} GB of disk space
          <span className="ml-auto truncate font-mono text-[11px]" style={{ color: "var(--text-1)" }}>{serverUrl}</span>
        </div>
      </section>

      <aside className="flex min-h-0 w-[360px] shrink-0 flex-col" style={{ background: "var(--surface-1)" }}>
        <div className="border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-2">
            <StatusPill state={state} />
            <button
              onClick={() => onSetApiServerRunning(!apiRunning)}
              disabled={apiBusy}
              className="ml-auto flex items-center gap-2 rounded-md px-3 py-1.5 text-xs transition"
              style={{
                background: "var(--surface-2)",
                color: "var(--text-0)",
                border: "1px solid var(--border)",
                cursor: apiBusy ? "wait" : "pointer",
                opacity: apiBusy ? 0.7 : 1,
              }}
            >
              <span>Serve</span>
              <span style={{ color: apiRunning ? "#34d399" : "var(--text-1)", fontWeight: 600 }}>{apiButtonState}</span>
            </button>
            <ToolBtn onClick={handleCopyUrl} icon={<CopyIcon />} label={copied ? "Copied!" : "URL"} />
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
          onLoad={onLoad}
          onSwap={onSwap}
          onUnload={onUnload}
          onSwapBack={() => onSwap()}
          onSyncSidecars={(model) => void handleSyncModelSidecars(model)}
        />
      </aside>
    </div>
  );

}

// Status pill

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

function hfCacheSummary(status: api.HfSidecarCacheStatus | undefined) {
  if (!status?.repo_id) return "No HF repo";
  const template = status.template_cached ? "template cached" : "template missing";
  return `${template}, ${status.sidecar_cached_count}/${status.sidecar_expected_count} sidecars`;
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
  return (
    <div
      onClick={onSelect}
      className="grid min-h-[45px] grid-cols-[120px_110px_minmax(220px,1fr)_130px_88px_116px] items-center border-b px-5 text-xs transition"
      style={{
        borderColor: "var(--border)",
        background: selected ? "rgba(99,102,241,0.14)" : loaded ? "rgba(34,211,238,0.08)" : "transparent",
        color: "var(--text-1)",
        boxShadow: selected ? "inset 3px 0 0 #6366f1" : loaded ? "inset 2px 0 0 rgba(34,211,238,0.5)" : "none",
        cursor: "pointer",
      }}
    >
      <div className="min-w-0">
        <span className="rounded px-1.5 py-0.5 font-mono text-[10px]" style={{ border: "1px solid var(--border)", color: "var(--text-0)" }}>
          {model.family || model.gguf_architecture || "gguf"}
        </span>
      </div>
      <div>
        <span className="rounded px-1.5 py-0.5 font-mono text-[10px]" style={{ border: "1px solid var(--border)", color: "var(--text-0)" }}>
          {modelParamsLabel(model)}
        </span>
      </div>
      <div className="min-w-0">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate font-mono text-xs font-semibold" style={{ color: "var(--text-0)" }}>
            {shortModelName(model)}
          </span>
          {loaded && <span className="h-2 w-2 shrink-0 rounded-full bg-emerald-400" />}
        </div>
        <div className="mt-0.5 flex flex-wrap items-center gap-1.5">
          {model.supports_reasoning && <MiniCap label="Reasoning" tone="#facc15" />}
          {model.supports_tools && <MiniCap label="Tools" tone="#34d399" />}
          {model.supports_vision && <MiniCap label="Vision" tone="#c4b5fd" />}
          {sidecarStatus?.repo_id && (
            <MiniCap
              label={sidecarStatus.template_cached ? `HF ${sidecarStatus.sidecar_cached_count}/${sidecarStatus.sidecar_expected_count}` : "HF missing"}
              tone={sidecarStatus.template_cached ? "#67e8f9" : "#94a3b8"}
            />
          )}
          {formatContext(model.context_window, model.max_context_window) && <span className="text-[10px]" style={{ color: "var(--text-2)" }}>{formatContext(model.context_window, model.max_context_window)}</span>}
        </div>
      </div>
      <span className="truncate font-mono text-[11px]" title={modelPublisher(model)}>{modelPublisher(model)}</span>
      <span className="font-mono text-[11px]" style={{ color: "#fbbf24" }}>{model.quant ?? "-"}</span>
      <div className="flex justify-end gap-1.5">
        {model.hf_repo && (
          <button
            onClick={(e) => { e.stopPropagation(); onSyncSidecars(); }}
            disabled={isLoading || sidecarSyncing}
            className="rounded-md px-2 py-1 text-[11px] font-semibold disabled:opacity-45"
            title={hfCacheSummary(sidecarStatus)}
            style={{ background: "var(--surface-2)", color: "#67e8f9", border: "1px solid var(--border)" }}
          >
            {sidecarSyncing ? "..." : "HF"}
          </button>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); showSwap ? onSwap() : onLoad(); }}
          disabled={isLoading || !model.provider_managed || loaded}
          className="rounded-md px-2 py-1 text-[11px] font-semibold disabled:opacity-45"
          style={{ background: "#22d3ee", color: "#061014", border: "none" }}
        >
          {loaded ? "Loaded" : showSwap ? "Swap" : "Load"}
        </button>
      </div>
    </div>
  );
}

function ModelInspector({
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
  const defaultContext = model?.context_window ?? model?.max_context_window ?? 8192;
  const maxContext = model?.max_context_window ?? Math.max(defaultContext * 4, 32768);
  const minContext = 512;
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
    const nextContext = model?.context_window ?? model?.max_context_window ?? 8192;
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
  }, [model?.filename, model?.context_window, model?.max_context_window, model?.template_mode, model?.supports_reasoning, model?.default_temperature, model?.default_top_p, model?.default_top_k, model?.default_min_p, model?.default_presence_penalty]);

  const setClampedContextSize = (value: number) => {
    if (!Number.isFinite(value)) {
      setContextSize(minContext);
      setFitMode("off");
      return;
    }
    setContextSize(Math.max(minContext, Math.min(maxContext, Math.round(value))));
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
        <div className="flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-base font-semibold" style={{ color: "var(--text-0)" }}>{shortModelName(model)}</h3>
            <p className="mt-1 truncate font-mono text-xs" style={{ color: "var(--text-1)" }}>{model.filename}</p>
          </div>
          {loaded && <span className="rounded px-2 py-0.5 text-[10px] font-bold uppercase" style={{ background: "rgba(52,211,153,0.14)", color: "#6ee7b7", border: "1px solid rgba(52,211,153,0.28)" }}>Loaded</span>}
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <ActionBtn label={loaded ? "Unload Model" : "Load Model"} disabled={isLoading || !model.provider_managed} variant={loaded ? "ghost" : "primary"} onClick={() => loaded ? onUnload() : onLoad(model.filename, loadOptions)} />
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
            ["load", "Load"],
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
          <InfoRow label="Context" value={liveContext ? `${fmtNum(liveContext)} live` : model.max_context_window ? `${fmtNum(model.max_context_window)} tokens` : "-"} />
          <InfoRow label="Size on disk" value={model.size_gb ? `${model.size_gb.toFixed(2)} GB` : "-"} />
          <InfoRow label="Provider" value={model.provider_name} />
          <InfoRow label="HF Repo" value={sidecarStatus?.repo_id ?? model.hf_repo ?? "-"} />
          <InfoRow label="HF Cache" value={hfCacheSummary(sidecarStatus)} />
        </section>
      )}

      {activeInspectorTab === "load" && (
        <section className="px-4 py-4">
          <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-0)" }}>Load Configuration</h4>
          <InfoRow label="State" value={loaded ? "Loaded" : "Not loaded"} />
          <InfoRow label="Context" value={launchPreview?.context_size ? `${fmtNum(launchPreview.context_size)} tokens` : model.max_context_window ? `${fmtNum(model.max_context_window)} max` : "Model default"} />
          <InfoRow label="Template" value={launchPreview?.template_source ?? model.template_source ?? model.template_mode ?? "-"} />
          <InfoRow label="Chat Template" value={model.has_chat_template ? "Embedded (uses --jinja)" : "Built-in fallback"} />
          <InfoRow label="Template Cache" value={sidecarStatus?.template_cached ? "Cached locally" : sidecarStatus?.repo_id ? "Missing locally" : "No HF repo"} />
          <InfoRow label="HF Sidecars" value={sidecarStatus?.repo_id ? `${sidecarStatus.sidecar_cached_count}/${sidecarStatus.sidecar_expected_count} cached` : "-"} />
          <InfoRow label="MMProj" value={launchPreview?.mmproj_path ? "Attached" : model.supports_vision ? "Not attached" : "Not required"} />
          <InfoRow label="Draft" value={launchPreview?.draft_model_path ? "Enabled" : "Disabled"} />
          {launchPreview?.draft_model_path && (
            <>
              <InfoRow label="Spec Type" value={launchPreview.spec_type || "-"} />
              <InfoRow label="Draft N" value={launchPreview.spec_draft_n_max ? String(launchPreview.spec_draft_n_max) : "-"} />
              <InfoRow label="Draft File" value={launchPreview.draft_model_path} />
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
                    step={512}
                    value={contextSize}
                    onChange={(e) => setClampedContextSize(Number(e.target.value))}
                    className="min-w-0 flex-1"
                  />
                  <input
                    type="number"
                    min={minContext}
                    max={maxContext}
                    step={512}
                    value={contextSize}
                    onChange={(e) => setClampedContextSize(Number(e.target.value) || minContext)}
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
                  <VramBar
                    usedMb={estimateContextVRAM(contextSize)}
                    dedicatedMb={gpuStats.dedicated_mb}
                    systemRamMb={gpuStats.system_ram_mb}
                    mode="estimate"
                  />
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

function ColumnFilter({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
}) {
  return (
    <input
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
      className="min-w-0 rounded px-2 py-1 text-[11px] outline-none"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        color: "var(--text-0)",
      }}
    />
  );
}

function MiniCap({ label, tone }: { label: string; tone: string }) {
  return <span className="rounded px-1.5 py-0.5 text-[9px] font-semibold" style={{ color: tone, border: `1px solid ${tone}55`, background: `${tone}18` }}>{label}</span>;
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
            background: "linear-gradient(90deg, #22d3ee, #38bdf8)",
          }}
        />
      </div>
    </div>
  );
}

function LoadErrorHint({ message }: { message: string }) {
  const lower = message.toLowerCase();
  let hint: string | null = null;
  if (lower.includes("chat_template.jinja") && lower.includes("404")) {
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
  const detectedCtx = model.context_window ?? 8192;
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
              step={512}
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
  variant: "primary" | "ghost" | "indigo";
}) {
  const styles: Record<string, { bg: string; border: string; color: string }> = {
    primary: {
      bg: "#22d3ee",
      border: "transparent",
      color: "#0a0a0a",
    },
    ghost: {
      bg: "var(--surface-2)",
      border: "var(--border)",
      color: "var(--text-1)",
    },
    indigo: {
      bg: "rgba(99,102,241,0.12)",
      border: "rgba(99,102,241,0.25)",
      color: "#a5b4fc",
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

void [Panel, Divider, LoadedModelRow, ModelRow];
