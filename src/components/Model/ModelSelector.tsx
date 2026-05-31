import { useEffect, useState, type ReactNode } from "react";
import type {
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

function fmtSamplingValue(v: number | null | undefined) {
  if (v == null) return "n/a";
  return Number.isInteger(v) ? v.toString() : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
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

// ─── Panel wrapper ─────────────────────────────────────────────────────────────

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

// ─── VRAM bar ──────────────────────────────────────────────────────────────────
// Full bar = dedicated VRAM + system RAM (spill zone).
// Green fill = used dedicated VRAM. Amber zone = system RAM overflow area.
// Divider marks the boundary between dedicated (fast) and spill (slow) memory.

function VramBar({
  usedMb,
  dedicatedMb,
  systemRamMb,
}: {
  usedMb: number;
  dedicatedMb: number;
  systemRamMb: number;
}) {
  const totalMb = dedicatedMb + Math.min(systemRamMb, dedicatedMb * 4); // cap spill zone at 4x VRAM
  const usedPct = totalMb > 0 ? (usedMb / totalMb) * 100 : 0;
  const dedicatedPct = totalMb > 0 ? (dedicatedMb / totalMb) * 100 : 100;

  const usedGb = (usedMb / 1024).toFixed(1);
  const dedicatedGb = (dedicatedMb / 1024).toFixed(1);
  const spillGb = (Math.min(systemRamMb, dedicatedMb * 4) / 1024).toFixed(0);

  // Fill colour: green while in dedicated zone, amber if spilling
  const fillColor = usedMb < dedicatedMb * 0.9 ? "#34d399" : "#f59e0b";

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] uppercase tracking-widest whitespace-nowrap" style={{ color: "var(--text-2)" }}>
        VRAM
      </span>
      <div
        className="relative rounded-full overflow-hidden"
        style={{ width: "110px", height: "6px", background: "var(--surface-3)" }}
        title={`${usedGb}/${dedicatedGb}GB dedicated | +${spillGb}GB RAM overflow`}
      >
        {/* Spill zone (right portion = system RAM) — always amber tint */}
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
      </div>
      <span className="text-[10px] whitespace-nowrap tabular-nums" style={{ color: "var(--text-1)" }}>
        {usedGb}/{dedicatedGb}GB
      </span>
    </div>
  );
}

// ─── Main component ─────────────────────────────────────────────────────────

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
  onScan,
  onOpenSettings,
}: Props) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterKey>("all");
  const [copied, setCopied] = useState(false);
  const serverUrl = buildServerUrl(settings);
  const gpuStats = useGpuStats();

  useEffect(() => {
    if (!copied) return;
    const t = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(t);
  }, [copied]);

  const filteredModels = models.filter((m) => {
    const q = query.trim().toLowerCase();
    const matchQ =
      !q ||
      m.filename.toLowerCase().includes(q) ||
      m.family.toLowerCase().includes(q) ||
      (m.quant ?? "").toLowerCase().includes(q);
    if (!matchQ) return false;
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
      })
    : null;
  const modelCards = filteredModels.filter((m) => m.filename !== loadedModel);

  const handleCopyUrl = async () => {
    try {
      await navigator.clipboard.writeText(serverUrl);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  const state = processStatus?.state ?? "Idle";
  const apiState = processStatus?.api_state ?? "Idle";
  const apiRunning = apiState === "Running" || apiState === "Starting";

  return (
    <div className="flex flex-col gap-3">
      <Panel>
        {/* ── Toolbar ── */}
        <div className="flex flex-wrap items-center gap-2 px-3 py-2.5">
          <StatusPill state={state} />
          <button
            onClick={() => onSetApiServerRunning(!apiRunning)}
            className="flex items-center gap-2 rounded px-3 py-1.5 text-xs transition"
            style={{
              background: "var(--surface-2)",
              color: "var(--text-0)",
              border: "1px solid var(--border)",
            }}
          >
            <span>Serve</span>
            <span style={{ color: apiRunning ? "#34d399" : "var(--text-1)", fontWeight: 600 }}>
              {apiRunning ? apiState : "Off"}
            </span>
            <span
              className="relative shrink-0 rounded-full transition"
              style={{
                width: "28px",
                height: "16px",
                background: apiRunning ? "#22d3ee" : "var(--surface-3)",
              }}
            >
              <span
                className="absolute rounded-full bg-white transition-all"
                style={{
                  width: "12px",
                  height: "12px",
                  top: "2px",
                  left: apiRunning ? "14px" : "2px",
                }}
              />
            </span>
          </button>
          <ToolBtn onClick={onOpenSettings} icon={<GearIcon />} label="Settings" />
          <ToolBtn
            onClick={handleCopyUrl}
            icon={<CopyIcon />}
            label={copied ? "Copied!" : "Copy URL"}
          />
          <div className="flex-1" />
          <span
            className="truncate font-mono text-xs"
            style={{ color: "var(--text-1)" }}
          >
            {serverUrl}
          </span>
          <span
            className="rounded px-2 py-0.5 text-xs"
            style={{
              background: "var(--surface-2)",
              color: "var(--text-1)",
              border: "1px solid var(--border)",
            }}
          >
            {models.length} model{models.length === 1 ? "" : "s"}
          </span>
          {settings && (
            <ProviderBadge
              providerName={settings.active_provider === "lm_studio" ? "LM Studio" : "Managed"}
              managed={settings.active_provider !== "lm_studio"}
            />
          )}
          {gpuStats && (
            <VramBar
              usedMb={gpuStats.used_mb}
              dedicatedMb={gpuStats.dedicated_mb}
              systemRamMb={gpuStats.system_ram_mb}
            />
          )}
          <button
            onClick={onScan}
            disabled={isLoading}
            className="flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-50"
            style={{
              background: "#22d3ee",
              color: "#0a0a0a",
              border: "none",
            }}
            onMouseEnter={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.filter = "brightness(1.1)")
            }
            onMouseLeave={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.filter = "")
            }
            aria-disabled={isLoading}
          >
            <PlusIcon />
            {isLoading && !loadProgress ? "Scanning..." : "Scan"}
          </button>
        </div>

        <Divider />

        {/* ── Search + filters ── */}
        <div className="flex flex-wrap items-center gap-2 px-3 py-2.5">
          <label className="relative min-w-[220px] flex-1">
            <span
              className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2"
              style={{ color: "var(--text-2)" }}
            >
              <SearchIcon />
            </span>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search name, family, or quant..."
              className="w-full rounded py-1.5 pl-8 pr-3 text-sm outline-none transition"
              style={{
                background: "var(--surface-2)",
                border: "1px solid var(--border)",
                color: "var(--text-0)",
              }}
              onFocus={(e) =>
                ((e.currentTarget as HTMLInputElement).style.borderColor =
                  "rgba(34,211,238,0.35)")
              }
              onBlur={(e) =>
                ((e.currentTarget as HTMLInputElement).style.borderColor =
                  "var(--border)")
              }
            />
          </label>

          <div className="flex items-center gap-1">
            {FILTERS.map((key) => (
              <button
                key={key}
                // onClick moved below to avoid duplicate
                className="rounded px-2.5 py-1 text-xs font-medium uppercase tracking-wider transition"
                style={{
                  background: filter === key ? "rgba(34,211,238,0.12)" : "transparent",
                  border:
                    filter === key
                      ? "1px solid rgba(34,211,238,0.25)"
                      : "1px solid transparent",
                  color: filter === key ? "#22d3ee" : "var(--text-1)",
                  cursor: isLoading ? "not-allowed" : "pointer",
                  opacity: isLoading ? 0.5 : 1,
                }}
                onClick={() => { if (!isLoading) setFilter(key); }}
                disabled={isLoading}
                aria-disabled={isLoading}
                onMouseEnter={(e) => {
                  if (!isLoading && filter !== key) {
                    (e.currentTarget as HTMLButtonElement).style.color =
                      "var(--text-0)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading && filter !== key) {
                    (e.currentTarget as HTMLButtonElement).style.color =
                      "var(--text-1)";
                  }
                }}
              >
                {key}
              </button>
            ))}
          </div>

          <span className="ml-auto text-xs" style={{ color: "var(--text-2)" }}>
            {filteredModels.length} / {models.length}
            {processStatus?.backend && (
              <> {" | "} <span style={{ color: "#22d3ee" }}>{processStatus.backend}</span></>
            )}
          </span>
        </div>

        {/* ── Error ── */}
        {error && (
          <>
            <Divider />
            <div
              className="px-3 py-2.5 text-sm"
              style={{
                background: "rgba(239,68,68,0.08)",
                color: "#fca5a5",
              }}
            >
              {error}
            </div>
          </>
        )}

        {/* ── Load progress ── */}
        {loadProgress && !loadProgress.done && (
          <>
            <Divider />
            <LoadingBar progress={loadProgress} />
          </>
        )}
        {loadProgress?.error && (
          <>
            <Divider />
            <div
              className="px-3 py-2.5 text-sm"
              style={{ background: "rgba(239,68,68,0.08)", color: "#fca5a5" }}
            >
              {loadProgress.error}
            </div>
          </>
        )}

        {/* ── Loaded model ── */}
        {activeModel && (
          <>
            <Divider />
            <LoadedModelRow
              model={activeModel}
              previousModel={previousModel}
              processStatus={processStatus}
              onUnload={onUnload}
              onSwapBack={() => onSwap()}
              isLoading={isLoading}
            />
          </>
        )}

        {/* ── Model list ── */}
        {models.length === 0 ? (
          <>
            <Divider />
            <EmptyMsg
              title="No models discovered yet"
              body="Set model directories in Settings then scan to populate the library."
            />
          </>
        ) : filter === "loaded" && !activeModel ? (
          <>
            <Divider />
            <EmptyMsg
              title="No model loaded"
              body="Load a model to see it here."
            />
          </>
        ) : filteredModels.length === 0 ? (
          <>
            <Divider />
            <EmptyMsg
              title="No matches"
              body="Clear the search or change the capability filter."
            />
          </>
        ) : modelCards.length > 0 ? (
          <>
            <Divider />
            {modelCards.map((m, i) => (
              <ModelRow
                key={m.path}
                model={m}
                isLoading={isLoading}
                showSwap={!!loadedModel && m.filename !== loadedModel}
                isLast={i === modelCards.length - 1}
                onLoad={(options) => onLoad(m.filename, options)}
                onSwap={(options) => onSwap(m.filename, options)}
              />
            ))}
          </>
        ) : null}
      </Panel>
    </div>
  );
}

// ─── Status pill ────────────────────────────────────────────────────────────

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

// ─── Tool button ─────────────────────────────────────────────────────────────

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

// ─── Loading bar ─────────────────────────────────────────────────────────────

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

// ─── Loaded model row ─────────────────────────────────────────────────────────

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

// ─── Model row ───────────────────────────────────────────────────────────────

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
  const defaultCtx = model.context_window ?? 8192;
  // Use the model's true training context ceiling from the profile, falling back to a conservative multiple
  const maxCtx = model.max_context_window ?? Math.max(defaultCtx * 4, 32768);
  const minCtx = 512;
  const [contextSize, setContextSize] = useState(defaultCtx);
  const [fitMode, setFitMode] = useState("on");
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
  // Default cache type is q8_0 (1 byte/element) — matches ProcessConfig default.
  // Formula: n_layers × 2 (K+V) × n_kv_heads × head_dim × bytes_per_element
  const KV_BPE = 1.0; // q8_0
  const kvBytesPerToken: number | null =
    model.n_layers != null && model.n_kv_heads != null && model.head_dim != null
      ? model.n_layers * 2 * model.n_kv_heads * model.head_dim * KV_BPE
      : null;

  function estimateContextVRAM(tokens: number): number {
    const modelMb = (model.size_gb || 0) * 1024;
    if (kvBytesPerToken != null) {
      return modelMb + (tokens * kvBytesPerToken) / (1024 * 1024);
    }
    // Fallback when GGUF metadata unavailable (external providers, unparseable files)
    return modelMb + (tokens * 2) / 1024;
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
                  onClick={() => setContextSize(defaultCtx)}
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
              onChange={(e) => setContextSize(Number(e.target.value))}
            />
            <div className="mt-1 flex justify-between text-[10px]" style={{ color: "var(--text-2)" }}>
              <span>{minCtx.toLocaleString()}</span>
              <span>Default: {defaultCtx.toLocaleString()}</span>
              <span>{maxCtx.toLocaleString()}</span>
            </div>
            {/* VRAM/overflow monitor as slider bar */}
            {gpuStats && (
              <div className="mt-2">
                <VramBar usedMb={estimateContextVRAM(contextSize)} dedicatedMb={gpuStats.dedicated_mb} systemRamMb={gpuStats.system_ram_mb} />
              </div>
            )}
          </div>}

          {/* Stats grid */}
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile label="File Size" value={model.size_gb && model.size_gb > 0 ? `${model.size_gb.toFixed(2)} GB` : 'N/A'} />
            <StatTile label="Provider" value={model.provider_name} />
            <StatTile label="Default Context" value={model.context_window ? `${fmtNum(model.context_window)} tokens` : '—'} />
            <StatTile label="Max Context" value={model.max_context_window ? `${fmtNum(model.max_context_window)} tokens` : '—'} />
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

// ─── Empty state ──────────────────────────────────────────────────────────────

function EmptyMsg({ title, body }: { title: string; body: string }) {
  return (
    <div className="px-4 py-12 text-center">
      <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
        {title}
      </p>
      <p className="mx-auto mt-1 max-w-sm text-xs" style={{ color: "var(--text-2)" }}>
        {body}
      </p>
    </div>
  );
}

// ─── Stat tile ────────────────────────────────────────────────────────────────

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

// ─── Capability badge ─────────────────────────────────────────────────────────

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

// ─── Action button ────────────────────────────────────────────────────────────

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

// ─── Icons ────────────────────────────────────────────────────────────────────

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
