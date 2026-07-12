import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import * as api from "../../lib/tauri";
import type { DownloadProgress, HubAccessStatus, HubModel, HubQuant } from "../../lib/tauri";
import type { ModelInfo } from "../../lib/types";

interface Props {
  models: ModelInfo[];
  onRefresh: () => void;
}

const PAGE_SIZE = 20;
const panelStyle = { background: "var(--surface-1)", border: "1px solid var(--border)" };
type HubSortMode = "downloads" | "lastModified" | "largest" | "smallest" | "likes" | "name";

const TAG_COLORS: Record<string, { bg: string; color: string; border: string }> = {
  reasoning: { bg: "rgba(251,191,36,0.10)", color: "#fde68a", border: "rgba(251,191,36,0.20)" },
  tools: { bg: "rgba(52,211,153,0.10)", color: "#6ee7b7", border: "rgba(52,211,153,0.20)" },
  thinking: { bg: "rgba(167,139,250,0.10)", color: "#c4b5fd", border: "rgba(167,139,250,0.20)" },
  chat: { bg: "rgba(34,211,238,0.08)", color: "#67e8f9", border: "rgba(34,211,238,0.18)" },
  math: { bg: "rgba(249,115,22,0.10)", color: "#fdba74", border: "rgba(249,115,22,0.20)" },
  moe: { bg: "rgba(99,102,241,0.10)", color: "#a5b4fc", border: "rgba(99,102,241,0.20)" },
  vision: { bg: "rgba(236,72,153,0.10)", color: "#f9a8d4", border: "rgba(236,72,153,0.20)" },
};

const QUANT_PREFERENCE = ["Q5_K_M", "Q4_K_M", "Q6_K", "Q5_K_S", "Q4_K_S", "Q8_0", "Q4_0", "Q3_K_M", "Q2_K", "F16", "BF16"];

function TagBadge({ tag }: { tag: string }) {
  const style = TAG_COLORS[tag] ?? { bg: "var(--surface-2)", color: "var(--text-1)", border: "var(--border)" };
  return (
    <span className="rounded px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider" style={{ background: style.bg, color: style.color, border: `1px solid ${style.border}` }}>
      {tag}
    </span>
  );
}

function basename(path: string) {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return (parts[parts.length - 1] ?? path).toLowerCase();
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / (1024 * 1024);
  if (mb >= 1) return `${mb.toFixed(0)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function formatSpeed(bytesPerSecond?: number | null) {
  if (!bytesPerSecond) return null;
  return `${formatBytes(bytesPerSecond)}/s`;
}

function formatEta(seconds?: number | null) {
  if (!seconds) return null;
  if (seconds < 60) return `${seconds}s left`;
  const minutes = Math.floor(seconds / 60);
  const rem = seconds % 60;
  if (minutes < 60) return `${minutes}m ${rem}s left`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours}h ${mins}m left`;
}

function downloadDetail(entry: DownloadProgress) {
  const parts = [
    `${formatBytes(entry.downloaded_bytes)} / ${formatBytes(entry.total_bytes)}`,
    `${Math.round(entry.percent * 100)}%`,
    formatSpeed(entry.speed_bps),
    formatEta(entry.eta_seconds),
  ].filter(Boolean);
  return parts.join(" · ");
}

function progressTone(status: string, error?: string | null) {
  if (error || status === "Failed") return "#f87171";
  if (status === "Completed") return "#34d399";
  if (status === "Retrying") return "#a5b4fc";
  if (status === "Cancelled" || status === "Paused" || status === "Pausing") return "#fbbf24";
  return "#22d3ee";
}

function quantSizeBytes(quant: HubQuant) {
  if (quant.size_bytes && quant.size_bytes > 0) return quant.size_bytes;
  if (quant.size_gb > 0) return Math.round(quant.size_gb * 1_073_741_824);
  return 0;
}

function modelHasMissingSize(model: HubModel) {
  return model.quants.some((quant) => quantSizeBytes(quant) <= 0);
}

function formatModelSizeRange(quants: HubQuant[], loading = false) {
  const sizes = quants.map(quantSizeBytes).filter((size) => size > 0).sort((a, b) => a - b);
  if (sizes.length === 0) return loading ? "checking..." : "size unknown";
  const min = sizes[0];
  const max = sizes[sizes.length - 1];
  if (Math.abs(min - max) < 64 * 1024 * 1024) return formatBytes(min);
  return `${formatBytes(min)}-${formatBytes(max)}`;
}

function formatQuantSize(quant: HubQuant | null | undefined, loading = false) {
  if (!quant) return "unknown";
  const size = quantSizeBytes(quant);
  if (size <= 0) return loading ? "checking..." : "unknown";
  return formatBytes(size);
}

function formatOptionSummary(model: HubModel, loading = false) {
  const sizes = model.quants.map(quantSizeBytes).filter((size) => size > 0).sort((a, b) => a - b);
  const count = model.quants.length;
  if (sizes.length === 0) return `${count} file${count === 1 ? "" : "s"} - ${loading ? "checking sizes" : "size unknown"}`;
  const min = sizes[0];
  const max = sizes[sizes.length - 1];
  const range = Math.abs(min - max) < 64 * 1024 * 1024 ? formatBytes(min) : `${formatBytes(min)}-${formatBytes(max)}`;
  return `${count} file${count === 1 ? "" : "s"} - ${range}`;
}

function modelTotalSize(quants: HubQuant[]) {
  const total = quants.reduce((sum, quant) => sum + quantSizeBytes(quant), 0);
  return total > 0 ? formatBytes(total) : "unknown";
}

function modelNeedsDetails(model: HubModel, includeReadme: boolean) {
  return modelHasMissingSize(model) || !model.license || !model.base_model || !model.pipeline_tag || (includeReadme && model.readme == null);
}

function readmePreview(text?: string | null) {
  if (!text) return "";
  return text
    .replace(/^---[\s\S]*?---\s*/m, "")
    .replace(/<[^>]+>/g, "")
    .replace(/!\[[^\]]*]\([^)]+\)/g, "")
    .trim();
}

function uniqueHubModels(models: HubModel[]) {
  const seen = new Set<string>();
  return models.filter((model) => {
    const key = model.id.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function modelMinSize(model: HubModel) {
  const sizes = model.quants.map(quantSizeBytes).filter((size) => size > 0);
  return sizes.length > 0 ? Math.min(...sizes) : Number.POSITIVE_INFINITY;
}

function modelMaxSize(model: HubModel) {
  const sizes = model.quants.map(quantSizeBytes).filter((size) => size > 0);
  return sizes.length > 0 ? Math.max(...sizes) : 0;
}

function abbrevCount(value: number) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}K`;
  return String(value);
}

function timeAgo(iso: string | null) {
  if (!iso) return "updated unknown";
  const timestamp = Date.parse(iso);
  if (!Number.isFinite(timestamp)) return "updated unknown";
  const days = Math.max(0, Math.floor((Date.now() - timestamp) / 86_400_000));
  if (days === 0) return "updated today";
  if (days === 1) return "updated yesterday";
  if (days < 30) return `updated ${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `updated ${months}mo ago`;
  return `updated ${Math.floor(months / 12)}y ago`;
}

function recommendedQuant(model: HubModel, isInstalled: (quant: HubQuant) => boolean) {
  const available = model.quants.filter((quant) => !isInstalled(quant));
  const candidates = available.length > 0 ? available : model.quants;
  return (
    QUANT_PREFERENCE.map((preferred) => candidates.find((quant) => quant.quant === preferred && !quant.filename.toLowerCase().includes("-mtp-"))).find(Boolean) ??
    QUANT_PREFERENCE.map((preferred) => candidates.find((quant) => quant.quant === preferred)).find(Boolean) ??
    candidates.find((quant) => quantSizeBytes(quant) > 0) ??
    candidates[0] ??
    null
  );
}

function quantLabel(quant: HubQuant) {
  const sizeBytes = quantSizeBytes(quant);
  const size = sizeBytes > 0 ? ` - ${formatBytes(sizeBytes)}` : "";
  return `${quant.quant}${size}`;
}

function modelCardBackground(selected: boolean, installed: boolean) {
  if (selected) return "linear-gradient(180deg, rgba(34,211,238,0.13), rgba(34,211,238,0.045)), var(--surface-1)";
  if (installed) return "linear-gradient(180deg, rgba(52,211,153,0.075), rgba(52,211,153,0.025)), var(--surface-1)";
  return "var(--surface-1)";
}

function HubStat({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-full px-3 py-1.5 text-xs" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
      <span style={{ color: "var(--text-2)" }}>{label}</span> <span className="font-semibold" style={{ color: "var(--text-0)" }}>{value}</span>
    </span>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-4 border-b py-2.5 last:border-b-0" style={{ borderColor: "var(--border)" }}>
      <span className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: "var(--text-2)" }}>{label}</span>
      <span className="min-w-0 truncate text-right font-mono text-xs" style={{ color: "var(--text-0)" }} title={value}>{value}</span>
    </div>
  );
}

function ReadmeMarkdown({ markdown }: { markdown: string }) {
  return (
    <div className="max-h-96 overflow-y-auto rounded-md px-4 py-3 text-sm leading-6" style={{ background: "#1b1c20", border: "1px solid var(--border)", color: "var(--text-1)" }}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => <h1 className="mb-3 mt-1 text-lg font-semibold leading-6" style={{ color: "var(--text-0)" }}>{children}</h1>,
          h2: ({ children }) => <h2 className="mb-2 mt-5 border-t pt-4 text-sm font-semibold uppercase tracking-[0.08em]" style={{ borderColor: "var(--border)", color: "var(--text-0)" }}>{children}</h2>,
          h3: ({ children }) => <h3 className="mb-2 mt-4 text-sm font-semibold" style={{ color: "var(--text-0)" }}>{children}</h3>,
          p: ({ children }) => <p className="mb-3" style={{ color: "var(--text-1)" }}>{children}</p>,
          a: ({ href, children }) => (
            <button
              type="button"
              onClick={() => href && href.startsWith("https://huggingface.co/") && void api.openExternalUrl(href)}
              className="font-medium underline decoration-dotted underline-offset-4"
              style={{ color: "#67e8f9" }}
            >
              {children}
            </button>
          ),
          ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => <li style={{ color: "var(--text-1)" }}>{children}</li>,
          code: ({ children }) => <code className="rounded px-1 py-0.5 font-mono text-[11px]" style={{ background: "#111216", color: "#a5f3fc" }}>{children}</code>,
          pre: ({ children }) => <pre className="mb-3 overflow-x-auto rounded-md p-3 text-[11px]" style={{ background: "#111216", border: "1px solid var(--border)", color: "var(--text-1)" }}>{children}</pre>,
          blockquote: ({ children }) => <blockquote className="mb-3 border-l-2 pl-3" style={{ borderColor: "#22d3ee", color: "var(--text-1)" }}>{children}</blockquote>,
          table: ({ children }) => <div className="mb-3 overflow-x-auto rounded-md" style={{ border: "1px solid var(--border)" }}><table className="w-full text-left text-xs">{children}</table></div>,
          th: ({ children }) => <th className="px-2 py-1.5 font-semibold" style={{ background: "#111216", color: "var(--text-0)" }}>{children}</th>,
          td: ({ children }) => <td className="border-t px-2 py-1.5" style={{ borderColor: "var(--border)", color: "var(--text-1)" }}>{children}</td>,
          hr: () => <hr className="my-4" style={{ borderColor: "var(--border)" }} />,
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}

function HubPreview({
  model,
  downloads,
  detailsLoading,
  isInstalled,
  onDownload,
  onCancel,
  onPause,
}: {
  model: HubModel | null;
  downloads: Record<string, DownloadProgress>;
  detailsLoading: boolean;
  isInstalled: (quant: HubQuant) => boolean;
  onDownload: (model: HubModel, quant: HubQuant) => void;
  onCancel: (id: string) => void;
  onPause: (id: string) => void;
}) {
  const [selectedQuantUrl, setSelectedQuantUrl] = useState<string | null>(null);
  const [panelTab, setPanelTab] = useState<"overview" | "files" | "readme">("overview");
  const recommended = model ? recommendedQuant(model, isInstalled) : null;
  const selectedQuant =
    model?.quants.find((quant) => quant.url === selectedQuantUrl) ??
    recommended ??
    model?.quants[0] ??
    null;
  const progress = selectedQuant ? downloads[selectedQuant.url] : null;
  const paused = progress?.status === "Paused";
  const downloading = progress && !progress.done && !paused;
  const selectedInstalled = selectedQuant ? isInstalled(selectedQuant) : false;
  const tone = progressTone(progress?.status ?? "", progress?.error);
  const selectedSize = formatQuantSize(selectedQuant, detailsLoading);

  useEffect(() => {
    setSelectedQuantUrl(null);
    setPanelTab("overview");
  }, [model?.id]);

  if (!model) {
    return (
      <aside className="flex min-h-[420px] items-center justify-center rounded text-sm" style={{ border: "1px solid var(--border)", color: "var(--text-2)" }}>
        Select a model to preview its details.
      </aside>
    );
  }

  return (
    <aside className="h-full overflow-y-auto rounded-md" style={{ border: "1px solid var(--border)", background: "#151619" }}>
      <div className="px-5 py-5" style={{ borderBottom: "1px solid var(--border)", background: "linear-gradient(180deg, rgba(34,211,238,0.055), rgba(34,211,238,0))" }}>
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-[0.22em]" style={{ color: "var(--text-2)" }}>{model.family}</div>
            <h3 className="mt-1 text-xl font-semibold leading-6" style={{ color: "var(--text-0)" }}>{model.name}</h3>
            <p className="mt-1 truncate font-mono text-xs" style={{ color: "var(--text-2)" }}>{model.id}</p>
          </div>
          {selectedInstalled && <span className="shrink-0 rounded-md px-2.5 py-1.5 text-[10px] font-bold uppercase" style={{ background: "rgba(52,211,153,0.12)", border: "1px solid rgba(52,211,153,0.28)", color: "#34d399" }}>Local</span>}
        </div>
        <div className="mt-4 grid grid-cols-3 gap-2">
          <button onClick={() => void api.openExternalUrl(model.hf_url)} className="rounded-md px-3 py-2 text-xs font-semibold" style={{ background: "#22d3ee", border: "none", color: "#041014" }}>Open HF</button>
          <button onClick={() => void navigator.clipboard?.writeText(model.id)} className="rounded-md px-3 py-2 text-xs font-medium" style={{ background: "#202126", border: "1px solid var(--border)", color: "var(--text-1)" }}>Copy repo</button>
          {selectedQuant && <button onClick={() => void api.openExternalUrl(selectedQuant.url)} className="rounded-md px-3 py-2 text-xs font-medium" style={{ background: "#202126", border: "1px solid var(--border)", color: "var(--text-1)" }}>Open file</button>}
        </div>
        <div className="mt-4 flex flex-wrap gap-1.5">{model.tags.slice(0, 7).map((tag) => <TagBadge key={tag} tag={tag} />)}</div>
        <div className="mt-5 grid grid-cols-4 gap-3 border-t pt-4 text-xs" style={{ borderColor: "var(--border)" }}>
          {[
            ["Downloads", abbrevCount(model.downloads ?? 0)],
            ["Likes", abbrevCount(model.likes ?? 0)],
            ["Files", String(model.quants.length)],
            ["Updated", timeAgo(model.last_modified ?? null).replace("updated ", "")],
          ].map(([label, value]) => (
            <div key={label} className="min-w-0">
              <div className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-2)" }}>{label}</div>
              <div className="mt-1 truncate font-semibold" style={{ color: "var(--text-0)" }}>{value}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="flex gap-5 border-b px-5" style={{ borderColor: "var(--border)" }}>
        {(["overview", "files", "readme"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setPanelTab(tab)}
            className="border-b-2 px-0 py-3 text-xs font-semibold capitalize"
            style={{
              background: "transparent",
              borderColor: panelTab === tab ? "#22d3ee" : "transparent",
              color: panelTab === tab ? "#a5f3fc" : "var(--text-1)",
            }}
          >
            {tab}
          </button>
        ))}
      </div>
      {panelTab === "overview" && (
        <div className="border-b px-5 py-4" style={{ borderColor: "var(--border)" }}>
          <div className="rounded-md px-3" style={{ background: "#1b1c20", border: "1px solid var(--border)" }}>
            <DetailRow label="Repo" value={model.id} />
            <DetailRow label="Pipeline" value={model.pipeline_tag ?? "-"} />
            <DetailRow label="License" value={model.license ?? "-"} />
            <DetailRow label="Base" value={model.base_model ?? "-"} />
            <DetailRow label="Total GGUF" value={modelTotalSize(model.quants)} />
            <DetailRow label="Vision" value={model.supports_vision ? "yes" : "no"} />
          </div>
        </div>
      )}
      {panelTab === "files" && (
        <div className="border-b px-5 py-4" style={{ borderColor: "var(--border)" }}>
          <div className="max-h-72 overflow-y-auto rounded-md" style={{ border: "1px solid var(--border)", background: "#1b1c20" }}>
            {model.quants.map((quant) => {
              const installed = isInstalled(quant);
              return (
                <button key={quant.url} onClick={() => setSelectedQuantUrl(quant.url)} className="grid w-full grid-cols-[minmax(0,1fr)_86px_72px] items-center gap-3 border-b px-3 py-2.5 text-left last:border-b-0" style={{ borderColor: "var(--border)", background: selectedQuant?.url === quant.url ? "rgba(34,211,238,0.08)" : "transparent" }}>
                  <span className="truncate font-mono text-[11px]" style={{ color: "var(--text-0)" }}>{quant.filename}</span>
                  <span className="text-[11px]" style={{ color: "var(--text-1)" }}>{formatQuantSize(quant, detailsLoading)}</span>
                  <span className="justify-self-start rounded px-1.5 py-0.5 text-[9px] font-bold uppercase" style={{ background: installed ? "rgba(52,211,153,0.12)" : "#22242a", border: `1px solid ${installed ? "rgba(52,211,153,0.24)" : "var(--border)"}`, color: installed ? "#34d399" : "var(--text-2)" }}>{installed ? "local" : quant.quant}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
      {panelTab === "readme" && (
        <div className="border-b px-5 py-4" style={{ borderColor: "var(--border)" }}>
          {detailsLoading && !model.readme ? (
            <div className="rounded-md px-3 py-6 text-sm" style={{ background: "#1b1c20", border: "1px solid var(--border)", color: "var(--text-2)" }}>Loading README...</div>
          ) : readmePreview(model.readme) ? (
            <ReadmeMarkdown markdown={readmePreview(model.readme)} />
          ) : (
            <div className="rounded-md px-3 py-6 text-sm" style={{ background: "#1b1c20", border: "1px solid var(--border)", color: "var(--text-2)" }}>No README preview available for this repo.</div>
          )}
        </div>
      )}
      <div className="px-5 py-4">
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="text-xs font-semibold uppercase tracking-[0.16em]" style={{ color: "var(--text-2)" }}>GGUF file</div>
          <span className="text-[11px]" style={{ color: "var(--text-2)" }}>{detailsLoading ? "checking file metadata" : `${model.quants.length} options`}</span>
        </div>
        <div className="rounded-md p-3" style={{ background: "#1b1c20", border: "1px solid var(--border)" }}>
          <div className="flex flex-col gap-3">
            <select
              value={selectedQuant?.url ?? ""}
              onChange={(event) => setSelectedQuantUrl(event.target.value)}
              className="w-full rounded px-3 py-2 text-sm outline-none"
              style={{ background: "#111216", border: "1px solid var(--border-mid)", color: "var(--text-0)" }}
            >
              {model.quants.map((quant) => (
                <option key={quant.url} value={quant.url}>
                  {quantLabel(quant)}{recommended?.url === quant.url ? " - recommended" : ""}{isInstalled(quant) ? " - installed" : ""}
                </option>
              ))}
            </select>
            {selectedQuant && (
              <div className="min-w-0 rounded-md px-3 py-3" style={{ background: "rgba(34,211,238,0.065)", border: "1px solid rgba(34,211,238,0.20)" }}>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ color: "#67e8f9" }}>Selected download</span>
                  <span className="rounded-full px-2 py-0.5 text-[11px] font-semibold" style={{ background: "rgba(34,211,238,0.12)", color: "#a5f3fc", border: "1px solid rgba(34,211,238,0.24)" }}>{selectedSize}</span>
                </div>
                <div className="mt-1 truncate font-mono text-[11px]" style={{ color: "var(--text-2)" }}>{selectedQuant.filename}</div>
                <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px]" style={{ color: "var(--text-1)" }}>
                  <span>{selectedQuant.quant}</span>
                  <span>{selectedSize === "unknown" ? "download size unknown" : selectedSize}</span>
                  {recommended?.url === selectedQuant.url && <span style={{ color: "#fde68a" }}>Recommended</span>}
                  {selectedInstalled && <span style={{ color: "#34d399" }}>Installed</span>}
                </div>
              </div>
            )}
            {progress && (
              <div>
                <div className="h-1.5 overflow-hidden rounded" style={{ background: "rgba(255,255,255,0.08)" }}>
                  <div className="h-full rounded transition-all" style={{ width: `${Math.max(4, Math.round(progress.percent * 100))}%`, background: tone }} />
                </div>
                <div className="mt-1 text-[11px]" style={{ color: progress.error ? "#f87171" : "var(--text-2)" }}>{progress.error ?? `${progress.status} · ${downloadDetail(progress)}`}</div>
              </div>
            )}
            <div className="flex justify-end">
              {paused && progress && selectedQuant ? (
                <div className="flex gap-2">
                  <button onClick={() => onDownload(model, selectedQuant)} className="rounded px-3 py-1.5 text-xs font-semibold" style={{ background: "#22d3ee", border: "none", color: "#041014" }}>Resume</button>
                  <button onClick={() => onCancel(progress.id)} className="rounded px-3 py-1.5 text-xs font-medium" style={{ background: "rgba(248,113,113,0.12)", border: "1px solid rgba(248,113,113,0.24)", color: "#f87171" }}>Cancel</button>
                </div>
              ) : downloading && progress ? (
                <div className="flex gap-2">
                  <button onClick={() => onPause(progress.id)} className="rounded px-3 py-1.5 text-xs font-medium" style={{ background: "rgba(251,191,36,0.12)", border: "1px solid rgba(251,191,36,0.24)", color: "#fde68a" }}>Pause</button>
                  <button onClick={() => onCancel(progress.id)} className="rounded px-3 py-1.5 text-xs font-medium" style={{ background: "rgba(248,113,113,0.12)", border: "1px solid rgba(248,113,113,0.24)", color: "#f87171" }}>Cancel</button>
                </div>
              ) : selectedInstalled ? (
                <span className="rounded-md px-3 py-2 text-xs font-medium" style={{ background: "rgba(52,211,153,0.10)", border: "1px solid rgba(52,211,153,0.22)", color: "#34d399" }}>Already on device</span>
              ) : (
                <button disabled={!selectedQuant} onClick={() => selectedQuant && onDownload(model, selectedQuant)} className="rounded-md px-4 py-2 text-xs font-semibold disabled:opacity-50" style={{ background: "#22d3ee", border: "none", color: "#041014" }}>
                  {selectedSize === "unknown" ? "Download selected" : `Download ${selectedSize}`}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

export function ModelBrowser({ models, onRefresh }: Props) {
  const [section, setSection] = useState<"search" | "local">("search");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<HubModel[]>([]);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>("rows");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [capabilityFilter, setCapabilityFilter] = useState("all");
  const [formatFilter, setFormatFilter] = useState("gguf");
  const [sortMode, setSortMode] = useState<HubSortMode>("lastModified");
  const [hubStatus, setHubStatus] = useState<HubAccessStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [downloads, setDownloads] = useState<Record<string, DownloadProgress>>({});
  const [downloadsOpen, setDownloadsOpen] = useState(false);
  const [detailLoadingIds, setDetailLoadingIds] = useState<Set<string>>(() => new Set());
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deletingPath, setDeletingPath] = useState<string | null>(null);
  const [deleteMessage, setDeleteMessage] = useState<string | null>(null);
  const [syncingMetadata, setSyncingMetadata] = useState(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const downloadsMenuRef = useRef<HTMLDivElement>(null);
  const detailFetchedIds = useRef<Set<string>>(new Set());

  const installedFilenames = useMemo(() => new Set(models.map((model) => basename(model.filename))), [models]);
  const tags = useMemo(() => Array.from(new Set(results.flatMap((model) => model.tags))).sort(), [results]);
  const visibleResults = useMemo(() => {
    const filtered = uniqueHubModels(results).filter((model) => {
      const tagOk = !selectedTag || model.tags.includes(selectedTag);
      const capabilityOk = capabilityFilter === "all" || model.tags.some((tag) => tag.toLowerCase() === capabilityFilter) || (capabilityFilter === "vision" && model.supports_vision);
      return tagOk && capabilityOk;
    });
    if (sortMode === "smallest") {
      return [...filtered].sort((a, b) => modelMinSize(a) - modelMinSize(b));
    }
    if (sortMode === "largest") {
      return [...filtered].sort((a, b) => modelMaxSize(b) - modelMaxSize(a));
    }
    if (sortMode === "name") {
      return [...filtered].sort((a, b) => a.name.localeCompare(b.name));
    }
    if (sortMode === "likes") {
      return [...filtered].sort((a, b) => (b.likes ?? 0) - (a.likes ?? 0));
    }
    return filtered;
  }, [capabilityFilter, results, selectedTag, sortMode]);
  const summary = query.trim()
    ? selectedTag
      ? `Results for "${query.trim()}" tagged ${selectedTag}`
      : `Results for "${query.trim()}"`
    : selectedTag
      ? `HF search for ${selectedTag}`
      : "Top GGUF models on Hugging Face";
  const downloadEntries = useMemo(() => Object.values(downloads).sort((a, b) => (a.done === b.done ? a.filename.localeCompare(b.filename) : a.done ? 1 : -1)), [downloads]);
  const activeDownloadCount = downloadEntries.filter((entry) => !entry.done).length;
  const completedDownloadCount = downloadEntries.filter((entry) => entry.done).length;
  const selectedHubModel = visibleResults.find((model) => model.id === selectedModelId) ?? visibleResults[0] ?? null;
  const totalLocalGb = useMemo(() => models.reduce((sum, model) => sum + (model.size_gb || 0), 0), [models]);
  const cacheGb = useMemo(() => downloadEntries.reduce((sum, entry) => sum + (entry.total_bytes || entry.downloaded_bytes || 0) / 1_073_741_824, 0), [downloadEntries]);
  const resultTitle = sortMode === "lastModified" ? "Latest Models" : sortMode === "largest" ? "Largest Models" : sortMode === "smallest" ? "Smallest Models" : sortMode === "likes" ? "Most Liked Models" : "Popular Models";

  const isInstalled = useCallback((quant: HubQuant) => installedFilenames.has(basename(quant.filename)), [installedFilenames]);
  const anyInstalled = useCallback((model: HubModel) => model.quants.some(isInstalled), [isInstalled]);

  const mergeHubDetail = useCallback((detail: HubModel | null) => {
    if (!detail) return;
    setResults((prev) => prev.map((model) => {
      if (model.id !== detail.id) return model;
      return {
        ...model,
        ...detail,
        tags: detail.tags.length > 0 ? detail.tags : model.tags,
        description: detail.description || model.description,
      };
    }));
  }, []);

  const requestHubDetails = useCallback((model: HubModel, includeReadme = false) => {
    if (!modelNeedsDetails(model, includeReadme)) return;
    const detailKey = `${model.id}:${includeReadme ? "readme" : "meta"}`;
    if (detailFetchedIds.current.has(detailKey)) return;
    detailFetchedIds.current.add(detailKey);
    setDetailLoadingIds((prev) => new Set(prev).add(model.id));
    void api.getHubModelDetails(model.id, includeReadme)
      .then(mergeHubDetail)
      .catch(() => {})
      .finally(() => {
        setDetailLoadingIds((prev) => {
          const next = new Set(prev);
          next.delete(model.id);
          return next;
        });
      });
  }, [mergeHubDetail]);

  const runSearch = useCallback(async (nextQuery: string, nextOffset: number, append: boolean) => {
    const trimmed = nextQuery.trim();
    append ? setLoadingMore(true) : (setLoading(true), setResults([]), setOffset(0), setHasMore(false));
    setError(null);
    try {
      const serverSort = sortMode === "lastModified" || sortMode === "likes" ? sortMode : "downloads";
      const found = uniqueHubModels(await api.searchHubModels(trimmed, nextOffset, serverSort, selectedTag));
      let addedCount = found.length;
      setResults((prev) => {
        const next = uniqueHubModels(append ? [...prev, ...found] : found);
        addedCount = append ? Math.max(0, next.length - prev.length) : next.length;
        return next;
      });
      setOffset(nextOffset + found.length);
      setHasMore(found.length === PAGE_SIZE && addedCount > 0);
    } catch (searchError) {
      setError(String(searchError));
      if (!append) setResults([]);
    } finally {
      append ? setLoadingMore(false) : setLoading(false);
    }
  }, [selectedTag, sortMode]);

  useEffect(() => {
    api.listDownloads().then((items) => setDownloads(Object.fromEntries(items.map((item) => [item.id, item])))).catch(() => {});
    api.getHubAccessStatus().then(setHubStatus).catch(() => setHubStatus({ configured: false, reachable: false, user: null, error: "Hub status unavailable" }));
  }, []);

  useEffect(() => {
    const unlisten = listen<DownloadProgress>("model-download-progress", (event) => {
      const progress = event.payload;
      setDownloads((prev) => ({ ...prev, [progress.id]: progress }));
      if (progress.done && !progress.error && progress.status === "Completed") onRefresh();
    });
    return () => { void unlisten.then((fn) => fn()); };
  }, [onRefresh]);

  useEffect(() => {
    if (!downloadsOpen) return undefined;
    function handlePointerDown(event: MouseEvent) {
      if (!downloadsMenuRef.current?.contains(event.target as Node)) setDownloadsOpen(false);
    }
    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, [downloadsOpen]);

  useEffect(() => {
    const timer = window.setTimeout(() => { void runSearch(query, 0, false); }, 250);
    return () => window.clearTimeout(timer);
  }, [query, selectedTag, runSearch]);

  useEffect(() => {
    if (selectedHubModel || visibleResults.length === 0) return;
    setSelectedModelId(visibleResults[0].id);
  }, [selectedHubModel, visibleResults]);

  useEffect(() => {
    if (!selectedHubModel) return;
    requestHubDetails(selectedHubModel, true);
  }, [requestHubDetails, selectedHubModel]);

  useEffect(() => {
    visibleResults.slice(0, 8).forEach((model) => requestHubDetails(model, false));
  }, [requestHubDetails, visibleResults]);

  useEffect(() => {
    const element = sentinelRef.current;
    if (!element || !hasMore || loadingMore) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting) void runSearch(query, offset, true);
    }, { threshold: 0.1 });
    observer.observe(element);
    return () => observer.disconnect();
  }, [hasMore, loadingMore, offset, query, runSearch]);

  async function handleDownload(model: HubModel, quant: HubQuant) {
    const id = quant.url;
    setDownloads((prev) => ({
      ...prev,
      [id]: prev[id] ?? {
        id,
        filename: quant.filename,
        dest_path: null,
        partial_path: null,
        supports_vision: model.supports_vision,
        repo_id: model.id,
        downloaded_bytes: 0,
        total_bytes: 0,
        percent: 0,
        speed_bps: null,
        eta_seconds: null,
        resumable: false,
        attempt: 1,
        done: false,
        status: "Starting",
        error: null,
      },
    }));
    void api.downloadHubModel(quant.url, quant.filename, model.supports_vision, model.id).catch((downloadError) => {
      const message = String(downloadError);
      if (message.toLowerCase().includes("cancelled") || message.toLowerCase().includes("paused")) return;
      setDownloads((prev) => ({
        ...prev,
        [id]: {
          id,
          filename: quant.filename,
          dest_path: prev[id]?.dest_path ?? null,
          partial_path: prev[id]?.partial_path ?? null,
          downloaded_bytes: prev[id]?.downloaded_bytes ?? 0,
          total_bytes: prev[id]?.total_bytes ?? 0,
          percent: prev[id]?.percent ?? 0,
          speed_bps: null,
          eta_seconds: null,
          resumable: prev[id]?.resumable ?? false,
          attempt: prev[id]?.attempt ?? 1,
          done: true,
          status: "Failed",
          error: message,
        },
      }));
    });
  }

  async function handleResumeDownload(entry: DownloadProgress) {
    setDownloads((prev) => ({ ...prev, [entry.id]: { ...entry, done: false, status: "Resuming", error: null } }));
    void api.downloadHubModel(entry.id, entry.filename, entry.supports_vision ?? undefined, entry.repo_id ?? undefined).catch((downloadError) => {
      const message = String(downloadError);
      if (message.toLowerCase().includes("cancelled") || message.toLowerCase().includes("paused")) return;
      setDownloads((prev) => prev[entry.id] ? { ...prev, [entry.id]: { ...prev[entry.id], done: true, status: "Failed", error: message } } : prev);
    });
  }

  async function handlePauseDownload(id: string) {
    try {
      await api.pauseDownload(id);
    } catch (pauseError) {
      const message = String(pauseError);
      setDownloads((prev) => prev[id] ? { ...prev, [id]: { ...prev[id], status: "Failed", error: message } } : prev);
    }
  }

  async function handleCancelDownload(id: string) {
    try {
      await api.cancelDownload(id);
    } catch (cancelError) {
      const message = String(cancelError);
      setDownloads((prev) => prev[id] ? { ...prev, [id]: { ...prev[id], done: true, status: "Failed", error: message } } : prev);
    }
  }

  async function handleDelete(model: ModelInfo) {
    setDeletingPath(model.path);
    setDeleteMessage(null);
    try {
      const message = await api.deleteModelFile(model.path);
      onRefresh();
      setDeleteConfirm(null);
      setDeleteMessage(message);
    } catch (deleteError) {
      setDeleteMessage(`Delete failed: ${String(deleteError)}`);
    } finally {
      setDeletingPath(null);
    }
  }

  async function handleSyncLocalMetadata() {
    setSyncingMetadata(true);
    setSyncMessage(null);
    try {
      const summary = await api.syncLocalModelMetadata();
      onRefresh();
      setSyncMessage(
        summary.matched_models > 0
          ? `Synced ${summary.updated_models} of ${summary.matched_models} Hugging Face matches across ${summary.scanned_models} local models.`
          : `No exact Hugging Face matches found for ${summary.scanned_models} local models.`
      );
    } catch (syncError) {
      setSyncMessage(`Sync failed: ${String(syncError)}`);
    } finally {
      setSyncingMetadata(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b px-5 py-3" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold" style={{ color: "var(--text-0)" }}>Model Hub</h2>
            <p className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>Search Hugging Face GGUF models, download resumably, and manage local files.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <HubStat label="HTTP" value={hubStatus?.configured ? "Auth" : "Public"} />
            <HubStat label="Cache" value={`${cacheGb.toFixed(1)} GB`} />
            <HubStat label="Local" value={String(models.length)} />
            <HubStat label="Disk" value={`${totalLocalGb.toFixed(1)} GB`} />
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex overflow-hidden rounded-md" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
            <button onClick={() => setSection("search")} className="px-6 py-2 text-sm font-medium transition" style={{ background: section === "search" ? "rgba(34,211,238,0.12)" : "transparent", color: section === "search" ? "#a5f3fc" : "var(--text-1)", border: "none", cursor: "pointer" }}>Discover</button>
            <button onClick={() => setSection("local")} className="px-6 py-2 text-sm font-medium transition" style={{ background: section === "local" ? "rgba(34,211,238,0.12)" : "transparent", color: section === "local" ? "#a5f3fc" : "var(--text-1)", border: "none", cursor: "pointer" }}>On Device</button>
          </div>

          {section === "search" && (
            <>
              <form className="min-w-[280px] flex-1" onSubmit={(event) => { event.preventDefault(); void runSearch(query, 0, false); }}>
                <input type="text" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models" className="w-full rounded-md px-4 py-2 text-sm outline-none" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }} autoFocus />
              </form>
              <select value={formatFilter} onChange={(event) => setFormatFilter(event.target.value)} className="rounded-md px-3 py-2 text-sm outline-none" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}>
                <option value="gguf">GGUF</option>
              </select>
              <select value={capabilityFilter} onChange={(event) => setCapabilityFilter(event.target.value)} className="rounded-md px-3 py-2 text-sm outline-none" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}>
                <option value="all">All capabilities</option>
                <option value="vision">Vision</option>
                <option value="reasoning">Reasoning</option>
                <option value="tools">Tools</option>
              </select>
              <select value={sortMode} onChange={(event) => setSortMode(event.target.value as HubSortMode)} className="rounded-md px-3 py-2 text-sm outline-none" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}>
                <option value="lastModified">Latest</option>
                <option value="downloads">Popular</option>
                <option value="likes">Most liked</option>
                <option value="largest">Largest</option>
                <option value="smallest">Smallest</option>
                <option value="name">Name</option>
              </select>
              <button onClick={() => void runSearch(query, 0, false)} disabled={loading} className="rounded-md px-3 py-2 text-sm" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)", cursor: loading ? "not-allowed" : "pointer" }}>{loading ? "..." : "Refresh"}</button>
            </>
          )}
        </div>

        {section === "search" && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="text-xs" style={{ color: "var(--text-2)" }}>{summary}</span>
            <span className="text-xs" style={{ color: hubStatus?.configured ? (hubStatus.reachable ? "#34d399" : "#f87171") : "var(--text-2)" }}>
              {hubStatus?.configured ? hubStatus.reachable ? `HF ${hubStatus.user ?? "token ready"}` : `HF error: ${hubStatus.error}` : "HF public mode"}
            </span>
            <button onClick={() => setSelectedTag(null)} className="rounded px-2.5 py-1 text-xs font-medium transition" style={{ background: selectedTag === null ? "rgba(34,211,238,0.12)" : "transparent", border: selectedTag === null ? "1px solid rgba(34,211,238,0.25)" : "1px solid transparent", color: selectedTag === null ? "#22d3ee" : "var(--text-1)", cursor: "pointer" }}>all</button>
            {tags.slice(0, 10).map((tag) => (
              <button key={tag} onClick={() => setSelectedTag(selectedTag === tag ? null : tag)} className="rounded px-2.5 py-1 text-xs font-medium transition" style={{ background: selectedTag === tag ? "rgba(34,211,238,0.12)" : "transparent", border: selectedTag === tag ? "1px solid rgba(34,211,238,0.25)" : "1px solid transparent", color: selectedTag === tag ? "#22d3ee" : "var(--text-1)", cursor: "pointer" }}>
                {tag}
              </button>
            ))}
          </div>
        )}

        <div className="relative ml-auto" ref={downloadsMenuRef}>
          <button onClick={() => setDownloadsOpen((open) => !open)} className="flex items-center gap-2 rounded px-3 py-1.5 text-xs font-medium transition" style={{ background: downloadsOpen ? "rgba(34,211,238,0.12)" : "var(--surface-2)", border: downloadsOpen ? "1px solid rgba(34,211,238,0.24)" : "1px solid var(--border)", color: downloadsOpen ? "#22d3ee" : "var(--text-1)", cursor: "pointer" }}>
            <span>Download Manager</span>
            {(activeDownloadCount > 0 || completedDownloadCount > 0) && <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold" style={{ background: activeDownloadCount > 0 ? "rgba(34,211,238,0.18)" : "rgba(255,255,255,0.08)", color: activeDownloadCount > 0 ? "#22d3ee" : "var(--text-1)" }}>{activeDownloadCount > 0 ? activeDownloadCount : completedDownloadCount}</span>}
          </button>

          {downloadsOpen && (
            <div className="absolute right-0 top-full z-20 mt-2 w-[460px] overflow-hidden rounded" style={{ background: "var(--surface-1)", border: "1px solid var(--border)", boxShadow: "0 14px 40px rgba(0,0,0,0.35)" }}>
              <div className="flex items-center justify-between border-b px-3 py-2" style={{ borderColor: "var(--border)" }}>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>Download Manager</div>
                  <div className="mt-1 text-xs" style={{ color: "var(--text-1)" }}>{activeDownloadCount > 0 ? `${activeDownloadCount} active · resumable transfers` : completedDownloadCount > 0 ? `${completedDownloadCount} recent` : "No downloads yet"}</div>
                </div>
                {completedDownloadCount > 0 && <button onClick={() => void api.clearCompletedDownloads().then(() => setDownloads((prev) => Object.fromEntries(Object.entries(prev).filter(([, entry]) => !entry.done))))} className="rounded px-2 py-1 text-[11px] font-medium transition" style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-1)", cursor: "pointer" }}>Clear Done</button>}
              </div>
              <div className="max-h-[420px] overflow-y-auto">
                {downloadEntries.length === 0 ? (
                  <div className="px-3 py-6 text-sm" style={{ color: "var(--text-2)" }}>Start a model download and it will show up here with progress and cancel controls.</div>
                ) : downloadEntries.map((entry) => {
                  const tone = progressTone(entry.status, entry.error);
                  const shortName = entry.filename.split(/[\\/]/).filter(Boolean).pop() ?? entry.filename;
                  return (
                    <div key={entry.id} className="border-b px-3 py-3 last:border-b-0" style={{ borderColor: "var(--border)" }}>
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm font-medium" style={{ color: "var(--text-0)" }}>{shortName}</div>
                          {entry.filename !== shortName && <div className="mt-0.5 truncate font-mono text-[11px]" style={{ color: "var(--text-2)" }}>{entry.filename}</div>}
                        </div>
                        <span className="shrink-0 text-[11px] font-medium" style={{ color: tone }}>{entry.status}</span>
                      </div>
                      {(!entry.done || entry.downloaded_bytes > 0 || entry.total_bytes > 0) && (
                        <div className="mt-2">
                          <div className="h-1.5 overflow-hidden rounded" style={{ background: "rgba(255,255,255,0.08)" }}>
                            <div className="h-full rounded transition-all" style={{ width: `${Math.max(4, Math.round(entry.percent * 100))}%`, background: tone }} />
                          </div>
                          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]" style={{ color: "var(--text-2)" }}>
                            <span>{downloadDetail(entry)}</span>
                            {entry.attempt > 1 && <span style={{ color: "#a5b4fc" }}>attempt {entry.attempt}/5</span>}
                            {entry.resumable && !entry.done && <span style={{ color: "#fde68a" }}>resume ready</span>}
                          </div>
                        </div>
                      )}
                      {entry.error && <div className="mt-2 text-[11px]" style={{ color: "#f87171" }}>{entry.error}</div>}
                      <div className="mt-3 flex items-center gap-2">
                        {!entry.done && entry.status === "Paused" ? (
                          <>
                            <button onClick={() => void handleResumeDownload(entry)} className="rounded px-2.5 py-1 text-[11px] font-semibold transition" style={{ background: "#22d3ee", border: "none", color: "#041014", cursor: "pointer" }}>Resume</button>
                            <button onClick={() => void handleCancelDownload(entry.id)} className="rounded px-2.5 py-1 text-[11px] font-medium transition" style={{ background: "rgba(248,113,113,0.12)", border: "1px solid rgba(248,113,113,0.24)", color: "#f87171", cursor: "pointer" }}>Cancel</button>
                          </>
                        ) : !entry.done ? (
                          <>
                            <button onClick={() => void handlePauseDownload(entry.id)} disabled={entry.status === "Pausing" || entry.status === "Cancelling"} className="rounded px-2.5 py-1 text-[11px] font-medium transition" style={{ background: "rgba(251,191,36,0.12)", border: "1px solid rgba(251,191,36,0.24)", color: "#fde68a", cursor: entry.status === "Pausing" || entry.status === "Cancelling" ? "not-allowed" : "pointer", opacity: entry.status === "Pausing" || entry.status === "Cancelling" ? 0.7 : 1 }}>{entry.status === "Pausing" ? "Pausing..." : "Pause"}</button>
                            <button onClick={() => void handleCancelDownload(entry.id)} disabled={entry.status === "Cancelling"} className="rounded px-2.5 py-1 text-[11px] font-medium transition" style={{ background: "rgba(248,113,113,0.12)", border: "1px solid rgba(248,113,113,0.24)", color: "#f87171", cursor: entry.status === "Cancelling" ? "not-allowed" : "pointer", opacity: entry.status === "Cancelling" ? 0.7 : 1 }}>{entry.status === "Cancelling" ? "Cancelling..." : "Cancel"}</button>
                          </>
                        ) : entry.done && entry.status === "Failed" && entry.resumable ? (
                          <>
                            <button onClick={() => void handleResumeDownload(entry)} className="rounded px-2.5 py-1 text-[11px] font-semibold transition" style={{ background: "#22d3ee", border: "none", color: "#041014", cursor: "pointer" }}>Resume</button>
                            <button onClick={() => void handleCancelDownload(entry.id)} className="rounded px-2.5 py-1 text-[11px] font-medium transition" style={{ background: "rgba(248,113,113,0.12)", border: "1px solid rgba(248,113,113,0.24)", color: "#f87171", cursor: "pointer" }}>Discard</button>
                          </>
                        ) : entry.dest_path ? (
                          <button onClick={() => void api.showInFolder(entry.dest_path as string)} className="rounded px-2.5 py-1 text-[11px] font-medium transition" style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-1)", cursor: "pointer" }}>Open Folder</button>
                        ) : null}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden p-4">
        {section === "search" ? (
          <div className="grid h-full min-h-0 gap-4 xl:grid-cols-[minmax(0,1fr)_500px]">
            <div className="min-h-0 overflow-y-auto pr-1">
              {error && <div className="mb-4 rounded px-4 py-3 text-sm" style={{ background: "rgba(248,113,113,0.10)", border: "1px solid rgba(248,113,113,0.25)", color: "#f87171" }}>{error}</div>}
              <div className="mb-3 flex items-center justify-between gap-2">
                <h3 className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>{resultTitle}</h3>
                <div className="flex overflow-hidden rounded-md" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
                  {(["rows", "cards"] as const).map((view) => (
                    <button key={view} onClick={() => setExpandedId(view)} className="px-3 py-1 text-xs" style={{ background: expandedId === view ? "rgba(34,211,238,0.12)" : "transparent", color: expandedId === view ? "#a5f3fc" : "var(--text-1)", border: "none" }}>{view === "rows" ? "list" : "cards"}</button>
                  ))}
                </div>
              </div>
              {loading ? (
                <div className="flex items-center justify-center rounded py-20 text-sm" style={{ border: "1px solid var(--border)", color: "var(--text-2)" }}>Searching Hugging Face...</div>
              ) : visibleResults.length === 0 && !error ? (
                <div className="flex flex-col items-center justify-center gap-3 rounded py-20 text-sm" style={{ border: "1px solid var(--border)", color: "var(--text-2)" }}>
                  <span>{selectedTag ? `No results matched the "${selectedTag}" filter.` : "Search Hugging Face for any GGUF model."}</span>
                  <span className="text-xs" style={{ color: "var(--text-2)" }}>Try llama, qwen, mistral, phi, gemma, or unsloth.</span>
                </div>
              ) : (
                <div className={expandedId === "rows" ? "overflow-hidden rounded-md" : "grid gap-3 lg:grid-cols-2 2xl:grid-cols-3"} style={expandedId === "rows" ? { border: "1px solid var(--border)", background: "var(--surface-1)" } : undefined}>
                  {expandedId === "rows" && (
                    <div className="grid grid-cols-[minmax(0,1.7fr)_92px_92px_120px_92px] gap-3 border-b px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.16em]" style={{ borderColor: "var(--border)", color: "var(--text-2)" }}>
                      <span>Model</span>
                      <span>Files</span>
                      <span>Size</span>
                      <span>Updated</span>
                      <span>Status</span>
                    </div>
                  )}
                  {visibleResults.map((model) => {
                    const installed = anyInstalled(model);
                    const selected = selectedHubModel?.id === model.id;
                    const topTags = model.tags.slice(0, expandedId === "rows" ? 4 : 5);
                    const detailsLoading = detailLoadingIds.has(model.id);
                    return (
                      <button
                        key={model.id}
                        onClick={() => setSelectedModelId(model.id)}
                        className={expandedId === "rows" ? "grid w-full grid-cols-[minmax(0,1.7fr)_92px_92px_120px_92px] items-center gap-3 border-b px-3 py-3 text-left transition last:border-b-0" : "flex min-h-[168px] flex-col rounded px-4 py-4 text-left transition"}
                        style={{
                          ...(expandedId === "rows" ? { background: selected ? "rgba(34,211,238,0.075)" : installed ? "rgba(52,211,153,0.045)" : "transparent", borderColor: "var(--border)" } : panelStyle),
                          ...(expandedId === "rows" ? {} : { background: modelCardBackground(selected, installed), border: `1px solid ${selected ? "rgba(34,211,238,0.72)" : installed ? "rgba(52,211,153,0.28)" : "var(--border)"}` }),
                          boxShadow: selected && expandedId !== "rows" ? "0 0 0 1px rgba(34,211,238,0.26) inset, 0 10px 28px rgba(34,211,238,0.08)" : "none",
                          cursor: "pointer",
                        }}
                      >
                        {expandedId === "rows" ? (
                          <>
                            <div className="min-w-0">
                              <div className="flex min-w-0 items-center gap-2">
                                <span className="truncate text-sm font-semibold" style={{ color: "var(--text-0)" }}>{model.name}</span>
                                {selected && <span className="shrink-0 rounded px-1.5 py-0.5 text-[9px] font-bold uppercase" style={{ background: "rgba(34,211,238,0.14)", color: "#67e8f9", border: "1px solid rgba(34,211,238,0.30)" }}>Selected</span>}
                              </div>
                              <div className="mt-1 truncate font-mono text-[11px]" style={{ color: "var(--text-2)" }}>{model.id}</div>
                              <div className="mt-2 flex min-w-0 flex-wrap gap-1.5">
                                {topTags.slice(0, 3).map((tag) => <TagBadge key={tag} tag={tag} />)}
                              </div>
                            </div>
                            <span className="text-xs" style={{ color: "var(--text-1)" }}>{model.quants.length}</span>
                            <span className="truncate text-xs" style={{ color: detailsLoading ? "#67e8f9" : "var(--text-1)" }}>{formatModelSizeRange(model.quants, detailsLoading)}</span>
                            <span className="text-xs" style={{ color: "var(--text-1)" }}>{timeAgo(model.last_modified ?? null).replace("updated ", "")}</span>
                            <span className="justify-self-start rounded px-2 py-1 text-[10px] font-bold uppercase" style={{ background: installed ? "rgba(52,211,153,0.10)" : "var(--surface-2)", color: installed ? "#34d399" : "var(--text-2)", border: `1px solid ${installed ? "rgba(52,211,153,0.22)" : "var(--border)"}` }}>{installed ? "On device" : `${abbrevCount(model.downloads ?? 0)} dl`}</span>
                          </>
                        ) : (
                          <>
                            <div className="flex items-start gap-3">
                              <div className="min-w-0 flex-1">
                                <div className="truncate text-sm font-semibold" style={{ color: "var(--text-0)" }}>{model.name}</div>
                                <div className="mt-1 truncate font-mono text-[11px]" style={{ color: "var(--text-2)" }}>{model.id}</div>
                                <div className="mt-2 flex flex-wrap gap-2 text-[11px]" style={{ color: "var(--text-1)" }}>
                                  <span>{abbrevCount(model.downloads ?? 0)} downloads</span>
                                  <span>{abbrevCount(model.likes ?? 0)} likes</span>
                                  <span>{formatOptionSummary(model, detailsLoading)}</span>
                                  <span>{timeAgo(model.last_modified ?? null)}</span>
                                </div>
                              </div>
                              <div className="flex shrink-0 flex-col items-end gap-1">
                                {selected && <span className="rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider" style={{ background: "rgba(34,211,238,0.14)", color: "#67e8f9", border: "1px solid rgba(34,211,238,0.30)" }}>Selected</span>}
                                {installed && <span className="rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider" style={{ background: "rgba(52,211,153,0.10)", color: "#34d399", border: "1px solid rgba(52,211,153,0.22)" }}>On device</span>}
                              </div>
                            </div>
                            <div className="mt-3 flex flex-wrap gap-1.5">
                              {topTags.map((tag) => <TagBadge key={tag} tag={tag} />)}
                            </div>
                            <div className="mt-auto grid grid-cols-3 gap-2 pt-4">
                              <div className="rounded px-2 py-1.5" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
                                <div className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-2)" }}>Params</div>
                                <div className="mt-0.5 truncate text-xs font-semibold" style={{ color: "var(--text-0)" }}>{model.params || "-"}</div>
                              </div>
                              <div className="rounded px-2 py-1.5" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
                                <div className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-2)" }}>Files</div>
                                <div className="mt-0.5 text-xs font-semibold" style={{ color: "var(--text-0)" }}>{model.quants.length}</div>
                              </div>
                              <div className="rounded px-2 py-1.5" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
                                <div className="text-[10px] uppercase tracking-wider" style={{ color: "var(--text-2)" }}>Size</div>
                                <div className="mt-0.5 truncate text-xs font-semibold" style={{ color: detailsLoading ? "#67e8f9" : "var(--text-0)" }}>{formatModelSizeRange(model.quants, detailsLoading)}</div>
                              </div>
                            </div>
                          </>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}

              {visibleResults.length > 0 && (
                <div className="mt-4 flex flex-col items-center gap-2">
                  {loadingMore && <span className="text-xs" style={{ color: "var(--text-2)" }}>Loading more...</span>}
                  {hasMore && !loadingMore && (
                    <button onClick={() => void runSearch(query, offset, true)} className="rounded-md px-3 py-1.5 text-xs font-medium" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
                      Load more from Hugging Face
                    </button>
                  )}
                  <div ref={sentinelRef} className="h-4 w-full" />
                  {!hasMore && <span className="pb-2 text-xs" style={{ color: "var(--text-2)" }}>All {visibleResults.length} visible results shown</span>}
                </div>
              )}
            </div>
            <HubPreview model={selectedHubModel} downloads={downloads} detailsLoading={selectedHubModel ? detailLoadingIds.has(selectedHubModel.id) : false} isInstalled={isInstalled} onDownload={(model, quant) => void handleDownload(model, quant)} onCancel={(id) => void handleCancelDownload(id)} onPause={(id) => void handlePauseDownload(id)} />
          </div>
        ) : (
          <div>
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs" style={{ color: "var(--text-2)" }}>These are your locally scanned .gguf files. Delete removes the file from disk.</p>
              <button
                onClick={() => void handleSyncLocalMetadata()}
                disabled={syncingMetadata || models.length === 0}
                className="rounded px-3 py-1.5 text-xs font-medium transition"
                style={{
                  background: syncingMetadata ? "rgba(34,211,238,0.12)" : "var(--surface-2)",
                  color: syncingMetadata ? "#22d3ee" : "var(--text-1)",
                  border: syncingMetadata ? "1px solid rgba(34,211,238,0.24)" : "1px solid var(--border)",
                  cursor: syncingMetadata || models.length === 0 ? "not-allowed" : "pointer",
                  opacity: syncingMetadata || models.length === 0 ? 0.7 : 1,
                }}
              >
                {syncingMetadata ? "Syncing HF Metadata..." : "Sync HF Metadata"}
              </button>
            </div>
            {syncMessage && (
              <div className="mb-3 rounded px-3 py-2 text-xs" style={{ background: "rgba(34,211,238,0.08)", border: "1px solid rgba(34,211,238,0.18)", color: "var(--text-1)" }}>
                {syncMessage}
              </div>
            )}
            {deleteMessage && (
              <div className="mb-3 rounded px-3 py-2 text-xs" style={{ background: deleteMessage.startsWith("Delete failed") ? "rgba(248,113,113,0.10)" : "rgba(52,211,153,0.08)", border: deleteMessage.startsWith("Delete failed") ? "1px solid rgba(248,113,113,0.25)" : "1px solid rgba(52,211,153,0.20)", color: deleteMessage.startsWith("Delete failed") ? "#f87171" : "var(--text-1)" }}>
                {deleteMessage}
              </div>
            )}
            {models.length === 0 ? (
              <div className="flex items-center justify-center py-16 text-sm" style={{ color: "var(--text-2)" }}>No local models found. Download some from Hugging Face, or add a directory in Settings.</div>
            ) : (
              <div className="flex flex-col gap-1">
                {models.map((model) => (
                  <div key={model.path} className="flex items-center gap-3 rounded px-3 py-2.5" style={panelStyle}>
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium" style={{ color: "var(--text-0)" }}>{model.filename}</div>
                      <div className="mt-0.5 truncate text-xs" style={{ color: "var(--text-2)" }}>{model.path}</div>
                    </div>
                    <div className="shrink-0 text-xs" style={{ color: "var(--text-1)" }}>
                      <div>{model.size_gb.toFixed(2)} GB</div>
                      <div style={{ color: "#fbbf24" }}>{model.quant ?? ""}</div>
                      <div style={{ color: "var(--text-2)" }}>{model.family}</div>
                    </div>
                    {deleteConfirm === model.path ? (
                      <div className="flex shrink-0 items-center gap-1">
                        <span className="text-xs" style={{ color: "#f87171" }}>Delete file?</span>
                        <button onClick={() => void handleDelete(model)} disabled={deletingPath === model.path} className="rounded px-2 py-1 text-xs font-medium" style={{ background: "rgba(248,113,113,0.15)", color: "#f87171", border: "1px solid rgba(248,113,113,0.25)", cursor: deletingPath === model.path ? "wait" : "pointer", opacity: deletingPath === model.path ? 0.7 : 1 }}>{deletingPath === model.path ? "Deleting..." : "Confirm"}</button>
                        <button onClick={() => setDeleteConfirm(null)} disabled={deletingPath === model.path} className="rounded px-2 py-1 text-xs" style={{ background: "var(--surface-2)", color: "var(--text-1)", border: "1px solid var(--border)", cursor: deletingPath === model.path ? "not-allowed" : "pointer", opacity: deletingPath === model.path ? 0.7 : 1 }}>Cancel</button>
                      </div>
                    ) : (
                      <button onClick={() => { setDeleteMessage(null); setDeleteConfirm(model.path); }} disabled={deletingPath !== null} className="shrink-0 rounded px-2 py-1 text-xs transition" style={{ background: "transparent", color: "var(--text-2)", border: "1px solid transparent", cursor: deletingPath ? "not-allowed" : "pointer", opacity: deletingPath ? 0.7 : 1 }}>Delete</button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
