import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import * as api from "../../lib/tauri";
import type { DownloadProgress, HubModel, HubQuant } from "../../lib/tauri";
import type { ModelInfo } from "../../lib/types";

interface Props {
  models: ModelInfo[];
  onRefresh: () => void;
}

const PAGE_SIZE = 20;
const panelStyle = { background: "var(--surface-1)", border: "1px solid var(--border)" };

const TAG_COLORS: Record<string, { bg: string; color: string; border: string }> = {
  reasoning: { bg: "rgba(251,191,36,0.10)", color: "#fde68a", border: "rgba(251,191,36,0.20)" },
  tools: { bg: "rgba(52,211,153,0.10)", color: "#6ee7b7", border: "rgba(52,211,153,0.20)" },
  thinking: { bg: "rgba(167,139,250,0.10)", color: "#c4b5fd", border: "rgba(167,139,250,0.20)" },
  chat: { bg: "rgba(34,211,238,0.08)", color: "#67e8f9", border: "rgba(34,211,238,0.18)" },
  math: { bg: "rgba(249,115,22,0.10)", color: "#fdba74", border: "rgba(249,115,22,0.20)" },
  moe: { bg: "rgba(99,102,241,0.10)", color: "#a5b4fc", border: "rgba(99,102,241,0.20)" },
  vision: { bg: "rgba(236,72,153,0.10)", color: "#f9a8d4", border: "rgba(236,72,153,0.20)" },
};

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

function progressTone(status: string, error?: string | null) {
  if (error || status === "Failed") return "#f87171";
  if (status === "Completed") return "#34d399";
  if (status === "Cancelled") return "#fbbf24";
  return "#22d3ee";
}

export function ModelBrowser({ models, onRefresh }: Props) {
  const [section, setSection] = useState<"search" | "local">("search");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<HubModel[]>([]);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [downloads, setDownloads] = useState<Record<string, DownloadProgress>>({});
  const [downloadsOpen, setDownloadsOpen] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const downloadsMenuRef = useRef<HTMLDivElement>(null);

  const installedFilenames = useMemo(() => new Set(models.map((model) => basename(model.filename))), [models]);
  const tags = useMemo(() => Array.from(new Set(results.flatMap((model) => model.tags))).sort(), [results]);
  const visibleResults = useMemo(() => results.filter((model) => !selectedTag || model.tags.includes(selectedTag)), [results, selectedTag]);
  const summary = query.trim() ? `Results for "${query.trim()}"` : "Top GGUF models on Hugging Face";
  const downloadEntries = useMemo(() => Object.values(downloads).sort((a, b) => (a.done === b.done ? a.filename.localeCompare(b.filename) : a.done ? 1 : -1)), [downloads]);
  const activeDownloadCount = downloadEntries.filter((entry) => !entry.done).length;
  const completedDownloadCount = downloadEntries.filter((entry) => entry.done).length;

  const isInstalled = useCallback((quant: HubQuant) => installedFilenames.has(basename(quant.filename)), [installedFilenames]);
  const anyInstalled = useCallback((model: HubModel) => model.quants.some(isInstalled), [isInstalled]);

  const runSearch = useCallback(async (nextQuery: string, nextOffset: number, append: boolean) => {
    const trimmed = nextQuery.trim();
    append ? setLoadingMore(true) : (setLoading(true), setResults([]), setOffset(0), setHasMore(false));
    setError(null);
    try {
      const found = await api.searchHubModels(trimmed, nextOffset);
      setResults((prev) => (append ? [...prev, ...found] : found));
      setOffset(nextOffset + found.length);
      setHasMore(found.length === PAGE_SIZE);
      if (!append && selectedTag && !found.some((model) => model.tags.includes(selectedTag))) {
        setSelectedTag(null);
      }
    } catch (searchError) {
      setError(String(searchError));
      if (!append) setResults([]);
    } finally {
      append ? setLoadingMore(false) : setLoading(false);
    }
  }, [selectedTag]);

  useEffect(() => {
    api.listDownloads().then((items) => setDownloads(Object.fromEntries(items.map((item) => [item.id, item])))).catch(() => {});
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
  }, [query, runSearch]);

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
    setDownloads((prev) => ({ ...prev, [id]: prev[id] ?? { id, filename: quant.filename, dest_path: null, downloaded_bytes: 0, total_bytes: 0, percent: 0, done: false, status: "Starting", error: null } }));
    void api.downloadHubModel(quant.url, quant.filename, model.supports_vision).catch((downloadError) => {
      const message = String(downloadError);
      if (message.toLowerCase().includes("cancelled")) return;
      setDownloads((prev) => ({ ...prev, [id]: { id, filename: quant.filename, dest_path: prev[id]?.dest_path ?? null, downloaded_bytes: prev[id]?.downloaded_bytes ?? 0, total_bytes: prev[id]?.total_bytes ?? 0, percent: prev[id]?.percent ?? 0, done: true, status: "Failed", error: message } }));
    });
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
    try {
      await api.deleteModelFile(model.path);
      onRefresh();
      setDeleteConfirm(null);
    } catch (deleteError) {
      alert(`Failed to delete: ${deleteError}`);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-start gap-2 border-b px-4 py-2.5" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
        <div className="flex rounded overflow-hidden" style={{ border: "1px solid var(--border)" }}>
          <button onClick={() => setSection("search")} className="px-3 py-1.5 text-xs font-medium transition" style={{ background: section === "search" ? "rgba(34,211,238,0.14)" : "transparent", color: section === "search" ? "#22d3ee" : "var(--text-1)", border: "none", cursor: "pointer" }}>Hugging Face</button>
          <button onClick={() => setSection("local")} className="px-3 py-1.5 text-xs font-medium transition" style={{ background: section === "local" ? "rgba(34,211,238,0.14)" : "transparent", color: section === "local" ? "#22d3ee" : "var(--text-1)", border: "none", borderLeft: "1px solid var(--border)", cursor: "pointer" }}>Local ({models.length})</button>
        </div>

        {section === "search" && (
          <div className="min-w-[320px] flex-1">
            <form className="flex items-center gap-2" onSubmit={(event) => { event.preventDefault(); void runSearch(query, 0, false); }}>
              <input type="text" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search Hugging Face GGUF models..." className="min-w-[240px] flex-1 rounded px-3 py-1.5 text-sm outline-none" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }} autoFocus />
              <button type="submit" disabled={loading} className="rounded px-4 py-1.5 text-xs font-medium transition" style={{ background: "#22d3ee", color: "#041014", border: "none", cursor: loading ? "not-allowed" : "pointer", opacity: loading ? 0.6 : 1 }}>{loading ? "Searching..." : "Refresh"}</button>
            </form>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span className="text-xs" style={{ color: "var(--text-2)" }}>{summary}</span>
              <button onClick={() => setSelectedTag(null)} className="rounded px-2.5 py-1 text-xs font-medium transition" style={{ background: selectedTag === null ? "rgba(34,211,238,0.12)" : "transparent", border: selectedTag === null ? "1px solid rgba(34,211,238,0.25)" : "1px solid transparent", color: selectedTag === null ? "#22d3ee" : "var(--text-1)", cursor: "pointer" }}>all</button>
              {tags.map((tag) => (
                <button key={tag} onClick={() => setSelectedTag(selectedTag === tag ? null : tag)} className="rounded px-2.5 py-1 text-xs font-medium transition" style={{ background: selectedTag === tag ? "rgba(34,211,238,0.12)" : "transparent", border: selectedTag === tag ? "1px solid rgba(34,211,238,0.25)" : "1px solid transparent", color: selectedTag === tag ? "#22d3ee" : "var(--text-1)", cursor: "pointer" }}>
                  {tag}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="relative ml-auto" ref={downloadsMenuRef}>
          <button onClick={() => setDownloadsOpen((open) => !open)} className="flex items-center gap-2 rounded px-3 py-1.5 text-xs font-medium transition" style={{ background: downloadsOpen ? "rgba(34,211,238,0.12)" : "var(--surface-2)", border: downloadsOpen ? "1px solid rgba(34,211,238,0.24)" : "1px solid var(--border)", color: downloadsOpen ? "#22d3ee" : "var(--text-1)", cursor: "pointer" }}>
            <span>Downloads</span>
            {(activeDownloadCount > 0 || completedDownloadCount > 0) && <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold" style={{ background: activeDownloadCount > 0 ? "rgba(34,211,238,0.18)" : "rgba(255,255,255,0.08)", color: activeDownloadCount > 0 ? "#22d3ee" : "var(--text-1)" }}>{activeDownloadCount > 0 ? activeDownloadCount : completedDownloadCount}</span>}
          </button>

          {downloadsOpen && (
            <div className="absolute right-0 top-full z-20 mt-2 w-[380px] overflow-hidden rounded" style={{ background: "var(--surface-1)", border: "1px solid var(--border)", boxShadow: "0 14px 40px rgba(0,0,0,0.35)" }}>
              <div className="flex items-center justify-between border-b px-3 py-2" style={{ borderColor: "var(--border)" }}>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>Download Manager</div>
                  <div className="mt-1 text-xs" style={{ color: "var(--text-1)" }}>{activeDownloadCount > 0 ? `${activeDownloadCount} active` : completedDownloadCount > 0 ? `${completedDownloadCount} recent` : "No downloads yet"}</div>
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
                          <div className="mt-1 text-[11px]" style={{ color: "var(--text-2)" }}>{formatBytes(entry.downloaded_bytes)} / {formatBytes(entry.total_bytes)} ({Math.round(entry.percent * 100)}%)</div>
                        </div>
                      )}
                      {entry.error && <div className="mt-2 text-[11px]" style={{ color: "#f87171" }}>{entry.error}</div>}
                      <div className="mt-3 flex items-center gap-2">
                        {!entry.done ? (
                          <button onClick={() => void handleCancelDownload(entry.id)} disabled={entry.status === "Cancelling"} className="rounded px-2.5 py-1 text-[11px] font-medium transition" style={{ background: "rgba(248,113,113,0.12)", border: "1px solid rgba(248,113,113,0.24)", color: "#f87171", cursor: entry.status === "Cancelling" ? "not-allowed" : "pointer", opacity: entry.status === "Cancelling" ? 0.7 : 1 }}>{entry.status === "Cancelling" ? "Cancelling..." : "Cancel"}</button>
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

      <div className="flex-1 overflow-y-auto p-4">
        {section === "search" ? (
          <div>
            {error && <div className="mb-4 rounded px-4 py-3 text-sm" style={{ background: "rgba(248,113,113,0.10)", border: "1px solid rgba(248,113,113,0.25)", color: "#f87171" }}>{error}</div>}
            {loading ? (
              <div className="flex items-center justify-center py-16 text-sm" style={{ color: "var(--text-2)" }}>Searching Hugging Face...</div>
            ) : visibleResults.length === 0 && !error ? (
              <div className="flex flex-col items-center justify-center gap-3 py-20 text-sm" style={{ color: "var(--text-2)" }}>
                <span className="text-3xl" style={{ opacity: 0.4 }}>&#x1F50D;</span>
                <span>{selectedTag ? `No results matched the "${selectedTag}" filter.` : "Search Hugging Face for any GGUF model."}</span>
                <span className="text-xs" style={{ color: "var(--text-2)" }}>Results are sorted by downloads. Try "llama", "qwen", "mistral", "phi", or "gemma".</span>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {visibleResults.map((model) => {
                  const expanded = expandedId === model.id;
                  const installed = anyInstalled(model);
                  return (
                    <div key={model.id} className="rounded overflow-hidden" style={{ ...panelStyle, border: `1px solid ${installed ? "rgba(34,211,238,0.25)" : "var(--border)"}` }}>
                      <button onClick={() => setExpandedId(expanded ? null : model.id)} className="flex w-full flex-wrap items-start gap-3 px-4 py-3 text-left transition" style={{ background: "none", border: "none", cursor: "pointer" }}>
                        <div className="min-w-0 flex-1">
                          <div className="mb-1 flex flex-wrap items-center gap-2">
                            <span className="max-w-xs truncate text-sm font-semibold" style={{ color: "var(--text-0)" }}>{model.name}</span>
                            {installed && <span className="rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider" style={{ background: "rgba(34,211,238,0.12)", color: "#22d3ee", border: "1px solid rgba(34,211,238,0.25)" }}>Installed</span>}
                            {model.tags.map((tag) => <TagBadge key={tag} tag={tag} />)}
                          </div>
                          <p className="text-xs leading-5 font-mono" style={{ color: "var(--text-2)" }}>{model.description}</p>
                        </div>
                        <span className="shrink-0 text-xs" style={{ color: "var(--text-2)", marginTop: "2px" }}>{expanded ? "v" : ">"}</span>
                      </button>
                      {expanded && (
                        <div style={{ borderTop: "1px solid var(--border)" }}>
                          {model.quants.map((quant) => {
                            const progress = downloads[quant.url];
                            const downloading = progress && !progress.done;
                            const downloaded = progress?.done && progress.status === "Completed" && !progress.error;
                            return (
                              <div key={quant.filename} className="flex flex-wrap items-center gap-3 px-4 py-2.5" style={{ borderTop: "1px solid var(--border)" }}>
                                <span className="text-xs font-semibold" style={{ color: "#fbbf24" }}>{quant.quant}</span>
                                {quant.size_gb > 0 && <span className="text-xs" style={{ color: "var(--text-1)" }}>{quant.size_gb.toFixed(2)} GB</span>}
                                <span className="min-w-0 flex-1 truncate font-mono text-xs" style={{ color: "var(--text-2)" }}>{quant.filename}</span>
                                {downloading ? (
                                  <div className="flex min-w-[240px] items-center gap-2">
                                    <div className="h-1.5 flex-1 overflow-hidden rounded" style={{ background: "rgba(255,255,255,0.08)" }}>
                                      <div className="h-full rounded transition-all" style={{ width: `${Math.round(progress.percent * 100)}%`, background: "#22d3ee" }} />
                                    </div>
                                    <span className="whitespace-nowrap text-xs" style={{ color: "var(--text-1)" }}>{formatBytes(progress.downloaded_bytes)} / {formatBytes(progress.total_bytes)} ({Math.round(progress.percent * 100)}%)</span>
                                    <button onClick={() => void handleCancelDownload(progress.id)} disabled={progress.status === "Cancelling"} className="rounded px-2 py-1 text-[11px] font-medium transition" style={{ background: "rgba(248,113,113,0.12)", border: "1px solid rgba(248,113,113,0.24)", color: "#f87171", cursor: progress.status === "Cancelling" ? "not-allowed" : "pointer", opacity: progress.status === "Cancelling" ? 0.7 : 1 }}>{progress.status === "Cancelling" ? "Cancelling..." : "Cancel"}</button>
                                  </div>
                                ) : downloaded ? (
                                  <span className="text-xs" style={{ color: "#34d399" }}>Downloaded</span>
                                ) : progress?.error ? (
                                  <span className="text-xs" style={{ color: "#f87171" }}>Error: {progress.error}</span>
                                ) : isInstalled(quant) ? (
                                  <span className="text-xs font-medium" style={{ color: "#22d3ee" }}>Installed</span>
                                ) : (
                                  <button onClick={() => void handleDownload(model, quant)} className="rounded px-3 py-1 text-xs font-medium transition" style={{ background: "#22d3ee", color: "#041014", border: "none", cursor: "pointer" }}>Download</button>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {visibleResults.length > 0 && (
              <div className="mt-4 flex flex-col items-center gap-2">
                {loadingMore && <span className="text-xs" style={{ color: "var(--text-2)" }}>Loading more...</span>}
                <div ref={sentinelRef} className="h-4 w-full" />
                {!hasMore && <span className="pb-2 text-xs" style={{ color: "var(--text-2)" }}>All {visibleResults.length} visible results shown</span>}
              </div>
            )}
          </div>
        ) : (
          <div>
            <p className="mb-3 text-xs" style={{ color: "var(--text-2)" }}>These are your locally scanned .gguf files. Delete removes the file from disk.</p>
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
                        <button onClick={() => void handleDelete(model)} className="rounded px-2 py-1 text-xs font-medium" style={{ background: "rgba(248,113,113,0.15)", color: "#f87171", border: "1px solid rgba(248,113,113,0.25)", cursor: "pointer" }}>Confirm</button>
                        <button onClick={() => setDeleteConfirm(null)} className="rounded px-2 py-1 text-xs" style={{ background: "var(--surface-2)", color: "var(--text-1)", border: "1px solid var(--border)", cursor: "pointer" }}>Cancel</button>
                      </div>
                    ) : (
                      <button onClick={() => setDeleteConfirm(model.path)} className="shrink-0 rounded px-2 py-1 text-xs transition" style={{ background: "transparent", color: "var(--text-2)", border: "1px solid transparent", cursor: "pointer" }}>Delete</button>
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
