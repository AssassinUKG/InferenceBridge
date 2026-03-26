import { useEffect, useRef, useState, useCallback } from "react";
import type { ModelInfo } from "../../lib/types";
import type { HubModel, HubQuant, DownloadProgress } from "../../lib/tauri";
import * as api from "../../lib/tauri";
import { listen } from "@tauri-apps/api/event";

interface Props {
  models: ModelInfo[];  // locally scanned models
  onRefresh: () => void;
}

const TAG_COLORS: Record<string, { bg: string; color: string; border: string }> = {
  reasoning: { bg: "rgba(251,191,36,0.1)", color: "#fde68a", border: "rgba(251,191,36,0.2)" },
  tools: { bg: "rgba(52,211,153,0.1)", color: "#6ee7b7", border: "rgba(52,211,153,0.2)" },
  thinking: { bg: "rgba(167,139,250,0.1)", color: "#c4b5fd", border: "rgba(167,139,250,0.2)" },
  chat: { bg: "rgba(34,211,238,0.08)", color: "#67e8f9", border: "rgba(34,211,238,0.18)" },
  math: { bg: "rgba(249,115,22,0.1)", color: "#fdba74", border: "rgba(249,115,22,0.2)" },
  moe: { bg: "rgba(99,102,241,0.1)", color: "#a5b4fc", border: "rgba(99,102,241,0.2)" },
  vision: { bg: "rgba(236,72,153,0.1)", color: "#f9a8d4", border: "rgba(236,72,153,0.2)" },
};

function TagBadge({ tag }: { tag: string }) {
  const style = TAG_COLORS[tag] ?? { bg: "var(--surface-2)", color: "var(--text-1)", border: "var(--border)" };
  return (
    <span
      className="rounded px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider"
      style={{ background: style.bg, color: style.color, border: `1px solid ${style.border}` }}
    >
      {tag}
    </span>
  );
}

function formatBytes(bytes: number) {
  if (bytes === 0) return "0 B";
  const gb = bytes / (1024 * 1024 * 1024);
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(0)} MB`;
}

function basename(path: string) {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return (parts[parts.length - 1] ?? path).toLowerCase();
}

function downloadTone(status: string, error?: string | null) {
  if (error || status === "Failed") {
    return "#f87171";
  }
  if (status === "Completed") {
    return "#34d399";
  }
  if (status === "Cancelled") {
    return "#fbbf24";
  }
  return "#22d3ee";
}

export function ModelBrowser({ models, onRefresh }: Props) {
  const [hubModels, setHubModels] = useState<HubModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [downloads, setDownloads] = useState<Record<string, DownloadProgress>>({});
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<"browse" | "search" | "local">("browse");
  const [hfQuery, setHfQuery] = useState("");
  const [hfResults, setHfResults] = useState<HubModel[]>([]);
  const [hfLoading, setHfLoading] = useState(false);
  const [hfLoadingMore, setHfLoadingMore] = useState(false);
  const [hfError, setHfError] = useState<string | null>(null);
  const [hfOffset, setHfOffset] = useState(0);
  const [hfHasMore, setHfHasMore] = useState(false);
  const [downloadsOpen, setDownloadsOpen] = useState(false);
  const PAGE_SIZE = 20;
  // Sentinel element at the bottom of search results; observed to trigger auto-load
  const sentinelRef = useRef<HTMLDivElement>(null);
  const downloadsMenuRef = useRef<HTMLDivElement>(null);

  // Load hub catalog
  useEffect(() => {
    api.listHubModels()
      .then((models) => {
        setHubModels(models);
        setCatalogError(null);
      })
      .catch((error) => {
        setHubModels([]);
        setCatalogError(String(error));
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    api.listDownloads()
      .then((items) => {
        setDownloads(Object.fromEntries(items.map((item) => [item.id, item])));
      })
      .catch(() => {
        // ignore startup download manager errors
      });
  }, []);

  // Listen for download progress events
  useEffect(() => {
    const unlisten = listen<DownloadProgress>("model-download-progress", (event) => {
      const prog = event.payload;
      setDownloads((prev) => ({
        ...prev,
        [prog.id]: prog,
      }));
      if (prog.done && !prog.error && prog.status === "Completed") {
        onRefresh();
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, [onRefresh]);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (!downloadsMenuRef.current?.contains(event.target as Node)) {
        setDownloadsOpen(false);
      }
    }

    if (downloadsOpen) {
      window.addEventListener("mousedown", handlePointerDown);
      return () => window.removeEventListener("mousedown", handlePointerDown);
    }

    return undefined;
  }, [downloadsOpen]);

  // Installed model filenames (lowercase for comparison)
  const installedFilenames = new Set(models.map((m) => basename(m.filename)));

  function isInstalled(quant: HubQuant) {
    return installedFilenames.has(basename(quant.filename));
  }

  function anyInstalled(model: HubModel) {
    return model.quants.some(isInstalled);
  }

  // All tags for filter
  const allTags = Array.from(new Set(hubModels.flatMap((m) => m.tags))).sort();

  const filteredModels = hubModels.filter((m) => {
    const q = searchQuery.toLowerCase();
    const matchesSearch = !q || m.name.toLowerCase().includes(q) || m.family.toLowerCase().includes(q) || m.description.toLowerCase().includes(q);
    const matchesTag = !selectedTag || m.tags.includes(selectedTag);
    return matchesSearch && matchesTag;
  });

  // Group by family
  const families = Array.from(new Set(filteredModels.map((m) => m.family)));
  const downloadEntries = Object.values(downloads).sort((left, right) => {
    if (left.done !== right.done) {
      return left.done ? 1 : -1;
    }
    return left.filename.localeCompare(right.filename);
  });
  const activeDownloadCount = downloadEntries.filter((entry) => !entry.done).length;
  const completedDownloadCount = downloadEntries.filter((entry) => entry.done).length;

  async function handleDownload(quant: HubQuant) {
    const id = quant.url;
    setDownloads((prev) => ({
      ...prev,
      [id]: prev[id] ?? {
        id,
        filename: quant.filename,
        dest_path: null,
        downloaded_bytes: 0,
        total_bytes: 0,
        percent: 0,
        done: false,
        status: "Starting",
        error: null,
      },
    }));
    void api.downloadHubModel(quant.url, quant.filename).catch((error) => {
      const message = String(error);
      if (message.toLowerCase().includes("cancelled")) {
        return;
      }
      setDownloads((prev) => ({
        ...prev,
        [id]: {
          id,
          filename: quant.filename,
          dest_path: prev[id]?.dest_path ?? null,
          downloaded_bytes: prev[id]?.downloaded_bytes ?? 0,
          total_bytes: prev[id]?.total_bytes ?? 0,
          percent: prev[id]?.percent ?? 0,
          done: true,
          status: "Failed",
          error: message,
        },
      }));
    });
  }

  async function handleCancelDownload(id: string) {
    try {
      await api.cancelDownload(id);
    } catch (error) {
      const message = String(error);
      setDownloads((prev) => {
        const existing = prev[id];
        if (!existing) {
          return prev;
        }
        return {
          ...prev,
          [id]: {
            ...existing,
            done: true,
            status: "Failed",
            error: message,
          },
        };
      });
    }
  }

  async function handleClearCompletedDownloads() {
    await api.clearCompletedDownloads();
    setDownloads((prev) =>
      Object.fromEntries(
        Object.entries(prev).filter(([, entry]) => !entry.done)
      )
    );
  }

  async function handleHfSearch() {
    if (!hfQuery.trim()) return;
    setHfLoading(true);
    setHfError(null);
    setHfResults([]);
    setHfOffset(0);
    setHfHasMore(false);
    try {
      const results = await api.searchHubModels(hfQuery.trim(), 0);
      setHfResults(results);
      setHfOffset(results.length);
      setHfHasMore(results.length === PAGE_SIZE);
    } catch (e) {
      setHfError(String(e));
    } finally {
      setHfLoading(false);
    }
  }

  const handleLoadMore = useCallback(async () => {
    if (!hfQuery.trim() || hfLoadingMore || !hfHasMore) return;
    setHfLoadingMore(true);
    try {
      const more = await api.searchHubModels(hfQuery.trim(), hfOffset);
      setHfResults((prev) => [...prev, ...more]);
      setHfOffset((prev) => prev + more.length);
      setHfHasMore(more.length === PAGE_SIZE);
    } catch (e) {
      setHfError(String(e));
    } finally {
      setHfLoadingMore(false);
    }
  }, [hfQuery, hfOffset, hfLoadingMore, hfHasMore]);

  // Auto-load more when the sentinel scrolls into view
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || !hfHasMore || hfLoadingMore) return;
    const obs = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) handleLoadMore(); },
      { threshold: 0.1 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [hfHasMore, hfLoadingMore, handleLoadMore]);

  async function handleDelete(model: ModelInfo) {
    try {
      await api.deleteModelFile(model.path);
      onRefresh();
      setDeleteConfirm(null);
    } catch (e) {
      alert(`Failed to delete: ${e}`);
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar */}
      <div
        className="flex flex-wrap items-center gap-2 border-b px-4 py-2.5"
        style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}
      >
        {/* Section toggle */}
        <div className="flex rounded overflow-hidden" style={{ border: "1px solid var(--border)" }}>
          <button
            onClick={() => setActiveSection("browse")}
            className="px-3 py-1.5 text-xs font-medium transition"
            style={{
              background: activeSection === "browse" ? "rgba(34,211,238,0.14)" : "transparent",
              color: activeSection === "browse" ? "#22d3ee" : "var(--text-1)",
              border: "none",
              cursor: "pointer",
            }}
          >
            Popular HF
          </button>
          <button
            onClick={() => setActiveSection("search")}
            className="px-3 py-1.5 text-xs font-medium transition"
            style={{
              background: activeSection === "search" ? "rgba(34,211,238,0.14)" : "transparent",
              color: activeSection === "search" ? "#22d3ee" : "var(--text-1)",
              borderLeft: "1px solid var(--border)",
              border: "none",
              borderLeftStyle: "solid",
              borderLeftWidth: "1px",
              borderLeftColor: "var(--border)",
              cursor: "pointer",
            }}
          >
            Search HF
          </button>
          <button
            onClick={() => setActiveSection("local")}
            className="px-3 py-1.5 text-xs font-medium transition"
            style={{
              background: activeSection === "local" ? "rgba(34,211,238,0.14)" : "transparent",
              color: activeSection === "local" ? "#22d3ee" : "var(--text-1)",
              borderLeft: "1px solid var(--border)",
              border: "none",
              borderLeftStyle: "solid",
              borderLeftWidth: "1px",
              borderLeftColor: "var(--border)",
              cursor: "pointer",
            }}
          >
            Local ({models.length})
          </button>
        </div>

        {activeSection === "browse" && (
          <>
            {/* Curated search */}
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter popular Hugging Face models..."
              className="min-w-[200px] flex-1 rounded px-3 py-1.5 text-sm outline-none"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
            />
            {/* Tag filters */}
            {allTags.map((tag) => (
              <button
                key={tag}
                onClick={() => setSelectedTag(selectedTag === tag ? null : tag)}
                className="rounded px-2.5 py-1 text-xs font-medium transition"
                style={{
                  background: selectedTag === tag ? "rgba(34,211,238,0.12)" : "transparent",
                  border: selectedTag === tag ? "1px solid rgba(34,211,238,0.25)" : "1px solid transparent",
                  color: selectedTag === tag ? "#22d3ee" : "var(--text-1)",
                  cursor: "pointer",
                }}
              >
                {tag}
              </button>
            ))}
          </>
        )}

        {activeSection === "search" && (
          <form
            className="flex flex-1 items-center gap-2"
            onSubmit={(e) => { e.preventDefault(); handleHfSearch(); }}
          >
            <input
              type="text"
              value={hfQuery}
              onChange={(e) => setHfQuery(e.target.value)}
              placeholder="Search all GGUF models on HuggingFace…"
              className="min-w-[240px] flex-1 rounded px-3 py-1.5 text-sm outline-none"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              autoFocus
            />
            <button
              type="submit"
              disabled={hfLoading || !hfQuery.trim()}
              className="rounded px-4 py-1.5 text-xs font-medium transition"
              style={{
                background: "#22d3ee",
                color: "#041014",
                border: "none",
                cursor: hfLoading || !hfQuery.trim() ? "not-allowed" : "pointer",
                opacity: hfLoading || !hfQuery.trim() ? 0.6 : 1,
              }}
            >
              {hfLoading ? "Searching…" : "Search"}
            </button>
          </form>
        )}

        <div className="relative ml-auto" ref={downloadsMenuRef}>
          <button
            onClick={() => setDownloadsOpen((open) => !open)}
            className="flex items-center gap-2 rounded px-3 py-1.5 text-xs font-medium transition"
            style={{
              background: downloadsOpen ? "rgba(34,211,238,0.12)" : "var(--surface-2)",
              border: downloadsOpen ? "1px solid rgba(34,211,238,0.24)" : "1px solid var(--border)",
              color: downloadsOpen ? "#22d3ee" : "var(--text-1)",
              cursor: "pointer",
            }}
          >
            <span>Downloads</span>
            {(activeDownloadCount > 0 || completedDownloadCount > 0) && (
              <span
                className="rounded px-1.5 py-0.5 text-[10px] font-semibold"
                style={{
                  background: activeDownloadCount > 0 ? "rgba(34,211,238,0.18)" : "rgba(255,255,255,0.08)",
                  color: activeDownloadCount > 0 ? "#22d3ee" : "var(--text-1)",
                }}
              >
                {activeDownloadCount > 0 ? activeDownloadCount : completedDownloadCount}
              </span>
            )}
          </button>

          {downloadsOpen && (
            <div
              className="absolute right-0 top-full z-20 mt-2 w-[380px] overflow-hidden rounded"
              style={{
                background: "var(--surface-1)",
                border: "1px solid var(--border)",
                boxShadow: "0 14px 40px rgba(0,0,0,0.35)",
              }}
            >
              <div
                className="flex items-center justify-between border-b px-3 py-2"
                style={{ borderColor: "var(--border)" }}
              >
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                    Download Manager
                  </div>
                  <div className="mt-1 text-xs" style={{ color: "var(--text-1)" }}>
                    {activeDownloadCount > 0
                      ? `${activeDownloadCount} active`
                      : completedDownloadCount > 0
                        ? `${completedDownloadCount} recent`
                        : "No downloads yet"}
                  </div>
                </div>
                {completedDownloadCount > 0 && (
                  <button
                    onClick={() => void handleClearCompletedDownloads()}
                    className="rounded px-2 py-1 text-[11px] font-medium transition"
                    style={{
                      background: "transparent",
                      border: "1px solid var(--border)",
                      color: "var(--text-1)",
                      cursor: "pointer",
                    }}
                  >
                    Clear Done
                  </button>
                )}
              </div>

              <div className="max-h-[420px] overflow-y-auto">
                {downloadEntries.length === 0 ? (
                  <div className="px-3 py-6 text-sm" style={{ color: "var(--text-2)" }}>
                    Start a model download and it will show up here with progress and cancel controls.
                  </div>
                ) : (
                  downloadEntries.map((entry) => {
                    const tone = downloadTone(entry.status, entry.error);
                    const isActive = !entry.done;
                    const shortName = entry.filename.split(/[\\/]/).filter(Boolean).pop() ?? entry.filename;
                    return (
                      <div
                        key={entry.id}
                        className="border-b px-3 py-3 last:border-b-0"
                        style={{ borderColor: "var(--border)" }}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-sm font-medium" style={{ color: "var(--text-0)" }}>
                              {shortName}
                            </div>
                            {entry.filename !== shortName && (
                              <div className="mt-0.5 truncate font-mono text-[11px]" style={{ color: "var(--text-2)" }}>
                                {entry.filename}
                              </div>
                            )}
                          </div>
                          <span className="shrink-0 text-[11px] font-medium" style={{ color: tone }}>
                            {entry.status}
                          </span>
                        </div>

                        {(isActive || entry.downloaded_bytes > 0 || entry.total_bytes > 0) && (
                          <div className="mt-2">
                            <div className="h-1.5 overflow-hidden rounded" style={{ background: "rgba(255,255,255,0.08)" }}>
                              <div
                                className="h-full rounded transition-all"
                                style={{
                                  width: `${Math.max(4, Math.round(entry.percent * 100))}%`,
                                  background: tone,
                                }}
                              />
                            </div>
                            <div className="mt-1 text-[11px]" style={{ color: "var(--text-2)" }}>
                              {formatBytes(entry.downloaded_bytes)} / {formatBytes(entry.total_bytes)} ({Math.round(entry.percent * 100)}%)
                            </div>
                          </div>
                        )}

                        {entry.error && (
                          <div className="mt-2 text-[11px]" style={{ color: "#f87171" }}>
                            {entry.error}
                          </div>
                        )}

                        <div className="mt-3 flex items-center gap-2">
                          {!entry.done ? (
                            <button
                              onClick={() => void handleCancelDownload(entry.id)}
                              disabled={entry.status === "Cancelling"}
                              className="rounded px-2.5 py-1 text-[11px] font-medium transition"
                              style={{
                                background: "rgba(248,113,113,0.12)",
                                border: "1px solid rgba(248,113,113,0.24)",
                                color: "#f87171",
                                cursor: entry.status === "Cancelling" ? "not-allowed" : "pointer",
                                opacity: entry.status === "Cancelling" ? 0.7 : 1,
                              }}
                            >
                              {entry.status === "Cancelling" ? "Cancelling..." : "Cancel"}
                            </button>
                          ) : entry.dest_path ? (
                            <button
                              onClick={() => void api.showInFolder(entry.dest_path!)}
                              className="rounded px-2.5 py-1 text-[11px] font-medium transition"
                              style={{
                                background: "transparent",
                                border: "1px solid var(--border)",
                                color: "var(--text-1)",
                                cursor: "pointer",
                              }}
                            >
                              Open Folder
                            </button>
                          ) : null}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {activeSection === "search" ? (
          /* ── HuggingFace live search ──────────────────────────────────── */
          <div>
            {hfError && (
              <div className="mb-4 rounded px-4 py-3 text-sm" style={{ background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.25)", color: "#f87171" }}>
                {hfError}
              </div>
            )}
            {hfLoading ? (
              <div className="flex items-center justify-center py-16 text-sm" style={{ color: "var(--text-2)" }}>
                Searching HuggingFace…
              </div>
            ) : hfResults.length === 0 && !hfError ? (
              <div className="flex flex-col items-center justify-center gap-3 py-20 text-sm" style={{ color: "var(--text-2)" }}>
                <span className="text-3xl" style={{ opacity: 0.4 }}>&#x1F50D;</span>
                <span>Search HuggingFace for any GGUF model.</span>
                <span className="text-xs" style={{ color: "var(--text-2)" }}>Results are sorted by downloads. Try "llama", "qwen", "mistral", "phi", "gemma"…</span>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {hfResults.map((model) => {
                  const expanded = expandedId === model.id;
                  const installed = anyInstalled(model);
                  return (
                    <div
                      key={model.id}
                      className="rounded overflow-hidden"
                      style={{ background: "var(--surface-1)", border: `1px solid ${installed ? "rgba(34,211,238,0.25)" : "var(--border)"}` }}
                    >
                      <button
                        onClick={() => setExpandedId(expanded ? null : model.id)}
                        className="w-full flex flex-wrap items-start gap-3 px-4 py-3 text-left transition"
                        style={{ background: "none", border: "none", cursor: "pointer" }}
                      >
                        <div className="flex-1 min-w-0">
                          <div className="flex flex-wrap items-center gap-2 mb-1">
                            <span className="text-sm font-semibold truncate max-w-xs" style={{ color: "var(--text-0)" }}>{model.name}</span>
                            {installed && (
                              <span className="rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider" style={{ background: "rgba(34,211,238,0.12)", color: "#22d3ee", border: "1px solid rgba(34,211,238,0.25)" }}>Installed</span>
                            )}
                            {model.tags.map((tag) => <TagBadge key={tag} tag={tag} />)}
                          </div>
                          <p className="text-xs leading-5 font-mono" style={{ color: "var(--text-2)" }}>{model.description}</p>
                        </div>
                        <span className="shrink-0 text-xs" style={{ color: "var(--text-2)", marginTop: "2px" }}>{expanded ? "▲" : "▼"}</span>
                      </button>
                      {expanded && (
                        <div style={{ borderTop: "1px solid var(--border)" }}>
                          {model.quants.map((quant) => {
                            const prog = downloads[quant.url];
                            const alreadyInstalled = isInstalled(quant);
                            const isDownloading = prog && !prog.done;
                            const justDone = prog?.done && prog.status === "Completed" && !prog?.error;
                            return (
                              <div key={quant.filename} className="flex flex-wrap items-center gap-3 px-4 py-2.5" style={{ borderTop: "1px solid var(--border)" }}>
                                <span className="text-xs font-semibold" style={{ color: "#fbbf24" }}>{quant.quant}</span>
                                {quant.size_gb > 0 && <span className="text-xs" style={{ color: "var(--text-1)" }}>{quant.size_gb.toFixed(2)} GB</span>}
                                <span className="text-xs truncate flex-1 font-mono" style={{ color: "var(--text-2)" }}>{quant.filename}</span>
                                {isDownloading ? (
                                  <div className="flex items-center gap-2 min-w-[240px]">
                                    <div className="flex-1 h-1.5 rounded overflow-hidden" style={{ background: "rgba(255,255,255,0.08)" }}>
                                      <div className="h-full rounded transition-all" style={{ width: `${Math.round(prog.percent * 100)}%`, background: "#22d3ee" }} />
                                    </div>
                                    <span className="text-xs whitespace-nowrap" style={{ color: "var(--text-1)" }}>
                                      {formatBytes(prog.downloaded_bytes)} / {formatBytes(prog.total_bytes)} ({Math.round(prog.percent * 100)}%)
                                    </span>
                                    <button
                                      onClick={() => void handleCancelDownload(prog.id)}
                                      disabled={prog.status === "Cancelling"}
                                      className="rounded px-2 py-1 text-[11px] font-medium transition"
                                      style={{
                                        background: "rgba(248,113,113,0.12)",
                                        border: "1px solid rgba(248,113,113,0.24)",
                                        color: "#f87171",
                                        cursor: prog.status === "Cancelling" ? "not-allowed" : "pointer",
                                        opacity: prog.status === "Cancelling" ? 0.7 : 1,
                                      }}
                                    >
                                      {prog.status === "Cancelling" ? "Cancelling..." : "Cancel"}
                                    </button>
                                  </div>
                                ) : justDone ? (
                                  <span className="text-xs" style={{ color: "#34d399" }}>Downloaded</span>
                                ) : prog?.error ? (
                                  <span className="text-xs" style={{ color: "#f87171" }}>Error: {prog.error}</span>
                                ) : alreadyInstalled ? (
                                  <span className="text-xs font-medium" style={{ color: "#22d3ee" }}>Installed</span>
                                ) : (
                                  <button
                                    onClick={() => handleDownload(quant)}
                                    className="rounded px-3 py-1 text-xs font-medium transition"
                                    style={{ background: "#22d3ee", color: "#041014", border: "none", cursor: "pointer" }}
                                    onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.filter = "brightness(1.1)")}
                                    onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.filter = "")}
                                  >
                                    Download
                                  </button>
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

            {/* Infinite scroll sentinel + status */}
            {hfResults.length > 0 && (
              <div className="mt-4 flex flex-col items-center gap-2">
                {hfLoadingMore && (
                  <span className="text-xs" style={{ color: "var(--text-2)" }}>
                    Loading more…
                  </span>
                )}
                {/* Sentinel div — IntersectionObserver fires handleLoadMore when visible */}
                <div ref={sentinelRef} className="h-4 w-full" />
                {!hfHasMore && (
                  <span className="text-xs pb-2" style={{ color: "var(--text-2)" }}>
                    All {hfResults.length} results shown
                  </span>
                )}
              </div>
            )}
          </div>
        ) : activeSection === "local" ? (
          /* ── Local models list ────────────────────────────────────────── */
          <div>
            <p className="mb-3 text-xs" style={{ color: "var(--text-2)" }}>
              These are your locally scanned .gguf files. Delete removes the file from disk.
            </p>
            {models.length === 0 ? (
              <div className="flex items-center justify-center py-16 text-sm" style={{ color: "var(--text-2)" }}>
                No local models found. Download some from Browse, or add a directory in Settings.
              </div>
            ) : (
              <div className="flex flex-col gap-1">
                {models.map((model) => (
                  <div
                    key={model.path}
                    className="flex items-center gap-3 rounded px-3 py-2.5"
                    style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate" style={{ color: "var(--text-0)" }}>{model.filename}</div>
                      <div className="text-xs mt-0.5 truncate" style={{ color: "var(--text-2)" }}>{model.path}</div>
                    </div>
                    <div className="flex items-center gap-2 text-xs shrink-0" style={{ color: "var(--text-1)" }}>
                      <span>{model.size_gb.toFixed(2)} GB</span>
                      <span style={{ color: "#fbbf24" }}>{model.quant ?? ""}</span>
                      <span style={{ color: "var(--text-2)" }}>{model.family}</span>
                    </div>
                    {deleteConfirm === model.path ? (
                      <div className="flex items-center gap-1 shrink-0">
                        <span className="text-xs" style={{ color: "#f87171" }}>Delete file?</span>
                        <button
                          onClick={() => handleDelete(model)}
                          className="rounded px-2 py-1 text-xs font-medium"
                          style={{ background: "rgba(248,113,113,0.15)", color: "#f87171", border: "1px solid rgba(248,113,113,0.25)", cursor: "pointer" }}
                        >
                          Confirm
                        </button>
                        <button
                          onClick={() => setDeleteConfirm(null)}
                          className="rounded px-2 py-1 text-xs"
                          style={{ background: "var(--surface-2)", color: "var(--text-1)", border: "1px solid var(--border)", cursor: "pointer" }}
                        >
                          Cancel
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setDeleteConfirm(model.path)}
                        className="rounded px-2 py-1 text-xs transition shrink-0"
                        style={{ background: "transparent", color: "var(--text-2)", border: "1px solid transparent", cursor: "pointer" }}
                        onMouseEnter={(e) => {
                          (e.currentTarget as HTMLButtonElement).style.color = "#f87171";
                          (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(248,113,113,0.25)";
                        }}
                        onMouseLeave={(e) => {
                          (e.currentTarget as HTMLButtonElement).style.color = "var(--text-2)";
                          (e.currentTarget as HTMLButtonElement).style.borderColor = "transparent";
                        }}
                      >
                        Delete
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          /* ── Browse / download hub ────────────────────────────────────── */
          <>
            {loading ? (
              <div className="flex items-center justify-center py-16 text-sm" style={{ color: "var(--text-2)" }}>
                Loading Hugging Face models...
              </div>
            ) : catalogError ? (
              <div className="rounded px-4 py-3 text-sm" style={{ background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.25)", color: "#f87171" }}>
                {catalogError}
              </div>
            ) : filteredModels.length === 0 ? (
              <div className="flex items-center justify-center py-16 text-sm" style={{ color: "var(--text-2)" }}>
                No downloadable Hugging Face GGUF models matched your filter.
              </div>
            ) : (
              families.map((family) => (
                <div key={family} className="mb-6">
                  <div
                    className="mb-2 text-[10px] uppercase tracking-widest font-semibold"
                    style={{ color: "var(--text-2)" }}
                  >
                    {family}
                  </div>
                  <div className="flex flex-col gap-2">
                    {filteredModels
                      .filter((m) => m.family === family)
                      .map((model) => {
                        const expanded = expandedId === model.id;
                        const installed = anyInstalled(model);
                        return (
                          <div
                            key={model.id}
                            className="rounded overflow-hidden"
                            style={{ background: "var(--surface-1)", border: `1px solid ${installed ? "rgba(34,211,238,0.25)" : "var(--border)"}` }}
                          >
                            {/* Header row */}
                            <button
                              onClick={() => setExpandedId(expanded ? null : model.id)}
                              className="w-full flex flex-wrap items-start gap-3 px-4 py-3 text-left transition"
                              style={{ background: "none", border: "none", cursor: "pointer" }}
                            >
                              <div className="flex-1 min-w-0">
                                <div className="flex flex-wrap items-center gap-2 mb-1">
                                  <span className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>{model.name}</span>
                                  <span className="text-xs" style={{ color: "var(--text-2)" }}>{model.params}</span>
                                  {installed && (
                                    <span
                                      className="rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider"
                                      style={{ background: "rgba(34,211,238,0.12)", color: "#22d3ee", border: "1px solid rgba(34,211,238,0.25)" }}
                                    >
                                      Installed
                                    </span>
                                  )}
                                  {model.tags.map((tag) => <TagBadge key={tag} tag={tag} />)}
                                </div>
                                <p className="text-xs leading-5" style={{ color: "var(--text-1)" }}>{model.description}</p>
                              </div>
                              <span className="shrink-0 text-xs" style={{ color: "var(--text-2)", marginTop: "2px" }}>{expanded ? "▲" : "▼"}</span>
                            </button>

                            {/* Quants */}
                            {expanded && (
                              <div style={{ borderTop: "1px solid var(--border)" }}>
                                {model.quants.map((quant) => {
                                  const prog = downloads[quant.url];
                                  const alreadyInstalled = isInstalled(quant);
                                  const isDownloading = prog && !prog.done;
                                  const justDone = prog?.done && prog.status === "Completed" && !prog?.error;

                                  return (
                                    <div
                                      key={quant.quant}
                                      className="flex flex-wrap items-center gap-3 px-4 py-2.5"
                                      style={{ borderTop: "1px solid var(--border)" }}
                                    >
                                      <span className="text-xs font-semibold" style={{ color: "#fbbf24" }}>{quant.quant}</span>
                                      <span className="text-xs" style={{ color: "var(--text-1)" }}>{quant.size_gb.toFixed(2)} GB</span>
                                      <span className="text-xs truncate flex-1 font-mono" style={{ color: "var(--text-2)" }}>{quant.filename}</span>

                                      {isDownloading ? (
                                        <div className="flex items-center gap-2 min-w-[240px]">
                                          <div className="flex-1 h-1.5 rounded overflow-hidden" style={{ background: "rgba(255,255,255,0.08)" }}>
                                            <div
                                              className="h-full rounded transition-all"
                                              style={{ width: `${Math.round(prog.percent * 100)}%`, background: "#22d3ee" }}
                                            />
                                          </div>
                                          <span className="text-xs whitespace-nowrap" style={{ color: "var(--text-1)" }}>
                                            {formatBytes(prog.downloaded_bytes)} / {formatBytes(prog.total_bytes)} ({Math.round(prog.percent * 100)}%)
                                          </span>
                                          <button
                                            onClick={() => void handleCancelDownload(prog.id)}
                                            disabled={prog.status === "Cancelling"}
                                            className="rounded px-2 py-1 text-[11px] font-medium transition"
                                            style={{
                                              background: "rgba(248,113,113,0.12)",
                                              border: "1px solid rgba(248,113,113,0.24)",
                                              color: "#f87171",
                                              cursor: prog.status === "Cancelling" ? "not-allowed" : "pointer",
                                              opacity: prog.status === "Cancelling" ? 0.7 : 1,
                                            }}
                                          >
                                            {prog.status === "Cancelling" ? "Cancelling..." : "Cancel"}
                                          </button>
                                        </div>
                                      ) : justDone ? (
                                        <span className="text-xs" style={{ color: "#34d399" }}>Downloaded</span>
                                      ) : prog?.error ? (
                                        <span className="text-xs" style={{ color: "#f87171" }}>Error: {prog.error}</span>
                                      ) : alreadyInstalled ? (
                                        <span className="text-xs font-medium" style={{ color: "#22d3ee" }}>Installed</span>
                                      ) : (
                                        <button
                                          onClick={() => handleDownload(quant)}
                                          className="rounded px-3 py-1 text-xs font-medium transition"
                                          style={{ background: "#22d3ee", color: "#041014", border: "none", cursor: "pointer" }}
                                          onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.filter = "brightness(1.1)")}
                                          onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.filter = "")}
                                        >
                                          Download
                                        </button>
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
                </div>
              ))
            )}
          </>
        )}
      </div>
    </div>
  );
}
