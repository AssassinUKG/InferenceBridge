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

export function ModelBrowser({ models, onRefresh }: Props) {
  const [hubModels, setHubModels] = useState<HubModel[]>([]);
  const [loading, setLoading] = useState(true);
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
  const PAGE_SIZE = 20;
  // Sentinel element at the bottom of search results; observed to trigger auto-load
  const sentinelRef = useRef<HTMLDivElement>(null);

  // Load hub catalog
  useEffect(() => {
    api.listHubModels()
      .then(setHubModels)
      .catch(() => setHubModels([]))
      .finally(() => setLoading(false));
  }, []);

  // Listen for download progress events
  useEffect(() => {
    const unlisten = listen<DownloadProgress>("model-download-progress", (event) => {
      const prog = event.payload;
      setDownloads((prev) => ({
        ...prev,
        [prog.filename]: prog,
      }));
      if (prog.done && !prog.error) {
        onRefresh();
        // Clear after 3s
        setTimeout(() => {
          setDownloads((prev) => {
            const next = { ...prev };
            delete next[prog.filename];
            return next;
          });
        }, 3000);
      }
    });
    return () => { unlisten.then((fn) => fn()); };
  }, [onRefresh]);

  // Installed model filenames (lowercase for comparison)
  const installedFilenames = new Set(models.map((m) => m.filename.toLowerCase()));

  function isInstalled(quant: HubQuant) {
    return installedFilenames.has(quant.filename.toLowerCase());
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

  async function handleDownload(quant: HubQuant) {
    try {
      await api.downloadHubModel(quant.url, quant.filename);
    } catch (e) {
      setDownloads((prev) => ({
        ...prev,
        [quant.filename]: {
          filename: quant.filename,
          downloaded_bytes: 0,
          total_bytes: 0,
          percent: 0,
          done: true,
          error: String(e),
        },
      }));
    }
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
            Browse Models
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
              placeholder="Filter featured models…"
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
                            const prog = downloads[quant.filename];
                            const alreadyInstalled = isInstalled(quant);
                            const isDownloading = prog && !prog.done;
                            const justDone = prog?.done && !prog?.error;
                            return (
                              <div key={quant.filename} className="flex flex-wrap items-center gap-3 px-4 py-2.5" style={{ borderTop: "1px solid var(--border)" }}>
                                <span className="text-xs font-semibold" style={{ color: "#fbbf24" }}>{quant.quant}</span>
                                {quant.size_gb > 0 && <span className="text-xs" style={{ color: "var(--text-1)" }}>{quant.size_gb.toFixed(2)} GB</span>}
                                <span className="text-xs truncate flex-1 font-mono" style={{ color: "var(--text-2)" }}>{quant.filename}</span>
                                {isDownloading ? (
                                  <div className="flex items-center gap-2 min-w-[200px]">
                                    <div className="flex-1 h-1.5 rounded overflow-hidden" style={{ background: "rgba(255,255,255,0.08)" }}>
                                      <div className="h-full rounded transition-all" style={{ width: `${Math.round(prog.percent * 100)}%`, background: "#22d3ee" }} />
                                    </div>
                                    <span className="text-xs whitespace-nowrap" style={{ color: "var(--text-1)" }}>
                                      {formatBytes(prog.downloaded_bytes)} / {formatBytes(prog.total_bytes)} ({Math.round(prog.percent * 100)}%)
                                    </span>
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
                Loading catalog…
              </div>
            ) : filteredModels.length === 0 ? (
              <div className="flex items-center justify-center py-16 text-sm" style={{ color: "var(--text-2)" }}>
                No models match your search.
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
                                  const prog = downloads[quant.filename];
                                  const alreadyInstalled = isInstalled(quant);
                                  const isDownloading = prog && !prog.done;
                                  const justDone = prog?.done && !prog?.error;

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
                                        <div className="flex items-center gap-2 min-w-[200px]">
                                          <div className="flex-1 h-1.5 rounded overflow-hidden" style={{ background: "rgba(255,255,255,0.08)" }}>
                                            <div
                                              className="h-full rounded transition-all"
                                              style={{ width: `${Math.round(prog.percent * 100)}%`, background: "#22d3ee" }}
                                            />
                                          </div>
                                          <span className="text-xs whitespace-nowrap" style={{ color: "var(--text-1)" }}>
                                            {formatBytes(prog.downloaded_bytes)} / {formatBytes(prog.total_bytes)} ({Math.round(prog.percent * 100)}%)
                                          </span>
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
