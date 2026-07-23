import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { Bell, LoaderCircle, Play, Square, X } from "lucide-react";
import { useModel } from "./hooks/useModel";
import { useSession } from "./hooks/useSession";
import { useChat } from "./hooks/useChat";
import { useContext } from "./hooks/useContext";
import { useGpuStats } from "./hooks/useGpuStats";
import { StatusBar } from "./components/common/StatusBar";
import { AppSidebar, type AppNavId } from "./components/common/AppSidebar";
import { CommandPalette } from "./components/common/CommandPalette";
import { AppContextMenu } from "./components/common/AppContextMenu";
import { Button, IconButton } from "./components/ui/Controls";
import { ChatPanel } from "./components/Chat/ChatPanel";
import { ModelLoadDialog, ModelSelector, type LoadDialogMode } from "./components/Model/ModelSelector";
import { RichModelPicker } from "./components/Model/RichModelPicker";
import { ModelArtwork, modelDisplayName } from "./components/Model/modelPresentation";
import { SettingsPanel } from "./components/Model/SettingsPanel";
import { ContextPanel } from "./components/Context/ContextPanel";
import { DebugInspector } from "./components/Debug/DebugInspector";
import { ModelBrowser } from "./components/Model/ModelBrowser";
import { BenchmarkPanel } from "./components/Benchmark/BenchmarkPanel";
import type { AppSettings, LiveStreamDelta, LiveStreamSnapshot, LoadProgress, ModelInfo, RuntimePackInfo } from "./lib/types";
import * as api from "./lib/tauri";
import type { DownloadProgress } from "./lib/tauri";
import { nextAutomaticChatName } from "./lib/chatNames";
import { conversationMarkdown, downloadConversation, safeConversationFilename } from "./lib/conversationUi";
import {
  activeLiveStream,
  applyLiveStreamDelta,
  appendLiveStreamDelta,
  isLiveStreamRunningStatus,
  formatLiveStreamTranscript,
  liveStreamInputText,
  liveStreamLogLevel,
  liveStreamLogSource,
  liveStreamTerminalRows,
  reconcileLiveStreams,
  tailLiveStream,
  upsertLiveStream,
} from "./lib/liveStreams";

type Tab = AppNavId;

const TAB_LABELS: Record<Tab, string> = {
  chat: "Chat",
  models: "Models",
  browse: "Browse",
  benchmark: "Benchmark",
  context: "Context",
  logs: "Logs",
  debug: "API",
  settings: "Settings",
};

const TAB_DESCRIPTIONS: Record<Tab, string> = {
  chat: "Local conversation",
  models: "On-device models",
  browse: "Hugging Face model hub",
  benchmark: "Performance workspace",
  context: "Context and KV state",
  logs: "Inference event stream",
  debug: "OpenAI-compatible developer API",
  settings: "Runtime and app preferences",
};

interface ContextWorkspaceProps {
  contextStatus: ReturnType<typeof useContext>;
  processStatus: ReturnType<typeof useModel>["processStatus"];
  gpuStats: ReturnType<typeof useGpuStats>;
}

interface UiNotice {
  id: string;
  tone: "info" | "success" | "error";
  title: string;
  message: string;
  dedupeKey?: string;
  action?: "settings";
}

interface ModelLoadRequest {
  model: ModelInfo;
  mode: LoadDialogMode;
  returnFocus: HTMLElement | null;
}

type UiNoticeInput = Omit<UiNotice, "id">;

function runtimeBuildNumber(version: string | null | undefined) {
  const match = version?.trim().match(/^b(\d+)$/i);
  return match ? Number.parseInt(match[1], 10) : null;
}

function formatRuntimeDownloadSize(bytes: number | null | undefined) {
  if (!bytes) return null;
  return `${(bytes / (1024 * 1024)).toFixed(bytes >= 100 * 1024 * 1024 ? 0 : 1)} MB`;
}

function chooseRuntimePack(packs: RuntimePackInfo[], preferredBackend: string | null | undefined) {
  const preferred = preferredBackend === "cpu" ? "cpu" : "cuda";
  const preferredPack = packs.find((pack) => pack.backend === preferred);
  if (preferredPack) return preferredPack;

  return [...packs].sort((a, b) => {
    const aBuild = runtimeBuildNumber(a.latest_version) ?? -1;
    const bBuild = runtimeBuildNumber(b.latest_version) ?? -1;
    return bBuild - aBuild;
  })[0];
}

function NotificationCenter({
  onOpenSettings,
  preferredBackend,
}: {
  onOpenSettings: () => void;
  preferredBackend?: string | null;
}) {
  const [notices, setNotices] = useState<UiNotice[]>([]);
  const [open, setOpen] = useState(false);
  const [seenIds, setSeenIds] = useState<Set<string>>(new Set());
  const menuRef = useRef<HTMLDivElement | null>(null);

  const pushNotice = useCallback((notice: UiNoticeInput) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setNotices((current) => {
      if (notice.dedupeKey) {
        const existing = current.find((item) => item.dedupeKey === notice.dedupeKey);
        if (existing) {
          const unchanged =
            existing.tone === notice.tone &&
            existing.title === notice.title &&
            existing.message === notice.message &&
            existing.action === notice.action;
          if (unchanged) return current;
          return current.map((item) =>
            item.id === existing.id ? { ...notice, id } : item
          );
        }
      }
      return [...current, { ...notice, id }].slice(-20);
    });
  }, []);

  const announceRuntimePacks = useCallback((packs: RuntimePackInfo[]) => {
    const comparable = packs.filter((pack) => pack.installed_version && pack.latest_version);
    const updatePack = chooseRuntimePack(
      comparable.filter((pack) => pack.update_available),
      preferredBackend,
    );
    const aheadPack = chooseRuntimePack(
      comparable.filter((pack) => {
        const installedBuild = runtimeBuildNumber(pack.installed_version);
        const latestBuild = runtimeBuildNumber(pack.latest_version);
        return installedBuild != null && latestBuild != null && installedBuild > latestBuild;
      }),
      preferredBackend,
    );
    const pack = updatePack ?? aheadPack;
    const activeKey = updatePack ? "runtime:update" : aheadPack ? "runtime:ahead" : null;

    setNotices((current) => current.filter((notice) => {
      if (notice.dedupeKey !== "runtime:update" && notice.dedupeKey !== "runtime:ahead") return true;
      return notice.dedupeKey === activeKey;
    }));

    if (!pack || !activeKey) return;
    const size = formatRuntimeDownloadSize(pack.size_bytes);
    const backend = pack.backend.toUpperCase();
    if (updatePack) {
      pushNotice({
        tone: "info",
        title: "Runtime update available",
        message: `${backend} llama.cpp ${pack.installed_version} -> ${pack.latest_version}${size ? ` (${size})` : ""} is ready to install.`,
        dedupeKey: activeKey,
        action: "settings",
      });
    } else {
      pushNotice({
        tone: "success",
        title: "Runtime ahead of stable",
        message: `Installed ${backend} llama.cpp ${pack.installed_version} is newer than stable ${pack.latest_version}. InferenceBridge will not offer a downgrade.`,
        dedupeKey: activeKey,
        action: "settings",
      });
    }
  }, [preferredBackend, pushNotice]);

  useEffect(() => {
    const cleanups: Array<Promise<() => void>> = [];
    cleanups.push(
      listen<LoadProgress>("model-load-progress", (event) => {
        const progress = event.payload;
        if (progress.error) {
          pushNotice({
            tone: "error",
            title: "Operation failed",
            message: progress.error,
          });
        } else if (progress.done) {
          pushNotice({
            tone: "success",
            title: "Operation complete",
            message: progress.message,
          });
        }
      })
    );
    cleanups.push(
      listen<DownloadProgress>("model-download-progress", (event) => {
        const progress = event.payload;
        if (progress.error) {
          pushNotice({
            tone: "error",
            title: "Download failed",
            message: `${progress.filename}: ${progress.error}`,
          });
        } else if (progress.done) {
          pushNotice({
            tone: "success",
            title: "Download complete",
            message: progress.filename,
          });
        }
      })
    );
    cleanups.push(
      listen<RuntimePackInfo[]>("runtime-packs-refreshed", (event) => {
        announceRuntimePacks(event.payload);
      })
    );
    cleanups.push(
      listen<UiNoticeInput>("runtime-operation-notice", (event) => {
        if (event.payload.title === "Runtime update installed") {
          setNotices((current) =>
            current.filter((notice) => !notice.dedupeKey?.startsWith("runtime:update"))
          );
        }
        pushNotice(event.payload);
      })
    );
    return () => {
      cleanups.forEach((cleanup) => cleanup.then((fn) => fn()));
    };
  }, [announceRuntimePacks, pushNotice]);

  useEffect(() => {
    let cancelled = false;
    const checkRuntimeUpdates = async () => {
      try {
        const packs = await api.listRuntimePacks();
        if (!cancelled) announceRuntimePacks(packs);
      } catch {
        // Update discovery is non-critical and should not create noisy errors.
      }
    };
    const initialTimer = window.setTimeout(() => void checkRuntimeUpdates(), 1200);
    const interval = window.setInterval(() => void checkRuntimeUpdates(), 30 * 60 * 1000);
    return () => {
      cancelled = true;
      window.clearTimeout(initialTimer);
      window.clearInterval(interval);
    };
  }, [announceRuntimePacks]);

  useEffect(() => {
    if (!open) return undefined;
    setSeenIds(new Set(notices.map((notice) => notice.id)));
    function handlePointerDown(event: MouseEvent) {
      if (!menuRef.current?.contains(event.target as Node)) setOpen(false);
    }
    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, [notices, open]);

  const unreadCount = notices.filter((notice) => !seenIds.has(notice.id)).length;

  return (
    <div className="relative" ref={menuRef}>
      <IconButton
        label={unreadCount > 0 ? `${unreadCount} unread notification${unreadCount === 1 ? "" : "s"}` : "Notifications"}
        size="sm"
        selected={open}
        className={unreadCount > 0 ? "bg-white/10 text-white" : ""}
        onClick={() => setOpen((value) => !value)}
      >
        <Bell size={16} fill={unreadCount > 0 ? "currentColor" : "none"} />
        {unreadCount > 0 && (
          <span className="absolute -right-1 -top-1 rounded-full bg-white px-1 text-[9px] font-bold leading-4 text-black" style={{ minWidth: 16, height: 16 }}>
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </IconButton>
      {open && (
        <div className="absolute right-0 top-full z-50 mt-2 w-[380px] overflow-hidden rounded" style={{ background: "var(--surface-1)", border: "1px solid var(--border)", boxShadow: "0 18px 48px rgba(0,0,0,0.42)" }}>
          <div className="flex items-center justify-between border-b px-3 py-2" style={{ borderColor: "var(--border)" }}>
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>Notifications</div>
              <div className="mt-0.5 text-xs" style={{ color: "var(--text-1)" }}>{notices.length > 0 ? `${notices.length} recent events` : "No events yet"}</div>
            </div>
            {notices.length > 0 && (
              <button onClick={() => setNotices([])} className="rounded-lg px-2 py-1 text-[11px] font-medium hover:bg-white/5" style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-1)", cursor: "pointer" }}>
                Clear
              </button>
            )}
          </div>
          <div className="max-h-[420px] overflow-y-auto">
            {notices.length === 0 ? (
              <div className="px-3 py-8 text-sm" style={{ color: "var(--text-2)" }}>Downloads, API actions, and runtime events will collect here.</div>
            ) : (
              [...notices].reverse().map((notice) => {
                const color = notice.tone === "error" ? "#f87171" : notice.tone === "success" ? "#34d399" : "#8ab4f8";
                return (
                  <div key={notice.id} className="border-b px-3 py-3 last:border-b-0" style={{ borderColor: "var(--border)" }}>
                    <div className="flex items-start gap-3">
                      <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
                      <div className="min-w-0 flex-1">
                        <div className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>{notice.title}</div>
                        <div className="mt-1 break-words text-xs leading-5" style={{ color: "var(--text-1)" }}>{notice.message}</div>
                        {notice.action === "settings" && (
                          <button
                            type="button"
                            onClick={() => {
                              onOpenSettings();
                              setOpen(false);
                            }}
                            className="mt-2 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1.5 text-[11px] font-medium text-[var(--text-0)] hover:bg-white/10"
                          >
                            Open runtime settings
                          </button>
                        )}
                      </div>
                      <button aria-label="Dismiss notification" title="Dismiss" onClick={() => setNotices((current) => current.filter((item) => item.id !== notice.id))} className="rounded-md p-1 hover:bg-white/5" style={{ color: "var(--text-2)", border: "none", background: "transparent", cursor: "pointer" }}>
                        <X size={14} />
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );

}

function LiveStreamFeed({ snapshot }: { snapshot: LiveStreamSnapshot | null }) {
  const [stream, setStream] = useState<LiveStreamSnapshot | null>(snapshot);
  const [mode, setMode] = useState<"full" | "visible" | "reasoning" | "events">("full");
  const outputRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    setStream(snapshot);
  }, [snapshot?.request_id, snapshot?.status, snapshot?.raw_output]);

  useEffect(() => {
    const cleanups: Array<Promise<() => void>> = [];
    cleanups.push(
      listen<LiveStreamSnapshot>("llm-stream-start", (event) => {
        setStream(event.payload);
      })
    );
    cleanups.push(
      listen<LiveStreamDelta>("llm-stream-delta", (event) => {
        setStream((current) => {
          if (!current || current.request_id !== event.payload.request_id) {
            return current;
          }
          return appendLiveStreamDelta(current, event.payload);
        });
      })
    );
    cleanups.push(
      listen<LiveStreamSnapshot>("llm-stream-done", (event) => {
        setStream(event.payload);
      })
    );
    return () => {
      cleanups.forEach((cleanup) => cleanup.then((fn) => fn()));
    };
  }, []);

  useEffect(() => {
    const node = outputRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [stream?.raw_output, stream?.events.length, mode]);

  const fullOutput = stream?.raw_output ?? "";
  const visibleOutput = stream?.visible_output ?? "";
  const reasoningOutput = stream?.reasoning_output ?? "";
  const eventsOutput =
    stream?.events
      .map((event) => `[${new Date(event.timestamp).toLocaleTimeString()}] ${event.kind}\n${event.text}`)
      .join("\n\n") ?? "";
  const output =
    mode === "full"
      ? fullOutput
      : mode === "visible"
      ? visibleOutput
      : mode === "reasoning"
      ? reasoningOutput
      : eventsOutput;
  const isRunning = stream?.status === "running";

  return (
    <section className="flex h-full min-h-0 flex-col" style={{ background: "var(--surface-1)" }}>
      <div
        className="flex shrink-0 flex-wrap items-center gap-2 px-3 py-2"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <span
          className={`h-2 w-2 rounded-full ${isRunning ? "animate-pulse" : ""}`}
          style={{ background: isRunning ? "#34d399" : stream ? "#64748b" : "#6b7280" }}
        />
        <div className="min-w-0 flex-1">
          <div className="truncate text-xs font-semibold uppercase tracking-[0.16em]" style={{ color: "var(--text-2)" }}>
            Live LLM Stream
          </div>
          <div className="truncate text-xs" style={{ color: "var(--text-1)" }}>
            {stream
              ? `${stream.source} - ${stream.model} - ${stream.status} - ${stream.request_id}`
              : "Waiting for the next generation."}
          </div>
        </div>
        {(["full", "visible", "reasoning", "events"] as const).map((item) => (
          <button
            key={item}
            onClick={() => setMode(item)}
            className="rounded px-2 py-1 text-xs"
            style={{
              background: mode === item ? "var(--surface-2)" : "transparent",
              border: mode === item ? "1px solid var(--border)" : "1px solid transparent",
              color: mode === item ? "var(--text-0)" : "var(--text-1)",
            }}
          >
            {item === "full" ? "Full" : item === "visible" ? "Text" : item === "reasoning" ? "Think" : "Events"}
          </button>
        ))}
        <button
          onClick={() => navigator.clipboard.writeText(output)}
          className="rounded px-2 py-1 text-xs"
          style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}
        >
          Copy
        </button>
      </div>
      <pre
        ref={outputRef}
        className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-words px-3 py-3 font-mono text-xs leading-5"
        style={{ color: output ? "var(--text-0)" : "var(--text-2)", background: "var(--bg)" }}
      >
        {output || "No stream output captured yet. Start a chat/API completion and this pane will fill live."}
      </pre>
    </section>
  );
}

void LiveStreamFeed;

type LiveStreamViewMode = "log" | "visible" | "reasoning" | "events" | "full" | "json";

function LiveStreamFeedV2({
  snapshot,
  snapshots = [],
}: {
  snapshot: LiveStreamSnapshot | null;
  snapshots?: LiveStreamSnapshot[];
}) {
  const [streams, setStreams] = useState<LiveStreamSnapshot[]>(() =>
    snapshots.length > 0 ? snapshots : snapshot ? [snapshot] : []
  );
  const [mode, setMode] = useState<LiveStreamViewMode>("log");
  const [showHistory, setShowHistory] = useState(true);
  const [query, setQuery] = useState("");
  const [expandOverride, setExpandOverride] = useState<Record<string, boolean>>({});
  const [followLive, setFollowLive] = useState(true);
  const [nowMs, setNowMs] = useState(() => Date.now());
  const outputRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (snapshots.length > 0) {
      setStreams((current) => reconcileLiveStreams(current, snapshots));
      return;
    }
    if (!snapshot) return;
    setStreams((current) => upsertLiveStream(current, snapshot));
  }, [snapshot?.request_id, snapshot?.status, snapshot?.raw_output, snapshots]);

  useEffect(() => {
    const cleanups: Array<Promise<() => void>> = [];
    cleanups.push(
      listen<LiveStreamSnapshot>("llm-stream-start", (event) => {
        setStreams((current) => upsertLiveStream(current, event.payload));
      })
    );
    cleanups.push(
      listen<LiveStreamDelta>("llm-stream-delta", (event) => {
        setStreams((current) => applyLiveStreamDelta(current, event.payload));
      })
    );
    cleanups.push(
      listen<LiveStreamSnapshot>("llm-stream-done", (event) => {
        setStreams((current) => upsertLiveStream(current, event.payload));
      })
    );
    return () => {
      cleanups.forEach((cleanup) => cleanup.then((fn) => fn()));
    };
  }, []);

  const active = activeLiveStream(streams);
  const latestCompletedOrActive = active ?? streams[streams.length - 1] ?? null;
  const isRunning = !!active;

  useEffect(() => {
    if (!active) return;
    setFollowLive(true);
    setNowMs(Date.now());
    const node = outputRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [active?.request_id]);

  useEffect(() => {
    if (!active) return;
    const interval = window.setInterval(() => setNowMs(Date.now()), 1_000);
    return () => window.clearInterval(interval);
  }, [active?.request_id]);

  useEffect(() => {
    if (!active || !followLive) return;
    const frame = window.requestAnimationFrame(() => {
      const node = outputRef.current;
      if (node) node.scrollTop = node.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
  }, [
    active?.request_id,
    active?.raw_output.length,
    active?.visible_output.length,
    active?.reasoning_output.length,
    active?.events.length,
    followLive,
    mode,
  ]);

  const historyStreams = showHistory
    ? streams
    : latestCompletedOrActive
      ? [latestCompletedOrActive]
      : [];
  const q = query.trim().toLowerCase();
  const filteredStreams = q
    ? historyStreams.filter((stream) =>
        `${stream.source} ${stream.model} ${stream.request_id} ${liveToolSummary(stream)}`
          .toLowerCase()
          .includes(q)
      )
    : historyStreams;
  const visibleStreams = tailLiveStream(
    filteredStreams,
    active && filteredStreams.some((stream) => stream.request_id === active.request_id)
      ? active
      : null,
  );
  const copyOutput = visibleStreams.map((stream) => liveStreamBlock(stream, mode)).join("\n\n");
  const isExpanded = (stream: LiveStreamSnapshot) =>
    expandOverride[stream.request_id] ??
    (stream.request_id === latestCompletedOrActive?.request_id || isLiveStreamRunningStatus(stream.status));
  const toggleExpanded = (stream: LiveStreamSnapshot) =>
    setExpandOverride((current) => ({
      ...current,
      [stream.request_id]: !isExpanded(stream),
    }));

  return (
    <section className="flex h-full min-h-0 flex-col" style={{ background: "var(--surface-1)" }}>
      <div
        className="flex shrink-0 flex-wrap items-center gap-2 px-3 py-2"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <span
          className={`h-2 w-2 rounded-full ${isRunning ? "animate-pulse" : ""}`}
          style={{ background: isRunning ? "#34d399" : latestCompletedOrActive ? "#64748b" : "#6b7280" }}
        />
        <div className="min-w-0 flex-1">
          <div className="truncate text-xs font-semibold uppercase tracking-[0.16em]" style={{ color: "var(--text-2)" }}>
            Live LLM Streams
          </div>
          <div className="truncate text-xs" style={{ color: "var(--text-1)" }}>
            {active
              ? `${streams.length} captured - generating ${formatLiveElapsed(active.started_at, nowMs)} - ${active.events.length === 0 ? "waiting for first token" : `${active.events.length} live events`}`
              : latestCompletedOrActive
              ? `${streams.length} captured - latest ${latestCompletedOrActive.status} - ${formatLiveDateTime(latestCompletedOrActive.started_at)}`
              : "Waiting for the next generation."}
          </div>
        </div>
        {active && (
          <button
            onClick={() => setFollowLive((value) => !value)}
            className="rounded px-2 py-1 text-xs"
            style={{
              background: followLive ? "rgba(52,211,153,0.1)" : "var(--surface-2)",
              border: followLive ? "1px solid rgba(52,211,153,0.25)" : "1px solid var(--border)",
              color: followLive ? "#34d399" : "var(--text-1)",
            }}
            title={followLive ? "Pause automatic scrolling" : "Resume automatic scrolling"}
          >
            {followLive ? "Following live" : "Resume live"}
          </button>
        )}
        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter streams..."
          className="w-[150px] rounded px-2 py-1 text-xs outline-none"
          style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
        />
        <button
          onClick={() => setShowHistory((value) => !value)}
          className="rounded px-2 py-1 text-xs"
          style={{
            background: showHistory ? "var(--surface-2)" : "transparent",
            border: "1px solid var(--border)",
            color: showHistory ? "var(--text-0)" : "var(--text-1)",
          }}
        >
          {showHistory ? "History" : "Latest"}
        </button>
        {(["log", "visible", "reasoning", "events", "full", "json"] as const).map((item) => (
          <button
            key={item}
            onClick={() => setMode(item)}
            className="rounded px-2 py-1 text-xs"
            style={{
              background: mode === item ? "var(--surface-2)" : "transparent",
              border: mode === item ? "1px solid var(--border)" : "1px solid transparent",
              color: mode === item ? "var(--text-0)" : "var(--text-1)",
            }}
          >
            {item === "log" ? "Log" : item === "full" ? "Raw" : item === "visible" ? "Text" : item === "reasoning" ? "Think" : item === "events" ? "Events" : "JSON"}
          </button>
        ))}
        <button
          onClick={() => navigator.clipboard.writeText(copyOutput)}
          className="rounded px-2 py-1 text-xs"
          style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}
        >
          Copy
        </button>
        <button
          onClick={() => setStreams([])}
          disabled={streams.length === 0}
          className="rounded px-2 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-40"
          style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}
        >
          Clear
        </button>
      </div>
      <div
        ref={outputRef}
        className="ib-live-log-terminal min-h-0 flex-1 overflow-auto"
        onScroll={(event) => {
          if (!active) return;
          const node = event.currentTarget;
          const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
          setFollowLive(distanceFromBottom <= 32);
        }}
      >
        {visibleStreams.length === 0 ? (
          <div className="px-3 py-3 font-mono text-xs leading-5" style={{ color: "var(--text-2)" }}>
            {q
              ? "No streams match the filter."
              : "No stream output captured yet. Start a chat/API completion and this pane will fill live."}
          </div>
        ) : (
          <div className="ib-live-log-streams">
            {visibleStreams.map((stream) => {
              const output = liveStreamOutput(stream, mode);
              const running = isLiveStreamRunningStatus(stream.status);
              const waitingForFirstToken = running && stream.events.length === 0;
              const completedAt = liveCompletedAt(stream);
              const expanded = isExpanded(stream);
              const level = liveStreamLogLevel(stream.status);
              return (
                <article
                  key={stream.request_id}
                  className={`ib-live-log-section ${running ? "is-running" : ""}`}
                >
                  <div
                    onClick={() => toggleExpanded(stream)}
                    className="ib-live-log-line"
                    role="button"
                    tabIndex={0}
                    aria-expanded={expanded}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        toggleExpanded(stream);
                      }
                    }}
                  >
                    <span className="ib-live-log-time">{formatLiveDateTime(stream.started_at)}</span>
                    <span className={`ib-live-log-level is-${level.toLowerCase()}`}>[{level}]</span>
                    <span className="ib-live-log-source">[{liveStreamLogSource(stream.source)}]</span>
                    <span className="ib-live-log-arrow">-&gt;</span>
                    <span className="ib-live-log-model">{stream.model}</span>
                    <span className="ib-live-log-status">{stream.status}</span>
                    <span className="ib-live-log-request" title={stream.request_id}>{shortLiveRequestId(stream.request_id)}</span>
                    {running && (
                      <span className="ib-live-log-live-detail">
                        {waitingForFirstToken
                          ? `waiting for first token - ${formatLiveElapsed(stream.started_at, nowMs)}`
                          : `live - ${stream.events.length} events - ${formatLiveElapsed(stream.started_at, nowMs)}`}
                      </span>
                    )}
                    {completedAt && <span className="ib-live-log-completed">done {formatLiveDateTime(completedAt)}</span>}
                    <span className="ib-live-log-chevron" aria-hidden="true">{expanded ? "▾" : "▸"}</span>
                  </div>
                  {expanded && (
                    <LiveStreamOutputView
                      stream={stream}
                      mode={mode}
                      output={output}
                      fallback={emptyLiveStreamMessage(mode, stream, nowMs)}
                    />
                  )}
                </article>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}

function LiveStreamOutputView({
  stream,
  mode,
  output,
  fallback,
}: {
  stream: LiveStreamSnapshot;
  mode: LiveStreamViewMode;
  output: string;
  fallback: string;
}) {
  if (!output) {
    return <div className="ib-live-log-output is-empty">{fallback}</div>;
  }

  if (mode === "log") {
    return (
      <div className="ib-live-log-output is-structured">
        {liveStreamTerminalRows(stream).map((row, index) => (
          <div className="ib-live-log-record" key={`${row.timestamp}-${row.kind}-${index}`}>
            <span className="ib-live-log-record-time">{formatLiveDateTime(row.timestamp)}</span>
            <span className={`ib-live-log-record-level is-${row.kind.toLowerCase()}`}>[{row.kind}]</span>
            <span className="ib-live-log-record-source">[{row.source}]</span>
            <span className="ib-live-log-record-direction">{row.direction}</span>
            <span className={`ib-live-log-record-message is-${row.kind.toLowerCase()}`}>
              {row.kind === "TOOL" ? <JsonSyntax text={row.message} /> : row.message}
            </span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={`ib-live-log-output is-code is-${mode}`}>
      {output.replace(/\r\n/g, "\n").split("\n").map((line, index) => (
        <div className="ib-live-log-code-line" key={index}>
          <ColouredLogLine line={line} mode={mode} />
        </div>
      ))}
    </div>
  );
}

function ColouredLogLine({ line, mode }: { line: string; mode: LiveStreamViewMode }) {
  const transcriptLabel = line.match(/^\[(INPUT|OUTPUT|RAW OUTPUT|REASONING)\]$/)?.[1];
  if (transcriptLabel) {
    return <span className={`ib-live-log-transcript-label is-${transcriptLabel.toLowerCase().replace(" ", "-")}`}>[{transcriptLabel}]</span>;
  }

  if (mode === "json") return <JsonSyntax text={line || " "} />;

  if (mode === "full") {
    const sse = line.match(/^(data:|event:)(\s*)(.*)$/);
    if (sse) {
      return (
        <>
          <span className="ib-live-log-sse-prefix">{sse[1]}</span>
          {sse[2]}
          <JsonSyntax text={sse[3]} />
        </>
      );
    }
  }

  if (mode === "events") {
    const event = line.match(/^(\[[^\]]+\])\s+(\S+)(.*)$/);
    if (event) {
      return (
        <>
          <span className="ib-live-log-event-time">{event[1]}</span>{" "}
          <span className={`ib-live-log-event-kind is-${eventKindClass(event[2])}`}>{event[2]}</span>
          {event[3]}
        </>
      );
    }
  }

  return <>{line || " "}</>;
}

function eventKindClass(kind: string) {
  const normalized = kind.toLowerCase();
  if (normalized.includes("error")) return "error";
  if (normalized.includes("tool")) return "tool";
  if (normalized.includes("reason")) return "think";
  if (normalized.includes("input")) return "input";
  if (normalized.includes("content")) return "output";
  return "info";
}

function JsonSyntax({ text }: { text: string }) {
  const pattern = /"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b/g;
  const tokens = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) tokens.push(text.slice(lastIndex, match.index));
    const token = match[0];
    let tokenClass = "literal";
    if (token.startsWith('"')) {
      tokenClass = /^\s*:/.test(text.slice(match.index + token.length)) ? "key" : "string";
    } else if (/^-?\d/.test(token)) {
      tokenClass = "number";
    }
    tokens.push(<span className={`ib-live-log-json-${tokenClass}`} key={`${match.index}-${tokenClass}`}>{token}</span>);
    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) tokens.push(text.slice(lastIndex));
  return <>{tokens}</>;
}

function liveStreamOutput(stream: LiveStreamSnapshot, mode: LiveStreamViewMode) {
  const input = liveStreamInputText(stream);
  if (mode === "log") {
    return liveStreamTerminalRows(stream)
      .map((row) => `${formatLiveDateTime(row.timestamp)} [${row.kind}] [${row.source}] ${row.direction} ${row.message}`)
      .join("\n");
  }
  if (mode === "full") {
    return formatLiveStreamTranscript(
      input,
      scrubToolLogNoise(stream.raw_output) || liveToolSummary(stream),
      "RAW OUTPUT",
    );
  }
  if (mode === "visible") {
    return formatLiveStreamTranscript(
      input,
      scrubToolLogNoise(stream.visible_output) || bufferedVisibleText(stream) || liveToolSummary(stream),
    );
  }
  if (mode === "reasoning") {
    return formatLiveStreamTranscript(input, stream.reasoning_output, "REASONING");
  }
  if (mode === "json") return JSON.stringify(stream, null, 2);
  return stream.events.map((event) => `[${formatLiveDateTime(event.timestamp)}] ${event.kind}\n${event.text}`).join("\n\n");
}

function liveStreamBlock(stream: LiveStreamSnapshot, mode: LiveStreamViewMode) {
  const level = liveStreamLogLevel(stream.status);
  const completedAt = liveCompletedAt(stream);
  return [
    `${formatLiveDateTime(stream.started_at)} [${level}] [${liveStreamLogSource(stream.source)}] -> ${stream.model} ${stream.status} ${stream.request_id}${completedAt ? ` done ${formatLiveDateTime(completedAt)}` : ""}`,
    liveStreamOutput(stream, mode),
  ]
    .filter(Boolean)
    .join("\n");
}

function liveCompletedAt(stream: LiveStreamSnapshot) {
  if (isLiveStreamRunningStatus(stream.status)) return null;
  return stream.events[stream.events.length - 1]?.timestamp ?? stream.started_at;
}

function formatLiveElapsed(startedAt: string, nowMs: number) {
  const startedMs = Date.parse(startedAt);
  if (Number.isNaN(startedMs)) return "elapsed unknown";
  const seconds = Math.max(0, Math.floor((nowMs - startedMs) / 1_000));
  const hours = Math.floor(seconds / 3_600);
  const minutes = Math.floor((seconds % 3_600) / 60);
  const remainingSeconds = seconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m ${remainingSeconds}s`;
  if (minutes > 0) return `${minutes}m ${remainingSeconds}s`;
  return `${remainingSeconds}s`;
}

function formatLiveDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  if (sameDay) {
    return date.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function shortLiveRequestId(id: string) {
  if (id.length <= 12) return id;
  return `${id.slice(0, 8)}...${id.slice(-6)}`;
}

function emptyLiveStreamMessage(mode: string, stream: LiveStreamSnapshot, nowMs: number) {
  if (isLiveStreamRunningStatus(stream.status) && stream.events.length === 0) {
    return `Preparing prompt / waiting for first token (${formatLiveElapsed(stream.started_at, nowMs)}). Live output will appear here as soon as llama-server emits it.`;
  }
  if (isLiveStreamRunningStatus(stream.status)) {
    if (mode === "reasoning") return "Generation is live; no reasoning tokens have been emitted yet.";
    if (mode === "visible") return "Generation is live; no visible text has been emitted yet.";
    if (mode === "events") return "Generation is live; waiting for the next event.";
    return "Generation is live; waiting for output in this view.";
  }
  if (mode === "reasoning") return "No reasoning tokens captured for this request.";
  if (mode === "visible") return "No visible text captured for this request.";
  if (mode === "events") return "No events captured for this request.";
  return "No output captured for this request.";
}

function scrubToolLogNoise(text: string) {
  let cleaned = text
    .replace(/<\|channel\|?>thought[\s\S]*?<channel\|>/g, "")
    .replace(/<\|channel>thought[\s\S]*?<channel\|>/g, "")
    .replace(/<\|tool_call>[\s\S]*?(?:<tool_call\|>|<\/tool_call>)/g, "")
    .replace(/<tool_call>[\s\S]*?(?:<tool_call\|>|<\/tool_call>)/g, "");
  const markerIndex = ["<|tool_call>", "<tool_call>", "<|channel>thought", "<|channel|>thought"]
    .map((marker) => cleaned.indexOf(marker))
    .filter((index) => index >= 0)
    .sort((a, b) => a - b)[0];
  if (markerIndex != null) {
    cleaned = cleaned.slice(0, markerIndex);
  }
  return cleaned.trim();
}

function bufferedVisibleText(stream: LiveStreamSnapshot) {
  return scrubToolLogNoise(
    stream.events
      .filter((event) => event.kind === "content_buffered")
      .map((event) => event.text)
      .join("")
  );
}

function liveToolSummary(stream: LiveStreamSnapshot) {
  const calls = stream.events.filter((event) => event.kind === "tool_call");
  if (calls.length === 0) return "";
  const names = calls.flatMap((event) => {
    try {
      const parsed = JSON.parse(event.text);
      const items = Array.isArray(parsed) ? parsed : [parsed];
      return items
        .map((item) => item?.function?.name ?? item?.name)
        .filter((name): name is string => typeof name === "string" && name.trim().length > 0);
    } catch {
      return [];
    }
  });
  return names.length > 0
    ? `Tool call${names.length === 1 ? "" : "s"}: ${names.join(", ")}`
    : `${calls.length} tool call event${calls.length === 1 ? "" : "s"} captured.`;
}

function LogsWorkspace({
  snapshot,
  snapshots,
}: {
  snapshot: LiveStreamSnapshot | null;
  snapshots: LiveStreamSnapshot[];
}) {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <LiveStreamFeedV2 snapshot={snapshot} snapshots={snapshots} />
    </div>
  );
}

function ContextWorkspace({ contextStatus, processStatus, gpuStats }: ContextWorkspaceProps) {
  return (
    <ContextPanel
      status={contextStatus}
      processStatus={processStatus}
      gpuStats={gpuStats}
    />
  );
}

function buildReachableApiUrl(settings: AppSettings | null) {
  const host =
    settings?.server_host === "0.0.0.0"
      ? "127.0.0.1"
      : settings?.server_host ?? "127.0.0.1";
  const port = settings?.server_port ?? 8800;
  return `http://${host}:${port}/v1`;
}

function launchPreviewMatchesLoadedModel(
  loadedModelName: string | null,
  previewModelPath: string | null | undefined
) {
  if (!loadedModelName || !previewModelPath) return false;
  const previewName =
    previewModelPath.split(/[\\/]/).pop()?.trim().toLowerCase() ??
    previewModelPath.trim().toLowerCase();
  const requested = loadedModelName.trim().toLowerCase();
  return (
    previewName === requested ||
    previewName.replace(/\.gguf$/i, "") === requested ||
    previewName === requested.replace(/\.gguf$/i, "") ||
    (!!requested && previewName.includes(requested))
  );
}

function filenameLooksVisionCapable(modelName: string | null) {
  const name = modelName?.trim().toLowerCase() ?? "";
  if (!name) return false;
  return (
    name.includes("gemma-4") ||
    name.includes("gemma4") ||
    name.includes("qwen3.6") ||
    name.includes("-vl") ||
    name.includes("_vl") ||
    name.includes("vision") ||
    name.includes("llava") ||
    name.includes("internvl") ||
    name.includes("minicpm-v") ||
    name.includes("minicpmv") ||
    name.includes("pixtral") ||
    name.includes("smolvlm")
  );
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  const [modelLoadRequest, setModelLoadRequest] = useState<ModelLoadRequest | null>(null);
  const [runtimeSettingsFocusRequest, setRuntimeSettingsFocusRequest] = useState(0);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [apiRecoveryMessage, setApiRecoveryMessage] = useState<string | null>(null);
  const [apiRecoveryBusy, setApiRecoveryBusy] = useState(false);

  const model = useModel();
  const session = useSession();
  const chat = useChat(session.activeId);
  const context = useContext();
  const gpuStats = useGpuStats();
  const modelPickerReturnFocus = useRef<HTMLElement | null>(null);

  const hasModel = !!model.processStatus?.model;
  const loadedModelName = model.processStatus?.model ?? null;
  const loadedModelEntry = loadedModelName
    ? model.models.find((entry) => entry.filename === loadedModelName) ?? null
    : null;
  const loadedModelVisionConfigured =
    loadedModelEntry?.supports_vision ?? filenameLooksVisionCapable(loadedModelName);
  const loadedModelSupportsVision =
    loadedModelEntry?.vision_runtime_ready ||
    (loadedModelVisionConfigured &&
      launchPreviewMatchesLoadedModel(
        loadedModelName,
        model.processStatus?.last_launch_preview?.model_path
      ) &&
      !!model.processStatus?.last_launch_preview?.mmproj_path);
  const loadedModelVisionStatusText =
    loadedModelVisionConfigured && !loadedModelSupportsVision && loadedModelName
      ? `The loaded model ${loadedModelName} was started without a matching mmproj sidecar, so pasted images will not be seen correctly. Reload a vision-ready model first.`
      : null;
  const debugApiUrl =
    model.processStatus?.api_url ??
    buildReachableApiUrl(settings);
  const modelTransition =
    model.loadProgress && !model.loadProgress.done
      ? model.loadProgress
      : model.processStatus?.model_load_progress && !model.processStatus.model_load_progress.done
      ? model.processStatus.model_load_progress
      : null;
  const modelTransitionActive =
    !!modelTransition ||
    model.isLoading ||
    ["Starting", "Stopping"].includes(model.processStatus?.state ?? "Idle") ||
    ["Loading", "Swapping", "Unloading"].includes(
      model.processStatus?.model_load_state ?? "Idle"
    );
  const apiStartupError =
    model.processStatus?.api_state === "Error" && !modelTransitionActive
      ? model.processStatus.api_error
      : null;
  const apiState = model.processStatus?.api_state ?? "Idle";
  const apiReachable = model.processStatus?.api_reachable ?? false;
  const apiStopping = model.apiAction === "stopping" || apiState === "Stopping";
  const apiStarting = model.apiAction === "starting" || apiState === "Starting";
  const apiRunning = (apiState === "Running" && apiReachable) || (apiState === "Running" && !apiStopping);
  const apiActive = apiRunning || apiStarting || apiStopping || apiReachable;
  const apiBusy = apiStarting || apiStopping;
  const apiPortOwner = model.processStatus?.api_port_owner ?? null;
  const configuredApiPort = settings?.server_port ?? 8800;

  const openModelPicker = useCallback((trigger: HTMLElement | null) => {
    modelPickerReturnFocus.current = trigger;
    setModelPickerOpen(true);
  }, []);

  const requestModelLoad = useCallback((entry: ModelInfo, returnFocus: HTMLElement | null) => {
    const mode: LoadDialogMode =
      entry.filename === loadedModelName ? "reload" : loadedModelName ? "swap" : "load";
    setModelPickerOpen(false);
    setModelLoadRequest({ model: entry, mode, returnFocus });
  }, [loadedModelName]);

  useEffect(() => {
    let cancelled = false;
    api
      .getSettings()
      .then((s) => { if (!cancelled) setSettings(s); })
      .catch(() => { if (!cancelled) setSettings(null); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (apiReachable) {
      setApiRecoveryMessage(null);
    }
  }, [apiReachable]);

  const handleNewSession = async () => {
    const name = nextAutomaticChatName(session.sessions);
    await session.createSession(name);
  };

  const handleExportSession = useCallback(async (sessionInfo: (typeof session.sessions)[number]) => {
    try {
      const messages = await api.getSessionMessages(sessionInfo.id);
      downloadConversation(
        safeConversationFilename(sessionInfo.name),
        conversationMarkdown(sessionInfo, messages),
      );
    } catch (error) {
      console.error("Failed to export conversation", error);
    }
  }, []);

  // Auto-create a session when a model becomes loaded and no sessions exist yet.
  useEffect(() => {
    if (hasModel && session.ready && session.sessions.length === 0 && !session.activeId && !session.isCreating) {
      session.createSession(nextAutomaticChatName(session.sessions));
    }
  }, [hasModel, session.ready, session.sessions.length, session.activeId]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg)", color: "var(--text-0)" }}>
      <AppContextMenu
        activeTab={activeTab}
        onNavigate={setActiveTab}
        onCreateChat={async () => {
          await handleNewSession();
          setActiveTab("chat");
        }}
      />
      <CommandPalette
        sessions={session.sessions}
        onCreateChat={() => { void handleNewSession(); setActiveTab("chat"); }}
        onSelectSession={session.setActiveId}
        onNavigate={setActiveTab}
        onChooseModel={() => openModelPicker(null)}
      />
      <AppSidebar
        activeTab={activeTab}
        sessions={session.sessions}
        activeSessionId={session.activeId}
        modelName={loadedModelName}
        apiState={apiState}
        sessionReady={session.ready}
        sessionError={session.error}
        creatingSession={session.isCreating}
        onNavigate={setActiveTab}
        onSelectSession={session.setActiveId}
        onCreateSession={handleNewSession}
        onDeleteSession={session.deleteSession}
        onRenameSession={session.renameSession}
        onSetSessionPinned={session.setSessionPinned}
        onExportSession={(sessionInfo) => { void handleExportSession(sessionInfo); }}
      />

      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="ib-page-header">
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-[var(--text-0)]">{TAB_LABELS[activeTab]}</div>
            <div className="truncate text-[11px] text-[var(--text-2)]">{TAB_DESCRIPTIONS[activeTab]}</div>
          </div>

          <div className="ml-auto flex min-w-0 items-center gap-1.5">
            {chat.isStreaming && (
              <button
                type="button"
                onClick={() => setActiveTab("chat")}
                className="mr-1 hidden items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs text-emerald-300 hover:bg-white/5 sm:flex"
              >
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                Generating
              </button>
            )}

            {activeTab !== "chat" && (
              <button
                type="button"
                onClick={(event) => openModelPicker(event.currentTarget)}
                aria-label={loadedModelEntry ? `Change model. Current model ${modelDisplayName(loadedModelEntry)}` : loadedModelName ? `Change model. Current model ${loadedModelName}` : "Choose a model"}
                aria-haspopup="dialog"
                aria-expanded={modelPickerOpen}
                aria-controls="rich-model-picker"
                className="hidden max-w-[280px] items-center gap-2 rounded-lg px-2 py-1 text-xs text-[var(--text-1)] hover:bg-white/5 lg:flex"
                title={loadedModelName ?? "Choose a model"}
              >
                {loadedModelEntry ? (
                  <ModelArtwork model={loadedModelEntry} size="xs" />
                ) : (
                  <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${loadedModelName ? "bg-emerald-400" : "bg-[var(--text-3)]"}`} />
                )}
                <span className="truncate">{loadedModelEntry ? modelDisplayName(loadedModelEntry) : loadedModelName ?? "Choose a model"}</span>
              </button>
            )}

            <NotificationCenter
              preferredBackend={settings?.backend_preference}
              onOpenSettings={() => {
                setRuntimeSettingsFocusRequest((request) => request + 1);
                setActiveTab("settings");
              }}
            />

            {activeTab !== "debug" && (
              <Button
                size="sm"
                variant={apiActive ? "secondary" : "primary"}
                onClick={() => model.setApiServerRunning(!apiActive)}
                disabled={apiBusy}
                icon={
                  apiBusy ? (
                    <LoaderCircle size={14} className="animate-spin" />
                  ) : apiActive ? (
                    <Square size={13} />
                  ) : (
                    <Play size={14} />
                  )
                }
              >
                {apiStopping ? "Stopping" : apiStarting ? "Starting" : apiActive ? "Stop API" : apiState === "Error" ? "Retry API" : "Start API"}
              </Button>
            )}
          </div>
        </header>

      {apiStartupError && (
        <div
          className="shrink-0 border-b px-4 py-3"
          style={{
            borderColor: "rgba(248, 113, 113, 0.24)",
            background: "rgba(127, 29, 29, 0.28)",
            color: "#fecaca",
          }}
        >
          <p className="text-sm font-semibold">API server issue</p>
          <p className="mt-1 text-sm" style={{ color: "#fca5a5" }}>
            {apiStartupError}
          </p>
          <p className="mt-1 text-xs" style={{ color: "#fda4af" }}>
            The desktop UI still talks to InferenceBridge directly. This only affects external API clients that expect the public endpoint at `{debugApiUrl}`.
          </p>
          {apiPortOwner && !apiReachable && (
            <p className="mt-2 text-xs" style={{ color: "#fecaca" }}>
              Port owner: {apiPortOwner.name ?? "Unknown process"} (PID {apiPortOwner.pid})
            </p>
          )}
          {apiRecoveryMessage && (
            <p className="mt-2 text-xs" style={{ color: "#fde68a" }}>
              {apiRecoveryMessage}
            </p>
          )}
          <div className="mt-3 flex items-center gap-2">
            <button
              onClick={() => model.setApiServerRunning(true)}
              className="rounded-lg px-3 py-1.5 text-xs font-medium transition"
              style={{
                background: "#f4f4f4",
                color: "#171717",
                border: "none",
                cursor: "pointer",
              }}
            >
              Retry API
            </button>
            <button
              onClick={() => setActiveTab("settings")}
              className="rounded px-3 py-1.5 text-xs font-medium transition"
              style={{
                background: "rgba(255,255,255,0.06)",
                color: "#fecaca",
                border: "1px solid rgba(248, 113, 113, 0.24)",
                cursor: "pointer",
              }}
            >
              Server Settings
            </button>
            {apiPortOwner && (apiPortOwner.killable || apiPortOwner.kind === "ghost") && (
              <button
                onClick={async () => {
                  setApiRecoveryBusy(true);
                  setApiRecoveryMessage(null);
                  try {
                    const message = await api.recoverApiPort(
                      apiPortOwner.pid,
                      configuredApiPort,
                      apiPortOwner.kind,
                    );
                    setApiRecoveryMessage(message);
                    await model.refresh();
                    await model.setApiServerRunning(true);
                  } catch (error) {
                    setApiRecoveryMessage(String(error));
                    await model.refresh();
                  } finally {
                    setApiRecoveryBusy(false);
                  }
                }}
                disabled={apiRecoveryBusy}
                className="rounded px-3 py-1.5 text-xs font-medium transition"
                style={{
                  background: "rgba(251,191,36,0.14)",
                  color: "#fde68a",
                  border: "1px solid rgba(251,191,36,0.24)",
                  cursor: apiRecoveryBusy ? "wait" : "pointer",
                  opacity: apiRecoveryBusy ? 0.7 : 1,
                }}
              >
                {apiRecoveryBusy
                  ? "Checking..."
                  : apiPortOwner.kind === "ghost"
                    ? `Show recovery steps (${apiPortOwner.pid})`
                    : `Kill ${apiPortOwner.kind === "llama-server" ? "llama-server" : "stale app"} (${apiPortOwner.pid})`}
              </button>
            )}
          </div>
        </div>
      )}

        <main className="flex min-h-0 flex-1 overflow-hidden">
        <div className="min-h-0 flex-1 overflow-hidden">
          <div className={`h-full ${activeTab === "chat" ? "block" : "hidden"}`}>
            <ChatPanel
              messages={chat.messages}
              isStreaming={chat.isStreaming}
              streamingText={chat.streamingText}
              streamingReasoning={chat.streamingReasoning}
              tokensPerSecond={chat.tokensPerSecond}
              processStatus={model.processStatus}
              error={chat.error}
              hasModel={hasModel}
              hasSession={!!session.activeId}
              sessionId={session.activeId}
              loadedModel={loadedModelName}
              loadedModelInfo={loadedModelEntry}
              modelPickerOpen={modelPickerOpen}
              loadedModelVisionConfigured={loadedModelVisionConfigured}
              loadedModelSupportsVision={loadedModelSupportsVision}
              loadedModelVisionStatusText={loadedModelVisionStatusText}
              onSend={chat.sendMessage}
              onStop={chat.stopGeneration}
              canCreateSession={session.ready}
              creatingSession={session.isCreating}
              onCreateSession={() => { void handleNewSession(); }}
              onOpenModelPicker={openModelPicker}
            />
          </div>

          <div className={`h-full overflow-y-auto ${activeTab === "browse" ? "block" : "hidden"}`}>
            <ModelBrowser
              models={model.models}
              onRefresh={model.scanModels}
            />
          </div>

          <div className={`h-full min-h-0 overflow-hidden ${activeTab === "benchmark" ? "block" : "hidden"}`}>
            <BenchmarkPanel models={model.models} processStatus={model.processStatus} />
          </div>

          <div className={`h-full min-h-0 overflow-hidden ${activeTab === "models" ? "block" : "hidden"}`}>
            <div className="box-border h-full min-h-0">
              <ModelSelector
                models={model.models}
                loadedModel={model.processStatus?.model ?? null}
                previousModel={model.processStatus?.previous_model ?? null}
                processStatus={model.processStatus}
                settings={settings}
                error={model.error}
                isLoading={model.isLoading}
                loadProgress={model.loadProgress}
                onUnload={model.unloadModel}
                onSwap={model.swapModel}
                onScan={model.scanModels}
                onOpenSettings={() => setActiveTab("settings")}
                onConfigureLoad={requestModelLoad}
              />
            </div>
          </div>

          <div className={`h-full ${activeTab === "context" ? "block" : "hidden"}`}>
            <ContextWorkspace
              contextStatus={context}
              processStatus={model.processStatus}
              gpuStats={gpuStats}
            />
          </div>

          <div className={`h-full ${activeTab === "logs" ? "block" : "hidden"}`}>
            <LogsWorkspace
              snapshot={model.processStatus?.live_stream ?? null}
              snapshots={model.processStatus?.live_streams ?? []}
            />
          </div>

          <div className={`h-full ${activeTab === "debug" ? "block" : "hidden"}`}>
            <DebugInspector
              apiUrl={debugApiUrl}
              processStatus={model.processStatus}
              loadProgress={model.loadProgress}
              models={model.models}
              onSetApiServerRunning={model.setApiServerRunning}
              apiAction={model.apiAction}
              onOpenSettings={() => setActiveTab("settings")}
              modelPickerOpen={modelPickerOpen}
              onOpenModelPicker={openModelPicker}
            />
          </div>

          <div className={`h-full ${activeTab === "settings" ? "block" : "hidden"}`}>
            <SettingsPanel
              onSaved={setSettings}
              processStatus={model.processStatus}
              loadProgress={model.loadProgress}
              apiAction={model.apiAction}
              onSetApiServerRunning={model.setApiServerRunning}
              runtimeFocusRequest={runtimeSettingsFocusRequest}
            />
          </div>
        </div>
        </main>

        <StatusBar
          processStatus={model.processStatus}
          contextStatus={context}
          settings={settings}
          loadProgress={model.loadProgress}
        />
      </div>

      <RichModelPicker
        open={modelPickerOpen}
        models={model.models}
        loadedModel={loadedModelName}
        processStatus={model.processStatus}
        isLoading={model.isLoading}
        loadProgress={model.loadProgress}
        error={model.error}
        returnFocus={modelPickerReturnFocus.current}
        switchingDisabledReason={chat.isStreaming ? "Stop the current generation before switching models." : null}
        onClose={() => setModelPickerOpen(false)}
        onConfigureLoad={(entry) => requestModelLoad(entry, modelPickerReturnFocus.current)}
        onOpenLibrary={() => {
          setModelPickerOpen(false);
          setActiveTab("models");
        }}
        onScan={model.scanModels}
      />

      {modelLoadRequest && (
        <ModelLoadDialog
          key={`${modelLoadRequest.model.provider_type}:${modelLoadRequest.model.path || modelLoadRequest.model.filename}`}
          model={modelLoadRequest.model}
          mode={modelLoadRequest.mode}
          loadedModel={loadedModelName}
          processStatus={model.processStatus}
          settings={settings}
          isLoading={model.isLoading}
          returnFocus={modelLoadRequest.returnFocus}
          onClose={() => setModelLoadRequest(null)}
          onSubmit={(options) => {
            const request = modelLoadRequest;
            setModelLoadRequest(null);
            if (request.mode === "swap") model.swapModel(request.model.filename, options);
            else model.loadModel(request.model.filename, options);
          }}
        />
      )}
    </div>
  );
}

export default App;
