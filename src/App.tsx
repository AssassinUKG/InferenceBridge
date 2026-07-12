import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useModel } from "./hooks/useModel";
import { useSession } from "./hooks/useSession";
import { useChat } from "./hooks/useChat";
import { useContext } from "./hooks/useContext";
import { useGpuStats } from "./hooks/useGpuStats";
import { StatusBar } from "./components/common/StatusBar";
import { Sidebar } from "./components/common/Sidebar";
import { ChatPanel } from "./components/Chat/ChatPanel";
import { ModelSelector } from "./components/Model/ModelSelector";
import { SettingsPanel } from "./components/Model/SettingsPanel";
import { ContextPanel } from "./components/Context/ContextPanel";
import { DebugInspector } from "./components/Debug/DebugInspector";
import { ModelBrowser } from "./components/Model/ModelBrowser";
import { BenchmarkPanel } from "./components/Benchmark/BenchmarkPanel";
import type { AppSettings, LiveStreamDelta, LiveStreamSnapshot, LoadProgress } from "./lib/types";
import * as api from "./lib/tauri";
import type { DownloadProgress } from "./lib/tauri";

type Tab = "chat" | "models" | "browse" | "benchmark" | "context" | "logs" | "debug" | "settings";

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

function streamHasDelta(stream: LiveStreamSnapshot, delta: LiveStreamDelta) {
  return stream.events.some(
    (event) =>
      event.timestamp === delta.timestamp &&
      event.kind === delta.kind &&
      event.text === delta.text
  );
}

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
}

function NotificationCenter() {
  const [notices, setNotices] = useState<UiNotice[]>([]);
  const [open, setOpen] = useState(false);
  const [seenIds, setSeenIds] = useState<Set<string>>(new Set());
  const menuRef = useRef<HTMLDivElement | null>(null);

  const pushNotice = (notice: Omit<UiNotice, "id">) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setNotices((current) => [...current, { ...notice, id }].slice(-20));
  };

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
    return () => {
      cleanups.forEach((cleanup) => cleanup.then((fn) => fn()));
    };
  }, []);

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
      <button
        onClick={() => setOpen((value) => !value)}
        className="relative flex h-8 w-8 items-center justify-center rounded transition"
        title="Notifications"
        style={{
          background: open ? "rgba(34,211,238,0.12)" : "var(--surface-2)",
          border: open ? "1px solid rgba(34,211,238,0.26)" : "1px solid var(--border)",
          color: open ? "#67e8f9" : "var(--text-1)",
          cursor: "pointer",
        }}
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M18 9A6 6 0 0 0 6 9c0 7-3 7-3 9h18c0-2-3-2-3-9Z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M10 21h4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
        {unreadCount > 0 && (
          <span className="absolute -right-1 -top-1 rounded-full px-1 text-[9px] font-bold leading-4" style={{ minWidth: 16, height: 16, background: "#22d3ee", color: "#041014" }}>
            {unreadCount > 9 ? "9+" : unreadCount}
          </span>
        )}
      </button>
      {open && (
        <div className="absolute right-0 top-full z-50 mt-2 w-[380px] overflow-hidden rounded" style={{ background: "var(--surface-1)", border: "1px solid var(--border)", boxShadow: "0 18px 48px rgba(0,0,0,0.42)" }}>
          <div className="flex items-center justify-between border-b px-3 py-2" style={{ borderColor: "var(--border)" }}>
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>Notifications</div>
              <div className="mt-0.5 text-xs" style={{ color: "var(--text-1)" }}>{notices.length > 0 ? `${notices.length} recent events` : "No events yet"}</div>
            </div>
            {notices.length > 0 && (
              <button onClick={() => setNotices([])} className="rounded px-2 py-1 text-[11px] font-medium" style={{ background: "transparent", border: "1px solid var(--border)", color: "var(--text-1)", cursor: "pointer" }}>
                Clear
              </button>
            )}
          </div>
          <div className="max-h-[420px] overflow-y-auto">
            {notices.length === 0 ? (
              <div className="px-3 py-8 text-sm" style={{ color: "var(--text-2)" }}>Downloads, API actions, and runtime events will collect here.</div>
            ) : (
              [...notices].reverse().map((notice) => {
                const color = notice.tone === "error" ? "#f87171" : notice.tone === "success" ? "#34d399" : "#22d3ee";
                return (
                  <div key={notice.id} className="border-b px-3 py-3 last:border-b-0" style={{ borderColor: "var(--border)" }}>
                    <div className="flex items-start gap-3">
                      <span className="mt-1 h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
                      <div className="min-w-0 flex-1">
                        <div className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>{notice.title}</div>
                        <div className="mt-1 break-words text-xs leading-5" style={{ color: "var(--text-1)" }}>{notice.message}</div>
                      </div>
                      <button onClick={() => setNotices((current) => current.filter((item) => item.id !== notice.id))} className="rounded px-1 text-xs" style={{ color: "var(--text-2)", border: "none", background: "transparent", cursor: "pointer" }}>
                        x
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

  if (notices.length === 0) {
    return null;
  }

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-[420px] max-w-[calc(100vw-32px)] flex-col gap-2">
      {notices.map((notice) => {
        const color =
          notice.tone === "error" ? "#f87171" : notice.tone === "success" ? "#34d399" : "#22d3ee";
        return (
          <div
            key={notice.id}
            className="pointer-events-auto rounded-lg px-4 py-3 shadow-2xl"
            style={{
              background: "rgba(58,58,60,0.98)",
              border: "1px solid rgba(255,255,255,0.14)",
              color: "var(--text-0)",
              boxShadow: "0 18px 48px rgba(0,0,0,0.45)",
            }}
          >
            <div className="flex items-start gap-2">
              <span className="mt-0.5 h-4 w-4 shrink-0 rounded-full text-center text-[10px] leading-4" style={{ background: color, color: "#111" }}>✓</span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>
                  {notice.title}
                </div>
                <div className="mt-2 break-words text-xs leading-5" style={{ color: "var(--text-1)" }}>
                  {notice.message}
                </div>
              </div>
              <button
                onClick={() => setNotices((current) => current.filter((item) => item.id !== notice.id))}
                className="rounded px-1 text-xs"
                style={{ color: "var(--text-2)", border: "none", background: "transparent", cursor: "pointer" }}
              >
                x
              </button>
            </div>
          </div>
        );
      })}
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
          if (streamHasDelta(current, event.payload)) {
            return current;
          }
          const next: LiveStreamSnapshot = {
            ...current,
            raw_output:
              event.payload.kind === "raw" ||
              event.payload.kind === "error"
                ? current.raw_output + event.payload.text
                : current.raw_output,
            visible_output:
              event.payload.kind === "content"
                ? current.visible_output + event.payload.text
                : current.visible_output,
            reasoning_output:
              event.payload.kind === "reasoning"
                ? current.reasoning_output + event.payload.text
                : current.reasoning_output,
            events: [...current.events, event.payload].slice(-500),
          };
          return next;
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
  const [mode, setMode] = useState<"full" | "visible" | "reasoning" | "events" | "json">("full");
  const [showHistory, setShowHistory] = useState(true);
  const [query, setQuery] = useState("");
  const [expandOverride, setExpandOverride] = useState<Record<string, boolean>>({});
  const outputRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (snapshots.length > 0) {
      setStreams(snapshots);
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
        setStreams((current) =>
          current.map((stream) =>
            stream.request_id === event.payload.request_id ? appendLiveDelta(stream, event.payload) : stream
          )
        );
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

  useEffect(() => {
    const node = outputRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [streams.map((stream) => `${stream.request_id}:${stream.raw_output.length}:${stream.events.length}:${stream.status}`).join("|"), mode]);

  const latest = streams[streams.length - 1] ?? null;
  const isRunning = latest?.status === "running";
  const historyStreams = showHistory ? streams : latest ? [latest] : [];
  const q = query.trim().toLowerCase();
  const visibleStreams = q
    ? historyStreams.filter((stream) =>
        `${stream.source} ${stream.model} ${stream.request_id} ${liveToolSummary(stream)}`
          .toLowerCase()
          .includes(q)
      )
    : historyStreams;
  const copyOutput = visibleStreams.map((stream) => liveStreamBlock(stream, mode)).join("\n\n");
  const isExpanded = (stream: LiveStreamSnapshot) =>
    expandOverride[stream.request_id] ??
    (stream.request_id === latest?.request_id || stream.status === "running");
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
          style={{ background: isRunning ? "#34d399" : latest ? "#64748b" : "#6b7280" }}
        />
        <div className="min-w-0 flex-1">
          <div className="truncate text-xs font-semibold uppercase tracking-[0.16em]" style={{ color: "var(--text-2)" }}>
            Live LLM Streams
          </div>
          <div className="truncate text-xs" style={{ color: "var(--text-1)" }}>
            {latest
              ? `${streams.length} captured - latest ${latest.status} - ${formatLiveDateTime(latest.started_at)}`
              : "Waiting for the next generation."}
          </div>
        </div>
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
        {(["full", "visible", "reasoning", "events", "json"] as const).map((item) => (
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
            {item === "full" ? "Raw" : item === "visible" ? "Text" : item === "reasoning" ? "Think" : item === "events" ? "Events" : "JSON"}
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
      <div ref={outputRef} className="min-h-0 flex-1 overflow-auto px-3 py-3" style={{ background: "var(--bg)" }}>
        {visibleStreams.length === 0 ? (
          <div className="font-mono text-xs leading-5" style={{ color: "var(--text-2)" }}>
            {q
              ? "No streams match the filter."
              : "No stream output captured yet. Start a chat/API completion and this pane will fill live."}
          </div>
        ) : (
          <div className="space-y-3">
            {visibleStreams.map((stream) => {
              const output = liveStreamOutput(stream, mode);
              const running = stream.status === "running";
              const completedAt = liveCompletedAt(stream);
              const expanded = isExpanded(stream);
              return (
                <article
                  key={stream.request_id}
                  className="overflow-hidden rounded-lg"
                  style={{ border: "1px solid var(--border)", background: "var(--surface-1)" }}
                >
                  <div
                    onClick={() => toggleExpanded(stream)}
                    className="flex cursor-pointer flex-wrap items-center gap-2 px-3 py-2 text-xs"
                    style={{ borderBottom: expanded ? "1px solid var(--border)" : "none", color: "var(--text-1)" }}
                  >
                    <span className="select-none" style={{ color: "var(--text-2)" }}>{expanded ? "▾" : "▸"}</span>
                    <span
                      className={`h-2 w-2 rounded-full ${running ? "animate-pulse" : ""}`}
                      style={{ background: running ? "#34d399" : stream.status === "error" ? "#f87171" : "#64748b" }}
                    />
                    <span className="font-semibold" style={{ color: "var(--text-0)" }}>{stream.source}</span>
                    <span>{stream.model}</span>
                    <span>{stream.status}</span>
                    <span title={stream.request_id}>{shortLiveRequestId(stream.request_id)}</span>
                    <span className="ml-auto">started {formatLiveDateTime(stream.started_at)}</span>
                    {completedAt && <span>done {formatLiveDateTime(completedAt)}</span>}
                  </div>
                  {expanded && (
                    <pre
                      className="max-h-[420px] overflow-auto whitespace-pre-wrap break-words px-3 py-3 font-mono text-xs leading-5"
                      style={{ color: output ? "var(--text-0)" : "var(--text-2)", background: "var(--bg)" }}
                    >
                      {output || emptyLiveStreamMessage(mode)}
                    </pre>
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

function upsertLiveStream(current: LiveStreamSnapshot[], next: LiveStreamSnapshot): LiveStreamSnapshot[] {
  const index = current.findIndex((stream) => stream.request_id === next.request_id);
  if (index >= 0) {
    const copy = [...current];
    copy[index] = next;
    return copy;
  }
  return [...current, next].slice(-30);
}

function appendLiveDelta(stream: LiveStreamSnapshot, delta: LiveStreamDelta): LiveStreamSnapshot {
  if (streamHasDelta(stream, delta)) {
    return stream;
  }
  return {
    ...stream,
    raw_output:
      delta.kind === "raw" || delta.kind === "error"
        ? stream.raw_output + delta.text
        : stream.raw_output,
    visible_output: delta.kind === "content" ? stream.visible_output + delta.text : stream.visible_output,
    reasoning_output: delta.kind === "reasoning" ? stream.reasoning_output + delta.text : stream.reasoning_output,
    events: [...stream.events, delta],
  };
}

function liveStreamOutput(stream: LiveStreamSnapshot, mode: "full" | "visible" | "reasoning" | "events" | "json") {
  if (mode === "full") return scrubToolLogNoise(stream.raw_output) || liveToolSummary(stream);
  if (mode === "visible") return scrubToolLogNoise(stream.visible_output) || bufferedVisibleText(stream) || liveToolSummary(stream);
  if (mode === "reasoning") return stream.reasoning_output;
  if (mode === "json") return JSON.stringify(stream, null, 2);
  return stream.events.map((event) => `[${formatLiveDateTime(event.timestamp)}] ${event.kind}\n${event.text}`).join("\n\n");
}

function liveStreamBlock(stream: LiveStreamSnapshot, mode: "full" | "visible" | "reasoning" | "events" | "json") {
  return [
    `# ${stream.source} - ${stream.model} - ${stream.status} - ${stream.request_id}`,
    `started: ${formatLiveDateTime(stream.started_at)}`,
    liveCompletedAt(stream) ? `completed: ${formatLiveDateTime(liveCompletedAt(stream)!)}` : "",
    "",
    liveStreamOutput(stream, mode),
  ]
    .filter(Boolean)
    .join("\n");
}

function liveCompletedAt(stream: LiveStreamSnapshot) {
  if (stream.status === "running") return null;
  return stream.events[stream.events.length - 1]?.timestamp ?? stream.started_at;
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

function emptyLiveStreamMessage(mode: string) {
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
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [apiRecoveryMessage, setApiRecoveryMessage] = useState<string | null>(null);
  const [apiRecoveryBusy, setApiRecoveryBusy] = useState(false);

  const model = useModel();
  const session = useSession();
  const chat = useChat(session.activeId);
  const context = useContext();
  const gpuStats = useGpuStats();

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
    const name = `Chat ${session.sessions.length + 1}`;
    await session.createSession(name);
  };

  // Auto-create a session when a model becomes loaded and no sessions exist yet.
  useEffect(() => {
    if (hasModel && session.sessions.length === 0 && !session.activeId) {
      session.createSession("Chat 1");
    }
  }, [hasModel, session.sessions.length, session.activeId]); // eslint-disable-line react-hooks/exhaustive-deps

  const tabs: Tab[] = ["chat", "models", "browse", "benchmark", "context", "logs", "debug", "settings"];

  return (
    <div className="flex h-screen flex-col" style={{ background: "var(--bg)", color: "var(--text-0)" }}>
      {/* Compact header */}
      <header
        className="flex shrink-0 items-center gap-4 px-4"
        style={{
          height: "44px",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface-1)",
        }}
      >
        {/* Logo */}
        <div className="flex items-center gap-2">
          <div
            className="flex h-6 w-6 items-center justify-center rounded text-[10px] font-bold"
            style={{
              background: "rgba(34,211,238,0.12)",
              border: "1px solid rgba(34,211,238,0.25)",
              color: "#22d3ee",
            }}
          >
            IB
          </div>
          <span className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>
            InferenceBridge
          </span>
        </div>

        {/* Divider */}
        <div className="h-4 w-px" style={{ background: "var(--border)" }} />

        {/* Nav tabs */}
        <nav className="flex items-center gap-0.5">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className="rounded px-3 py-1 text-sm transition"
              style={{
                background: activeTab === tab ? "rgba(34,211,238,0.14)" : "transparent",
                color: activeTab === tab ? "#a5f3fc" : "var(--text-1)",
                border: activeTab === tab ? "1px solid rgba(34,211,238,0.26)" : "1px solid transparent",
                boxShadow: activeTab === tab ? "inset 0 -2px 0 rgba(34,211,238,0.65)" : "none",
                cursor: "pointer",
              }}
              onMouseEnter={(e) => {
                if (activeTab !== tab) {
                  (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.05)";
                  (e.currentTarget as HTMLButtonElement).style.color = "var(--text-0)";
                }
              }}
              onMouseLeave={(e) => {
                if (activeTab !== tab) {
                  (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                  (e.currentTarget as HTMLButtonElement).style.color = "var(--text-1)";
                }
              }}
            >
              {TAB_LABELS[tab]}
            </button>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <NotificationCenter />

          <button
            onClick={() => setActiveTab("settings")}
            className="flex items-center gap-2 rounded px-3 py-1 text-xs transition"
            style={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text-1)",
              cursor: "pointer",
            }}
          >
            <span
              className={`h-2 w-2 rounded-full ${apiStarting || apiStopping ? "animate-pulse" : ""}`}
              style={{
                background:
                  apiRunning ? "#34d399" : apiStarting || apiStopping ? "#fde68a" : apiState === "Error" ? "#f87171" : "#6b7280",
              }}
            />
            <span>{apiRunning ? "Serve Running" : apiStopping ? "Serve Stopping" : apiStarting ? "Serve Starting" : apiState === "Error" ? "Serve Unreachable" : "Serve Off"}</span>
          </button>

          <button
            onClick={() => model.setApiServerRunning(!apiActive)}
            disabled={apiBusy}
            className="rounded px-3 py-1 text-xs font-medium transition"
            style={{
              background: "#22d3ee",
              color: "#0a0a0a",
              border: "none",
              cursor: apiBusy ? "wait" : "pointer",
              opacity: apiBusy ? 0.7 : 1,
            }}
          >
            {apiStopping ? "Stopping API..." : apiStarting ? "Starting API..." : apiActive ? "Stop API" : apiState === "Error" ? "Retry API" : "Start API"}
          </button>

          {chat.isStreaming && (
            <button
              onClick={() => setActiveTab("chat")}
              className="flex items-center gap-1.5 rounded px-2 py-1 text-xs transition"
              style={{
                background: "rgba(52,211,153,0.1)",
                border: "1px solid rgba(52,211,153,0.2)",
                color: "#34d399",
              }}
            >
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
              Generating
            </button>
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
              className="rounded px-3 py-1.5 text-xs font-medium transition"
              style={{
                background: "#22d3ee",
                color: "#0a0a0a",
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
            {apiPortOwner?.killable && (
              <button
                onClick={async () => {
                  setApiRecoveryBusy(true);
                  setApiRecoveryMessage(null);
                  try {
                    const message = await api.recoverApiPort(apiPortOwner.pid, configuredApiPort);
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
                    ? `Diagnose ghost owner (${apiPortOwner.pid})`
                    : `Kill ${apiPortOwner.kind === "llama-server" ? "llama-server" : "stale app"} (${apiPortOwner.pid})`}
              </button>
            )}
          </div>
        </div>
      )}

      {/* Main content */}
      <main className="flex min-h-0 flex-1 overflow-hidden">
        {/* Sidebar - chat only */}
        <div className={activeTab === "chat" ? "min-h-0 shrink-0" : "hidden"}>
          <Sidebar
            sessions={session.sessions}
            activeId={session.activeId}
            onSelect={session.setActiveId}
            onCreate={handleNewSession}
            onDelete={session.deleteSession}
          />
        </div>

        {/* Content area */}
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
              loadedModel={loadedModelName}
              loadedModelSupportsVision={loadedModelSupportsVision}
              loadedModelVisionStatusText={loadedModelVisionStatusText}
              onSend={chat.sendMessage}
              onStop={chat.stopGeneration}
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
            <div className="box-border h-full min-h-0 p-3">
              <ModelSelector
                models={model.models}
                loadedModel={model.processStatus?.model ?? null}
                previousModel={model.processStatus?.previous_model ?? null}
                processStatus={model.processStatus}
                settings={settings}
                error={model.error}
                isLoading={model.isLoading}
                loadProgress={model.loadProgress}
                onLoad={model.loadModel}
                onUnload={model.unloadModel}
                onSwap={model.swapModel}
                onSetApiServerRunning={model.setApiServerRunning}
                apiAction={model.apiAction}
                onScan={model.scanModels}
                onOpenSettings={() => setActiveTab("settings")}
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
            />
          </div>

          <div className={`h-full ${activeTab === "settings" ? "block" : "hidden"}`}>
            <SettingsPanel
              onSaved={setSettings}
              processStatus={model.processStatus}
              loadProgress={model.loadProgress}
              apiAction={model.apiAction}
              onSetApiServerRunning={model.setApiServerRunning}
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
  );
}

export default App;
