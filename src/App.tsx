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
import type { AppSettings, LiveStreamDelta, LiveStreamSnapshot, LoadProgress } from "./lib/types";
import * as api from "./lib/tauri";
import type { DownloadProgress } from "./lib/tauri";

type Tab = "chat" | "models" | "browse" | "context" | "logs" | "debug" | "settings";

const TAB_LABELS: Record<Tab, string> = {
  chat: "Chat",
  models: "Models",
  browse: "Browse",
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

  const pushNotice = (notice: Omit<UiNotice, "id">) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setNotices((current) => [...current, { ...notice, id }].slice(-5));
    if (notice.tone !== "error") {
      window.setTimeout(() => {
        setNotices((current) => current.filter((item) => item.id !== id));
      }, 6500);
    }
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
  const visibleStreams = showHistory ? streams : latest ? [latest] : [];
  const copyOutput = visibleStreams.map((stream) => liveStreamBlock(stream, mode)).join("\n\n");

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
            No stream output captured yet. Start a chat/API completion and this pane will fill live.
          </div>
        ) : (
          <div className="space-y-3">
            {visibleStreams.map((stream) => {
              const output = liveStreamOutput(stream, mode);
              const running = stream.status === "running";
              const completedAt = liveCompletedAt(stream);
              return (
                <article
                  key={stream.request_id}
                  className="overflow-hidden rounded-lg"
                  style={{ border: "1px solid var(--border)", background: "var(--surface-1)" }}
                >
                  <div
                    className="flex flex-wrap items-center gap-2 px-3 py-2 text-xs"
                    style={{ borderBottom: "1px solid var(--border)", color: "var(--text-1)" }}
                  >
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
                  <pre
                    className="max-h-[420px] overflow-auto whitespace-pre-wrap break-words px-3 py-3 font-mono text-xs leading-5"
                    style={{ color: output ? "var(--text-0)" : "var(--text-2)", background: "var(--bg)" }}
                  >
                    {output || emptyLiveStreamMessage(mode)}
                  </pre>
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
      <div
        className="flex h-12 shrink-0 items-center justify-between gap-3 px-4"
        style={{ borderBottom: "1px solid var(--border)", background: "var(--surface-1)" }}
      >
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold" style={{ color: "var(--text-0)" }}>
            Logs
          </div>
          <div className="truncate text-xs" style={{ color: "var(--text-2)" }}>
            Live LLM streams, raw transport, visible text, reasoning, events, and JSON snapshots.
          </div>
        </div>
        <div className="shrink-0 text-xs" style={{ color: "var(--text-1)" }}>
          {snapshot ? `${snapshot.source} - ${snapshot.status}` : "No active stream"}
        </div>
      </div>
      <div className="min-h-0 flex-1">
        <LiveStreamFeedV2 snapshot={snapshot} snapshots={snapshots} />
      </div>
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
  const apiRunning = apiState === "Running" && apiReachable;
  const apiStarting = apiState === "Starting";
  const apiPortOwner = model.processStatus?.api_port_owner ?? null;

  useEffect(() => {
    let cancelled = false;
    api
      .getSettings()
      .then((s) => { if (!cancelled) setSettings(s); })
      .catch(() => { if (!cancelled) setSettings(null); });
    return () => { cancelled = true; };
  }, []);

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

  const tabs: Tab[] = ["chat", "models", "browse", "context", "logs", "debug", "settings"];

  return (
    <div className="flex h-screen flex-col" style={{ background: "var(--bg)", color: "var(--text-0)" }}>
      <NotificationCenter />
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
                background: activeTab === tab ? "rgba(255,255,255,0.1)" : "transparent",
                color: activeTab === tab ? "var(--text-0)" : "var(--text-1)",
                border: "none",
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
              className={`h-2 w-2 rounded-full ${apiState === "Starting" ? "animate-pulse" : ""}`}
              style={{
                background:
                  apiRunning ? "#34d399" : apiStarting ? "#fde68a" : apiState === "Error" ? "#f87171" : "#6b7280",
              }}
            />
            <span>{apiRunning ? "Serve Running" : apiStarting ? "Serve Starting" : apiState === "Error" ? "Serve Unreachable" : "Serve Off"}</span>
          </button>

          <button
            onClick={() => model.setApiServerRunning(apiRunning || apiStarting ? false : true)}
            className="rounded px-3 py-1 text-xs font-medium transition"
            style={{
              background: "#22d3ee",
              color: "#0a0a0a",
              border: "none",
              cursor: "pointer",
            }}
          >
            {apiRunning || apiStarting ? "Stop API" : apiState === "Error" ? "Retry API" : "Start API"}
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
                  await api.killProcess(apiPortOwner.pid);
                  await model.refresh();
                  await model.setApiServerRunning(true);
                }}
                className="rounded px-3 py-1.5 text-xs font-medium transition"
                style={{
                  background: "rgba(251,191,36,0.14)",
                  color: "#fde68a",
                  border: "1px solid rgba(251,191,36,0.24)",
                  cursor: "pointer",
                }}
              >
                Kill {apiPortOwner.kind === "llama-server" ? "llama-server" : "stale app"} ({apiPortOwner.pid})
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

          <div className={`h-full overflow-y-auto ${activeTab === "models" ? "block" : "hidden"}`}>
            <div className="p-3">
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
              onOpenSettings={() => setActiveTab("settings")}
            />
          </div>

          <div className={`h-full ${activeTab === "settings" ? "block" : "hidden"}`}>
            <SettingsPanel
              onSaved={setSettings}
              processStatus={model.processStatus}
              loadProgress={model.loadProgress}
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
