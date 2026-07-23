import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  Box,
  ChevronDown,
  ChevronRight,
  Copy,
  LoaderCircle,
  Play,
  RefreshCw,
  Server,
  Settings,
  Square,
  Terminal,
  Trash2,
} from "lucide-react";
import type {
  ApiServerAction,
  DebugApiResponse,
  EffectiveProfileInfo,
  LoadProgress,
  LogEntry,
  ModelInfo,
  ProcessStatusInfo,
  RuntimeDoctorReport,
} from "../../lib/types";
import * as api from "../../lib/tauri";

type DebugTab = "server" | "api" | "doctor" | "profile" | "launch" | "docs" | "logs" | "prompt" | "trace";
type Example = {
  label: string;
  method: string;
  path: string;
  body?: string;
  description: string;
};

interface Props {
  apiUrl: string;
  processStatus: ProcessStatusInfo | null;
  loadProgress: LoadProgress | null;
  models: ModelInfo[];
  onSetApiServerRunning: (running: boolean) => Promise<void> | void;
  apiAction?: ApiServerAction;
  onOpenSettings: () => void;
  modelPickerOpen: boolean;
  onOpenModelPicker: (trigger: HTMLElement | null) => void;
}

const TABS: Array<{ key: DebugTab; label: string }> = [
  { key: "server", label: "Local Server" },
  { key: "api", label: "API Editor" },
  { key: "doctor", label: "Doctor" },
  { key: "profile", label: "Profile" },
  { key: "launch", label: "Launch" },
  { key: "docs", label: "Docs" },
  { key: "logs", label: "Logs" },
  { key: "prompt", label: "Raw Prompt" },
  { key: "trace", label: "Parse Trace" },
];

const SUPPORTED_ENDPOINTS = [
  { method: "GET", path: "/v1/health" },
  { method: "GET", path: "/v1/models" },
  { method: "POST", path: "/v1/models/load" },
  { method: "POST", path: "/v1/models/unload" },
  { method: "POST", path: "/v1/models/stats" },
  { method: "POST", path: "/v1/chat/completions" },
  { method: "POST", path: "/v1/responses" },
  { method: "POST", path: "/v1/completions" },
  { method: "POST", path: "/v1/messages" },
  { method: "POST", path: "/v1/embeddings" },
  { method: "POST", path: "/v1/rerank" },
  { method: "GET", path: "/v1/context/status" },
  { method: "GET", path: "/v1/runtime/status" },
  { method: "GET", path: "/v1/runtime/doctor" },
  { method: "GET", path: "/v1/metrics" },
  { method: "POST", path: "/v1/inference/cancel" },
] as const;

function Divider() {
  return <div style={{ height: "1px", background: "var(--border)" }} />;
}

function Panel({
  title,
  description,
  actions,
  children,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
        borderRadius: "8px",
        overflow: "hidden",
      }}
    >
      <div className="flex items-start justify-between gap-3 px-4 py-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            {title}
          </div>
          {description && (
            <p className="mt-1 text-sm" style={{ color: "var(--text-1)" }}>
              {description}
            </p>
          )}
        </div>
        {actions}
      </div>
      <Divider />
      {children}
    </section>
  );
}

function FieldRow({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-2 px-4 py-3 md:grid-cols-[140px_minmax(0,1fr)] md:items-start">
      <div className="pt-1 text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
        {label}
      </div>
      <div>{children}</div>
    </div>
  );
}

function ActionButton({
  label,
  onClick,
  primary,
  disabled,
}: {
  label: string;
  onClick: () => void;
  primary?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded-lg px-3 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-50"
      style={
        primary
          ? {
              background: "#f4f4f4",
              color: "#171717",
              border: "none",
              cursor: disabled ? "not-allowed" : "pointer",
            }
          : {
              background: "var(--surface-2)",
              color: "var(--text-1)",
              border: "1px solid var(--border)",
              cursor: disabled ? "not-allowed" : "pointer",
            }
      }
    >
      {label}
    </button>
  );
}

function CompactCode({
  value,
  emptyLabel,
}: {
  value: string;
  emptyLabel?: string;
}) {
  return (
    <pre
      className="overflow-x-auto rounded px-3 py-3 text-xs leading-6"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        color: "var(--text-0)",
        minHeight: "44px",
      }}
    >
      <code>{value || emptyLabel || "No data yet."}</code>
    </pre>
  );
}

function StatusDot({ running, starting, error }: { running: boolean; starting?: boolean; error?: boolean }) {
  const color = running ? "#34d399" : error ? "#f87171" : "#6b7280";
  return (
    <span
      className={starting ? "animate-pulse" : ""}
      style={{
        display: "inline-block",
        width: "8px",
        height: "8px",
        borderRadius: "999px",
        background: color,
      }}
    />
  );
}

function Metric({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-md px-3 py-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
        {label}
      </div>
      <div className={`mt-1 text-sm ${mono ? "break-all font-mono" : "font-medium"}`} style={{ color: "var(--text-0)" }}>
        {value}
      </div>
    </div>
  );
}

function ServerMetric({
  label,
  value,
  detail,
  tone,
  mono,
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "ok" | "warn" | "error" | "neutral";
  mono?: boolean;
}) {
  const color = tone === "ok" ? "#34d399" : tone === "warn" ? "#fbbf24" : tone === "error" ? "#f87171" : "var(--text-0)";
  return (
    <div className="min-w-0 rounded-md px-3 py-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
      <div className="text-[10px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>{label}</div>
      <div className={`mt-1 truncate text-sm font-semibold ${mono ? "font-mono" : ""}`} title={value} style={{ color }}>{value}</div>
      {detail && <div className="mt-1 truncate text-xs" title={detail} style={{ color: "var(--text-2)" }}>{detail}</div>}
    </div>
  );
}

function DocCard({ title, body }: { title: string; body: string[] }) {
  return (
    <div className="rounded px-3 py-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
      <div className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>
        {title}
      </div>
      <div className="mt-2 space-y-1 text-sm" style={{ color: "var(--text-1)" }}>
        {body.map((line) => (
          <div key={line}>{line}</div>
        ))}
      </div>
    </div>
  );
}

function prettyJson(value: string) {
  if (!value.trim()) return "";
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

type NormalizedTraceEvent = {
  kind?: string;
  text?: string | null;
  raw_span?: string | null;
  parser_stage?: string;
  decision?: string;
  tool_call?: {
    id?: string;
    namespace?: string | null;
    name?: string;
    arguments?: unknown;
    raw_span?: string | null;
    target_channel?: string | null;
  } | null;
};

type ParsedTrace = {
  parser_type?: string;
  tool_call_format?: string;
  think_tag_style?: string;
  visible_text?: string;
  reasoning_text?: string;
  normalized?: {
    events?: NormalizedTraceEvent[];
    decisions?: string[];
    visible_text?: string;
    reasoning_text?: string;
    tool_calls?: unknown[];
    parser_type?: string;
  };
};

function parseTraceJson(trace: string): ParsedTrace | null {
  if (!trace.trim()) return null;
  try {
    const parsed = JSON.parse(trace);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function stageTone(stage?: string) {
  if (stage === "strict_profile") return "#34d399";
  if (stage === "recovery") return "#fbbf24";
  if (stage === "fallback") return "#f87171";
  return "var(--text-2)";
}

function ParseTraceSummary({ trace }: { trace: ParsedTrace | null }) {
  const events = trace?.normalized?.events ?? [];
  const decisions = trace?.normalized?.decisions ?? [];
  const counts = events.reduce<Record<string, number>>((acc, event) => {
    const kind = event.kind ?? "unknown";
    acc[kind] = (acc[kind] ?? 0) + 1;
    return acc;
  }, {});

  if (!trace?.normalized) {
    return (
      <div className="rounded px-3 py-3 text-sm" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
        No normalized parser events captured yet.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid gap-2 md:grid-cols-4">
        <Metric label="Parser" value={trace.normalized.parser_type ?? trace.parser_type ?? "Unknown"} />
        <Metric label="Events" value={String(events.length)} />
        <Metric label="Tools" value={String(trace.normalized.tool_calls?.length ?? 0)} />
        <Metric label="Reasoning" value={trace.normalized.reasoning_text?.trim() ? "Captured" : "None"} />
      </div>

      {Object.keys(counts).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(counts).map(([kind, count]) => (
            <span key={kind} className="rounded px-2 py-1 text-xs" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
              {kind}: {count}
            </span>
          ))}
        </div>
      )}

      {decisions.length > 0 && (
        <div className="rounded px-3 py-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            Parser Decisions
          </div>
          <div className="space-y-1 text-xs" style={{ color: "var(--text-1)" }}>
            {decisions.map((decision, index) => (
              <div key={`${decision}-${index}`}>{decision}</div>
            ))}
          </div>
        </div>
      )}

      <div className="overflow-hidden rounded" style={{ border: "1px solid var(--border)" }}>
        {events.length === 0 ? (
          <div className="px-3 py-4 text-sm" style={{ color: "var(--text-2)" }}>No normalized events.</div>
        ) : (
          events.map((event, index) => (
            <div key={`${event.kind}-${index}`} className="px-3 py-3" style={{ borderTop: index === 0 ? "none" : "1px solid var(--border)" }}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-[0.12em]" style={{ color: "var(--text-0)" }}>{event.kind ?? "unknown"}</span>
                <span className="rounded px-2 py-0.5 text-[10px] font-semibold" style={{ background: "rgba(255,255,255,0.08)", color: stageTone(event.parser_stage) }}>
                  {event.parser_stage ?? "unlabeled"}
                </span>
                <span className="text-xs" style={{ color: "var(--text-2)" }}>{event.decision}</span>
              </div>
              {event.tool_call ? (
                <CompactCode
                  value={JSON.stringify(
                    {
                      namespace: event.tool_call.namespace,
                      name: event.tool_call.name,
                      arguments: event.tool_call.arguments,
                      target_channel: event.tool_call.target_channel,
                    },
                    null,
                    2
                  )}
                />
              ) : event.text ? (
                <div className="mt-2 whitespace-pre-wrap rounded px-3 py-2 text-xs leading-5" style={{ background: "var(--surface-2)", color: "var(--text-1)" }}>
                  {event.text}
                </div>
              ) : null}
              {event.raw_span && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs" style={{ color: "var(--text-2)" }}>Raw span</summary>
                  <CompactCode value={event.raw_span} />
                </details>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function parseHeaders(input: string): Record<string, string> {
  if (!input.trim()) return {};
  const parsed = JSON.parse(input);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Headers must be a JSON object.");
  }
  return Object.fromEntries(Object.entries(parsed).map(([key, value]) => [key, String(value)]));
}

function normalizeApiPath(path: string) {
  const trimmed = path.trim() || "/v1/health";
  const withSlash = trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
  return withSlash.startsWith("/v1") ? withSlash : `/v1${withSlash}`;
}

function buildPublicUrl(baseApiUrl: string, path: string) {
  const parsed = new URL(baseApiUrl);
  const normalizedPath = normalizeApiPath(path);
  return `${parsed.protocol}//${parsed.host}${normalizedPath}`;
}

function formatHeaders(headers: Headers | Array<[string, string]>) {
  if (headers instanceof Headers) {
    return Array.from(headers.entries()).map(([key, value]) => `${key}: ${value}`).join("\n");
  }
  return headers.map(([key, value]) => `${key}: ${value}`).join("\n");
}

function exampleList(activeModel: string | null, selectedModel: string | null): Example[] {
  const modelName = selectedModel || activeModel || "Your-Model.gguf";
  return [
    { label: "Health", method: "GET", path: "/v1/health", description: "Fast serve check for the public endpoint." },
    { label: "Models", method: "GET", path: "/v1/models", description: "List discovered models and mark the active one." },
    { label: "Model Detail", method: "GET", path: `/v1/models/${encodeURIComponent(modelName)}`, description: "Inspect one model entry by exact filename." },
    { label: "Model Stats", method: "POST", path: "/v1/models/stats", body: JSON.stringify({ model: modelName }, null, 2), description: "Poll load progress and backend stats for one model." },
    { label: "Context", method: "GET", path: "/v1/context/status", description: "See KV usage, context pressure, and compaction state." },
    { label: "Runtime", method: "GET", path: "/v1/runtime/status", description: "Inspect backend, launch, startup, and slot status." },
    { label: "Doctor", method: "GET", path: "/v1/runtime/doctor", description: "Probe managed llama.cpp, OpenAI-compatible providers, Ollama, and external local providers." },
    { label: "Profile", method: "GET", path: `/v1/debug/profile?model=${encodeURIComponent(modelName)}`, description: "Show the effective profile after detection and overrides." },
    { label: "Load Model", method: "POST", path: "/v1/models/load", body: JSON.stringify({ model: modelName }, null, 2), description: "Start loading a model in the background." },
    { label: "Unload", method: "POST", path: "/v1/models/unload", body: "{}", description: "Unload the active backend model." },
    { label: "Chat", method: "POST", path: "/v1/chat/completions", body: JSON.stringify({ model: modelName, messages: [{ role: "user", content: "Reply with exactly: InferenceBridge OK" }], temperature: 0.2, stream: false }, null, 2), description: "Run a standard OpenAI-compatible chat completion request." },
  ];
}

export function DebugInspector({
  apiUrl,
  processStatus,
  loadProgress,
  models,
  onSetApiServerRunning,
  apiAction = null,
  onOpenSettings,
  modelPickerOpen,
  onOpenModelPicker,
}: Props) {
  const [activeTab, setActiveTab] = useState<DebugTab>("server");
  const [method, setMethod] = useState("GET");
  const [path, setPath] = useState("/v1/health");
  const [headersText, setHeadersText] = useState(JSON.stringify({ Accept: "application/json", "Content-Type": "application/json" }, null, 2));
  const [bodyText, setBodyText] = useState("");
  const [response, setResponse] = useState<DebugApiResponse | null>(null);
  const [sending, setSending] = useState(false);
  const [requestDurationMs, setRequestDurationMs] = useState<number | null>(null);
  const [recentRequests, setRecentRequests] = useState<Array<{ method: string; path: string; status: number; at: string }>>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logQuery, setLogQuery] = useState("");
  const [logLevel, setLogLevel] = useState("ALL");
  const [rawPrompt, setRawPrompt] = useState("");
  const [parseTrace, setParseTrace] = useState("");
  const [launchPreview, setLaunchPreview] = useState("");
  const [effectiveProfile, setEffectiveProfile] = useState<EffectiveProfileInfo | null>(null);
  const [runtimeDoctor, setRuntimeDoctor] = useState<RuntimeDoctorReport | null>(null);
  const [selectedProfileModel, setSelectedProfileModel] = useState("");
  const [selectedServerModel, setSelectedServerModel] = useState<string | null>(null);
  const [serverEndpointsOpen, setServerEndpointsOpen] = useState(false);
  const [serverLogsOpen, setServerLogsOpen] = useState(true);
  const [workspaceVisible, setWorkspaceVisible] = useState(false);
  const [autoRefreshLogs, setAutoRefreshLogs] = useState(true);
  const [tabError, setTabError] = useState<string | null>(null);
  const workspaceRootRef = useRef<HTMLDivElement | null>(null);

  const activeModel = processStatus?.model ?? null;
  const serveState = processStatus?.api_state ?? "Idle";
  const serveReachable = processStatus?.api_reachable ?? false;
  const modelTransition =
    loadProgress && !loadProgress.done
      ? loadProgress
      : processStatus?.model_load_progress ?? null;
  const modelTransitionActive =
    (!!modelTransition && !modelTransition.done) ||
    ["Starting", "Stopping"].includes(processStatus?.state ?? "Idle") ||
    ["Loading", "Swapping", "Unloading"].includes(
      processStatus?.model_load_state ?? "Idle"
  );
  const serveStopping = apiAction === "stopping" || serveState === "Stopping";
  const serveStarting = apiAction === "starting" || serveState === "Starting";
  const serveRunning = (serveState === "Running" && serveReachable) || (serveState === "Running" && !serveStopping);
  const serveActive = serveRunning || serveStarting || serveStopping || serveReachable;
  const serveBusy = serveStarting || serveStopping;
  const selectedModel = selectedProfileModel || activeModel || models[0]?.filename || null;
  const activeModelInfo = useMemo(
    () => (activeModel ? models.find((model) => model.filename === activeModel) ?? null : null),
    [activeModel, models]
  );
  const selectedServerModelInfo = useMemo(
    () => (selectedServerModel ? models.find((model) => model.filename === selectedServerModel) ?? null : null),
    [models, selectedServerModel]
  );
  const examples = useMemo(() => exampleList(activeModel, selectedModel), [activeModel, selectedModel]);
  const parsedTrace = useMemo(() => parseTraceJson(parseTrace), [parseTrace]);
  const logsSurfaceActive =
    workspaceVisible &&
    (activeTab === "logs" || (activeTab === "server" && serverLogsOpen));

  useEffect(() => {
    const node = workspaceRootRef.current;
    if (!node) return;
    const observer = new IntersectionObserver(([entry]) => {
      setWorkspaceVisible(entry.isIntersecting);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setSelectedServerModel(activeModel);
  }, [activeModel]);

  useEffect(() => {
    if (!selectedProfileModel && (activeModel || models[0]?.filename)) {
      setSelectedProfileModel(activeModel || models[0]?.filename || "");
    }
  }, [activeModel, models, selectedProfileModel]);

  const refreshLogs = async () => {
    try {
      setLogs(await api.getLogs(400));
      setTabError(null);
    } catch (error) {
      setTabError(String(error));
    }
  };

  const refreshPrompt = async () => {
    try {
      setRawPrompt(await api.getRawPrompt());
      setTabError(null);
    } catch (error) {
      setTabError(String(error));
    }
  };

  const refreshTrace = async () => {
    try {
      setParseTrace(await api.getParseTrace());
      setTabError(null);
    } catch (error) {
      setTabError(String(error));
    }
  };

  const refreshLaunch = async () => {
    try {
      setLaunchPreview(await api.getLaunchPreview());
      setTabError(null);
    } catch (error) {
      setLaunchPreview("");
      setTabError(String(error));
    }
  };

  const refreshProfile = async (modelName?: string) => {
    try {
      setEffectiveProfile(await api.getEffectiveProfile(modelName || undefined));
      setTabError(null);
    } catch (error) {
      setEffectiveProfile(null);
      setTabError(String(error));
    }
  };

  const refreshDoctor = async () => {
    try {
      setRuntimeDoctor(await api.getRuntimeDoctor());
      setTabError(null);
    } catch (error) {
      setRuntimeDoctor(null);
      setTabError(String(error));
    }
  };

  useEffect(() => {
    if (logsSurfaceActive) refreshLogs();
    if (activeTab === "doctor") refreshDoctor();
    if (activeTab === "prompt") refreshPrompt();
    if (activeTab === "trace") refreshTrace();
    if (activeTab === "launch") refreshLaunch();
    if (activeTab === "profile") refreshProfile(selectedProfileModel || activeModel || undefined);
  }, [activeTab, logsSurfaceActive]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (activeTab !== "profile") return;
    refreshProfile(selectedProfileModel || activeModel || undefined);
  }, [selectedProfileModel, activeModel]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!logsSurfaceActive || !autoRefreshLogs) return;
    const timer = window.setInterval(() => {
      refreshLogs();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [logsSurfaceActive, autoRefreshLogs]); // eslint-disable-line react-hooks/exhaustive-deps

  // Live: llama-server pushes a `llama-server-log` event per output line.
  // Coalesce bursts into a debounced refresh so the console tracks the server
  // in ~real time without refetching the buffer on every single line. The 3s
  // poll above stays as a backstop.
  useEffect(() => {
    if (!logsSurfaceActive || !autoRefreshLogs) return;
    let timer: number | undefined;
    const unlisten = listen("llama-server-log", () => {
      if (timer != null) return;
      timer = window.setTimeout(() => {
        timer = undefined;
        refreshLogs();
      }, 250);
    });
    return () => {
      if (timer != null) window.clearTimeout(timer);
      unlisten.then((fn) => fn());
    };
  }, [logsSurfaceActive, autoRefreshLogs]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadExample = (example: Example) => {
    setMethod(example.method);
    setPath(example.path);
    setBodyText(example.body ?? "");
  };

  const openSupportedEndpoint = (endpoint: (typeof SUPPORTED_ENDPOINTS)[number]) => {
    const matchingExample = examples.find(
      (example) => example.method === endpoint.method && example.path === endpoint.path
    );
    if (matchingExample) {
      loadExample(matchingExample);
    } else {
      setMethod(endpoint.method);
      setPath(endpoint.path);
      setBodyText("");
    }
    setActiveTab("api");
  };

  const sendRequest = async () => {
    setSending(true);
    setTabError(null);
    const started = performance.now();
    const normalizedPath = normalizeApiPath(path);

    try {
      parseHeaders(headersText);
    } catch (error) {
      setSending(false);
      setResponse({ status: 0, headers: [], body: String(error), transport: "editor" });
      return;
    }

    try {
      const publicUrl = buildPublicUrl(apiUrl, normalizedPath);
      const responseInit = await fetch(publicUrl, {
        method,
        headers: parseHeaders(headersText),
        body: method === "GET" ? undefined : bodyText || undefined,
      });
      const text = await responseInit.text();
      const duration = Math.round(performance.now() - started);
      setRequestDurationMs(duration);
      setResponse({ status: responseInit.status, headers: Array.from(responseInit.headers.entries()), body: prettyJson(text), transport: "public" });
      setRecentRequests((current) => [{ method, path: normalizedPath, status: responseInit.status, at: new Date().toLocaleTimeString() }, ...current].slice(0, 8));
    } catch {
      const direct = await api.debugApiRequest({ method, path: normalizedPath, body: method === "GET" ? undefined : bodyText || undefined });
      const duration = Math.round(performance.now() - started);
      setRequestDurationMs(duration);
      setResponse({ ...direct, body: prettyJson(direct.body) });
      setRecentRequests((current) => [{ method, path: normalizedPath, status: direct.status, at: new Date().toLocaleTimeString() }, ...current].slice(0, 8));
    } finally {
      setSending(false);
    }
  };

  const copyCurl = async () => {
    const normalizedPath = normalizeApiPath(path);
    const publicUrl = buildPublicUrl(apiUrl, normalizedPath);
    let curl = `curl -X ${method} "${publicUrl}"`;
    try {
      const headers = parseHeaders(headersText);
      for (const [key, value] of Object.entries(headers)) {
        curl += ` -H "${key}: ${value.replace(/"/g, '\\"')}"`;
      }
    } catch {
      // ignore invalid headers here; send path will surface the error
    }
    if (method !== "GET" && bodyText.trim()) {
      curl += ` -d "${bodyText.replace(/"/g, '\\"').replace(/\n/g, "\\n")}"`;
    }
    await navigator.clipboard.writeText(curl);
  };

  const filteredLogs = logs.filter((entry) => {
    const matchesLevel = logLevel === "ALL" || entry.level.toUpperCase() === logLevel;
    const haystack = `${entry.target} ${entry.message}`.toLowerCase();
    const matchesQuery = !logQuery.trim() || haystack.includes(logQuery.trim().toLowerCase());
    return matchesLevel && matchesQuery;
  });

  const serveDisplayState = serveRunning
    ? "Running"
    : serveStopping
      ? "Stopping"
      : modelTransitionActive
        ? processStatus?.model_load_state ?? "Loading"
        : serveStarting
          ? "Starting"
          : serveState === "Error"
            ? "Unreachable"
            : "Stopped";
  const serveTone = serveRunning
    ? "#34d399"
    : serveStarting || serveStopping || modelTransitionActive
      ? "#fbbf24"
      : serveState === "Error"
        ? "#f87171"
        : "var(--text-2)";
  const selectedServerModelIsActive =
    !!selectedServerModel && selectedServerModel === activeModel;
  const selectedContext =
    processStatus?.last_launch_preview?.context_size ??
    selectedServerModelInfo?.context_window ??
    null;
  const selectedCapabilities = selectedServerModelInfo
    ? [
        selectedServerModelInfo.supports_tools ? "Tools" : null,
        selectedServerModelInfo.supports_reasoning ? "Reasoning" : null,
        selectedServerModelInfo.supports_vision ? "Vision" : null,
      ].filter((capability): capability is string => capability != null)
    : [];

  if (activeTab === "server") {
    return (
      <div ref={workspaceRootRef} className="flex h-full min-h-0 flex-col overflow-hidden" style={{ background: "var(--bg)" }}>
        <header
          className="flex min-h-[46px] shrink-0 flex-wrap items-center gap-2 px-3 py-1.5"
          style={{ background: "var(--surface-1)", borderBottom: "1px solid var(--border)" }}
        >
          <button
            type="button"
            onClick={() => onSetApiServerRunning(!serveActive)}
            disabled={serveBusy}
            className="flex h-8 shrink-0 items-center gap-2 rounded-lg px-2.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-55"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: serveTone }}
            title={processStatus?.api_error ?? `API server ${serveDisplayState.toLowerCase()}`}
          >
            {serveBusy ? (
              <LoaderCircle size={14} className="animate-spin" />
            ) : serveActive ? (
              <Square size={12} fill="currentColor" />
            ) : (
              <Play size={14} fill="currentColor" />
            )}
            <span>Status: {serveDisplayState}</span>
            <span
              className="relative h-4 w-7 rounded-full"
              style={{ background: serveActive ? "rgba(52,211,153,0.28)" : "rgba(255,255,255,0.10)" }}
              aria-hidden="true"
            >
              <span
                className="absolute top-0.5 h-3 w-3 rounded-full transition-all"
                style={{ left: serveActive ? "14px" : "2px", background: serveActive ? "#34d399" : "var(--text-2)" }}
              />
            </span>
          </button>

          <button
            type="button"
            onClick={onOpenSettings}
            className="flex h-8 shrink-0 items-center gap-2 rounded-lg px-2.5 text-xs font-medium transition hover:bg-white/5"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}
          >
            <Settings size={14} />
            Server Settings
          </button>

          <div className="ml-auto flex min-w-0 items-center gap-2">
            <span className="hidden text-[11px] font-medium lg:inline" style={{ color: serveReachable ? "#34d399" : "var(--text-2)" }}>
              {serveReachable ? "Reachable at" : "Configured at"}
            </span>
            <code className="max-w-[260px] truncate rounded-md px-2 py-1 text-[11px]" title={apiUrl} style={{ background: "var(--surface-2)", color: "var(--text-0)" }}>
              {apiUrl}
            </code>
            <button
              type="button"
              onClick={() => navigator.clipboard.writeText(apiUrl)}
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition hover:bg-white/5"
              style={{ border: "1px solid var(--border)", color: "var(--text-1)" }}
              aria-label="Copy API URL"
              title="Copy API URL"
            >
              <Copy size={14} />
            </button>
            <button
              type="button"
              onClick={(event) => onOpenModelPicker(event.currentTarget)}
              aria-haspopup="dialog"
              aria-controls="rich-model-picker"
              aria-expanded={modelPickerOpen}
              className="flex h-8 shrink-0 items-center gap-2 rounded-lg px-3 text-xs font-semibold transition hover:bg-white"
              style={{ background: "var(--accent)", color: "var(--accent-contrast)", border: "1px solid transparent" }}
            >
              <ChevronDown size={14} />
              Choose model
            </button>
          </div>
        </header>

        <nav
          className="flex shrink-0 items-center gap-1 overflow-x-auto px-2 py-1.5"
          style={{ background: "var(--surface-1)", borderBottom: "1px solid var(--border)" }}
          aria-label="Developer tools"
          role="tablist"
        >
          {TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.key}
              onClick={() => setActiveTab(tab.key)}
              className="shrink-0 rounded-md px-2.5 py-1 text-xs font-medium transition"
              style={{
                background: activeTab === tab.key ? "rgba(255,255,255,0.10)" : "transparent",
                color: activeTab === tab.key ? "var(--text-0)" : "var(--text-1)",
                border: activeTab === tab.key ? "1px solid var(--border-mid)" : "1px solid transparent",
              }}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {processStatus?.api_error && !modelTransitionActive && (
          <div
            className="shrink-0 px-3 py-2 text-xs"
            style={{
              background: serveReachable ? "rgba(251,191,36,0.08)" : "rgba(248,113,113,0.08)",
              borderBottom: `1px solid ${serveReachable ? "rgba(251,191,36,0.20)" : "rgba(248,113,113,0.20)"}`,
              color: serveReachable ? "#fde68a" : "#fca5a5",
            }}
          >
            {processStatus.api_error}
          </div>
        )}
        {tabError && (
          <div className="shrink-0 px-3 py-2 text-xs" style={{ background: "rgba(248,113,113,0.08)", borderBottom: "1px solid rgba(248,113,113,0.20)", color: "#fca5a5" }}>
            {tabError}
          </div>
        )}

        <div className="grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[minmax(0,1fr)_280px]">
          <section className="flex min-h-0 flex-col overflow-hidden">
            <div className="flex h-10 shrink-0 items-center justify-between px-3" style={{ borderBottom: "1px solid var(--border)" }}>
              <div className="flex items-center gap-2 text-xs font-semibold" style={{ color: "var(--text-0)" }}>
                <Server size={14} />
                Loaded model
                <span className="rounded-full px-1.5 py-0.5 text-[10px]" style={{ background: "var(--surface-2)", color: "var(--text-2)" }}>
                  {activeModel ? 1 : 0}
                </span>
              </div>
              <span className="text-[11px]" style={{ color: "var(--text-2)" }}>
                {processStatus?.backend ?? "Runtime idle"}
              </span>
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto p-3">
              {activeModel ? (
                <button
                  type="button"
                  onClick={() => setSelectedServerModel(activeModel)}
                  className="w-full rounded-lg p-3 text-left transition"
                  style={{
                    background: selectedServerModelIsActive ? "rgba(255,255,255,0.075)" : "var(--surface-1)",
                    border: selectedServerModelIsActive ? "1px solid var(--border-mid)" : "1px solid var(--border)",
                  }}
                  aria-pressed={selectedServerModelIsActive}
                >
                  <div className="flex min-w-0 items-start gap-3">
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg" style={{ background: "var(--surface-2)", color: serveRunning ? "#34d399" : "var(--text-1)" }}>
                      <Box size={17} />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex min-w-0 flex-wrap items-center gap-2">
                        <span className="min-w-0 truncate text-sm font-semibold" title={activeModel} style={{ color: "var(--text-0)" }}>
                          {activeModel}
                        </span>
                        <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em]" style={{ background: "rgba(52,211,153,0.10)", border: "1px solid rgba(52,211,153,0.20)", color: "#34d399" }}>
                          Active
                        </span>
                      </div>
                      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]" style={{ color: "var(--text-2)" }}>
                        <span>{activeModelInfo?.family || "Loaded via API"}</span>
                        {activeModelInfo?.quant && <span>{activeModelInfo.quant}</span>}
                        {!!activeModelInfo?.size_gb && <span>{activeModelInfo.size_gb.toFixed(1)} GB</span>}
                        {processStatus?.last_launch_preview?.context_size && (
                          <span>{processStatus.last_launch_preview.context_size.toLocaleString()} ctx</span>
                        )}
                      </div>
                      {activeModelInfo && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {activeModelInfo.supports_tools && <span className="rounded px-1.5 py-0.5 text-[10px]" style={{ background: "var(--surface-2)", color: "var(--text-1)" }}>Tools</span>}
                          {activeModelInfo.supports_reasoning && <span className="rounded px-1.5 py-0.5 text-[10px]" style={{ background: "var(--surface-2)", color: "var(--text-1)" }}>Reasoning</span>}
                          {activeModelInfo.supports_vision && <span className="rounded px-1.5 py-0.5 text-[10px]" style={{ background: "var(--surface-2)", color: "var(--text-1)" }}>Vision</span>}
                        </div>
                      )}
                    </div>
                    <div className="shrink-0 text-right text-[11px]" style={{ color: "var(--text-2)" }}>
                      <div style={{ color: serveTone }}>{processStatus?.model_load_state ?? "Loaded"}</div>
                      <div className="mt-1">{processStatus?.active_requests ?? 0} active</div>
                    </div>
                  </div>
                  {modelTransitionActive && modelTransition && (
                    <div className="mt-3">
                      <div className="mb-1 flex items-center justify-between gap-3 text-[10px]" style={{ color: "#fde68a" }}>
                        <span className="truncate">{modelTransition.message}</span>
                        <span>{Math.round(modelTransition.progress * 100)}%</span>
                      </div>
                      <div className="h-1 overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.08)" }}>
                        <div className="h-full rounded-full bg-amber-400" style={{ width: `${Math.max(2, Math.min(100, modelTransition.progress * 100))}%` }} />
                      </div>
                    </div>
                  )}
                </button>
              ) : (
                <div className="flex h-full min-h-[180px] flex-col items-center justify-center text-center">
                  <span className="flex h-12 w-12 items-center justify-center rounded-xl" style={{ background: "var(--surface-1)", border: "1px solid var(--border)", color: "var(--text-2)" }}>
                    {modelTransitionActive ? <LoaderCircle size={21} className="animate-spin" /> : <Box size={21} />}
                  </span>
                  <p className="mt-3 text-sm font-semibold" style={{ color: "var(--text-0)" }}>
                    {modelTransitionActive ? processStatus?.model_load_state ?? "Loading model" : "No model loaded"}
                  </p>
                  <p className="mt-1 max-w-[360px] text-xs leading-5" style={{ color: "var(--text-2)" }}>
                    {modelTransitionActive
                      ? modelTransition?.message ?? "Preparing the local runtime."
                      : "Load a local model to serve it through the OpenAI-compatible API."}
                  </p>
                  {!modelTransitionActive && (
                    <button
                      type="button"
                      onClick={(event) => onOpenModelPicker(event.currentTarget)}
                      aria-haspopup="dialog"
                      aria-controls="rich-model-picker"
                      aria-expanded={modelPickerOpen}
                      className="mt-4 flex h-8 items-center gap-2 rounded-lg px-3 text-xs font-semibold transition hover:bg-white"
                      style={{ background: "var(--accent)", color: "var(--accent-contrast)" }}
                    >
                      <ChevronDown size={14} />
                      Choose a model
                    </button>
                  )}
                </div>
              )}
            </div>
          </section>

          <aside className="hidden min-h-0 flex-col overflow-hidden lg:flex" style={{ background: "var(--surface-1)", borderLeft: "1px solid var(--border)" }}>
            <div className="flex h-10 shrink-0 items-center px-3 text-xs font-semibold" style={{ borderBottom: "1px solid var(--border)", color: "var(--text-0)" }}>
              Model & runtime
            </div>
            {selectedServerModel ? (
              <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
                <div className="border-b py-2" style={{ borderColor: "var(--border)" }}>
                  <p className="break-words text-xs font-semibold leading-5" style={{ color: "var(--text-0)" }}>{selectedServerModel}</p>
                  <p className="mt-1 text-[11px]" style={{ color: selectedServerModelIsActive ? "#34d399" : "var(--text-2)" }}>
                    {selectedServerModelIsActive ? "Active runtime model" : "Not active"}
                  </p>
                </div>
                {([
                  { label: "Family", value: selectedServerModelInfo?.family || "Unknown" },
                  { label: "Quant", value: selectedServerModelInfo?.quant || "Unknown" },
                  { label: "Size on disk", value: selectedServerModelInfo?.size_gb ? `${selectedServerModelInfo.size_gb.toFixed(1)} GB` : "Unknown" },
                  { label: "Provider", value: selectedServerModelInfo?.provider_name || "Managed runtime" },
                  { label: "Backend", value: processStatus?.backend || "Unknown" },
                  { label: "Context", value: selectedContext ? selectedContext.toLocaleString() : "Unknown" },
                  { label: "Runtime", value: processStatus?.state || "Idle" },
                  { label: "Requests", value: `${processStatus?.active_requests ?? 0} active / ${processStatus?.queued_requests ?? 0} queued` },
                  { label: "Slots", value: `${processStatus?.slot_count ?? processStatus?.parallel_slots ?? 0}` },
                  { label: "Startup", value: processStatus?.startup_duration_ms != null ? `${processStatus.startup_duration_ms} ms` : "Unknown" },
                  { label: "llama.cpp", value: processStatus?.server_version || "Unknown" },
                ] as Array<{ label: string; value: string }>).map((item) => (
                  <div key={item.label} className="grid grid-cols-[88px_minmax(0,1fr)] gap-2 border-b py-2 text-[11px]" style={{ borderColor: "var(--border)" }}>
                    <span style={{ color: "var(--text-2)" }}>{item.label}</span>
                    <span className="min-w-0 break-words text-right" title={item.value} style={{ color: "var(--text-0)" }}>{item.value}</span>
                  </div>
                ))}
                <div className="py-2">
                  <div className="text-[11px]" style={{ color: "var(--text-2)" }}>Capabilities</div>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {selectedCapabilities.length > 0 ? selectedCapabilities.map((capability) => (
                      <span key={capability} className="rounded px-1.5 py-0.5 text-[10px]" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
                        {capability}
                      </span>
                    )) : <span className="text-[11px]" style={{ color: "var(--text-3)" }}>None reported</span>}
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex min-h-0 flex-1 items-center justify-center px-4 text-center text-xs" style={{ color: "var(--text-2)" }}>
                No model selected
              </div>
            )}
          </aside>
        </div>

        <section className="shrink-0" style={{ background: "var(--surface-1)", borderTop: "1px solid var(--border)" }}>
          <button
            type="button"
            onClick={() => setServerEndpointsOpen((open) => !open)}
            className="flex h-10 w-full items-center gap-2 px-3 text-left text-xs font-medium transition hover:bg-white/[0.03]"
            style={{ color: "var(--text-1)" }}
            aria-expanded={serverEndpointsOpen}
          >
            {serverEndpointsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <span className="font-semibold" style={{ color: "var(--text-0)" }}>Supported endpoints</span>
            <span className="rounded-full px-1.5 py-0.5 text-[10px]" style={{ background: "var(--surface-2)", color: "var(--text-2)" }}>
              {SUPPORTED_ENDPOINTS.length}
            </span>
            <span className="ml-auto text-[11px]" style={{ color: "var(--text-3)" }}>Select a route to open it in API Editor</span>
          </button>
          {serverEndpointsOpen && (
            <div className="grid max-h-[136px] grid-cols-1 gap-1 overflow-y-auto border-t p-2 sm:grid-cols-2 xl:grid-cols-3" style={{ borderColor: "var(--border)" }}>
              {SUPPORTED_ENDPOINTS.map((endpoint) => (
                <button
                  key={`${endpoint.method}-${endpoint.path}`}
                  type="button"
                  onClick={() => openSupportedEndpoint(endpoint)}
                  className="flex min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-left transition hover:bg-white/5"
                  style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
                >
                  <span className="w-9 shrink-0 text-[10px] font-bold" style={{ color: endpoint.method === "GET" ? "#34d399" : "#60a5fa" }}>{endpoint.method}</span>
                  <code className="min-w-0 truncate text-[10px]" title={endpoint.path} style={{ color: "var(--text-1)" }}>{endpoint.path}</code>
                </button>
              ))}
            </div>
          )}
        </section>

        <section
          className={`ib-api-log-dock flex shrink-0 flex-col overflow-hidden ${serverLogsOpen ? "is-open" : ""}`}
          style={{ background: "#171717", borderTop: "1px solid var(--border)" }}
        >
          <div className="flex h-10 shrink-0 items-center gap-2 px-3">
            <button
              type="button"
              onClick={() => setServerLogsOpen((open) => !open)}
              className="flex min-w-0 items-center gap-2 text-xs font-semibold"
              style={{ color: "var(--text-0)" }}
              aria-expanded={serverLogsOpen}
            >
              {serverLogsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              <Terminal size={14} />
              Developer logs
              <span className="text-[10px] font-normal" style={{ color: "var(--text-3)" }}>{filteredLogs.length}</span>
            </button>
            <div className="ml-auto flex items-center gap-1">
              <button
                type="button"
                onClick={() => setAutoRefreshLogs((current) => !current)}
                className="rounded-md px-2 py-1 text-[10px] transition hover:bg-white/5"
                style={{ color: autoRefreshLogs ? "#34d399" : "var(--text-2)" }}
                title="Toggle automatic log refresh"
              >
                Auto {autoRefreshLogs ? "on" : "off"}
              </button>
              <button type="button" onClick={refreshLogs} className="flex h-7 w-7 items-center justify-center rounded-md transition hover:bg-white/5" style={{ color: "var(--text-2)" }} aria-label="Refresh developer logs" title="Refresh">
                <RefreshCw size={13} />
              </button>
              <button
                type="button"
                onClick={async () => {
                  await api.clearLogs();
                  await refreshLogs();
                }}
                className="flex h-7 w-7 items-center justify-center rounded-md transition hover:bg-white/5"
                style={{ color: "var(--text-2)" }}
                aria-label="Clear developer logs"
                title="Clear"
              >
                <Trash2 size={13} />
              </button>
            </div>
          </div>
          {serverLogsOpen && (
            <>
              <div className="flex shrink-0 items-center gap-2 border-y px-3 py-1.5" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
                <select
                  value={logLevel}
                  onChange={(event) => setLogLevel(event.target.value)}
                  className="h-7 rounded-md px-2 text-[10px] outline-none"
                  style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)" }}
                >
                  <option value="ALL">ALL</option>
                  <option value="INFO">INFO</option>
                  <option value="WARN">WARN</option>
                  <option value="ERROR">ERROR</option>
                  <option value="DEBUG">DEBUG</option>
                </select>
                <input
                  value={logQuery}
                  onChange={(event) => setLogQuery(event.target.value)}
                  placeholder="Filter logs..."
                  className="h-7 min-w-0 flex-1 rounded-md px-2 text-[10px] outline-none"
                  style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
                />
              </div>
              <div className="min-h-0 flex-1 overflow-auto px-3 py-2 font-mono text-[10px] leading-[1.65]">
                {filteredLogs.length === 0 ? (
                  <div style={{ color: "var(--text-3)" }}>No developer logs match the current filter.</div>
                ) : filteredLogs.map((entry, index) => (
                  <div key={`${entry.timestamp}-${index}`} className="grid grid-cols-[142px_46px_minmax(110px,180px)_minmax(0,1fr)] gap-2 border-b py-0.5" style={{ borderColor: "rgba(255,255,255,0.035)" }}>
                    <span className="truncate" title={entry.timestamp} style={{ color: "var(--text-3)" }}>{entry.timestamp}</span>
                    <span style={{ color: entry.level === "ERROR" ? "#f87171" : entry.level === "WARN" ? "#fbbf24" : entry.level === "INFO" ? "#34d399" : "#60a5fa" }}>[{entry.level}]</span>
                    <span className="truncate" title={entry.target} style={{ color: "var(--text-2)" }}>{entry.target}</span>
                    <span className="min-w-0 whitespace-pre-wrap break-words" style={{ color: "var(--text-1)" }}>{entry.message}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      </div>
    );
  }

  return (
    <div ref={workspaceRootRef} className="h-full overflow-y-auto">
      <div className="flex flex-col gap-3 p-3">
        <Panel
          title="Developer Console"
          description="Manage the embedded OpenAI-compatible server and inspect the public API used by external tools."
          actions={
            <div className="flex flex-wrap items-center gap-2">
              <ActionButton label="Copy URL" onClick={() => navigator.clipboard.writeText(apiUrl)} />
              <ActionButton label="Server Settings" onClick={onOpenSettings} />
              <ActionButton
                label={serveStopping ? "Stopping API..." : serveStarting ? "Starting API..." : serveActive ? "Stop API" : serveState === "Error" ? "Retry API" : "Start API"}
                onClick={() => onSetApiServerRunning(!serveActive)}
                disabled={serveBusy}
                primary
              />
            </div>
          }
        >
          <div className="grid gap-3 px-4 py-3 xl:grid-cols-[1.25fr_1.45fr_1.6fr_1fr]">
            <ServerMetric
              label="Server"
              value={
                serveRunning
                  ? "Running"
                  : serveStopping
                    ? "Stopping"
                    : modelTransitionActive
                      ? processStatus?.model_load_state ?? "Loading"
                      : serveStarting
                        ? "Starting"
                        : serveState === "Error"
                          ? "Unreachable"
                          : "Stopped"
              }
              detail={
                modelTransitionActive
                  ? modelTransition?.message ?? "Model transition in progress"
                  : serveRunning
                    ? "Public API is reachable"
                    : serveState === "Error"
                      ? processStatus?.api_error ?? "API is not bound"
                      : "External clients are not being served"
              }
              tone={serveRunning ? "ok" : serveStarting || serveStopping || modelTransitionActive ? "warn" : serveState === "Error" ? "error" : "neutral"}
            />
            <ServerMetric label={serveReachable ? "Reachable at" : "Configured at"} value={apiUrl} tone={serveReachable ? "ok" : "neutral"} mono />
            <ServerMetric label="Active model" value={activeModel ?? "No model loaded"} detail={processStatus?.last_launch_preview?.context_size ? `${processStatus.last_launch_preview.context_size.toLocaleString()} ctx` : undefined} />
            <ServerMetric
              label="Runtime"
              value={processStatus?.backend ?? "Unknown backend"}
              detail={processStatus?.startup_duration_ms != null ? `${processStatus.startup_duration_ms} ms startup` : undefined}
            />
          </div>
          <div className="hidden">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold" style={{ color: serveRunning ? "#34d399" : serveStarting || serveStopping || modelTransitionActive ? "#fde68a" : serveState === "Error" ? "#f87171" : "var(--text-0)" }}>
                <StatusDot running={serveRunning} starting={serveStarting || serveStopping || modelTransitionActive} error={serveState === "Error" && !modelTransitionActive} />
                <span>
                  {serveRunning
                    ? "Running"
                    : serveStopping
                      ? "Stopping"
                    : modelTransitionActive
                      ? processStatus?.model_load_state ?? "Loading"
                      : serveStarting
                        ? "Starting"
                        : serveState === "Error"
                          ? "Unreachable"
                          : "Stopped"}
                </span>
              </div>
              <p className="mt-2 text-xs" style={{ color: "var(--text-1)" }}>
                {modelTransitionActive
                  ? modelTransition?.message ?? "Model transition in progress."
                  : serveRunning
                  ? "Public API is reachable for external tools."
                  : serveStopping
                    ? "Public API is stopping. Waiting for the server task and port release."
                  : serveStarting
                    ? "Public API is starting up."
                    : serveState === "Error"
                    ? processStatus?.api_error ?? "The public API is not bound right now."
                    : "The public API is currently off."}
              </p>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                {serveReachable ? "Reachable At" : "Configured At"}
              </div>
              <div className="mt-1 font-mono text-sm" style={{ color: serveReachable ? "#34d399" : "var(--text-0)" }}>
                {apiUrl}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                Active Model
              </div>
              <div className="mt-1 text-sm font-medium" style={{ color: "var(--text-0)" }}>
                {activeModel ?? "No model loaded"}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                Runtime
              </div>
              <div className="mt-1 text-sm" style={{ color: "var(--text-1)" }}>
                {processStatus?.backend ?? "Unknown backend"}
                {processStatus?.startup_duration_ms != null && ` · ${processStatus.startup_duration_ms} ms`}
              </div>
            </div>
          </div>
        </Panel>

        <section
          className="flex flex-wrap items-center gap-1 rounded-md px-2 py-2"
          style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
          role="tablist"
          aria-label="Developer tools"
        >
          {TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.key}
              onClick={() => setActiveTab(tab.key)}
              className="rounded px-3 py-1.5 text-sm font-medium transition"
              style={{
                background: activeTab === tab.key ? "rgba(255,255,255,0.10)" : "transparent",
                color: activeTab === tab.key ? "var(--text-0)" : "var(--text-1)",
                border: activeTab === tab.key ? "1px solid var(--border-mid)" : "1px solid transparent",
                cursor: "pointer",
              }}
            >
              {tab.label}
            </button>
          ))}
          <div className="ml-auto font-mono text-xs" style={{ color: "var(--text-1)" }}>
            Embedded API: {apiUrl}
          </div>
        </section>

        {activeTab === "api" && (
          <div className="grid gap-3 xl:grid-cols-[360px_minmax(0,1fr)]">
            <Panel title="Examples" description="Use these as one-click starting points for real requests.">
              <div>
                {examples.map((example, index) => (
                  <div key={example.label}>
                    <button
                      onClick={() => loadExample(example)}
                      className="flex w-full items-start justify-between gap-3 px-4 py-3 text-left transition"
                      style={{ background: "transparent", border: "none", cursor: "pointer" }}
                    >
                      <div className="min-w-0">
                        <div className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>
                          {example.label}
                        </div>
                        <div className="mt-1 font-mono text-xs" style={{ color: "var(--text-2)" }}>
                          {example.method} {example.path}
                        </div>
                        <div className="mt-2 text-xs" style={{ color: "var(--text-1)" }}>
                          {example.description}
                        </div>
                      </div>
                      <span
                        className="rounded px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.12em]"
                        style={{
                          background: "rgba(255,255,255,0.07)",
                          border: "1px solid var(--border)",
                          color: "var(--text-1)",
                        }}
                      >
                        Example
                      </span>
                    </button>
                    {index < examples.length - 1 && <Divider />}
                  </div>
                ))}
              </div>
            </Panel>

            <div className="flex flex-col gap-3">
              <Panel
                title="API Editor"
                description="Send real requests against the embedded OpenAI-compatible API."
                actions={
                  <div className="flex items-center gap-2">
                    <ActionButton label="Copy cURL" onClick={copyCurl} />
                    <ActionButton label={sending ? "Sending..." : "Send"} onClick={sendRequest} primary disabled={sending} />
                  </div>
                }
              >
                <FieldRow label="Method">
                  <select
                    value={method}
                    onChange={(event) => setMethod(event.target.value)}
                    className="w-full rounded px-3 py-2 text-sm outline-none"
                    style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
                  >
                    <option value="GET">GET</option>
                    <option value="POST">POST</option>
                    <option value="DELETE">DELETE</option>
                  </select>
                </FieldRow>
                <Divider />
                <FieldRow label="Path">
                  <input
                    value={path}
                    onChange={(event) => setPath(event.target.value)}
                    className="w-full rounded px-3 py-2 text-sm outline-none"
                    style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
                  />
                </FieldRow>
                <Divider />
                <FieldRow label="Headers">
                  <textarea
                    value={headersText}
                    onChange={(event) => setHeadersText(event.target.value)}
                    rows={5}
                    className="w-full rounded px-3 py-2 text-sm outline-none"
                    style={{
                      background: "var(--surface-2)",
                      border: "1px solid var(--border)",
                      color: "var(--text-0)",
                      resize: "vertical",
                      fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace",
                    }}
                  />
                </FieldRow>
                <Divider />
                <FieldRow label="Body">
                  <textarea
                    value={bodyText}
                    onChange={(event) => setBodyText(event.target.value)}
                    rows={10}
                    className="w-full rounded px-3 py-2 text-sm outline-none"
                    style={{
                      background: "var(--surface-2)",
                      border: "1px solid var(--border)",
                      color: "var(--text-0)",
                      resize: "vertical",
                      fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace",
                    }}
                  />
                </FieldRow>
              </Panel>

              <Panel title="Response" description="What external clients would receive from the embedded API.">
                <div className="flex items-center justify-between px-4 py-3 text-sm">
                  <div style={{ color: response?.status && response.status < 400 ? "#34d399" : response?.status ? "#f87171" : "var(--text-1)" }}>
                    {response ? response.status : "No request yet"}
                  </div>
                  <div className="text-xs" style={{ color: "var(--text-2)" }}>
                    {requestDurationMs != null ? `${requestDurationMs} ms` : ""}
                    {response ? ` | ${response.transport === "public" ? "Public API" : "Direct fallback"}` : ""}
                  </div>
                </div>
                <Divider />
                <FieldRow label="Headers">
                  <CompactCode value={response ? formatHeaders(response.headers) : ""} emptyLabel="Run a request to inspect response headers." />
                </FieldRow>
                <Divider />
                <FieldRow label="Body">
                  <CompactCode value={response?.body ?? ""} emptyLabel="Run a request to inspect the live embedded API response." />
                </FieldRow>
              </Panel>

              <Panel title="Recent Requests" description="Jump back to recent endpoints and resend them quickly.">
                {recentRequests.length === 0 ? (
                  <div className="px-4 py-4 text-sm" style={{ color: "var(--text-1)" }}>
                    No requests sent yet.
                  </div>
                ) : (
                  <div>
                    {recentRequests.map((item, index) => (
                      <div key={`${item.at}-${index}`}>
                        <button
                          onClick={() => {
                            setMethod(item.method);
                            setPath(item.path);
                          }}
                          className="flex w-full items-center gap-3 px-4 py-3 text-left transition"
                          style={{ background: "transparent", border: "none", cursor: "pointer" }}
                        >
                          <span className="text-xs tabular-nums" style={{ color: "var(--text-2)" }}>
                            {item.at}
                          </span>
                          <span className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>
                            {item.method}
                          </span>
                          <span className="min-w-0 flex-1 truncate font-mono text-xs" style={{ color: "var(--text-1)" }}>
                            {item.path}
                          </span>
                          <span className="text-sm font-semibold" style={{ color: item.status < 400 ? "#34d399" : "#f87171" }}>
                            {item.status}
                          </span>
                        </button>
                        {index < recentRequests.length - 1 && <Divider />}
                      </div>
                    ))}
                  </div>
                )}
              </Panel>
            </div>
          </div>
        )}

        {activeTab === "doctor" && (
          <div className="grid gap-3 lg:grid-cols-[320px_minmax(0,1fr)]">
            <Panel
              title="Runtime Doctor"
              description="Probe local provider endpoints and report what is actually reachable."
              actions={<ActionButton label="Refresh Doctor" onClick={refreshDoctor} primary />}
            >
              <div className="grid gap-3 px-4 py-3">
                <Metric
                  label="Providers"
                  value={
                    runtimeDoctor
                      ? `${runtimeDoctor.summary.reachable_providers}/${runtimeDoctor.summary.total_providers} reachable`
                      : "Not checked"
                  }
                />
                <Metric
                  label="Public API"
                  value={
                    runtimeDoctor
                      ? runtimeDoctor.app_api.reachable
                        ? "Reachable"
                        : runtimeDoctor.app_api.state
                      : "Unknown"
                  }
                />
                <Metric
                  label="Active Runtime"
                  value={
                    runtimeDoctor?.active_runtime.model ??
                    runtimeDoctor?.active_runtime.state ??
                    "No runtime"
                  }
                />
                <Metric
                  label="Next Step"
                  value={runtimeDoctor?.summary.preferred_next_step ?? "Run doctor to inspect provider state."}
                />
              </div>
            </Panel>

            <Panel title="Provider Probes" description="Managed llama.cpp, standalone llama.cpp, Ollama, and OpenAI-compatible provider probes.">
              {!runtimeDoctor ? (
                <div className="px-4 py-4 text-sm" style={{ color: "var(--text-1)" }}>
                  Run doctor to probe local providers.
                </div>
              ) : (
                <div>
                  {runtimeDoctor.providers.map((provider, index) => (
                    <div key={provider.id}>
                      <div className="grid gap-3 px-4 py-3 xl:grid-cols-[minmax(0,1fr)_220px_220px]">
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            {(() => {
                              const warning = provider.status === "warn";
                              const badgeStyle = warning
                                ? {
                                    background: "rgba(251,191,36,0.10)",
                                    border: "1px solid rgba(251,191,36,0.24)",
                                    color: "#fbbf24",
                                  }
                                : {
                                    background: provider.reachable ? "rgba(52,211,153,0.1)" : "rgba(107,114,128,0.12)",
                                    border: provider.reachable ? "1px solid rgba(52,211,153,0.22)" : "1px solid var(--border)",
                                    color: provider.reachable ? "#34d399" : "var(--text-2)",
                                  };
                              return (
                                <>
                                  <StatusDot running={provider.reachable && !warning} error={!provider.reachable && provider.status !== "idle"} />
                                  <span className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>
                                    {provider.name}
                                  </span>
                                  <span
                                    className="rounded px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.12em]"
                                    style={badgeStyle}
                                  >
                                    {provider.status}
                                  </span>
                                </>
                              );
                            })()}
                          </div>
                          <div className="mt-2 font-mono text-xs break-all" style={{ color: "var(--text-2)" }}>
                            {provider.base_url}
                          </div>
                          {provider.error && (
                            <div className="mt-2 text-xs" style={{ color: "#fca5a5" }}>
                              {provider.error}
                            </div>
                          )}
                          {provider.hints.length > 0 && (
                            <div className="mt-2 space-y-1 text-xs" style={{ color: "var(--text-1)" }}>
                              {provider.hints.map((hint) => (
                                <div key={hint}>{hint}</div>
                              ))}
                            </div>
                          )}
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <Metric label="Models" value={provider.model_count.toString()} />
                          <Metric label="Context" value={provider.context_limit ? provider.context_limit.toLocaleString() : "Unknown"} />
                        </div>

                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <Metric label="Health" value={provider.endpoints.health ? "Yes" : "No"} />
                          <Metric label="Models API" value={provider.endpoints.openai_models ? "Yes" : "No"} />
                          <Metric label="Props" value={provider.endpoints.props ? "Yes" : "No"} />
                          <Metric label="Slots" value={provider.endpoints.slots ? "Yes" : "No"} />
                        </div>
                      </div>
                      {provider.models.length > 0 && (
                        <>
                          <Divider />
                          <div className="px-4 py-3">
                            <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                              Models
                            </div>
                            <CompactCode
                              value={provider.models.slice(0, 12).map((model) => model.id).join("\n")}
                              emptyLabel="No models reported."
                            />
                          </div>
                        </>
                      )}
                      {index < runtimeDoctor.providers.length - 1 && <Divider />}
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          </div>
        )}

        {activeTab === "profile" && (
          <div className="grid gap-3 lg:grid-cols-[260px_minmax(0,1fr)]">
            <Panel title="Profile Target" description="Choose which model to inspect after detection and overrides.">
              <FieldRow label="Model">
                <select
                  value={selectedProfileModel}
                  onChange={(event) => setSelectedProfileModel(event.target.value)}
                  className="w-full rounded px-3 py-2 text-sm outline-none"
                  style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
                >
                  {models.map((model) => (
                    <option key={model.filename} value={model.filename}>
                      {model.filename}
                    </option>
                  ))}
                </select>
              </FieldRow>
              <Divider />
              <div className="px-4 py-3">
                <ActionButton label="Refresh Profile" onClick={() => refreshProfile(selectedProfileModel || activeModel || undefined)} primary />
              </div>
            </Panel>

            <Panel title="Effective Profile" description="Resolved family, parser, renderer, tool style, and any override entry.">
              <div className="grid gap-3 px-4 py-3 md:grid-cols-2 xl:grid-cols-4">
                <Metric label="Family" value={effectiveProfile?.profile.family ?? "Unknown"} />
                <Metric label="Parser" value={effectiveProfile?.profile.parser_type ?? "Unknown"} />
                <Metric label="Renderer" value={effectiveProfile?.profile.renderer_type ?? "Unknown"} />
                <Metric label="Vision" value={effectiveProfile ? (effectiveProfile.profile.supports_vision ? "Yes" : "No") : "Unknown"} />
                <Metric label="Think Style" value={effectiveProfile?.profile.think_tag_style ?? "Unknown"} />
                <Metric label="Tool Format" value={effectiveProfile?.profile.tool_call_format ?? "Unknown"} />
                <Metric label="Context" value={effectiveProfile?.profile.default_context_window?.toString() ?? "Unknown"} />
                <Metric label="Max Output" value={effectiveProfile?.profile.default_max_output_tokens?.toString() ?? "Unknown"} />
              </div>
              <Divider />
              <div className="grid gap-3 px-4 py-3 xl:grid-cols-2">
                <div>
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                    Profile
                  </div>
                  <CompactCode value={effectiveProfile ? JSON.stringify(effectiveProfile.profile, null, 2) : ""} emptyLabel="No effective profile loaded yet." />
                </div>
                <div>
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                    Override Entry
                  </div>
                  <CompactCode value={effectiveProfile?.override_entry ? JSON.stringify(effectiveProfile.override_entry, null, 2) : ""} emptyLabel="No per-model override exists for this model." />
                </div>
              </div>
            </Panel>
          </div>
        )}

        {activeTab === "launch" && (
          <div className="grid gap-3 lg:grid-cols-[320px_minmax(0,1fr)]">
            <Panel title="Runtime Snapshot" description="Last resolved launch inputs from the model runtime.">
              <div className="grid gap-3 px-4 py-3">
                <Metric label="Backend" value={processStatus?.backend ?? "Unknown"} />
                <Metric label="Startup" value={processStatus?.startup_duration_ms != null ? `${processStatus.startup_duration_ms} ms` : "Unknown"} />
                <Metric label="Crash Count" value={String(processStatus?.crash_count ?? 0)} />
                <Metric label="Slots" value={processStatus?.slot_count != null ? `${processStatus.slot_count} total` : `${processStatus?.parallel_slots ?? 0} configured`} />
              </div>
              <Divider />
              <div className="px-4 py-3">
                <ActionButton label="Refresh Launch Preview" onClick={refreshLaunch} primary />
              </div>
            </Panel>

            <Panel title="Launch Preview" description="Exact resolved launch configuration before spawning llama.cpp.">
              <div className="px-4 py-3">
                <CompactCode value={launchPreview} emptyLabel="Load a model to capture a launch preview." />
              </div>
              {processStatus?.last_launch_preview && (
                <>
                  <Divider />
                  <div className="grid gap-3 px-4 py-3 md:grid-cols-2">
                    <Metric label="Server Path" value={processStatus.last_launch_preview.server_path} mono />
                    <Metric label="Model Path" value={processStatus.last_launch_preview.model_path} mono />
                    <Metric label="mmproj" value={processStatus.last_launch_preview.mmproj_path ?? "None"} mono />
                    <Metric label="Port" value={String(processStatus.last_launch_preview.port)} />
                  </div>
                </>
              )}
            </Panel>
          </div>
        )}

        {activeTab === "docs" && (
          <Panel title="Docs" description="Quick reference for the embedded public API.">
            <div className="grid gap-3 px-4 py-3 lg:grid-cols-2">
              <DocCard title="Basic Flow" body={["1. GET /v1/models", "2. POST /v1/models/load if needed", "3. POST /v1/models/stats while loading", "4. POST /v1/chat/completions"]} />
              <DocCard title="Serve Modes" body={["The GUI uses direct app state.", "External tools use the public /v1 API on port 8800.", "If the public port is down, the in-app editor can still fall back to direct debug transport."]} />
            </div>
            <Divider />
            <div className="grid gap-3 px-4 py-3 xl:grid-cols-2">
              {examples.slice(0, 6).map((example) => (
                <div key={example.label}>
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                    {example.label}
                  </div>
                  <CompactCode value={`curl -X ${example.method} "${buildPublicUrl(apiUrl, example.path)}"${example.body ? `\n  -H "Content-Type: application/json"\n  -d '${example.body}'` : ""}`} />
                </div>
              ))}
            </div>
          </Panel>
        )}

        {activeTab === "logs" && (
          <Panel
            title="Logs"
            description="Live Rust tracing plus HTTP middleware events."
            actions={
              <div className="flex items-center gap-2">
                <ActionButton label={autoRefreshLogs ? "Auto-refresh On" : "Auto-refresh Off"} onClick={() => setAutoRefreshLogs((current) => !current)} />
                <ActionButton label="Refresh" onClick={refreshLogs} />
                <ActionButton
                  label="Clear"
                  onClick={async () => {
                    await api.clearLogs();
                    await refreshLogs();
                  }}
                />
              </div>
            }
          >
            <div className="grid gap-2 px-4 py-3 md:grid-cols-[220px_minmax(0,1fr)_auto]">
              <select
                value={logLevel}
                onChange={(event) => setLogLevel(event.target.value)}
                className="rounded px-3 py-2 text-sm outline-none"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              >
                <option value="ALL">ALL</option>
                <option value="INFO">INFO</option>
                <option value="WARN">WARN</option>
                <option value="ERROR">ERROR</option>
                <option value="DEBUG">DEBUG</option>
              </select>
              <input
                value={logQuery}
                onChange={(event) => setLogQuery(event.target.value)}
                placeholder="Search target or message..."
                className="rounded px-3 py-2 text-sm outline-none"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              />
              <div className="self-center text-xs" style={{ color: "var(--text-2)" }}>
                {filteredLogs.length} entries
              </div>
            </div>
            <Divider />
            <div className="max-h-[620px] overflow-y-auto">
              {filteredLogs.length === 0 ? (
                <div className="px-4 py-4 text-sm" style={{ color: "var(--text-1)" }}>
                  No log entries match the current filter.
                </div>
              ) : (
                filteredLogs.map((entry, index) => (
                  <div key={`${entry.timestamp}-${index}`}>
                    <div className="px-4 py-3">
                      <div className="flex flex-wrap items-center gap-3 text-xs">
                        <span style={{ color: "var(--text-2)" }}>{entry.timestamp}</span>
                        <span style={{ color: entry.level === "ERROR" ? "#f87171" : entry.level === "WARN" ? "#fbbf24" : "#22d3ee" }}>
                          {entry.level}
                        </span>
                        <span style={{ color: "var(--text-2)" }}>{entry.target}</span>
                      </div>
                      <p className="mt-2 whitespace-pre-wrap font-mono text-xs leading-6" style={{ color: "var(--text-0)" }}>
                        {entry.message}
                      </p>
                    </div>
                    {index < filteredLogs.length - 1 && <Divider />}
                  </div>
                ))
              )}
            </div>
          </Panel>
        )}

        {activeTab === "prompt" && (
          <Panel title="Raw Prompt" description="The last rendered prompt sent into the backend." actions={<ActionButton label="Refresh" onClick={refreshPrompt} />}>
            <div className="px-4 py-3">
              <CompactCode value={rawPrompt} emptyLabel="No prompt captured yet." />
            </div>
          </Panel>
        )}

        {activeTab === "trace" && (
          <Panel title="Parse Trace" description="Last normalization / parsing trace captured during generation." actions={<ActionButton label="Refresh" onClick={refreshTrace} />}>
            <div className="space-y-3 px-4 py-3">
              {parseTrace && <ParseTraceSummary trace={parsedTrace} />}
              <CompactCode value={parseTrace} emptyLabel="No parse trace captured yet." />
            </div>
          </Panel>
        )}

        {tabError && (
          <div
            className="rounded px-4 py-3 text-sm"
            style={{
              background: "rgba(248,113,113,0.08)",
              border: "1px solid rgba(248,113,113,0.22)",
              color: "#fca5a5",
            }}
          >
            {tabError}
          </div>
        )}
      </div>
    </div>
  );
}
