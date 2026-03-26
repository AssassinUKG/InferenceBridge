import { useEffect, useMemo, useState, type ReactNode } from "react";
import type {
  DebugApiResponse,
  EffectiveProfileInfo,
  LogEntry,
  ModelInfo,
  ProcessStatusInfo,
} from "../../lib/types";
import * as api from "../../lib/tauri";

type DebugTab = "api" | "profile" | "launch" | "docs" | "logs" | "prompt" | "trace";
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
  models: ModelInfo[];
  onSetApiServerRunning: (running: boolean) => Promise<void> | void;
  onOpenSettings: () => void;
}

const TABS: Array<{ key: DebugTab; label: string }> = [
  { key: "api", label: "API Editor" },
  { key: "profile", label: "Profile" },
  { key: "launch", label: "Launch" },
  { key: "docs", label: "Docs" },
  { key: "logs", label: "Logs" },
  { key: "prompt", label: "Raw Prompt" },
  { key: "trace", label: "Parse Trace" },
];

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
        borderRadius: "10px",
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
      className="rounded px-3 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-50"
      style={
        primary
          ? {
              background: "#22d3ee",
              color: "#0a0a0a",
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
    <div className="rounded px-3 py-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
        {label}
      </div>
      <div className={`mt-1 text-sm ${mono ? "break-all font-mono" : "font-medium"}`} style={{ color: "var(--text-0)" }}>
        {value}
      </div>
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
    { label: "Profile", method: "GET", path: `/v1/debug/profile?model=${encodeURIComponent(modelName)}`, description: "Show the effective profile after detection and overrides." },
    { label: "Load Model", method: "POST", path: "/v1/models/load", body: JSON.stringify({ model: modelName }, null, 2), description: "Start loading a model in the background." },
    { label: "Unload", method: "POST", path: "/v1/models/unload", body: "{}", description: "Unload the active backend model." },
    { label: "Chat", method: "POST", path: "/v1/chat/completions", body: JSON.stringify({ model: modelName, messages: [{ role: "user", content: "Reply with exactly: InferenceBridge OK" }], temperature: 0.2, stream: false }, null, 2), description: "Run a standard OpenAI-compatible chat completion request." },
  ];
}

export function DebugInspector({
  apiUrl,
  processStatus,
  models,
  onSetApiServerRunning,
  onOpenSettings,
}: Props) {
  const [activeTab, setActiveTab] = useState<DebugTab>("api");
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
  const [selectedProfileModel, setSelectedProfileModel] = useState("");
  const [autoRefreshLogs, setAutoRefreshLogs] = useState(true);
  const [tabError, setTabError] = useState<string | null>(null);

  const activeModel = processStatus?.model ?? null;
  const serveState = processStatus?.api_state ?? "Idle";
  const serveReachable = processStatus?.api_reachable ?? false;
  const serveRunning = serveState === "Running" && serveReachable;
  const serveStarting = serveState === "Starting";
  const selectedModel = selectedProfileModel || activeModel || models[0]?.filename || null;
  const examples = useMemo(() => exampleList(activeModel, selectedModel), [activeModel, selectedModel]);

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

  useEffect(() => {
    if (activeTab === "logs") refreshLogs();
    if (activeTab === "prompt") refreshPrompt();
    if (activeTab === "trace") refreshTrace();
    if (activeTab === "launch") refreshLaunch();
    if (activeTab === "profile") refreshProfile(selectedProfileModel || activeModel || undefined);
  }, [activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (activeTab !== "profile") return;
    refreshProfile(selectedProfileModel || activeModel || undefined);
  }, [selectedProfileModel, activeModel]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (activeTab !== "logs" || !autoRefreshLogs) return;
    const timer = window.setInterval(() => {
      refreshLogs();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [activeTab, autoRefreshLogs]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadExample = (example: Example) => {
    setMethod(example.method);
    setPath(example.path);
    setBodyText(example.body ?? "");
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

  return (
    <div className="h-full overflow-y-auto">
      <div className="flex flex-col gap-3 p-3">
        <Panel
          title="Developer Server"
          description="Run the app like LM Studio and inspect the same public API your external tools use."
          actions={
            <div className="flex flex-wrap items-center gap-2">
              <ActionButton label="Copy URL" onClick={() => navigator.clipboard.writeText(apiUrl)} />
              <ActionButton label="Server Settings" onClick={onOpenSettings} />
              <ActionButton
                label={serveRunning || serveStarting ? "Stop API" : serveState === "Error" ? "Retry API" : "Start API"}
                onClick={() => onSetApiServerRunning(serveRunning || serveStarting ? false : true)}
                primary
              />
            </div>
          }
        >
          <div className="grid gap-3 px-4 py-3 lg:grid-cols-4">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold" style={{ color: serveRunning ? "#34d399" : serveStarting ? "#fde68a" : serveState === "Error" ? "#f87171" : "var(--text-0)" }}>
                <StatusDot running={serveRunning} starting={serveStarting} error={serveState === "Error"} />
                <span>{serveRunning ? "Running" : serveStarting ? "Starting" : serveState === "Error" ? "Unreachable" : "Stopped"}</span>
              </div>
              <p className="mt-2 text-xs" style={{ color: "var(--text-1)" }}>
                {serveRunning
                  ? "Public API is reachable for external tools."
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
          className="flex flex-wrap items-center gap-2 rounded px-3 py-2"
          style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
        >
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className="rounded px-3 py-1.5 text-sm font-medium transition"
              style={{
                background: activeTab === tab.key ? "var(--surface-2)" : "transparent",
                color: activeTab === tab.key ? "var(--text-0)" : "var(--text-1)",
                border: activeTab === tab.key ? "1px solid var(--border)" : "1px solid transparent",
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
                          background: "rgba(34,211,238,0.08)",
                          border: "1px solid rgba(34,211,238,0.18)",
                          color: "#22d3ee",
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
            <div className="px-4 py-3">
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
