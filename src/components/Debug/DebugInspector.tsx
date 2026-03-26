import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import type { DebugApiResponse, LogEntry, ModelInfo, ProcessStatusInfo } from "../../lib/types";
import * as api from "../../lib/tauri";

type DebugTab = "api" | "docs" | "logs" | "prompt" | "trace";

interface Props {
  apiUrl: string;
  processStatus: ProcessStatusInfo | null;
  models: ModelInfo[];
  onSetApiServerRunning: (running: boolean) => void;
  onOpenSettings: () => void;
}

interface ExamplePreset {
  id: string;
  label: string;
  method: string;
  path: string;
  body?: string;
  description: string;
  note: string;
}

interface RecentRequest {
  id: string;
  method: string;
  path: string;
  status: number;
  durationMs: number;
  transport: string;
  timestamp: string;
  body: string;
}

const DEFAULT_HEADERS = `{
  "Accept": "application/json",
  "Content-Type": "application/json"
}`;

const panelStyle = {
  background: "var(--surface-1)",
  border: "1px solid var(--border)",
  borderRadius: "10px",
  overflow: "hidden",
} as const;

function Panel({ children }: { children: ReactNode }) {
  return <section style={panelStyle}>{children}</section>;
}

function Header({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div
      className="flex flex-wrap items-start justify-between gap-2 px-4 py-3"
      style={{ borderBottom: "1px solid var(--border)" }}
    >
      <div>
        <div className="text-[11px] uppercase tracking-[0.22em]" style={{ color: "var(--text-2)" }}>
          {title}
        </div>
        {subtitle && (
          <p className="mt-1 text-xs leading-5" style={{ color: "var(--text-1)" }}>
            {subtitle}
          </p>
        )}
      </div>
      {actions}
    </div>
  );
}

function Button({
  label,
  onClick,
  primary = false,
  disabled = false,
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
      className="rounded px-3 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-60"
      style={{
        background: primary ? "#22d3ee" : "var(--surface-2)",
        color: primary ? "#041014" : "var(--text-0)",
        border: primary ? "1px solid transparent" : "1px solid var(--border)",
      }}
    >
      {label}
    </button>
  );
}

function Tab({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="rounded px-3 py-1.5 text-xs font-medium transition"
      style={{
        background: active ? "var(--surface-2)" : "transparent",
        color: active ? "var(--text-0)" : "var(--text-1)",
        border: active ? "1px solid var(--border)" : "1px solid transparent",
      }}
    >
      {label}
    </button>
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
    <div
      className="grid gap-4 px-4 py-3"
      style={{
        gridTemplateColumns: "140px minmax(0,1fr)",
        borderTop: "1px solid var(--border)",
      }}
    >
      <div className="pt-0.5">
        <div className="text-[11px] uppercase tracking-[0.22em]" style={{ color: "var(--text-2)" }}>
          {label}
        </div>
      </div>
      <div>{children}</div>
    </div>
  );
}

function InfoPill({ label, tone = "default" }: { label: string; tone?: "default" | "good" | "warn" | "bad" }) {
  const tones = {
    default: {
      background: "var(--surface-2)",
      border: "1px solid var(--border)",
      color: "var(--text-1)",
    },
    good: {
      background: "rgba(52,211,153,0.1)",
      border: "1px solid rgba(52,211,153,0.18)",
      color: "#86efac",
    },
    warn: {
      background: "rgba(251,191,36,0.1)",
      border: "1px solid rgba(251,191,36,0.18)",
      color: "#fde68a",
    },
    bad: {
      background: "rgba(248,113,113,0.1)",
      border: "1px solid rgba(248,113,113,0.18)",
      color: "#fca5a5",
    },
  } as const;

  return (
    <span
      className="inline-flex items-center rounded px-2 py-1 text-[11px] font-medium"
      style={tones[tone]}
    >
      {label}
    </span>
  );
}

function normalizePath(path: string) {
  if (!path.trim()) return "/v1/health";
  return path.startsWith("/") ? path : `/${path}`;
}

function buildPublicUrl(apiUrl: string, path: string) {
  const base = apiUrl.replace(/\/$/, "");
  const normalized = normalizePath(path);

  if (base.endsWith("/v1") && normalized === "/v1") {
    return base;
  }

  if (base.endsWith("/v1") && normalized.startsWith("/v1/")) {
    return `${base}${normalized.slice(3)}`;
  }

  return `${base}${normalized}`;
}

function formatJsonMaybe(value: string) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

function parseHeaders(raw: string) {
  if (!raw.trim()) return {};
  const parsed = JSON.parse(raw);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Headers must be a JSON object.");
  }
  return Object.entries(parsed).reduce<Record<string, string>>((acc, [key, value]) => {
    acc[key] = String(value);
    return acc;
  }, {});
}

function buildCurlCommand(apiUrl: string, method: string, path: string, headers: string, body: string) {
  let headerArgs = "";
  try {
    headerArgs = Object.entries(parseHeaders(headers))
      .map(([key, value]) => `-H "${key}: ${value.replace(/"/g, '\\"')}"`)
      .join(" ");
  } catch {
    headerArgs = "";
  }
  const bodyArg =
    body.trim() && method !== "GET"
      ? ` --data-raw "${body.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\r?\n/g, "")}"`
      : "";
  return `curl -X ${method} "${buildPublicUrl(apiUrl, path)}"${
    headerArgs ? ` ${headerArgs}` : ""
  }${bodyArg}`;
}

function apiStatusMeta(apiState: string) {
  if (apiState === "Running") {
    return { dot: "#34d399", label: "Running", detail: "Public API is reachable for external tools." };
  }
  if (apiState === "Starting") {
    return { dot: "#fbbf24", label: "Starting", detail: "Bringing up the embedded OpenAI-compatible listener." };
  }
  if (apiState === "Error") {
    return { dot: "#f87171", label: "Port Busy", detail: "Desktop UI is alive, but the public serve port needs attention." };
  }
  return { dot: "#6b7280", label: "Stopped", detail: "Desktop UI is running without the public API listener." };
}

function transportLabel(transport: string) {
  return transport === "public" ? "Public API" : "Direct App";
}

function statusColor(status: number) {
  if (status >= 200 && status < 300) return "#34d399";
  if (status >= 400) return "#f87171";
  return "#fbbf24";
}

export function DebugInspector({
  apiUrl,
  processStatus,
  models,
  onSetApiServerRunning,
  onOpenSettings,
}: Props) {
  const [activeTab, setActiveTab] = useState<DebugTab>("api");
  const [selectedExampleId, setSelectedExampleId] = useState("models");
  const [method, setMethod] = useState("GET");
  const [path, setPath] = useState("/v1/models");
  const [headers, setHeaders] = useState(DEFAULT_HEADERS);
  const [body, setBody] = useState("");
  const [response, setResponse] = useState<DebugApiResponse | null>(null);
  const [responseDurationMs, setResponseDurationMs] = useState<number | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [copiedUrl, setCopiedUrl] = useState(false);
  const [copiedCurl, setCopiedCurl] = useState(false);
  const [recentRequests, setRecentRequests] = useState<RecentRequest[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [logsQuery, setLogsQuery] = useState("");
  const [logsLevel, setLogsLevel] = useState("ALL");
  const [logsAutoRefresh, setLogsAutoRefresh] = useState(true);
  const [rawPrompt, setRawPrompt] = useState("Loading raw prompt...");
  const [parseTrace, setParseTrace] = useState("Loading parse trace...");

  const apiState = processStatus?.api_state ?? "Idle";
  const apiRunning = apiState === "Running" || apiState === "Starting";
  const activeModel = processStatus?.model ?? null;
  const fallbackModel = activeModel ?? models[0]?.filename ?? "replace-with-model-from-v1-models";
  const encodedFallbackModel = encodeURIComponent(fallbackModel);
  const serveMeta = apiStatusMeta(apiState);

  const examples = useMemo<ExamplePreset[]>(
    () => [
      {
        id: "health",
        label: "Health",
        method: "GET",
        path: "/v1/health",
        description: "Fast serve check for the public endpoint.",
        note: "Use this first to confirm the embedded API is responding.",
      },
      {
        id: "models",
        label: "Models",
        method: "GET",
        path: "/v1/models",
        description: "List discovered models and mark the active one.",
        note: "This mirrors what external clients should inspect before load or chat requests.",
      },
      {
        id: "model-detail",
        label: "Model Detail",
        method: "GET",
        path: `/v1/models/${encodedFallbackModel}`,
        description: "Fetch one model record by filename or ID.",
        note: "Use this after GET /v1/models when you want the metadata for one exact model entry.",
      },
      {
        id: "model-stats",
        label: "Model Stats",
        method: "POST",
        path: "/v1/models/stats",
        body: JSON.stringify({ model: fallbackModel }, null, 2),
        description: "Inspect load progress and runtime stats for one named model.",
        note: "Best follow-up after a load request because model loading is asynchronous and model-targeted now.",
      },
      {
        id: "context",
        label: "Context",
        method: "GET",
        path: "/v1/context/status",
        description: "See current KV usage and context pressure.",
        note: "Useful once a model is loaded and taking prompts.",
      },
      {
        id: "sessions",
        label: "Sessions",
        method: "GET",
        path: "/v1/sessions",
        description: "List saved chat sessions from the app database.",
        note: "Handy for checking shared state between the GUI and API.",
      },
      {
        id: "load",
        label: "Load Model",
        method: "POST",
        path: "/v1/models/load",
        body: JSON.stringify({ model: fallbackModel }, null, 2),
        description: "Start loading a model in the background.",
        note: "A 200 response here means accepted. Follow with Model Stats until stage=ready.",
      },
      {
        id: "unload",
        label: "Unload",
        method: "POST",
        path: "/v1/models/unload",
        body: JSON.stringify({}, null, 2),
        description: "Unload the active backend model.",
        note: "Use this before testing a different load or swap path.",
      },
      {
        id: "chat",
        label: "Chat",
        method: "POST",
        path: "/v1/chat/completions",
        body: JSON.stringify(
          {
            model: fallbackModel,
            messages: [
              { role: "system", content: "You are a concise local assistant." },
              { role: "user", content: "Reply with exactly: InferenceBridge ready" },
            ],
            temperature: 0.2,
            max_tokens: 96,
            stream: false,
          },
          null,
          2
        ),
        description: "Run a standard OpenAI-compatible completion request.",
        note: activeModel
          ? "You can remove the model field if you want to target the currently loaded model."
          : "If no model is active, the API will try to load the named model before answering.",
      },
      {
        id: "chat-current",
        label: "Chat Current",
        method: "POST",
        path: "/v1/chat/completions",
        body: JSON.stringify(
          {
            messages: [{ role: "user", content: "Summarize this local API in one sentence." }],
            temperature: 0.4,
            max_tokens: 96,
            stream: false,
          },
          null,
          2
        ),
        description: "Send a request against the currently loaded model without naming it.",
        note: "Only works when a model is already active, which makes it a great post-load smoke test.",
      },
      {
        id: "tool-call",
        label: "Tool Calling",
        method: "POST",
        path: "/v1/chat/completions",
        body: JSON.stringify(
          {
            model: fallbackModel,
            messages: [
              { role: "user", content: "What's the weather in London right now?" },
            ],
            tools: [
              {
                type: "function",
                function: {
                  name: "get_weather",
                  description: "Get the current weather for a given city.",
                  parameters: {
                    type: "object",
                    properties: {
                      city: { type: "string", description: "City name" },
                      unit: { type: "string", enum: ["celsius", "fahrenheit"], description: "Temperature unit" },
                    },
                    required: ["city"],
                  },
                },
              },
            ],
            temperature: 0.1,
            max_tokens: 256,
            stream: false,
          },
          null,
          2
        ),
        description: "Test tool/function calling with a weather lookup example.",
        note: "Models that support tools (Qwen3, Llama 3.1+, Mistral) will return a tool_calls array. The model won't actually call the function — you handle tool dispatch in your app.",
      },
    ],
    [activeModel, fallbackModel]
  );

  const selectedExample = useMemo(
    () => examples.find((example) => example.id === selectedExampleId) ?? examples[0],
    [examples, selectedExampleId]
  );

  const docs = useMemo(
    () => [
      {
        title: "Quick Flow",
        body:
          "Treat this like a local LM Studio style server: list models, inspect a chosen model, load it, watch targeted stats, then call chat completions. The desktop UI and the public API reflect the same backend state.",
        code: `GET  /v1/models\nGET  /v1/models/{model}\nPOST /v1/models/load\nPOST /v1/models/stats\nPOST /v1/chat/completions`,
      },
      {
        title: "Quick cURL",
        body:
          "These snippets match the built-in examples so you can move from the editor to scripts without rewriting the request shape.",
        code: `curl "${apiUrl}/models"\n\ncurl "${apiUrl}/models/${encodedFallbackModel}"\n\ncurl -X POST "${apiUrl}/models/load" \\\n  -H "Content-Type: application/json" \\\n  --data-raw "{\\"model\\":\\"${fallbackModel}\\"}"\n\ncurl -X POST "${apiUrl}/models/stats" \\\n  -H "Content-Type: application/json" \\\n  --data-raw "{\\"model\\":\\"${fallbackModel}\\"}"\n\ncurl -X POST "${apiUrl}/chat/completions" \\\n  -H "Content-Type: application/json" \\\n  --data-raw "{\\"model\\":\\"${fallbackModel}\\",\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"Reply with exactly: InferenceBridge ready\\"}],\\"stream\\":false}"`,
      },
      {
        title: "Public API vs Direct App",
        body:
          "Public API means the request actually hit the HTTP listener on port 8800. Direct App means the Debug editor fell back to the same backend route in-process, which keeps debugging useful even when the public port is off or temporarily blocked.",
      },
      {
        title: "Load Requests",
        body:
          "Model loads are asynchronous. A successful load response means the request was accepted. Check `POST /v1/models/stats` with the same model name, the Models tab, or the footer badges to confirm the model is fully ready.",
      },
    ],
    [apiUrl, encodedFallbackModel, fallbackModel]
  );

  const applyExample = useCallback((example: ExamplePreset) => {
    setSelectedExampleId(example.id);
    setMethod(example.method);
    setPath(example.path);
    setBody(example.body ?? "");
    setRequestError(null);
  }, []);

  useEffect(() => {
    if (selectedExample) {
      setMethod(selectedExample.method);
      setPath(selectedExample.path);
      setBody(selectedExample.body ?? "");
    }
  }, [selectedExample]);

  useEffect(() => {
    if (!copiedUrl) return;
    const timer = window.setTimeout(() => setCopiedUrl(false), 1600);
    return () => window.clearTimeout(timer);
  }, [copiedUrl]);

  useEffect(() => {
    if (!copiedCurl) return;
    const timer = window.setTimeout(() => setCopiedCurl(false), 1600);
    return () => window.clearTimeout(timer);
  }, [copiedCurl]);

  const loadLogs = useCallback(async () => {
    const entries = await api.getLogs(300);
    setLogs(entries);
  }, []);

  useEffect(() => {
    if (activeTab !== "logs") return;
    loadLogs().catch(() => undefined);
    if (!logsAutoRefresh) return;
    const interval = window.setInterval(() => loadLogs().catch(() => undefined), 2000);
    return () => window.clearInterval(interval);
  }, [activeTab, loadLogs, logsAutoRefresh]);

  useEffect(() => {
    if (activeTab !== "prompt") return;
    api.getRawPrompt().then(setRawPrompt).catch((err) => setRawPrompt(String(err)));
  }, [activeTab]);

  useEffect(() => {
    if (activeTab !== "trace") return;
    api.getParseTrace().then(setParseTrace).catch((err) => setParseTrace(String(err)));
  }, [activeTab]);

  const filteredLogs = useMemo(() => {
    const q = logsQuery.trim().toLowerCase();
    return logs.filter((entry) => {
      if (logsLevel !== "ALL" && entry.level.toUpperCase() !== logsLevel) return false;
      if (!q) return true;
      return `${entry.target} ${entry.message}`.toLowerCase().includes(q);
    });
  }, [logs, logsLevel, logsQuery]);

  const logLevels = useMemo(() => {
    const values = new Set<string>(["ALL"]);
    logs.forEach((entry) => values.add(entry.level.toUpperCase()));
    return Array.from(values);
  }, [logs]);

  const handleCopyUrl = useCallback(async () => {
    await navigator.clipboard.writeText(apiUrl);
    setCopiedUrl(true);
  }, [apiUrl]);

  const handleCopyCurl = useCallback(async () => {
    await navigator.clipboard.writeText(buildCurlCommand(apiUrl, method, path, headers, body));
    setCopiedCurl(true);
  }, [apiUrl, body, headers, method, path]);

  const sendRequest = useCallback(async () => {
    setIsSending(true);
    setRequestError(null);
    const requestPath = normalizePath(path);
    const methodUpper = method.toUpperCase();
    const started = performance.now();

    try {
      const requestHeaders = parseHeaders(headers);
      const useBody = methodUpper !== "GET" && body.trim().length > 0;
      const url = buildPublicUrl(apiUrl, requestPath);

      try {
        const httpResponse = await fetch(url, {
          method: methodUpper,
          headers: requestHeaders,
          body: useBody ? body : undefined,
        });
        const text = await httpResponse.text();
        const finalResponse: DebugApiResponse = {
          status: httpResponse.status,
          headers: Array.from(httpResponse.headers.entries()),
          body: formatJsonMaybe(text),
          transport: "public",
        };
        const durationMs = Math.round(performance.now() - started);
        setResponse(finalResponse);
        setResponseDurationMs(durationMs);
        setRecentRequests((current) =>
          [
            {
              id: `${Date.now()}-${Math.random()}`,
              method: methodUpper,
              path: requestPath,
              status: finalResponse.status,
              durationMs,
              transport: "public",
              timestamp: new Date().toLocaleTimeString(),
              body,
            },
            ...current,
          ].slice(0, 10)
        );
        return;
      } catch {
        const directResponse = await api.debugApiRequest({
          method: methodUpper,
          path: requestPath,
          body: useBody ? body : null,
        });
        const durationMs = Math.round(performance.now() - started);
        const finalResponse = {
          ...directResponse,
          body: formatJsonMaybe(directResponse.body),
        };
        setResponse(finalResponse);
        setResponseDurationMs(durationMs);
        setRecentRequests((current) =>
          [
            {
              id: `${Date.now()}-${Math.random()}`,
              method: methodUpper,
              path: requestPath,
              status: finalResponse.status,
              durationMs,
              transport: finalResponse.transport,
              timestamp: new Date().toLocaleTimeString(),
              body,
            },
            ...current,
          ].slice(0, 10)
        );
      }
    } catch (error) {
      setRequestError(String(error));
      setResponse(null);
      setResponseDurationMs(null);
    } finally {
      setIsSending(false);
    }
  }, [apiUrl, body, headers, method, path]);

  return (
    <div className="h-full overflow-y-auto p-3">
      <div className="mx-auto flex max-w-7xl flex-col gap-3">
        <Panel>
          <Header
            title="Developer Server"
            subtitle="Serve the same OpenAI-compatible API external tools use, with direct inspection from inside the app."
            actions={
              <div className="flex flex-wrap gap-2">
                <Button label={copiedUrl ? "Copied URL" : "Copy URL"} onClick={() => void handleCopyUrl()} />
                <Button label="Server Settings" onClick={onOpenSettings} />
                <Button
                  label={apiRunning ? "Stop API" : "Start API"}
                  onClick={() => onSetApiServerRunning(!apiRunning)}
                  primary
                />
              </div>
            }
          />
          <FieldRow label="Status">
            <div className="flex flex-wrap items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: serveMeta.dot }} />
              <span className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
                {serveMeta.label}
              </span>
              <InfoPill
                label={apiRunning ? "Serve On" : "Serve Off"}
                tone={apiState === "Error" ? "bad" : apiRunning ? "good" : "default"}
              />
              <span className="text-xs" style={{ color: "var(--text-1)" }}>
                {serveMeta.detail}
              </span>
            </div>
          </FieldRow>
          <FieldRow label="Reachable At">
            <div className="flex flex-wrap items-center gap-3">
              <span className="font-mono text-sm" style={{ color: "var(--text-0)" }}>
                {apiUrl}
              </span>
            </div>
          </FieldRow>
          <FieldRow label="Active Model">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
                {activeModel ?? "No model loaded"}
              </span>
              <span className="text-xs" style={{ color: "var(--text-1)" }}>
                Load from the editor or Models tab, then use chat completions here.
              </span>
            </div>
          </FieldRow>
          <FieldRow label="Quick Flow">
            <div className="flex flex-wrap items-center gap-2 text-xs" style={{ color: "var(--text-1)" }}>
              <InfoPill label="1. /v1/models" />
              <InfoPill label="2. /v1/models/{id}" />
              <InfoPill label="3. POST /v1/models/stats" />
              <InfoPill label="4. /v1/chat/completions" />
            </div>
          </FieldRow>
        </Panel>

        <Panel>
          <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2.5">
            <div className="flex flex-wrap gap-1">
              <Tab active={activeTab === "api"} label="API Editor" onClick={() => setActiveTab("api")} />
              <Tab active={activeTab === "docs"} label="Docs" onClick={() => setActiveTab("docs")} />
              <Tab active={activeTab === "logs"} label="Logs" onClick={() => setActiveTab("logs")} />
              <Tab active={activeTab === "prompt"} label="Raw Prompt" onClick={() => setActiveTab("prompt")} />
              <Tab active={activeTab === "trace"} label="Parse Trace" onClick={() => setActiveTab("trace")} />
            </div>
            <div className="text-xs" style={{ color: "var(--text-1)" }}>
              Embedded API: <span className="font-mono" style={{ color: "var(--text-0)" }}>{apiUrl}</span>
            </div>
          </div>
        </Panel>

        {activeTab === "api" && (
          <div className="grid gap-3" style={{ gridTemplateColumns: "minmax(260px, 320px) minmax(0,1fr)" }}>
            <Panel>
              <Header title="Examples" subtitle="Use these as one-click starting points for real requests." />
              <div>
                {examples.map((example) => (
                  <button
                    key={example.id}
                    onClick={() => applyExample(example)}
                    className="w-full px-4 py-3 text-left transition"
                    style={{
                      background: selectedExampleId === example.id ? "var(--surface-2)" : "transparent",
                      borderBottom: "1px solid var(--border)",
                    }}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="text-[13px] font-medium" style={{ color: "var(--text-0)" }}>
                          {example.label}
                        </div>
                        <div className="mt-1 text-xs" style={{ color: "var(--text-2)" }}>
                          {example.method} {example.path}
                        </div>
                      </div>
                      <span
                        className="rounded px-2 py-1 text-[11px] uppercase tracking-[0.16em]"
                        style={{
                          background: "rgba(34,211,238,0.1)",
                          color: "#67e8f9",
                          border: "1px solid rgba(34,211,238,0.16)",
                        }}
                      >
                        Example
                      </span>
                    </div>
                    <p className="mt-1.5 text-xs leading-5" style={{ color: "var(--text-1)" }}>
                      {example.description}
                    </p>
                  </button>
                ))}
              </div>
              <div className="px-4 py-3">
                <div className="text-[11px] uppercase tracking-[0.2em]" style={{ color: "var(--text-2)" }}>
                  Example Notes
                </div>
                <p className="mt-1.5 text-xs leading-5" style={{ color: "var(--text-1)" }}>
                  {selectedExample.note}
                </p>
              </div>
            </Panel>

            <div className="flex flex-col gap-4">
              <Panel>
                <Header
                  title="API Editor"
                  subtitle="Send real requests against the embedded OpenAI-compatible API."
                  actions={
                    <div className="flex flex-wrap gap-2">
                      <Button label={copiedCurl ? "Copied cURL" : "Copy cURL"} onClick={() => void handleCopyCurl()} />
                      <Button label={isSending ? "Sending..." : "Send"} onClick={() => void sendRequest()} primary disabled={isSending} />
                    </div>
                  }
                />
                <div className="grid gap-0">
                  <div className="grid gap-3 px-4 py-3" style={{ gridTemplateColumns: "120px minmax(0,1fr)" }}>
                    <label className="pt-2.5 text-[11px] uppercase tracking-[0.22em]" style={{ color: "var(--text-2)" }}>Method</label>
                    <select
                      value={method}
                      onChange={(event) => setMethod(event.target.value)}
                      className="rounded px-3 py-2 text-sm outline-none"
                      style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
                    >
                      <option value="GET">GET</option>
                      <option value="POST">POST</option>
                    </select>
                  </div>
                  <div style={{ borderTop: "1px solid var(--border)" }} />
                  <div className="grid gap-3 px-4 py-3" style={{ gridTemplateColumns: "120px minmax(0,1fr)" }}>
                    <label className="pt-2.5 text-[11px] uppercase tracking-[0.22em]" style={{ color: "var(--text-2)" }}>Path</label>
                    <input
                      value={path}
                      onChange={(event) => setPath(event.target.value)}
                      className="rounded px-3 py-2 text-sm outline-none"
                      style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
                    />
                  </div>
                  <div style={{ borderTop: "1px solid var(--border)" }} />
                  <div className="grid gap-3 px-4 py-3" style={{ gridTemplateColumns: "120px minmax(0,1fr)" }}>
                    <label className="pt-2.5 text-[11px] uppercase tracking-[0.22em]" style={{ color: "var(--text-2)" }}>Headers</label>
                    <textarea
                      value={headers}
                      onChange={(event) => setHeaders(event.target.value)}
                      rows={5}
                      className="rounded px-3 py-2 font-mono text-xs outline-none"
                      style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)", resize: "vertical" }}
                    />
                  </div>
                  <div style={{ borderTop: "1px solid var(--border)" }} />
                  <div className="grid gap-3 px-4 py-3" style={{ gridTemplateColumns: "120px minmax(0,1fr)" }}>
                    <label className="pt-2.5 text-[11px] uppercase tracking-[0.22em]" style={{ color: "var(--text-2)" }}>Body</label>
                    <textarea
                      value={body}
                      onChange={(event) => setBody(event.target.value)}
                      rows={8}
                      className="rounded px-3 py-2 font-mono text-xs outline-none"
                      style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)", resize: "vertical" }}
                    />
                  </div>
                </div>
              </Panel>
              <Panel>
                <Header
                  title="Response"
                  subtitle="What external clients would receive from the embedded API."
                  actions={
                    response ? (
                      <div className="text-right">
                        <div className="text-xl font-semibold" style={{ color: statusColor(response.status) }}>
                          {response.status}
                        </div>
                        <div className="text-xs" style={{ color: "var(--text-1)" }}>
                          {responseDurationMs ?? 0} ms | {transportLabel(response.transport)}
                        </div>
                      </div>
                    ) : (
                      <div className="text-right">
                        <div className="text-xl font-semibold" style={{ color: "var(--text-1)" }}>Idle</div>
                        <div className="text-xs" style={{ color: "var(--text-2)" }}>No request yet</div>
                      </div>
                    )
                  }
                />
                <div className="px-4 py-3">
                  {requestError && (
                    <div
                      className="mb-3 rounded px-3 py-2.5 text-xs"
                      style={{
                        background: "rgba(127, 29, 29, 0.28)",
                        border: "1px solid rgba(248, 113, 113, 0.24)",
                        color: "#fecaca",
                      }}
                    >
                      {requestError}
                    </div>
                  )}
                  {response?.status === 200 && normalizePath(path) === "/v1/models/load" && (
                    <div
                      className="mb-3 rounded px-3 py-2.5 text-xs"
                      style={{
                        background: "rgba(34,211,238,0.08)",
                        border: "1px solid rgba(34,211,238,0.16)",
                        color: "var(--text-1)",
                      }}
                    >
                      `POST /v1/models/load` accepts the request and then loads in the background. Use `POST /v1/models/stats` with the same model name until the stage becomes `ready`.
                    </div>
                  )}
                  <div className="grid gap-4">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.2em]" style={{ color: "var(--text-2)" }}>Headers</div>
                      <pre className="mt-1.5 overflow-x-auto rounded px-3 py-2.5 text-xs" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}>
                        <code>{response ? response.headers.map(([key, value]) => `${key}: ${value}`).join("\n") : "Run a request to inspect response headers."}</code>
                      </pre>
                    </div>
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.2em]" style={{ color: "var(--text-2)" }}>Body</div>
                      <pre className="mt-1.5 overflow-x-auto overflow-y-auto rounded px-3 py-2.5 text-xs" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)", maxHeight: "420px" }}>
                        <code>{response?.body ?? "Run a request to inspect the live embedded API response."}</code>
                      </pre>
                    </div>
                  </div>
                </div>
              </Panel>

              <Panel>
                <Header title="Recent Requests" subtitle="Jump back to recent endpoints and resend them quickly." />
                {recentRequests.length === 0 ? (
                  <div className="px-4 py-5 text-xs" style={{ color: "var(--text-1)" }}>
                    No requests sent yet.
                  </div>
                ) : (
                  recentRequests.map((entry) => (
                    <button
                      key={entry.id}
                      onClick={() => {
                        setMethod(entry.method);
                        setPath(entry.path);
                        setBody(entry.body);
                        setRequestError(null);
                      }}
                      className="grid w-full items-center gap-3 px-4 py-3 text-left transition"
                      style={{
                        gridTemplateColumns: "92px minmax(0,1fr) 96px 90px",
                        borderTop: "1px solid var(--border)",
                        color: "var(--text-0)",
                      }}
                    >
                      <span style={{ color: "var(--text-2)" }}>{entry.timestamp}</span>
                      <span>
                        <span className="font-semibold">{entry.method}</span>
                        <span className="ml-3 font-mono text-sm" style={{ color: "var(--text-1)" }}>
                          {entry.path}
                        </span>
                      </span>
                      <span style={{ color: statusColor(entry.status) }}>{entry.status}</span>
                      <span className="text-right" style={{ color: "var(--text-2)" }}>
                        {entry.durationMs} ms
                      </span>
                    </button>
                  ))
                )}
              </Panel>
            </div>
          </div>
        )}

        {activeTab === "docs" && (
          <div className="grid gap-4">
            {docs.map((doc) => (
              <Panel key={doc.title}>
                <div className="px-5 py-4">
                  <div className="text-[13px] font-semibold" style={{ color: "var(--text-0)" }}>{doc.title}</div>
                  <p className="mt-1.5 text-xs leading-5" style={{ color: "var(--text-1)" }}>{doc.body}</p>
                  {doc.code && (
                    <pre className="mt-3 overflow-x-auto rounded px-3 py-2.5 text-xs" style={{ background: "rgba(15,23,42,0.55)", border: "1px solid rgba(255,255,255,0.06)", color: "var(--text-0)" }}>
                      <code>{doc.code}</code>
                    </pre>
                  )}
                </div>
              </Panel>
            ))}
          </div>
        )}

        {activeTab === "logs" && (
          <Panel>
            <Header
              title="Logs"
              subtitle="Live Rust tracing plus HTTP middleware events."
              actions={
                <div className="flex flex-wrap gap-2">
                  <Button label={logsAutoRefresh ? "Auto-refresh On" : "Auto-refresh Off"} onClick={() => setLogsAutoRefresh((value) => !value)} />
                  <Button label="Refresh" onClick={() => void loadLogs()} />
                  <Button label="Clear" onClick={() => { api.clearLogs().then(() => loadLogs()).catch(() => undefined); }} />
                </div>
              }
            />
            <div className="grid gap-3 px-4 py-3 md:grid-cols-[200px_minmax(0,1fr)_110px]">
              <select
                value={logsLevel}
                onChange={(event) => setLogsLevel(event.target.value)}
                className="rounded px-3 py-2 text-xs outline-none"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              >
                {logLevels.map((level) => (
                  <option key={level} value={level}>{level}</option>
                ))}
              </select>
              <input
                value={logsQuery}
                onChange={(event) => setLogsQuery(event.target.value)}
                placeholder="Search target or message..."
                className="rounded px-3 py-2 text-xs outline-none"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
              />
              <div className="flex items-center justify-end text-xs" style={{ color: "var(--text-2)" }}>
                {filteredLogs.length} entries
              </div>
            </div>
            <div className="px-4 pb-4">
              <div className="flex max-h-[40rem] flex-col gap-3 overflow-y-auto">
                {filteredLogs.length === 0 && (
                  <div className="rounded px-3 py-5 text-xs" style={{ background: "var(--surface-2)", color: "var(--text-1)" }}>
                    No log entries match the current filter.
                  </div>
                )}
                {filteredLogs.map((entry, index) => (
                  <div
                    key={`${entry.timestamp}-${index}`}
                    className="rounded px-3 py-2.5"
                    style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
                  >
                    <div className="flex flex-wrap items-center gap-2.5 text-xs">
                      <span style={{ color: "var(--text-2)" }}>{entry.timestamp}</span>
                      <span
                        className="font-semibold uppercase"
                        style={{
                          color:
                            entry.level.toUpperCase() === "ERROR"
                              ? "#f87171"
                              : entry.level.toUpperCase() === "WARN"
                                ? "#fbbf24"
                                : entry.level.toUpperCase() === "INFO"
                                  ? "#22d3ee"
                                  : "var(--text-1)",
                        }}
                      >
                        {entry.level}
                      </span>
                      <span style={{ color: "var(--text-1)" }}>{entry.target}</span>
                    </div>
                    <pre className="mt-2 whitespace-pre-wrap text-xs" style={{ color: "var(--text-0)" }}>
                      <code>{entry.message}</code>
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          </Panel>
        )}

        {activeTab === "prompt" && (
          <Panel>
            <Header title="Raw Prompt" subtitle="The last rendered prompt sent to the inference backend." />
            <div className="px-4 py-3">
              <pre className="max-h-[42rem] overflow-auto rounded px-3 py-3 text-xs" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}>
                <code>{rawPrompt}</code>
              </pre>
            </div>
          </Panel>
        )}

        {activeTab === "trace" && (
          <Panel>
            <Header title="Parse Trace" subtitle="Inspect normalization and tool parsing output from the last response." />
            <div className="px-4 py-3">
              <pre className="max-h-[42rem] overflow-auto rounded px-3 py-3 text-xs" style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}>
                <code>{parseTrace}</code>
              </pre>
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}
