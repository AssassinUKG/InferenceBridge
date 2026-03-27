import { useEffect, useState } from "react";
import { useModel } from "./hooks/useModel";
import { useSession } from "./hooks/useSession";
import { useChat } from "./hooks/useChat";
import { useContext } from "./hooks/useContext";
import { StatusBar } from "./components/common/StatusBar";
import { Sidebar } from "./components/common/Sidebar";
import { ChatPanel } from "./components/Chat/ChatPanel";
import { ModelSelector } from "./components/Model/ModelSelector";
import { SettingsPanel } from "./components/Model/SettingsPanel";
import { ContextPanel } from "./components/Context/ContextPanel";
import { DebugInspector } from "./components/Debug/DebugInspector";
import { ModelBrowser } from "./components/Model/ModelBrowser";
import type { AppSettings } from "./lib/types";
import * as api from "./lib/tauri";

type Tab = "chat" | "models" | "browse" | "context" | "debug" | "settings";

const TAB_LABELS: Record<Tab, string> = {
  chat: "Chat",
  models: "Models",
  browse: "Browse",
  context: "Context",
  debug: "API",
  settings: "Settings",
};

function buildReachableApiUrl(settings: AppSettings | null) {
  const host =
    settings?.server_host === "0.0.0.0"
      ? "127.0.0.1"
      : settings?.server_host ?? "127.0.0.1";
  const port = settings?.server_port ?? 8800;
  return `http://${host}:${port}/v1`;
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");
  const [settings, setSettings] = useState<AppSettings | null>(null);

  const model = useModel();
  const session = useSession();
  const chat = useChat(session.activeId);
  const context = useContext();

  const hasModel = !!model.processStatus?.model;
  const loadedModelName = model.processStatus?.model ?? null;
  const loadedModelSupportsVision = loadedModelName
    ? model.models.find((entry) => entry.filename === loadedModelName)?.supports_vision ?? false
    : false;
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

  const tabs: Tab[] = ["chat", "models", "browse", "context", "debug", "settings"];

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
        {/* Sidebar — chat only */}
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
              error={chat.error}
              hasModel={hasModel}
              hasSession={!!session.activeId}
              loadedModel={loadedModelName}
              loadedModelSupportsVision={loadedModelSupportsVision}
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
            <ContextPanel status={context} />
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
