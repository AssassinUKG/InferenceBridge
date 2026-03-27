import { useCallback, useEffect, useState, type ReactNode } from "react";
import type { ApiAccessInfo, AppSettings, LlamaServerInfo, LoadProgress, ProcessStatusInfo } from "../../lib/types";
import * as api from "../../lib/tauri";

interface Props {
  onSaved?: (settings: AppSettings) => void;
  processStatus: ProcessStatusInfo | null;
  loadProgress: LoadProgress | null;
  onSetApiServerRunning: (running: boolean) => void | Promise<void>;
}

// ─── Shared primitives ────────────────────────────────────────────────────────

function SectionPanel({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border)",
        borderRadius: "10px",
        overflow: "hidden",
      }}
    >
      {children}
    </div>
  );
}

function SectionHeader({ title, description }: { title: string; description?: string }) {
  return (
    <div
      className="px-4 py-3"
      style={{ borderBottom: "1px solid var(--border)" }}
    >
      <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
        {title}
      </p>
      {description && (
        <p className="mt-0.5 text-xs" style={{ color: "var(--text-1)" }}>
          {description}
        </p>
      )}
    </div>
  );
}

function Divider() {
  return <div style={{ height: "1px", background: "var(--border)" }} />;
}

function FieldRow({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-center gap-4 px-4 py-3">
      <div className="w-40 shrink-0">
        <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
          {label}
        </p>
        {hint && (
          <p className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>
            {hint}
          </p>
        )}
      </div>
      <div className="flex-1">{children}</div>
    </div>
  );
}

function FlatInput({
  value,
  onChange,
  type = "text",
  placeholder,
  min,
  max,
}: {
  value: string | number;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
  min?: number;
  max?: number;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      min={min}
      max={max}
      className="w-full rounded py-1.5 px-3 text-sm outline-none transition"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        color: "var(--text-0)",
      }}
      onFocus={(e) =>
        ((e.currentTarget as HTMLInputElement).style.borderColor = "rgba(34,211,238,0.35)")
      }
      onBlur={(e) =>
        ((e.currentTarget as HTMLInputElement).style.borderColor = "var(--border)")
      }
    />
  );
}

function FlatSelect({
  value,
  onChange,
  children,
}: {
  value: string;
  onChange: (v: string) => void;
  children: ReactNode;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded py-1.5 px-3 text-sm outline-none transition"
      style={{
        background: "var(--surface-2)",
        border: "1px solid var(--border)",
        color: "var(--text-0)",
        cursor: "pointer",
      }}
    >
      {children}
    </select>
  );
}

function InfoTile({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="px-4 py-2.5">
      <p className="text-[10px] uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
        {label}
      </p>
      <p
        className={`mt-0.5 text-sm ${mono ? "font-mono break-all" : "font-medium"}`}
        style={{ color: "var(--text-0)" }}
      >
        {value}
      </p>
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: () => void }) {
  return (
    <button
      onClick={onChange}
      className="relative shrink-0 rounded-full transition"
      style={{
        width: "40px",
        height: "22px",
        background: checked ? "#22d3ee" : "var(--surface-3)",
        border: "none",
        cursor: "pointer",
      }}
    >
      <span
        className="absolute rounded-full bg-white transition-all"
        style={{
          width: "16px",
          height: "16px",
          top: "3px",
          left: checked ? "21px" : "3px",
        }}
      />
    </button>
  );
}

function FlatBtn({
  label,
  onClick,
  disabled,
  primary,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="rounded px-3 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-50"
      style={
        primary
          ? { background: "#22d3ee", color: "#0a0a0a", border: "none", cursor: disabled ? "not-allowed" : "pointer" }
          : {
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text-1)",
              cursor: disabled ? "not-allowed" : "pointer",
            }
      }
      onMouseEnter={(e) => {
        if (!disabled && primary) (e.currentTarget as HTMLButtonElement).style.filter = "brightness(1.08)";
      }}
      onMouseLeave={(e) => {
        if (primary) (e.currentTarget as HTMLButtonElement).style.filter = "";
      }}
    >
      {label}
    </button>
  );
}

// ─── API Key field ─────────────────────────────────────────────────────────────

function ApiKeyRow({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [visible, setVisible] = useState(false);
  const [copied, setCopied] = useState(false);

  function generate() {
    // ib- prefix + 32 random hex chars (128 bits of entropy)
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    const hex = Array.from(bytes).map((b) => b.toString(16).padStart(2, "0")).join("");
    onChange(`ib-${hex}`);
  }

  function copy() {
    if (!value) return;
    navigator.clipboard.writeText(value).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="flex items-start gap-4 px-4 py-3">
      <div className="w-40 shrink-0">
        <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>API Key</p>
        <p className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>
          Bearer token for external clients. Leave blank for open access.
        </p>
      </div>
      <div className="flex flex-1 flex-col gap-2">
        <div className="flex items-center gap-2">
          <input
            type={visible ? "text" : "password"}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="No key set — public access"
            className="flex-1 rounded px-3 py-1.5 text-sm font-mono outline-none"
            style={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              color: "var(--text-0)",
            }}
          />
          <button
            onClick={() => setVisible((v) => !v)}
            title={visible ? "Hide key" : "Show key"}
            className="rounded px-2.5 py-1.5 text-xs transition"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-1)", cursor: "pointer" }}
          >
            {visible ? "Hide" : "Show"}
          </button>
          <button
            onClick={copy}
            disabled={!value}
            className="rounded px-2.5 py-1.5 text-xs transition"
            style={{
              background: copied ? "rgba(52,211,153,0.15)" : "var(--surface-2)",
              border: copied ? "1px solid rgba(52,211,153,0.3)" : "1px solid var(--border)",
              color: copied ? "#34d399" : "var(--text-1)",
              cursor: value ? "pointer" : "not-allowed",
              opacity: value ? 1 : 0.5,
            }}
          >
            {copied ? "Copied" : "Copy"}
          </button>
          <button
            onClick={generate}
            className="rounded px-2.5 py-1.5 text-xs font-medium transition"
            style={{ background: "#22d3ee", color: "#0a0a0a", border: "none", cursor: "pointer" }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.filter = "brightness(1.08)")}
            onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.filter = "")}
          >
            Generate
          </button>
        </div>
        {value && (
          <p className="text-[10px]" style={{ color: "var(--text-2)" }}>
            Use as: <span className="font-mono">Authorization: Bearer {visible ? value : value.slice(0, 6) + "…"}</span>
          </p>
        )}
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function SettingsPanel({ onSaved, processStatus, loadProgress, onSetApiServerRunning }: Props) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [persistedSettings, setPersistedSettings] = useState<AppSettings | null>(null);
  const [accessInfo, setAccessInfo] = useState<ApiAccessInfo | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [llamaInfo, setLlamaInfo] = useState<LlamaServerInfo | null>(null);
  const [llamaLoading, setLlamaLoading] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<string | null>(null);

  const loadSettings = useCallback(async () => {
    try {
      const s = await api.getSettings();
      setSettings(s);
      setPersistedSettings(s);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const loadApiAccessInfo = useCallback(async () => {
    try {
      setAccessInfo(await api.getApiAccessInfo());
    } catch {
      // Non-critical; LAN preview can be omitted.
    }
  }, []);

  const loadLlamaInfo = useCallback(async () => {
    setLlamaLoading(true);
    try {
      setLlamaInfo(await api.getLlamaInfo());
    } catch {
      // non-critical
    } finally {
      setLlamaLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
    loadApiAccessInfo();
    loadLlamaInfo();
  }, [loadSettings, loadApiAccessInfo, loadLlamaInfo]);

  const saveSettings = useCallback(async (nextSettings?: AppSettings) => {
    const settingsToSave = nextSettings ?? settings;
    if (!settingsToSave) return false;
    setSaving(true);
    setSaved(false);
    try {
      await api.updateSettings(settingsToSave);
      setSaved(true);
      setError(null);
      setPersistedSettings(settingsToSave);
      await loadApiAccessInfo();
      onSaved?.(settingsToSave);
      setTimeout(() => setSaved(false), 2000);
      return true;
    } catch (e) {
      setError(String(e));
      return false;
    } finally {
      setSaving(false);
    }
  }, [settings, loadApiAccessInfo, onSaved]);

  const handleDownload = async (backend: string) => {
    setDownloadStatus(`Downloading ${backend} build…`);
    try {
      const result = await api.downloadLlamaBuild(backend);
      setDownloadStatus(result);
      loadLlamaInfo();
      setTimeout(() => setDownloadStatus(null), 5000);
    } catch (e) {
      setDownloadStatus(`Error: ${String(e)}`);
    }
  };

  const handleUpdate = async () => {
    setDownloadStatus("Checking for updates…");
    try {
      const result = await api.updateLlamaServer();
      setDownloadStatus(result);
      loadLlamaInfo();
      setTimeout(() => setDownloadStatus(null), 5000);
    } catch (e) {
      setDownloadStatus(`Error: ${String(e)}`);
    }
  };

  if (!settings) {
    return (
      <div className="p-4 text-sm" style={{ color: "var(--text-1)" }}>
        {error ? `Error: ${error}` : "Loading…"}
      </div>
    );
  }

  const serverUrl = `http://${settings.server_host}:${settings.server_port}/v1`;
  const persistedServerUrl = persistedSettings
    ? `http://${persistedSettings.server_host}:${persistedSettings.server_port}/v1`
    : processStatus?.api_url ?? serverUrl;
  const apiState = processStatus?.api_state ?? "Idle";
  const apiReachable = processStatus?.api_reachable ?? false;
  const apiRunning = apiState === "Running" && apiReachable;
  const apiStarting = apiState === "Starting";
  const apiError = processStatus?.api_error ?? null;
  const modelTransition =
    loadProgress && !loadProgress.done
      ? loadProgress
      : processStatus?.model_load_progress ?? null;
  const modelTransitionActive =
    (!!modelTransition && !modelTransition.done) ||
    ["Loading", "Swapping", "Unloading"].includes(
      processStatus?.model_load_state ?? "Idle"
    );
  const isLanMode = settings.server_host === "0.0.0.0";
  const lanUrl =
    accessInfo?.lan_host ? `http://${accessInfo.lan_host}:${settings.server_port}/v1` : null;
  const apiConfigDirty =
    !!persistedSettings &&
    (persistedSettings.server_host !== settings.server_host ||
      persistedSettings.server_port !== settings.server_port ||
      (persistedSettings.api_key ?? "") !== (settings.api_key ?? ""));
  const statusMessage = apiConfigDirty
    ? `Unsaved API changes: ${serverUrl} will not be used until you save. Current saved endpoint is ${persistedServerUrl}.`
    : modelTransitionActive
      ? modelTransition?.message ??
        `Model transition in progress. Public API will come back on ${serverUrl} once loading finishes.`
    : apiRunning
      ? `Public API reachable on ${serverUrl}`
      : apiStarting
        ? `Public API is starting on ${serverUrl}`
        : apiState === "Error"
          ? apiError ?? `Public API is not reachable on ${serverUrl}.`
          : "Public API is currently off.";
  const apiActionLabel = apiRunning || apiStarting
    ? "Stop API"
    : apiState === "Error"
      ? apiConfigDirty ? "Save & Retry API" : "Retry API"
      : apiConfigDirty ? "Save & Start API" : "Start API";

  const handleApiServerToggle = async () => {
    const shouldStart = !(apiRunning || apiStarting);
    if (shouldStart && apiConfigDirty) {
      const saved = await saveSettings(settings);
      if (!saved) {
        return;
      }
    }

    try {
      await onSetApiServerRunning(shouldStart);
    } catch (e) {
      setError(String(e));
    }
  };

  const applyNetworkMode = async (host: string) => {
    const nextSettings = { ...settings, server_host: host };
    setSettings(nextSettings);
    const saved = await saveSettings(nextSettings);
    if (!saved) {
      return;
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-3 flex flex-col gap-3">

        {/* ── API Surface ── */}
        <SectionPanel>
          <SectionHeader title="API Surface" description="OpenAI-compatible endpoint for external clients. The desktop UI talks to InferenceBridge directly. Save API changes before starting or retrying the public endpoint." />

          <FieldRow label="Host">
            <div className="flex flex-col gap-2">
              <FlatInput
                value={settings.server_host}
                onChange={(v) => setSettings({ ...settings, server_host: v })}
                placeholder="127.0.0.1"
              />
              <div className="flex flex-wrap items-center gap-2">
                <FlatBtn
                  label="Local only"
                  onClick={() => {
                    void applyNetworkMode("127.0.0.1");
                  }}
                  disabled={saving || settings.server_host === "127.0.0.1"}
                />
                <FlatBtn
                  label="Local network"
                  onClick={() => {
                    void applyNetworkMode("0.0.0.0");
                  }}
                  disabled={saving || isLanMode}
                  primary
                />
                <span className="text-xs" style={{ color: "var(--text-2)" }}>
                  {isLanMode
                    ? "Binding on all interfaces so other devices on your LAN can reach the API."
                    : "Local network mode binds the API on 0.0.0.0 for other devices on your LAN."}
                </span>
              </div>
            </div>
          </FieldRow>
          <Divider />
          <FieldRow label="Port">
            <FlatInput
              type="number"
              value={settings.server_port}
              onChange={(v) => setSettings({ ...settings, server_port: Number(v) || 8800 })}
              min={1}
              max={65535}
            />
          </FieldRow>
          <Divider />
          <ApiKeyRow
            value={settings.api_key ?? ""}
            onChange={(v) => setSettings({ ...settings, api_key: v || null })}
          />
          <Divider />
          <div className="flex items-center justify-between px-4 py-3">
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
                Server Status
              </p>
              <p
                className="mt-0.5 text-xs"
                style={{
                  color: modelTransitionActive
                    ? "#fcd34d"
                    : apiRunning
                      ? "#34d399"
                      : apiStarting
                        ? "#fde68a"
                        : apiState === "Error"
                          ? "#f87171"
                          : "var(--text-2)",
                }}
              >
                {statusMessage}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <FlatBtn
                label={apiActionLabel}
                onClick={handleApiServerToggle}
                primary
              />
            </div>
          </div>
          {((apiError && !modelTransitionActive) || apiConfigDirty) && (
            <>
              <Divider />
              <div className="px-4 py-2.5 text-xs" style={{ color: "#fca5a5", background: "rgba(248,113,113,0.06)" }}>
                {apiConfigDirty ? statusMessage : apiError}
              </div>
            </>
          )}
          <Divider />
          <div
            className="flex items-center gap-2 px-4 py-2.5"
            style={{ background: apiReachable ? "rgba(52,211,153,0.04)" : "rgba(255,255,255,0.03)" }}
          >
            <span className="text-xs" style={{ color: "var(--text-1)" }}>
              {apiReachable ? "Reachable at" : apiConfigDirty ? "Edited endpoint" : "Configured endpoint"}
            </span>
            <span className="font-mono text-xs" style={{ color: apiReachable ? "#34d399" : "var(--text-1)" }}>{serverUrl}</span>
          </div>
          {apiConfigDirty && (
            <>
              <Divider />
              <div className="flex items-center gap-2 px-4 py-2.5" style={{ background: "rgba(255,255,255,0.02)" }}>
                <span className="text-xs" style={{ color: "var(--text-2)" }}>
                  Current saved endpoint
                </span>
                <span className="font-mono text-xs" style={{ color: "var(--text-1)" }}>{persistedServerUrl}</span>
              </div>
            </>
          )}
          {isLanMode && (
            <>
              <Divider />
              <div className="flex items-center gap-2 px-4 py-2.5" style={{ background: "rgba(34,211,238,0.05)" }}>
                <span className="text-xs" style={{ color: "var(--text-2)" }}>
                  LAN URL
                </span>
                <span className="font-mono text-xs" style={{ color: "#22d3ee" }}>
                  {lanUrl ?? `http://<your-lan-ip>:${settings.server_port}/v1`}
                </span>
              </div>
            </>
          )}
          <Divider />
          <div className="flex items-center justify-between px-4 py-3">
            <div>
              <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
                Start API on launch
              </p>
              <p className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>
                Automatically expose the public OpenAI-compatible server when the app opens.
              </p>
            </div>
            <Toggle
              checked={settings.api_autostart}
              onChange={() => setSettings({ ...settings, api_autostart: !settings.api_autostart })}
            />
          </div>
          <Divider />
          <div className="px-4 py-2.5 text-xs" style={{ color: "var(--text-2)" }}>
            Run either the GUI app or the headless `serve` command, not both on the same port.
          </div>
        </SectionPanel>

        <div className="grid gap-3 xl:grid-cols-2">
          {/* ── llama.cpp Server ── */}
          <SectionPanel>
            <SectionHeader title="llama.cpp Server" description="Managed binary details and update controls." />

            {llamaLoading ? (
              <div className="px-4 py-3 text-xs" style={{ color: "var(--text-2)" }}>Loading…</div>
            ) : (
              <>
                <InfoTile label="Version" value={llamaInfo?.version ?? "Not installed"} />
                <Divider />
                <InfoTile label="Managed Dir" value={llamaInfo?.managed_dir ?? "Unknown"} mono />
                {llamaInfo?.binary_path && (
                  <>
                    <Divider />
                    <InfoTile label="Binary Path" value={llamaInfo.binary_path} mono />
                  </>
                )}
                {llamaInfo?.update_available && llamaInfo.latest_version && (
                  <>
                    <Divider />
                    <div
                      className="px-4 py-2 text-xs"
                      style={{
                        background: "rgba(251,191,36,0.06)",
                        color: "#fbbf24",
                      }}
                    >
                      Update available: {llamaInfo.latest_version}
                    </div>
                  </>
                )}
              </>
            )}

            <Divider />
            <div className="flex flex-wrap items-center gap-2 px-4 py-3">
              <FlatBtn label="Check Updates" onClick={handleUpdate} disabled={!!downloadStatus} />
              <FlatBtn
                label="Download CUDA"
                onClick={() => handleDownload("cuda")}
                disabled={!!downloadStatus}
                primary
              />
              <FlatBtn
                label="Download CPU"
                onClick={() => handleDownload("cpu")}
                disabled={!!downloadStatus}
              />
            </div>

            {downloadStatus && (
              <>
                <Divider />
                <div
                  className="px-4 py-2.5 text-xs"
                  style={{
                    color: downloadStatus.startsWith("Error") ? "#f87171" : "#22d3ee",
                    background: downloadStatus.startsWith("Error")
                      ? "rgba(248,113,113,0.06)"
                      : "rgba(34,211,238,0.06)",
                  }}
                >
                  {downloadStatus}
                </div>
              </>
            )}
          </SectionPanel>

          {/* ── Execution Defaults ── */}
          <SectionPanel>
            <SectionHeader title="Execution Defaults" description="Runtime preferences used when launching llama-server." />

            <FieldRow label="Backend" hint="Which build to prefer.">
              <FlatSelect
                value={settings.backend_preference}
                onChange={(v) => setSettings({ ...settings, backend_preference: v })}
              >
                <option value="auto">Auto (CUDA → CPU fallback)</option>
                <option value="cuda">Force CUDA</option>
                <option value="cpu">CPU only (AVX2)</option>
              </FlatSelect>
            </FieldRow>
            <Divider />
            <FieldRow label="GPU Layers" hint="-1 = all to GPU">
              <FlatInput
                type="number"
                value={settings.gpu_layers}
                onChange={(v) => setSettings({ ...settings, gpu_layers: Number(v) || 0 })}
              />
            </FieldRow>
            <Divider />
            <FieldRow label="Main GPU" hint="Device index (multi-GPU)">
              <FlatInput
                type="number"
                value={settings.main_gpu}
                onChange={(v) => setSettings({ ...settings, main_gpu: Number(v) || 0 })}
                min={0}
              />
            </FieldRow>
            <Divider />
            <FieldRow label="CPU Threads" hint="0 = auto">
              <FlatInput
                type="number"
                value={settings.threads}
                onChange={(v) => setSettings({ ...settings, threads: Number(v) || 0 })}
                min={0}
              />
            </FieldRow>
            <Divider />
            <FieldRow label="Batch Threads" hint="0 = same as threads">
              <FlatInput
                type="number"
                value={settings.threads_batch}
                onChange={(v) => setSettings({ ...settings, threads_batch: Number(v) || 0 })}
                min={0}
              />
            </FieldRow>
            <Divider />
            <FieldRow label="Theme">
              <FlatInput
                value={settings.theme}
                onChange={(v) => setSettings({ ...settings, theme: v })}
              />
            </FieldRow>
          </SectionPanel>
        </div>

        {/* ── Inference Engine ── */}
        <SectionPanel>
          <SectionHeader title="Inference Engine" description="llama-server parameters — take effect on next model load." />

          <div className="grid gap-0 xl:grid-cols-2">
            <div>
              <FieldRow label="Batch Size" hint="-b: logical batch (0=default 2048)">
                <FlatInput
                  type="number"
                  value={settings.batch_size}
                  onChange={(v) => setSettings({ ...settings, batch_size: Number(v) || 0 })}
                  min={0}
                  placeholder="0"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Micro-Batch" hint="-ub: physical batch (0=default 512)">
                <FlatInput
                  type="number"
                  value={settings.ubatch_size}
                  onChange={(v) => setSettings({ ...settings, ubatch_size: Number(v) || 0 })}
                  min={0}
                  placeholder="0"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Parallel Slots" hint="--parallel: concurrent requests">
                <FlatInput
                  type="number"
                  value={settings.parallel_slots}
                  onChange={(v) => setSettings({ ...settings, parallel_slots: Math.max(1, Number(v) || 1) })}
                  min={1}
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Defrag Threshold" hint="KV cache defrag (0=off, 0.1=default)">
                <FlatInput
                  type="number"
                  value={settings.defrag_thold}
                  onChange={(v) => setSettings({ ...settings, defrag_thold: parseFloat(v) || 0 })}
                  min={0}
                  max={1}
                  placeholder="0.1"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="RoPE Scale" hint="--rope-freq-scale (0=auto)">
                <FlatInput
                  type="number"
                  value={settings.rope_freq_scale}
                  onChange={(v) => setSettings({ ...settings, rope_freq_scale: parseFloat(v) || 0 })}
                  min={0}
                  placeholder="0"
                />
              </FieldRow>
            </div>

            <div>
              {[
                {
                  label: "Flash Attention",
                  hint: "-fa: faster attention (requires compatible GPU)",
                  key: "flash_attn" as const,
                },
                {
                  label: "Memory Map",
                  hint: "--mmap: map model file into memory (default on)",
                  key: "use_mmap" as const,
                },
                {
                  label: "Memory Lock",
                  hint: "--mlock: pin model in RAM, prevents swapping",
                  key: "use_mlock" as const,
                },
                {
                  label: "Continuous Batching",
                  hint: "-cb: process multiple requests together",
                  key: "cont_batching" as const,
                },
              ].map((item, i) => (
                <div key={item.key}>
                  {i > 0 && <Divider />}
                  <div className="flex items-center justify-between px-4 py-3">
                    <div>
                      <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
                        {item.label}
                      </p>
                      <p className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>
                        {item.hint}
                      </p>
                    </div>
                    <Toggle
                      checked={!!settings[item.key]}
                      onChange={() => setSettings({ ...settings, [item.key]: !settings[item.key] })}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </SectionPanel>

        {/* ── Model Directories ── */}
        <SectionPanel>
          <SectionHeader
            title="Model Directories"
            description="Folders scanned for .gguf files. Changes trigger an immediate re-scan on save."
          />
          <div className="px-4 py-3 flex flex-col gap-2">
            {(settings.scan_dirs ?? []).map((dir, i) => (
              <div key={i} className="flex items-center gap-2">
                <FlatInput
                  value={dir}
                  onChange={(v) => {
                    const dirs = [...(settings.scan_dirs ?? [])];
                    dirs[i] = v;
                    setSettings({ ...settings, scan_dirs: dirs });
                  }}
                  placeholder="/path/to/models"
                />
                <button
                  onClick={() => {
                    const dirs = (settings.scan_dirs ?? []).filter((_, j) => j !== i);
                    setSettings({ ...settings, scan_dirs: dirs });
                  }}
                  className="shrink-0 rounded px-2 py-1.5 text-xs transition"
                  style={{
                    background: "rgba(248,113,113,0.08)",
                    border: "1px solid rgba(248,113,113,0.22)",
                    color: "#f87171",
                    cursor: "pointer",
                  }}
                  title="Remove directory"
                >
                  ✕
                </button>
              </div>
            ))}
            <button
              onClick={() =>
                setSettings({
                  ...settings,
                  scan_dirs: [...(settings.scan_dirs ?? []), ""],
                })
              }
              className="self-start rounded px-3 py-1.5 text-xs transition"
              style={{
                background: "var(--surface-2)",
                border: "1px solid var(--border)",
                color: "var(--text-1)",
                cursor: "pointer",
              }}
            >
              + Add Directory
            </button>
          </div>
          <Divider />
          <div className="px-4 py-2.5 text-xs" style={{ color: "var(--text-2)" }}>
            Save settings to apply directory changes and trigger a model re-scan.
          </div>
        </SectionPanel>

        {/* ── Lifecycle ── */}
        <SectionPanel>
          <SectionHeader title="Lifecycle" />
          <div className="px-4 py-3">
            <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
              Managed backend exit
            </p>
            <p className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>
              The desktop app now always unloads the current model and stops the managed llama-server when the window closes.
            </p>
          </div>
        </SectionPanel>

        {/* ── Save bar ── */}
        <div className="flex items-center gap-3 pb-2">
          <FlatBtn
            label={saving ? "Saving…" : "Save Settings"}
            onClick={() => {
              void saveSettings();
            }}
            disabled={saving}
            primary
          />
          {saved && (
            <span className="text-xs" style={{ color: "#34d399" }}>Saved</span>
          )}
          {error && (
            <span className="text-xs" style={{ color: "#f87171" }}>{error}</span>
          )}
        </div>

      </div>
    </div>
  );
}
