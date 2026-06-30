import { useCallback, useEffect, useState, type ReactNode } from "react";
import type { ApiAccessInfo, ApiServerAction, AppSettings, LlamaServerInfo, LoadProgress, ProcessStatusInfo, RuntimeDoctorReport, RuntimePackInfo, TemplateDryRunReport } from "../../lib/types";
import * as api from "../../lib/tauri";
import { formatCliArgs, parseCliArgs } from "../../lib/args";

interface Props {
  onSaved?: (settings: AppSettings) => void;
  processStatus: ProcessStatusInfo | null;
  loadProgress: LoadProgress | null;
  apiAction?: ApiServerAction;
  onSetApiServerRunning: (running: boolean) => void | Promise<void>;
}

// Shared primitives

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

type SpecPreset = "disabled" | "self-mtp" | "dflash" | "custom";

function specPresetFor(settings: AppSettings): SpecPreset {
  const specType = settings.spec_type.trim();
  const draftPath = settings.draft_model_path.trim();
  if (!specType && !draftPath) return "disabled";
  if (specType === "draft-mtp" && !draftPath) return "self-mtp";
  if (specType === "draft-dflash") return "dflash";
  return "custom";
}

function applySpecPreset(settings: AppSettings, preset: SpecPreset): AppSettings {
  switch (preset) {
    case "disabled":
      return {
        ...settings,
        draft_model_path: "",
        spec_type: "",
        spec_draft_n_max: 0,
      };
    case "self-mtp":
      return {
        ...settings,
        draft_model_path: "",
        spec_type: "draft-mtp",
        spec_draft_n_max: settings.spec_draft_n_max > 0 ? settings.spec_draft_n_max : 2,
      };
    case "dflash":
      return {
        ...settings,
        spec_type: "draft-dflash",
        spec_draft_n_max: settings.spec_draft_n_max > 0 ? settings.spec_draft_n_max : 8,
      };
    default:
      return settings;
  }
}

function applyQwenToolReliabilityPreset(settings: AppSettings): AppSettings {
  return {
    ...settings,
    use_jinja: true,
    template_mode: "repo",
    template_name: null,
    reasoning_mode: "off",
    reasoning_preserve: false,
    chat_template_kwargs_json: JSON.stringify({ enable_thinking: false, preserve_thinking: false }),
    parallel_slots: 1,
    cont_batching: true,
    flash_attn: true,
    draft_model_path: "",
    spec_type: "",
    spec_draft_n_max: 0,
    draft_max_tokens: 0,
    draft_min_tokens: 0,
    draft_p_min: 0,
  };
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

function formatBytes(value: number | null | undefined) {
  if (!value || value <= 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}

function VersionChip({ value }: { value: string }) {
  return (
    <span
      className="rounded-full px-2 py-0.5 text-[10px] font-semibold"
      style={{ background: "rgba(255,255,255,0.10)", color: "var(--text-1)" }}
    >
      {value}
    </span>
  );
}

function RuntimeManager({
  settings,
  setSettings,
  llamaInfo,
  llamaLoading,
  runtimePacks,
  runtimeLoading,
  runtimeError,
  runtimeSearch,
  setRuntimeSearch,
  runtimeFilter,
  setRuntimeFilter,
  downloadStatus,
  downloadingBackend,
  loadProgress,
  onRefresh,
  onInstall,
}: {
  settings: AppSettings;
  setSettings: (settings: AppSettings) => void;
  llamaInfo: LlamaServerInfo | null;
  llamaLoading: boolean;
  runtimePacks: RuntimePackInfo[];
  runtimeLoading: boolean;
  runtimeError: string | null;
  runtimeSearch: string;
  setRuntimeSearch: (value: string) => void;
  runtimeFilter: string;
  setRuntimeFilter: (value: string) => void;
  downloadStatus: string | null;
  downloadingBackend: string | null;
  loadProgress: LoadProgress | null;
  onRefresh: () => void;
  onInstall: (backend: string) => void;
}) {
  const selectedBackend = settings.backend_preference === "cpu" ? "cpu" : "cuda";
  const selectedPack = runtimePacks.find((pack) => pack.backend === selectedBackend) ?? runtimePacks[0];
  const query = runtimeSearch.trim().toLowerCase();
  const rows = runtimePacks.filter((pack) => {
    const matchesSearch =
      !query ||
      pack.name.toLowerCase().includes(query) ||
      pack.description.toLowerCase().includes(query) ||
      pack.backend.toLowerCase().includes(query);
    const matchesFilter =
      runtimeFilter === "all" ||
      (runtimeFilter === "updates" && pack.update_available) ||
      (runtimeFilter === "installed" && pack.installed_version) ||
      (runtimeFilter === "compatible" && pack.available);
    return matchesSearch && matchesFilter;
  });
  const activeDownload = loadProgress?.stage === "downloading" && !loadProgress.done;
  const flagSupport = llamaInfo?.flag_support ?? null;
  const missingFlags = flagSupport?.missing_critical_flags ?? [];

  return (
    <SectionPanel>
      <SectionHeader title="Runtime" description="Managed llama.cpp engines, updates, and install state." />
      <div className="space-y-3 p-4">
        <div className="overflow-hidden rounded" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <div className="flex items-center gap-4 px-3 py-2.5">
            <div className="w-32 shrink-0 text-sm font-medium" style={{ color: "var(--text-0)" }}>GGUF</div>
            <FlatSelect
              value={selectedBackend}
              onChange={(backend) => setSettings({ ...settings, backend_preference: backend })}
            >
              {runtimePacks.map((pack) => (
                <option key={pack.id} value={pack.backend}>
                  {pack.name}{pack.latest_version ? ` ${pack.latest_version}` : ""}
                </option>
              ))}
              {runtimePacks.length === 0 && <option value={selectedBackend}>Managed llama.cpp</option>}
            </FlatSelect>
          </div>
          <Divider />
          <div className="flex items-center justify-between gap-4 px-3 py-2.5">
            <div>
              <div className="text-sm" style={{ color: "var(--text-0)" }}>Use selected runtime when launching models</div>
              <div className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>
                {selectedPack?.description ?? "Install a runtime pack to manage llama.cpp locally."}
              </div>
            </div>
            <Toggle
              checked={settings.backend_preference !== "cpu"}
              onChange={() => setSettings({ ...settings, backend_preference: settings.backend_preference === "cpu" ? "cuda" : "cpu" })}
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 rounded px-3 py-2.5" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <div className="mr-auto text-sm" style={{ color: "var(--text-0)" }}>Runtime updates channel</div>
          <FlatBtn label={runtimeLoading ? "Checking..." : "Check for updates"} onClick={onRefresh} disabled={runtimeLoading || !!downloadingBackend} />
          <select
            value="stable"
            onChange={() => {}}
            className="rounded px-2.5 py-1.5 text-xs outline-none"
            style={{ background: "var(--surface-3)", border: "1px solid var(--border)", color: "var(--text-1)" }}
          >
            <option value="stable">Stable</option>
          </select>
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>Engines & Frameworks</div>
              <div className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>
                Installed: {llamaLoading ? "checking..." : llamaInfo?.version ?? "none"}
              </div>
            </div>
            {runtimeError && <div className="text-xs" style={{ color: "#f87171" }}>{runtimeError}</div>}
          </div>
          {flagSupport && (
            <div
              className="mb-2 rounded px-3 py-2 text-xs"
              style={{
                background: missingFlags.length > 0 || flagSupport.error ? "rgba(251,191,36,0.08)" : "rgba(52,211,153,0.08)",
                border: missingFlags.length > 0 || flagSupport.error ? "1px solid rgba(251,191,36,0.24)" : "1px solid rgba(52,211,153,0.18)",
                color: "var(--text-1)",
              }}
            >
              <div className="font-semibold" style={{ color: missingFlags.length > 0 || flagSupport.error ? "#fbbf24" : "#34d399" }}>
                llama.cpp flags: {flagSupport.checked ? `${flagSupport.supported_flags.length} checked` : "not checked"}
              </div>
              <div className="mt-1 break-all" style={{ color: "var(--text-2)" }}>
                {flagSupport.binary_path ?? llamaInfo?.binary_path ?? "No llama-server binary found"}
              </div>
              {flagSupport.error ? (
                <div className="mt-1" style={{ color: "#fbbf24" }}>{flagSupport.error}</div>
              ) : missingFlags.length > 0 ? (
                <div className="mt-1">
                  Missing watched flags: {missingFlags.slice(0, 10).join(", ")}
                  {missingFlags.length > 10 ? ` +${missingFlags.length - 10} more` : ""}
                </div>
              ) : (
                <div className="mt-1">All watched reliability flags are advertised by this binary.</div>
              )}
            </div>
          )}
          <div className="mb-2 grid gap-2 md:grid-cols-[1fr_180px]">
            <input
              value={runtimeSearch}
              onChange={(event) => setRuntimeSearch(event.target.value)}
              placeholder="Search runtimes..."
              className="rounded px-3 py-1.5 text-sm outline-none"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
            />
            <select
              value={runtimeFilter}
              onChange={(event) => setRuntimeFilter(event.target.value)}
              className="rounded px-3 py-1.5 text-sm outline-none"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-0)" }}
            >
              <option value="compatible">Compatible only</option>
              <option value="updates">Updates only</option>
              <option value="installed">Installed</option>
              <option value="all">All types</option>
            </select>
          </div>

          <div className="overflow-hidden rounded" style={{ border: "1px solid var(--border)" }}>
            {rows.length === 0 ? (
              <div className="px-3 py-8 text-center text-sm" style={{ color: "var(--text-2)" }}>
                {runtimeLoading ? "Checking runtime packs..." : "No runtime packs matched."}
              </div>
            ) : (
              rows.map((pack, index) => {
                const installing = downloadingBackend === pack.backend;
                const isLatest = pack.available && !pack.update_available;
                const installLabel = pack.installed_version ? "Update" : "Install";
                return (
                  <div key={pack.id} className="px-3 py-3" style={{ borderTop: index === 0 ? "none" : "1px solid var(--border)" }}>
                    <div className="flex items-start gap-3">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-sm font-semibold" style={{ color: "var(--text-0)" }}>{pack.name}</span>
                          {pack.installed_version && <VersionChip value={pack.installed_version} />}
                          {pack.latest_version && pack.installed_version && pack.latest_version !== pack.installed_version && <span className="text-xs" style={{ color: "var(--text-2)" }}>→</span>}
                          {pack.latest_version && <VersionChip value={pack.latest_version} />}
                        </div>
                        <div className="mt-1 text-xs" style={{ color: "var(--text-1)" }}>{pack.description}</div>
                        {pack.error ? (
                          <div className="mt-1 text-[11px]" style={{ color: "#f87171" }}>{pack.error}</div>
                        ) : pack.latest_version ? (
                          <div className="mt-1 text-[11px]" style={{ color: "var(--text-2)" }}>{pack.latest_version} release asset ready</div>
                        ) : null}
                        {installing && activeDownload && (
                          <div className="mt-2">
                            <div className="h-1.5 overflow-hidden rounded" style={{ background: "rgba(255,255,255,0.08)" }}>
                              <div className="h-full rounded transition-all" style={{ width: `${Math.max(4, Math.round((loadProgress?.progress ?? 0) * 100))}%`, background: "#22d3ee" }} />
                            </div>
                            <div className="mt-1 text-[11px]" style={{ color: "var(--text-2)" }}>{loadProgress?.message ?? downloadStatus}</div>
                          </div>
                        )}
                      </div>
                      <div className="flex shrink-0 items-center gap-3">
                        {isLatest ? (
                          <span className="text-xs font-semibold" style={{ color: "#34d399" }}>✓ Latest version</span>
                        ) : pack.available ? (
                          <button
                            onClick={() => onInstall(pack.backend)}
                            disabled={!!downloadingBackend}
                            className="rounded px-3 py-2 text-xs font-semibold transition"
                            style={{ background: "#3b82f6", color: "#fff", border: "none", cursor: downloadingBackend ? "not-allowed" : "pointer", opacity: downloadingBackend && !installing ? 0.55 : 1 }}
                          >
                            {installing ? "Installing..." : `${installLabel}${pack.size_bytes ? `  ${formatBytes(pack.size_bytes)}` : ""}`}
                          </button>
                        ) : (
                          <span className="text-xs" style={{ color: "var(--text-2)" }}>Unavailable</span>
                        )}
                        <button className="rounded px-2 py-1 text-sm" style={{ background: "transparent", border: "none", color: "var(--text-2)", cursor: "pointer" }}>⋮</button>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {downloadStatus && !activeDownload && (
          <div
            className="rounded px-3 py-2 text-xs"
            style={{
              color: downloadStatus.startsWith("Error") ? "#f87171" : "#22d3ee",
              background: downloadStatus.startsWith("Error") ? "rgba(248,113,113,0.08)" : "rgba(34,211,238,0.08)",
              border: downloadStatus.startsWith("Error") ? "1px solid rgba(248,113,113,0.20)" : "1px solid rgba(34,211,238,0.18)",
            }}
          >
            {downloadStatus}
          </div>
        )}
      </div>
    </SectionPanel>
  );
}

// API key field

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
            placeholder="No key set - public access"
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
            Use as: <span className="font-mono">Authorization: Bearer {visible ? value : value.slice(0, 6) + "..."}</span>
          </p>
        )}
      </div>
    </div>
  );
}

// Main component

export function SettingsPanel({ onSaved, processStatus, loadProgress, apiAction = null, onSetApiServerRunning }: Props) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [persistedSettings, setPersistedSettings] = useState<AppSettings | null>(null);
  const [accessInfo, setAccessInfo] = useState<ApiAccessInfo | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [llamaInfo, setLlamaInfo] = useState<LlamaServerInfo | null>(null);
  const [llamaLoading, setLlamaLoading] = useState(false);
  const [runtimePacks, setRuntimePacks] = useState<RuntimePackInfo[]>([]);
  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [runtimeSearch, setRuntimeSearch] = useState("");
  const [runtimeFilter, setRuntimeFilter] = useState("compatible");
  const [downloadStatus, setDownloadStatus] = useState<string | null>(null);
  const [downloadingBackend, setDownloadingBackend] = useState<string | null>(null);
  const [providerCheck, setProviderCheck] = useState<RuntimeDoctorReport | null>(null);
  const [providerChecking, setProviderChecking] = useState(false);
  const [templateDryRun, setTemplateDryRun] = useState<TemplateDryRunReport | null>(null);
  const [templateDryRunLoading, setTemplateDryRunLoading] = useState(false);
  const [configPath, setConfigPath] = useState<string | null>(null);
  const [configPathCopied, setConfigPathCopied] = useState(false);

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

  const loadConfigPath = useCallback(async () => {
    try {
      setConfigPath(await api.getConfigFilePath());
    } catch {
      setConfigPath(null);
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

  const loadRuntimePacks = useCallback(async () => {
    setRuntimeLoading(true);
    setRuntimeError(null);
    try {
      setRuntimePacks(await api.listRuntimePacks());
    } catch (e) {
      setRuntimeError(String(e));
    } finally {
      setRuntimeLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
    loadApiAccessInfo();
    loadConfigPath();
    loadLlamaInfo();
    loadRuntimePacks();
  }, [loadSettings, loadApiAccessInfo, loadConfigPath, loadLlamaInfo, loadRuntimePacks]);

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
    setDownloadingBackend(backend);
    setDownloadStatus(`Installing ${backend.toUpperCase()} runtime...`);
    try {
      const result = await api.downloadLlamaBuild(backend);
      setDownloadStatus(result);
      await Promise.all([loadLlamaInfo(), loadRuntimePacks()]);
      setTimeout(() => setDownloadStatus(null), 5000);
    } catch (e) {
      setDownloadStatus(`Error: ${String(e)}`);
    } finally {
      setDownloadingBackend(null);
    }
  };

  const handleUpdate = async () => {
    setDownloadStatus("Checking runtime updates...");
    await Promise.all([loadLlamaInfo(), loadRuntimePacks()]);
    setDownloadStatus("Runtime update check complete.");
    setTimeout(() => setDownloadStatus(null), 3000);
  };

  const checkProviders = async () => {
    setProviderChecking(true);
    try {
      setProviderCheck(await api.getRuntimeDoctor());
    } catch (e) {
      setError(String(e));
    } finally {
      setProviderChecking(false);
    }
  };

  const runTemplateDryRun = async () => {
    if (!settings) return;
    setTemplateDryRunLoading(true);
    try {
      setTemplateDryRun(await api.templateDryRun({
        modelName: processStatus?.model ?? null,
        useJinja: settings.use_jinja,
        templateMode: settings.template_mode,
        templateName: settings.template_name,
        customTemplatePath: settings.custom_template_path,
        chatTemplateKwargsJson: settings.chat_template_kwargs_json,
        reasoningMode: settings.reasoning_mode,
        parallelSlots: settings.parallel_slots,
      }));
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setTemplateDryRunLoading(false);
    }
  };

  if (!settings) {
    return (
      <div className="p-4 text-sm" style={{ color: "var(--text-1)" }}>
        {error ? `Error: ${error}` : "Loading..."}
      </div>
    );
  }

  const buildReachableServerUrl = (host: string, port: number) =>
    `http://${host === "0.0.0.0" ? "127.0.0.1" : host}:${port}/v1`;
  const serverUrl = buildReachableServerUrl(settings.server_host, settings.server_port);
  const persistedServerUrl = persistedSettings
    ? buildReachableServerUrl(persistedSettings.server_host, persistedSettings.server_port)
    : processStatus?.api_url ?? serverUrl;
  const apiState = processStatus?.api_state ?? "Idle";
  const apiReachable = processStatus?.api_reachable ?? false;
  const apiStopping = apiAction === "stopping" || apiState === "Stopping";
  const apiStarting = apiAction === "starting" || apiState === "Starting";
  const apiRunning = (apiState === "Running" && apiReachable) || (apiState === "Running" && !apiStopping);
  const apiActive = apiRunning || apiStarting || apiStopping || apiReachable;
  const apiBusy = apiStarting || apiStopping;
  const apiError = processStatus?.api_error ?? null;
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
  const isLanMode = settings.server_host === "0.0.0.0";
  const lanUrl =
    accessInfo?.lan_host ? `http://${accessInfo.lan_host}:${settings.server_port}/v1` : null;
  const apiConfigDirty =
    !!persistedSettings &&
    (persistedSettings.server_host !== settings.server_host ||
      persistedSettings.server_port !== settings.server_port ||
      (persistedSettings.api_key ?? "") !== (settings.api_key ?? ""));
  const dirtyApiMessage = `Unsaved API changes: ${serverUrl} will not be used until you save. Current saved endpoint is ${persistedServerUrl}.`;
  const statusMessage = apiStopping
    ? `Stopping public API on ${persistedServerUrl}. This can take a few seconds while the port is released.`
    : apiStarting
      ? `Public API is starting on ${serverUrl}.`
      : apiConfigDirty
        ? `Saved API is still using ${persistedServerUrl}. Save settings to apply ${serverUrl}.`
        : modelTransitionActive
          ? modelTransition?.message ??
            `Model transition in progress. Public API will come back on ${serverUrl} once loading finishes.`
          : apiRunning
            ? isLanMode && lanUrl
              ? `Public API reachable locally on ${serverUrl} and on your LAN at ${lanUrl}`
              : `Public API reachable on ${serverUrl}`
            : apiState === "Error"
              ? apiError ?? `Public API is not reachable on ${serverUrl}.`
              : "Public API is currently off.";
  const apiActionLabel = apiStopping
    ? "Stopping API..."
    : apiStarting
      ? "Starting API..."
      : apiActive
        ? "Stop API"
        : apiState === "Error"
          ? apiConfigDirty ? "Save & Retry API" : "Retry API"
          : apiConfigDirty ? "Save & Start API" : "Start API";
  const providerDirty =
    !!persistedSettings &&
    (persistedSettings.active_provider !== settings.active_provider ||
      persistedSettings.lm_studio_enabled !== settings.lm_studio_enabled ||
      persistedSettings.lm_studio_base_url !== settings.lm_studio_base_url ||
      (persistedSettings.lm_studio_api_key ?? "") !== (settings.lm_studio_api_key ?? "") ||
      persistedSettings.sglang_enabled !== settings.sglang_enabled ||
      persistedSettings.sglang_base_url !== settings.sglang_base_url ||
      (persistedSettings.sglang_api_key ?? "") !== (settings.sglang_api_key ?? ""));
  const lmStudioBaseUrl = settings.lm_studio_base_url.trim() || "http://127.0.0.1:1234/v1";
  const normalizedLmStudioUrl = lmStudioBaseUrl.endsWith("/v1")
    ? lmStudioBaseUrl
    : `${lmStudioBaseUrl.replace(/\/$/, "")}/v1`;
  const sglangBaseUrl = settings.sglang_base_url.trim() || "http://127.0.0.1:30000/v1";
  const normalizedSglangUrl = sglangBaseUrl.endsWith("/v1")
    ? sglangBaseUrl
    : `${sglangBaseUrl.replace(/\/$/, "")}/v1`;
  const configuredLmStudioProbe = providerCheck?.providers.find((provider) => provider.id === "lm-studio-configured");
  const configuredSglangProbe = providerCheck?.providers.find((provider) => provider.id === "sglang-configured");

  const handleApiServerToggle = async () => {
    if (apiBusy) {
      return;
    }
    const shouldStart = !apiActive;
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

        {/* API Surface */}
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
                    : apiStopping || apiStarting
                      ? "#fde68a"
                    : apiRunning
                      ? "#34d399"
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
                disabled={apiBusy || saving}
                primary
              />
            </div>
          </div>
          {((apiError && !modelTransitionActive) || apiConfigDirty) && (
            <>
              <Divider />
              <div className="px-4 py-2.5 text-xs" style={{ color: "#fca5a5", background: "rgba(248,113,113,0.06)" }}>
                {apiConfigDirty ? dirtyApiMessage : apiError}
              </div>
            </>
          )}
          <Divider />
          <div
            className="flex items-center gap-2 px-4 py-2.5"
            style={{ background: apiReachable ? "rgba(52,211,153,0.04)" : "rgba(255,255,255,0.03)" }}
          >
            <span className="text-xs" style={{ color: "var(--text-1)" }}>
              {apiReachable
                ? isLanMode
                  ? "Local URL"
                  : "Reachable at"
                : apiConfigDirty
                  ? "Edited endpoint"
                  : isLanMode
                    ? "Local URL"
                    : "Configured endpoint"}
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
          {configPath && (
            <>
              <Divider />
              <div className="flex items-center gap-3 px-4 py-3">
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--text-2)" }}>
                    Config File
                  </p>
                  <p className="mt-1 truncate font-mono text-xs" style={{ color: "var(--text-1)" }} title={configPath}>
                    {configPath}
                  </p>
                </div>
                <FlatBtn
                  label={configPathCopied ? "Copied" : "Copy"}
                  onClick={async () => {
                    await navigator.clipboard.writeText(configPath);
                    setConfigPathCopied(true);
                    window.setTimeout(() => setConfigPathCopied(false), 1500);
                  }}
                />
                <FlatBtn
                  label="Open Folder"
                  onClick={() => {
                    void api.showInFolder(configPath);
                  }}
                  primary
                />
              </div>
            </>
          )}
        </SectionPanel>

        {/* Providers */}
        <SectionPanel>
          <SectionHeader
            title="Providers"
            description="Choose the backend behind InferenceBridge's own API. Keep the API server running as the stable front door."
          />
          <FieldRow label="Active Provider" hint="Routes /v1/chat/completions">
            <FlatSelect
              value={settings.active_provider}
              onChange={(v) => setSettings({ ...settings, active_provider: v })}
            >
              <option value="managed_llamacpp">Managed llama.cpp</option>
              <option value="lm_studio">LM Studio</option>
              <option value="sglang">SGLang</option>
            </FlatSelect>
          </FieldRow>
          <Divider />
          <FieldRow label="LM Studio" hint="OpenAI-compatible local server">
            <div className="flex items-center gap-3">
              <Toggle
                checked={settings.lm_studio_enabled}
                onChange={() =>
                  setSettings({
                    ...settings,
                    lm_studio_enabled: !settings.lm_studio_enabled,
                    active_provider: !settings.lm_studio_enabled ? "lm_studio" : settings.active_provider,
                  })
                }
              />
              <span className="text-sm" style={{ color: "var(--text-1)" }}>
                {settings.lm_studio_enabled ? "Enabled" : "Disabled"}
              </span>
            </div>
          </FieldRow>
          <Divider />
          <FieldRow label="LM Studio URL" hint="Usually ends with /v1">
            <FlatInput
              value={settings.lm_studio_base_url}
              onChange={(v) => setSettings({ ...settings, lm_studio_base_url: v })}
              placeholder="http://127.0.0.1:1234/v1"
            />
            <p className="mt-1 text-xs font-mono" style={{ color: "var(--text-2)" }}>
              Target: {normalizedLmStudioUrl}
            </p>
          </FieldRow>
          <Divider />
          <FieldRow label="LM Studio Key" hint="Usually blank">
            <FlatInput
              value={settings.lm_studio_api_key ?? ""}
              onChange={(v) => setSettings({ ...settings, lm_studio_api_key: v || null })}
              placeholder="Optional Bearer token"
            />
          </FieldRow>
          <Divider />
          <FieldRow label="SGLang" hint="External OpenAI-compatible server">
            <div className="flex items-center gap-3">
              <Toggle
                checked={settings.sglang_enabled}
                onChange={() =>
                  setSettings({
                    ...settings,
                    sglang_enabled: !settings.sglang_enabled,
                    active_provider: !settings.sglang_enabled ? "sglang" : settings.active_provider,
                  })
                }
              />
              <span className="text-sm" style={{ color: "var(--text-1)" }}>
                {settings.sglang_enabled ? "Enabled" : "Disabled"}
              </span>
            </div>
          </FieldRow>
          <Divider />
          <FieldRow label="SGLang URL" hint="Default server port is 30000">
            <FlatInput
              value={settings.sglang_base_url}
              onChange={(v) => setSettings({ ...settings, sglang_base_url: v })}
              placeholder="http://127.0.0.1:30000/v1"
            />
            <p className="mt-1 text-xs font-mono" style={{ color: "var(--text-2)" }}>
              Target: {normalizedSglangUrl}
            </p>
          </FieldRow>
          <Divider />
          <FieldRow label="SGLang Key" hint="Usually blank">
            <FlatInput
              value={settings.sglang_api_key ?? ""}
              onChange={(v) => setSettings({ ...settings, sglang_api_key: v || null })}
              placeholder="Optional Bearer token"
            />
          </FieldRow>
          <Divider />
          <FieldRow label="Hugging Face" hint="Used for Hub search and gated downloads">
            <FlatInput
              value={settings.hf_api_key ?? ""}
              onChange={(v) => setSettings({ ...settings, hf_api_key: v || null })}
              placeholder="hf_..."
            />
            <p className="mt-1 text-xs" style={{ color: "var(--text-2)" }}>
              Adds Authorization to Hugging Face model discovery, metadata sync, and downloads.
            </p>
          </FieldRow>
          <Divider />
          <div className="flex items-center justify-between gap-3 px-4 py-3">
            <span className="text-xs" style={{ color: providerDirty ? "#fbbf24" : "var(--text-2)" }}>
              {providerDirty
                ? "Save Settings before requests use this provider config."
                : settings.active_provider === "lm_studio"
                  ? "Chat completions and model listing will route through LM Studio."
                  : settings.active_provider === "sglang"
                    ? "Chat completions and model listing will route through SGLang."
                  : "Chat completions use the managed llama.cpp runtime."}
            </span>
            <div className="flex items-center gap-2">
              <FlatBtn label={providerChecking ? "Checking..." : "Test"} onClick={() => void checkProviders()} disabled={providerChecking || providerDirty} />
              <FlatBtn label={saving ? "Saving..." : "Save Settings"} onClick={() => void saveSettings()} disabled={saving} primary />
            </div>
          </div>
          {configuredLmStudioProbe && (
            <>
              <Divider />
              <div className="px-4 py-2.5 text-xs" style={{ color: configuredLmStudioProbe.reachable ? "#34d399" : "#f87171", background: configuredLmStudioProbe.reachable ? "rgba(52,211,153,0.05)" : "rgba(248,113,113,0.06)" }}>
                LM Studio {configuredLmStudioProbe.status}: {configuredLmStudioProbe.model_count} model{configuredLmStudioProbe.model_count === 1 ? "" : "s"} at {configuredLmStudioProbe.base_url}
                {configuredLmStudioProbe.error ? ` (${configuredLmStudioProbe.error})` : ""}
              </div>
            </>
          )}
          {configuredSglangProbe && (
            <>
              <Divider />
              <div className="px-4 py-2.5 text-xs" style={{ color: configuredSglangProbe.reachable ? "#34d399" : "#f87171", background: configuredSglangProbe.reachable ? "rgba(52,211,153,0.05)" : "rgba(248,113,113,0.06)" }}>
                SGLang {configuredSglangProbe.status}: {configuredSglangProbe.model_count} model{configuredSglangProbe.model_count === 1 ? "" : "s"} at {configuredSglangProbe.base_url}
                {configuredSglangProbe.error ? ` (${configuredSglangProbe.error})` : ""}
              </div>
            </>
          )}
        </SectionPanel>

        <RuntimeManager
          settings={settings}
          setSettings={setSettings}
          llamaInfo={llamaInfo}
          llamaLoading={llamaLoading}
          runtimePacks={runtimePacks}
          runtimeLoading={runtimeLoading}
          runtimeError={runtimeError}
          runtimeSearch={runtimeSearch}
          setRuntimeSearch={setRuntimeSearch}
          runtimeFilter={runtimeFilter}
          setRuntimeFilter={setRuntimeFilter}
          downloadStatus={downloadStatus}
          downloadingBackend={downloadingBackend}
          loadProgress={loadProgress}
          onRefresh={() => void handleUpdate()}
          onInstall={(backend) => void handleDownload(backend)}
        />

        <div className="grid gap-3 xl:grid-cols-2">
          {/* Execution Defaults */}
          <SectionPanel>
            <SectionHeader title="Execution Defaults" description="Runtime preferences used when launching llama-server." />

            <FieldRow label="Backend" hint="Which build to prefer.">
              <FlatSelect
                value={settings.backend_preference}
                onChange={(v) => setSettings({ ...settings, backend_preference: v })}
              >
                <option value="auto">Auto (CUDA / CPU fallback)</option>
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

        {/* Inference Engine */}
        <SectionPanel>
          <SectionHeader title="Inference Engine" description="llama-server parameters and template controls. These take effect on the next model load or swap." />

          <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3" style={{ borderBottom: "1px solid var(--border)" }}>
            <div className="min-w-0">
              <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>Qwen tool reliability</p>
              <p className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>
                Applies the conservative llama.cpp setup for Qwen tool calls and previews the rendered fallback prompt.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <FlatBtn
                label="Apply Qwen Preset"
                onClick={() => {
                  setSettings(applyQwenToolReliabilityPreset(settings));
                  setTemplateDryRun(null);
                }}
              />
              <FlatBtn
                label={templateDryRunLoading ? "Dry Running..." : "Template Dry Run"}
                onClick={() => void runTemplateDryRun()}
                disabled={templateDryRunLoading}
                primary
              />
            </div>
          </div>

          {templateDryRun && (
            <>
              <div className="grid gap-3 px-4 py-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]" style={{ borderBottom: "1px solid var(--border)" }}>
                <div className="min-w-0">
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                    Template Checks
                  </div>
                  <div className="space-y-1 text-xs">
                    <div style={{ color: "var(--text-1)" }}>
                      {templateDryRun.model_name} / {templateDryRun.family} / {templateDryRun.renderer} / {templateDryRun.tool_format}
                    </div>
                    {templateDryRun.warnings.map((warning) => (
                      <div key={warning} style={{ color: "#fbbf24" }}>WARN: {warning}</div>
                    ))}
                    {templateDryRun.checks.map((check) => (
                      <div key={check} style={{ color: "#34d399" }}>OK: {check}</div>
                    ))}
                  </div>
                </div>
                <div className="min-w-0">
                  <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
                    Rendered Prompt
                  </div>
                  <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded p-3 font-mono text-[11px] leading-5" style={{ background: "var(--bg)", border: "1px solid var(--border)", color: "var(--text-1)" }}>
                    {templateDryRun.prompt}
                  </pre>
                </div>
              </div>
            </>
          )}

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
              <Divider />
              <FieldRow label="Fit Mode" hint="--fit: on, off, or auto (blank = unset)">
                <FlatInput
                  value={settings.fit_mode}
                  onChange={(v) => setSettings({ ...settings, fit_mode: v })}
                  placeholder="on"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Cache RAM (MiB)" hint="--cache-ram: RAM budget for KV cache">
                <FlatInput
                  type="number"
                  value={settings.cache_ram_mb ?? ""}
                  onChange={(v) =>
                    setSettings({
                      ...settings,
                      cache_ram_mb: v.trim() ? Math.max(0, Number(v) || 0) : null,
                    })
                  }
                  min={0}
                  placeholder="4096"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Ctx Copy" hint="-ctxcp: context copy checkpoints (blank = unset)">
                <FlatInput
                  type="number"
                  value={settings.ctxcp ?? ""}
                  onChange={(v) =>
                    setSettings({
                      ...settings,
                      ctxcp: v.trim() ? Math.max(0, Number(v) || 0) : null,
                    })
                  }
                  min={0}
                  placeholder="2"
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
                {
                  label: "Use Jinja",
                  hint: "--jinja: let llama.cpp render repo/custom templates",
                  key: "use_jinja" as const,
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

              <Divider />
              <FieldRow label="Reasoning Mode" hint="--reasoning: on, off, or auto (blank = unset)">
                <FlatInput
                  value={settings.reasoning_mode}
                  onChange={(v) => setSettings({ ...settings, reasoning_mode: v })}
                  placeholder="on"
                />
              </FieldRow>
              <Divider />
              <div className="flex items-center justify-between px-4 py-3">
                <div>
                  <p className="text-sm font-medium" style={{ color: "var(--text-0)" }}>
                    Preserve Reasoning
                  </p>
                  <p className="mt-0.5 text-xs" style={{ color: "var(--text-2)" }}>
                    --reasoning-preserve: keep supported template reasoning output instead of stripping it.
                  </p>
                </div>
                <Toggle
                  checked={settings.reasoning_preserve}
                  onChange={() =>
                    setSettings({
                      ...settings,
                      reasoning_preserve: !settings.reasoning_preserve,
                    })
                  }
                />
              </div>
              <Divider />
              <FieldRow label="Template Mode" hint="repo, custom, or builtin">
                <FlatInput
                  value={settings.template_mode}
                  onChange={(v) => setSettings({ ...settings, template_mode: v })}
                  placeholder="repo"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Built-in Template" hint="Used when template mode is builtin">
                <FlatInput
                  value={settings.template_name ?? ""}
                  onChange={(v) =>
                    setSettings({
                      ...settings,
                      template_name: v.trim() ? v : null,
                    })
                  }
                  placeholder="chatml"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Custom Template Path" hint="Used when template mode is custom">
                <FlatInput
                  value={settings.custom_template_path ?? ""}
                  onChange={(v) =>
                    setSettings({
                      ...settings,
                      custom_template_path: v.trim() ? v : null,
                    })
                  }
                  placeholder="C:\\path\\to\\chat_template.jinja"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Template Kwargs JSON" hint="Passed to --chat-template-kwargs">
                <FlatInput
                  value={settings.chat_template_kwargs_json ?? ""}
                  onChange={(v) =>
                    setSettings({
                      ...settings,
                      chat_template_kwargs_json: v.trim() ? v : null,
                    })
                  }
                  placeholder='{"preserve_thinking": true}'
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Spec Preset" hint="Quick presets for llama.cpp speculative decoding">
                <FlatSelect
                  value={specPresetFor(settings)}
                  onChange={(v) => setSettings(applySpecPreset(settings, v as SpecPreset))}
                >
                  <option value="disabled">Disabled</option>
                  <option value="self-mtp">Self MTP / draft-mtp</option>
                  <option value="dflash">DFlash / draft-dflash</option>
                  <option value="custom">Custom raw args</option>
                </FlatSelect>
              </FieldRow>
              <Divider />
              <FieldRow label="Draft Model Path" hint="-md: GGUF draft model for speculative decoding">
                <FlatInput
                  value={settings.draft_model_path}
                  onChange={(v) => setSettings({ ...settings, draft_model_path: v })}
                  placeholder="C:\\path\\to\\draft-model.gguf"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Spec Type" hint="--spec-type: draft-mtp for self-MTP, draft-dflash for DFlash">
                <FlatInput
                  value={settings.spec_type}
                  onChange={(v) => setSettings({ ...settings, spec_type: v })}
                  placeholder="draft-mtp or draft-dflash"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Spec Draft N Max" hint="--spec-draft-n-max: 2 for Qwen self-MTP; 8-15 for DFlash testing">
                <FlatInput
                  type="number"
                  value={settings.spec_draft_n_max}
                  onChange={(v) =>
                    setSettings({
                      ...settings,
                      spec_draft_n_max: Math.max(0, Number(v) || 0),
                    })
                  }
                  min={0}
                  placeholder="3"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="Draft Legacy Args" hint="Optional --draft-max / --draft-min / --draft-p-min">
                <div className="grid grid-cols-3 gap-2">
                  <FlatInput
                    type="number"
                    value={settings.draft_max_tokens}
                    onChange={(v) =>
                      setSettings({ ...settings, draft_max_tokens: Math.max(0, Number(v) || 0) })
                    }
                    min={0}
                    placeholder="16"
                  />
                  <FlatInput
                    type="number"
                    value={settings.draft_min_tokens}
                    onChange={(v) =>
                      setSettings({ ...settings, draft_min_tokens: Math.max(0, Number(v) || 0) })
                    }
                    min={0}
                    placeholder="5"
                  />
                  <FlatInput
                    type="number"
                    value={settings.draft_p_min}
                    onChange={(v) =>
                      setSettings({ ...settings, draft_p_min: Math.max(0, Number(v) || 0) })
                    }
                    min={0}
                    placeholder="0.0"
                  />
                </div>
              </FieldRow>
              <Divider />
              <FieldRow label="Extra Args" hint="Raw llama-server args appended last">
                <FlatInput
                  value={formatCliArgs(settings.extra_args)}
                  onChange={(v) =>
                    setSettings({
                      ...settings,
                      extra_args: parseCliArgs(v),
                    })
                  }
                  placeholder="--temp 0.6 --top-p 0.95"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="DiffusionGemma Runner" hint="Path to llama-diffusion-cli from the DiffusionGemma llama.cpp build">
                <FlatInput
                  value={settings.llama_diffusion_cli_path}
                  onChange={(v) =>
                    setSettings({
                      ...settings,
                      llama_diffusion_cli_path: v,
                    })
                  }
                  placeholder="C:\\llama.cpp\\build\\bin\\Release\\llama-diffusion-cli.exe"
                />
              </FieldRow>
              <Divider />
              <FieldRow label="DiffusionGemma Defaults" hint="-n and --diffusion-kv-cache for llama-diffusion-cli">
                <div className="grid grid-cols-3 gap-2">
                  <FlatInput
                    type="number"
                    value={settings.diffusion_n_predict}
                    onChange={(v) =>
                      setSettings({
                        ...settings,
                        diffusion_n_predict: Math.max(1, Number(v) || 2048),
                      })
                    }
                    min={1}
                    placeholder="2048"
                  />
                  <select
                    className="w-full bg-[var(--bg-3)] border border-[var(--border-1)] rounded px-2 py-1.5 text-sm"
                    value={settings.diffusion_kv_cache}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        diffusion_kv_cache: e.target.value,
                      })
                    }
                  >
                    <option value="auto">KV auto</option>
                    <option value="on">KV on</option>
                    <option value="off">KV off</option>
                  </select>
                  <label className="flex items-center gap-2 text-sm text-[var(--text-2)]">
                    <input
                      type="checkbox"
                      checked={settings.diffusion_visual}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          diffusion_visual: e.target.checked,
                        })
                      }
                    />
                    Visual
                  </label>
                </div>
              </FieldRow>
              <Divider />
              <FieldRow label="DiffusionGemma Extra Args" hint="Raw llama-diffusion-cli args appended last">
                <FlatInput
                  value={formatCliArgs(settings.diffusion_extra_args)}
                  onChange={(v) =>
                    setSettings({
                      ...settings,
                      diffusion_extra_args: parseCliArgs(v),
                    })
                  }
                  placeholder="--diffusion-max-steps 48"
                />
              </FieldRow>
            </div>
          </div>
        </SectionPanel>

        {/* Model Directories */}
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
                  Remove
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

        {/* Lifecycle */}
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

        {/* Save bar */}
        <div className="flex items-center gap-3 pb-2">
          <FlatBtn
            label={saving ? "Saving..." : "Save Settings"}
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


