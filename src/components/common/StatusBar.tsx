import type {
  AppSettings,
  ContextStatus,
  LoadProgress,
  ProcessStatusInfo,
} from "../../lib/types";
import { Box, Braces, Database, TriangleAlert } from "lucide-react";

interface Props {
  processStatus: ProcessStatusInfo | null;
  contextStatus: ContextStatus;
  settings: AppSettings | null;
  loadProgress: LoadProgress | null;
}

function buildApiUrl(settings: AppSettings | null) {
  if (!settings) {
    return "http://127.0.0.1:8800/v1";
  }
  const host = settings.server_host === "0.0.0.0" ? "127.0.0.1" : settings.server_host;
  return `http://${host}:${settings.server_port}/v1`;
}

export function StatusBar({ processStatus, contextStatus, settings }: Props) {
  const model = processStatus?.model ?? "No model loaded";
  const hasModel = !!processStatus?.model;
  const crashes = processStatus?.crash_count ?? 0;
  const pct = Math.round(contextStatus.fill_ratio * 100);
  const apiUrl = processStatus?.api_url ?? buildApiUrl(settings);
  const apiState = processStatus?.api_state ?? "Idle";

  const kvColor =
    pct > 95
      ? "text-rose-300"
      : pct > 80
        ? "text-amber-300"
        : pct > 0
          ? "text-emerald-300"
          : "text-slate-500";

  return (
    <footer className="flex h-8 shrink-0 items-center gap-4 border-t border-[var(--border)] bg-[var(--surface-0)] px-4 text-[11px] text-[var(--text-2)]">
      <span className={`flex min-w-0 items-center gap-1.5 ${hasModel ? "text-emerald-300" : ""}`} title={model}>
        <Box size={12} />
        <span className="max-w-[280px] truncate">{model}</span>
        {processStatus?.backend && <span className="text-[var(--text-3)]">/{processStatus.backend}</span>}
      </span>

      <span className="hidden items-center gap-1.5 md:flex" title={apiUrl}>
        <Braces size={12} />
        <span className={apiState === "Running" ? "text-emerald-300" : apiState === "Error" ? "text-rose-300" : ""}>
          API {apiState}
        </span>
        <span className="max-w-[240px] truncate text-[var(--text-3)]">{apiUrl}</span>
      </span>

      {contextStatus.total_tokens > 0 ? (
        <span className={`ml-auto flex items-center gap-1.5 ${contextStatus.used_tokens > 0 ? kvColor : "text-[var(--text-2)]"}`}>
          <Database size={12} />
          {contextStatus.used_tokens > 0
            ? `${contextStatus.used_tokens.toLocaleString()}/${contextStatus.total_tokens.toLocaleString()} (${pct}%)`
            : `${contextStatus.total_tokens.toLocaleString()} context`}
        </span>
      ) : (
        <span className="ml-auto" />
      )}

      {crashes > 0 && (
        <span className="flex items-center gap-1.5 text-amber-300">
          <TriangleAlert size={12} />
          {crashes} crash{crashes === 1 ? "" : "es"}
        </span>
      )}

      {processStatus?.server_version && (
        <span className="hidden text-[var(--text-3)] xl:inline">llama.cpp {processStatus.server_version}</span>
      )}

      {processStatus?.api_error && (
        <span className="max-w-[260px] truncate text-rose-300" title={processStatus.api_error}>
          {processStatus.api_error}
        </span>
      )}
    </footer>
  );
}
