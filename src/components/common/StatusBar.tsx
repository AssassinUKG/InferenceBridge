import type {
  AppSettings,
  ContextStatus,
  LoadProgress,
  ProcessStatusInfo,
} from "../../lib/types";

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

function apiStateTone(state: string) {
  if (state === "Running") {
    return "border-emerald-400/18 bg-emerald-400/10 text-emerald-200";
  }
  if (state === "Starting") {
    return "border-amber-400/18 bg-amber-400/10 text-amber-200";
  }
  if (state === "Error") {
    return "border-rose-400/18 bg-rose-400/10 text-rose-200";
  }
  return "border-white/8 bg-white/5 text-slate-400";
}

export function StatusBar({ processStatus, contextStatus, settings }: Props) {
  const model = processStatus?.model ?? "No model loaded";
  const state = processStatus?.state ?? "Idle";
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
    <footer className="border-t border-white/8 bg-slate-950/82 px-4 py-2 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl flex-col gap-2 text-xs text-slate-400 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-wrap items-center gap-3">
          <span className="inline-flex items-center gap-2 rounded-full border border-white/8 bg-white/5 px-3 py-1">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                state === "Running"
                  ? "bg-emerald-400 shadow-[0_0_14px_rgba(74,222,128,0.9)]"
                  : state === "Starting"
                    ? "animate-pulse bg-amber-400"
                    : state === "Stopping"
                      ? "bg-orange-400"
                      : "bg-slate-600"
              }`}
            />
            <span className="font-medium text-slate-200">{model}</span>
          </span>

          {processStatus?.backend && (
            <span className="rounded-full border border-cyan-400/18 bg-cyan-400/10 px-3 py-1 text-cyan-200">
              {processStatus.backend}
            </span>
          )}

          {processStatus?.previous_model &&
            processStatus.previous_model !== processStatus.model && (
              <span className="text-indigo-300/85">
                prev: {processStatus.previous_model}
              </span>
            )}

          {crashes > 0 && (
            <span className="text-orange-300">Crashes: {crashes}</span>
          )}

          <span className={`rounded-full border px-3 py-1 ${apiStateTone(apiState)}`}>
            API {apiState}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <span className="text-slate-500">{apiUrl}</span>

          {processStatus?.api_error && (
            <span className="max-w-[28rem] truncate text-rose-300" title={processStatus.api_error}>
              {processStatus.api_error}
            </span>
          )}

          {contextStatus.total_tokens > 0 ? (
            <span className={contextStatus.used_tokens > 0 ? kvColor : "text-slate-400"}>
              KV:{" "}
              {contextStatus.used_tokens > 0
                ? `${contextStatus.used_tokens.toLocaleString()}/${contextStatus.total_tokens.toLocaleString()} (${pct}%)`
                : `${contextStatus.total_tokens.toLocaleString()} ctx`}
            </span>
          ) : state === "Running" ? (
            <span className="text-slate-500">KV: waiting...</span>
          ) : null}

          {processStatus?.server_version && (
            <span className="text-slate-500">
              llama.cpp {processStatus.server_version}
            </span>
          )}
        </div>
      </div>
    </footer>
  );
}
