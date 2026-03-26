import type { AppSettings, ProcessStatusInfo } from "../../lib/types";

interface Props {
  status: ProcessStatusInfo | null;
  settings: AppSettings | null;
}

function buildServerUrl(settings: AppSettings | null) {
  if (!settings) {
    return "http://127.0.0.1:8800/v1";
  }

  return `http://${settings.server_host}:${settings.server_port}/v1`;
}

function apiTone(state: string) {
  if (state === "Running") {
    return "text-emerald-200";
  }
  if (state === "Starting") {
    return "text-amber-200";
  }
  if (state === "Error") {
    return "text-rose-200";
  }
  return "text-slate-200";
}

function statusTone(state: string) {
  if (state === "Running") {
    return "text-emerald-200";
  }
  if (state === "Starting") {
    return "text-amber-200";
  }
  if (state === "Stopping") {
    return "text-orange-200";
  }
  if (state === "Crashed") {
    return "text-rose-200";
  }
  return "text-slate-200";
}

export function ProcessStatus({ status, settings }: Props) {
  if (!status) {
    return (
      <section className="rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(15,23,42,0.82),rgba(8,15,29,0.94))] p-5">
        <h3 className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
          Process Status
        </h3>
        <p className="mt-3 text-sm text-slate-400">
          No process information is available yet.
        </p>
      </section>
    );
  }

  const apiUrl = status.api_url ?? buildServerUrl(settings);

  return (
    <section className="rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(15,23,42,0.82),rgba(8,15,29,0.94))] p-5 shadow-[0_12px_40px_rgba(2,6,23,0.28)]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
            Process Status
          </h3>
          <p className="mt-2 text-sm text-slate-400">
            Runtime health, binary, and currently loaded model metadata.
          </p>
        </div>
        <span className={`rounded-full border border-white/10 bg-white/5 px-3 py-1 text-sm font-medium ${statusTone(status.state)}`}>
          {status.state}
        </span>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <InfoTile label="Current model" value={status.model ?? "None"} />
        <InfoTile label="Previous model" value={status.previous_model ?? "None"} accent="indigo" />
        <InfoTile label="Backend" value={status.backend ?? "Unknown"} accent="cyan" />
        <InfoTile label="llama.cpp version" value={status.server_version ?? "Unknown"} />
        <InfoTile label="API URL" value={apiUrl} mono />
        <InfoTile label="API state" value={status.api_state} accent={status.api_state === "Error" ? "rose" : "default"} />
        <InfoTile label="Crash count" value={String(status.crash_count)} accent={status.crash_count > 0 ? "rose" : "default"} />
      </div>

      {status.api_error && (
        <div className="mt-4 rounded-2xl border border-rose-400/20 bg-rose-400/10 px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.2em] text-rose-200">API server issue</p>
          <p className={`mt-1 text-sm ${apiTone("Error")}`}>{status.api_error}</p>
        </div>
      )}

      {status.server_path && (
        <div className="mt-4 rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
          <p className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Binary path</p>
          <p className="mt-1 break-all font-mono text-sm text-slate-300">{status.server_path}</p>
        </div>
      )}
    </section>
  );
}

function InfoTile({
  label,
  value,
  mono = false,
  accent = "default",
}: {
  label: string;
  value: string;
  mono?: boolean;
  accent?: "default" | "cyan" | "indigo" | "rose";
}) {
  const accentClass =
    accent === "cyan"
      ? "text-cyan-100"
      : accent === "indigo"
        ? "text-indigo-100"
        : accent === "rose"
          ? "text-rose-100"
          : "text-slate-100";

  return (
    <div className="rounded-2xl border border-white/8 bg-white/5 px-4 py-3">
      <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={`mt-1 text-sm font-semibold ${accentClass} ${mono ? "break-all font-mono" : ""}`}>
        {value}
      </p>
    </div>
  );
}
