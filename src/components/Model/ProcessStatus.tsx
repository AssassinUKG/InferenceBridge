import type { AppSettings, ProcessStatusInfo } from "../../lib/types";

interface Props {
  status: ProcessStatusInfo | null;
  settings: AppSettings | null;
}

function buildServerUrl(settings: AppSettings | null) {
  if (!settings) return "http://127.0.0.1:8800/v1";
  return `http://${settings.server_host}:${settings.server_port}/v1`;
}

function toneForState(state: string) {
  if (state === "Running") return "#86efac";
  if (state === "Starting") return "#fde68a";
  if (state === "Error") return "#fca5a5";
  return "var(--text-1)";
}

function Stat({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded px-4 py-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
      <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
        {label}
      </div>
      <div className="mt-1 text-sm font-semibold" style={{ color: "var(--text-0)" }}>
        {value}
      </div>
    </div>
  );
}

export function ProcessStatus({ status, settings }: Props) {
  if (!status) {
    return (
      <section className="rounded px-4 py-4" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
        <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
          Process Status
        </div>
        <p className="mt-2 text-sm" style={{ color: "var(--text-1)" }}>
          No runtime information is available yet.
        </p>
      </section>
    );
  }

  const apiUrl = status.api_url ?? buildServerUrl(settings);
  const apiReachable = status.api_reachable;

  return (
    <section className="rounded px-4 py-4" style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            Process Status
          </div>
          <div className="mt-1 text-lg font-semibold" style={{ color: toneForState(status.state) }}>
            {status.state}
          </div>
        </div>
        <div className="text-right">
          <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            API
          </div>
          <div className="mt-1 text-sm font-medium" style={{ color: toneForState(apiReachable ? "Running" : status.api_state) }}>
            {apiReachable ? "Running" : status.api_state}
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <Stat label="Current Model" value={status.model ?? "None"} />
        <Stat label="Previous Model" value={status.previous_model ?? "None"} />
        <Stat label="Backend" value={status.backend ?? "Unknown"} />
        <Stat label="llama.cpp" value={status.server_version ?? "Unknown"} />
        <Stat label="API URL" value={apiUrl} />
        <Stat label="Crash Count" value={String(status.crash_count)} />
        <Stat
          label="Startup Time"
          value={status.startup_duration_ms != null ? `${status.startup_duration_ms} ms` : "Unknown"}
        />
        <Stat
          label="Slots"
          value={
            status.slot_count != null
              ? `${status.slot_count} total / ${status.parallel_slots ?? status.slot_count} configured`
              : `${status.parallel_slots ?? 0} configured`
          }
        />
      </div>

      {status.api_error && (
        <div className="mt-4 rounded px-4 py-3" style={{ background: "rgba(127,29,29,0.28)", border: "1px solid rgba(248,113,113,0.22)" }}>
          <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "#fecaca" }}>
            API Server Issue
          </div>
          <p className="mt-1 text-sm" style={{ color: "#fca5a5" }}>
            {status.api_error}
          </p>
        </div>
      )}

      {status.last_launch_preview && (
        <div className="mt-4 rounded px-4 py-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            Last Launch Preview
          </div>
          <pre className="mt-2 overflow-x-auto text-xs" style={{ color: "var(--text-0)" }}>
            <code>{status.last_launch_preview.args.join(" ")}</code>
          </pre>
        </div>
      )}

      {status.server_path && (
        <div className="mt-4 rounded px-4 py-3" style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}>
          <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            Binary Path
          </div>
          <p className="mt-1 break-all font-mono text-xs" style={{ color: "var(--text-0)" }}>
            {status.server_path}
          </p>
        </div>
      )}
    </section>
  );
}
