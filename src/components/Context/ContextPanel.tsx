import { useEffect, useMemo, useState } from "react";
import type {
  ContextStatus,
  GenerationRequest,
  GpuStats,
  ProcessStatusInfo,
  RuntimePerformanceMetrics,
} from "../../lib/types";

interface Props {
  status: ContextStatus;
  processStatus: ProcessStatusInfo | null;
  gpuStats: GpuStats | null;
}

function formatNumber(value: number | null | undefined, digits = 0) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatRate(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${value.toFixed(value >= 100 ? 0 : 1)} tok/s`;
}

function formatDuration(ms: number | null | undefined) {
  if (ms == null || Number.isNaN(ms)) return "n/a";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

function formatTime(value: string | null | undefined) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "n/a";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function shortModelName(value: string | null | undefined) {
  if (!value) return "No model loaded";
  const file = value.split(/[\\/]/).pop() ?? value;
  return file.replace(/\.gguf$/i, "");
}

function StatCard({
  label,
  value,
  detail,
  tone = "neutral",
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "neutral" | "good" | "warn" | "bad" | "info";
}) {
  const color =
    tone === "good"
      ? "#34d399"
      : tone === "warn"
      ? "#fbbf24"
      : tone === "bad"
      ? "#f87171"
      : tone === "info"
      ? "#38bdf8"
      : "var(--text-0)";

  return (
    <div
      className="min-w-0 rounded px-4 py-3"
      style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
    >
      <div className="text-[10px] uppercase tracking-[0.16em]" style={{ color: "var(--text-2)" }}>
        {label}
      </div>
      <div className="mt-1 truncate text-lg font-semibold" style={{ color }} title={value}>
        {value}
      </div>
      {detail && (
        <div className="mt-0.5 truncate text-xs" style={{ color: "var(--text-1)" }} title={detail}>
          {detail}
        </div>
      )}
    </div>
  );
}

function LiveActivityPanel({
  activeGeneration,
  last,
  activeAgeMs,
}: {
  activeGeneration: GenerationRequest | null;
  last: RuntimePerformanceMetrics | null;
  activeAgeMs: number | null;
}) {
  const isActive = !!activeGeneration;
  const activeMatchesLast =
    !!activeGeneration?.id && !!last?.request_id && activeGeneration.id === last.request_id;
  const liveTokens = activeMatchesLast ? last?.completion_tokens ?? 0 : 0;
  const liveRate = activeMatchesLast ? last?.decode_tokens_per_second : null;
  const liveElapsed = activeMatchesLast ? last?.elapsed_ms : activeAgeMs;
  const pulseColor = isActive ? "#34d399" : "#64748b";

  return (
    <section
      className="rounded px-4 py-4"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            Live Activity
          </div>
          <div className="mt-1 flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{
                background: pulseColor,
                boxShadow: isActive ? `0 0 0 6px rgba(52, 211, 153, 0.12)` : "none",
              }}
            />
            <span className="text-lg font-semibold" style={{ color: "var(--text-0)" }}>
              {isActive ? "Generating" : "Idle"}
            </span>
          </div>
        </div>
        <div className="grid min-w-[280px] grid-cols-3 gap-2 text-right text-xs">
          <div>
            <div style={{ color: "var(--text-2)" }}>Elapsed</div>
            <div className="mt-1 font-medium tabular-nums" style={{ color: "var(--text-0)" }}>
              {formatDuration(liveElapsed)}
            </div>
          </div>
          <div>
            <div style={{ color: "var(--text-2)" }}>Output</div>
            <div className="mt-1 font-medium tabular-nums" style={{ color: "var(--text-0)" }}>
              {isActive ? formatNumber(liveTokens) : formatNumber(last?.completion_tokens)}
            </div>
          </div>
          <div>
            <div style={{ color: "var(--text-2)" }}>Rate</div>
            <div className="mt-1 font-medium tabular-nums" style={{ color: isActive ? "#34d399" : "var(--text-0)" }}>
              {formatRate(isActive ? liveRate : last?.decode_tokens_per_second)}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.14em]" style={{ color: "var(--text-2)" }}>
            Source
          </div>
          <div className="mt-1 truncate text-sm" style={{ color: "var(--text-0)" }}>
            {activeGeneration?.source ?? last?.source ?? "none"}
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.14em]" style={{ color: "var(--text-2)" }}>
            Status
          </div>
          <div className="mt-1 truncate text-sm" style={{ color: "var(--text-0)" }}>
            {activeGeneration?.status ?? "no active request"}
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.14em]" style={{ color: "var(--text-2)" }}>
            Request
          </div>
          <div
            className="mt-1 truncate font-mono text-xs"
            style={{ color: "var(--text-1)" }}
            title={activeGeneration?.id ?? last?.request_id ?? ""}
          >
            {activeGeneration?.id ?? last?.request_id ?? "n/a"}
          </div>
        </div>
      </div>
    </section>
  );
}

function Meter({
  value,
  label,
  detail,
  color,
}: {
  value: number;
  label: string;
  detail: string;
  color: string;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="text-xs" style={{ color: "var(--text-1)" }}>
          {label}
        </span>
        <span className="text-xs tabular-nums" style={{ color }}>
          {detail}
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full" style={{ background: "var(--surface-2)" }}>
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${Math.max(0, Math.min(value, 100))}%`, background: color }}
        />
      </div>
    </div>
  );
}

export function ContextPanel({ status, processStatus, gpuStats }: Props) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);

  const activeGeneration = processStatus?.active_generation ?? null;
  const last = processStatus?.last_generation_metrics ?? null;
  const kvPct = Math.round((status.fill_ratio || 0) * 100);
  const kvColor =
    kvPct > 95 ? "#f87171" : kvPct > 80 ? "#fbbf24" : kvPct > 50 ? "#38bdf8" : "#34d399";
  const gpuPct = gpuStats?.dedicated_mb ? (gpuStats.used_mb / gpuStats.dedicated_mb) * 100 : 0;
  const gpuColor = gpuPct > 92 ? "#f87171" : gpuPct > 80 ? "#fbbf24" : "#38bdf8";
  const activeAgeMs = useMemo(() => {
    if (!activeGeneration?.started_at) return null;
    const started = new Date(activeGeneration.started_at).getTime();
    return Number.isNaN(started) ? null : now - started;
  }, [activeGeneration?.started_at, now]);
  const requestPressure =
    processStatus?.scheduler_limit && processStatus.scheduler_limit > 0
      ? Math.round(((processStatus.active_requests ?? 0) / processStatus.scheduler_limit) * 100)
      : 0;

  if (status.total_tokens === 0 && !processStatus?.model) {
    return (
      <div className="flex h-full items-center justify-center" style={{ color: "var(--text-1)" }}>
        <div className="text-center">
          <p className="text-xl font-semibold" style={{ color: "var(--text-0)" }}>
            Runtime Stats
          </p>
          <p className="mt-2 text-sm">Load a model to inspect live context, throughput, and backend health.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex h-full max-w-6xl flex-col gap-4 overflow-y-auto p-4">
      <section
        className="rounded px-4 py-4"
        style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
              Runtime
            </div>
            <div className="mt-1 truncate text-xl font-semibold" style={{ color: "var(--text-0)" }}>
              {shortModelName(processStatus?.model)}
            </div>
            <div className="mt-1 text-xs" style={{ color: "var(--text-1)" }}>
              {processStatus?.backend ?? "backend n/a"} - {processStatus?.server_version ?? "version n/a"}
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <span
              className="rounded px-2 py-1 text-xs"
              style={{
                background:
                  processStatus?.state === "Running"
                    ? "rgba(52,211,153,0.1)"
                    : "rgba(251,191,36,0.12)",
                border: "1px solid var(--border)",
                color: processStatus?.state === "Running" ? "#34d399" : "#fbbf24",
              }}
            >
              {processStatus?.state ?? "Unknown"}
            </span>
            <span
              className="rounded px-2 py-1 text-xs"
              style={{
                background:
                  processStatus?.api_state === "Running"
                    ? "rgba(34,211,238,0.1)"
                    : "rgba(107,114,128,0.12)",
                border: "1px solid var(--border)",
                color: processStatus?.api_state === "Running" ? "#22d3ee" : "var(--text-1)",
              }}
            >
              API {processStatus?.api_state ?? "Unknown"}
            </span>
            {activeGeneration && (
              <span
                className="rounded px-2 py-1 text-xs"
                style={{
                  background: "rgba(52,211,153,0.1)",
                  border: "1px solid rgba(52,211,153,0.22)",
                  color: "#34d399",
                }}
              >
                Generating {formatDuration(activeAgeMs)}
              </span>
            )}
          </div>
        </div>
      </section>

      <section className="grid gap-3 lg:grid-cols-4 md:grid-cols-2">
        <StatCard
          label="Decode Speed"
          value={formatRate(last?.decode_tokens_per_second)}
          detail="last completion"
          tone="good"
        />
        <StatCard
          label="End-to-End"
          value={formatRate(last?.end_to_end_tokens_per_second)}
          detail={`elapsed ${formatDuration(last?.elapsed_ms)}`}
          tone="info"
        />
        <StatCard
          label="Prefill"
          value={formatRate(last?.prompt_tokens_per_second)}
          detail={`${formatNumber(last?.prompt_tokens)} prompt tokens`}
          tone="info"
        />
        <StatCard
          label="Output"
          value={formatNumber(last?.completion_tokens)}
          detail={`${formatNumber(last?.total_tokens)} total tokens`}
          tone="neutral"
        />
      </section>

      <section className="grid gap-3 lg:grid-cols-3">
        <div
          className="rounded px-4 py-4"
          style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
        >
          <div className="text-[10px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            KV Cache
          </div>
          <div className="mt-3">
            <Meter
              value={kvPct}
              label={`${status.used_tokens.toLocaleString()} / ${status.total_tokens.toLocaleString()} tokens`}
              detail={`${kvPct}% fill`}
              color={kvColor}
            />
          </div>
          <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
            <div>
              <div style={{ color: "var(--text-2)" }}>Pinned</div>
              <div className="mt-1 font-medium" style={{ color: "var(--text-0)" }}>
                {status.pinned_tokens.toLocaleString()}
              </div>
            </div>
            <div>
              <div style={{ color: "var(--text-2)" }}>Rolling</div>
              <div className="mt-1 font-medium" style={{ color: "var(--text-0)" }}>
                {status.rolling_tokens.toLocaleString()}
              </div>
            </div>
            <div>
              <div style={{ color: "var(--text-2)" }}>Free</div>
              <div className="mt-1 font-medium" style={{ color: "var(--text-0)" }}>
                {Math.max(status.total_tokens - status.used_tokens, 0).toLocaleString()}
              </div>
            </div>
          </div>
        </div>

        <div
          className="rounded px-4 py-4"
          style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
        >
          <div className="text-[10px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            Request Scheduler
          </div>
          <div className="mt-3">
            <Meter
              value={requestPressure}
              label={`${processStatus?.active_requests ?? 0} active - ${processStatus?.queued_requests ?? 0} queued`}
              detail={`${processStatus?.scheduler_limit ?? 0} slots`}
              color={requestPressure > 80 ? "#fbbf24" : "#34d399"}
            />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
            <div>
              <div style={{ color: "var(--text-2)" }}>Parallel</div>
              <div className="mt-1 font-medium" style={{ color: "var(--text-0)" }}>
                {formatNumber(processStatus?.parallel_slots)}
              </div>
            </div>
            <div>
              <div style={{ color: "var(--text-2)" }}>Slots Seen</div>
              <div className="mt-1 font-medium" style={{ color: "var(--text-0)" }}>
                {formatNumber(processStatus?.slot_count)}
              </div>
            </div>
          </div>
        </div>

        <div
          className="rounded px-4 py-4"
          style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
        >
          <div className="text-[10px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
            GPU Memory
          </div>
          {gpuStats ? (
            <>
              <div className="mt-3">
                <Meter
                  value={gpuPct}
                  label={gpuStats.name}
                  detail={`${gpuStats.used_mb.toLocaleString()} / ${gpuStats.dedicated_mb.toLocaleString()} MB`}
                  color={gpuColor}
                />
              </div>
              <div className="mt-4 text-xs" style={{ color: "var(--text-1)" }}>
                Free {gpuStats.free_mb.toLocaleString()} MB - System RAM {gpuStats.system_ram_mb.toLocaleString()} MB
              </div>
            </>
          ) : (
            <div className="mt-3 text-sm" style={{ color: "var(--text-1)" }}>
              GPU stats unavailable. Install NVIDIA tools or use a supported GPU backend.
            </div>
          )}
        </div>
      </section>

      <section className="grid gap-3 lg:grid-cols-4 md:grid-cols-2">
        <StatCard
          label="Active Source"
          value={activeGeneration?.source ?? "idle"}
          detail={activeGeneration?.status ?? "no active request"}
          tone={activeGeneration ? "good" : "neutral"}
        />
        <StatCard
          label="Last Request"
          value={last?.source ?? "n/a"}
          detail={`finished ${formatTime(last?.finished_at)}`}
        />
        <StatCard
          label="Model Load"
          value={processStatus?.model_load_state ?? "n/a"}
          detail={`startup ${formatDuration(processStatus?.startup_duration_ms)}`}
        />
        <StatCard
          label="Launch Context"
          value={formatNumber(processStatus?.last_launch_preview?.context_size)}
          detail={`${processStatus?.last_launch_preview?.parallel_slots ?? "n/a"} parallel slots`}
        />
      </section>

      <LiveActivityPanel
        activeGeneration={activeGeneration}
        last={last}
        activeAgeMs={activeAgeMs}
      />

      <section
        className="rounded px-4 py-4"
        style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
      >
        <div className="text-[10px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
          Compaction And Pressure
        </div>
        <p className="mt-2 text-sm leading-6" style={{ color: "var(--text-1)" }}>
          {status.last_compaction_action ??
            "No compaction action recorded yet. Runtime pressure events will appear here when usage crosses strategy thresholds."}
        </p>
      </section>
    </div>
  );
}
