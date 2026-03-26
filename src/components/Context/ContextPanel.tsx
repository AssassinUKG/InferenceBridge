import type { ContextStatus } from "../../lib/types";

interface Props {
  status: ContextStatus;
}

function StatCard({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit?: string;
}) {
  return (
    <div
      className="rounded px-4 py-3"
      style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
    >
      <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
        {label}
      </div>
      <div className="mt-1 text-lg font-semibold" style={{ color: "var(--text-0)" }}>
        {value}
      </div>
      {unit && (
        <div className="text-xs" style={{ color: "var(--text-1)" }}>
          {unit}
        </div>
      )}
    </div>
  );
}

export function ContextPanel({ status }: Props) {
  const pct = Math.round(status.fill_ratio * 100);
  const barColor =
    pct > 95 ? "#f87171" : pct > 80 ? "#fbbf24" : pct > 50 ? "#38bdf8" : "#34d399";

  if (status.total_tokens === 0) {
    return (
      <div className="flex h-full items-center justify-center" style={{ color: "var(--text-1)" }}>
        <div className="text-center">
          <p className="text-xl font-semibold" style={{ color: "var(--text-0)" }}>
            Context
          </p>
          <p className="mt-2 text-sm">Load a model to inspect KV cache status and memory pressure.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex h-full max-w-5xl flex-col gap-4 p-4">
      <section
        className="rounded px-4 py-4"
        style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
      >
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
              KV Cache
            </div>
            <div className="mt-1 text-lg font-semibold" style={{ color: "var(--text-0)" }}>
              {status.used_tokens.toLocaleString()} / {status.total_tokens.toLocaleString()} tokens
            </div>
          </div>
          <div className="text-right">
            <div className="text-xl font-semibold" style={{ color: barColor }}>
              {pct}%
            </div>
            <div className="text-xs" style={{ color: "var(--text-1)" }}>
              fill ratio
            </div>
          </div>
        </div>

        <div className="mt-3 h-3 overflow-hidden rounded-full" style={{ background: "var(--surface-2)" }}>
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${Math.min(pct, 100)}%`, background: barColor }}
          />
        </div>
      </section>

      <section className="grid gap-3 md:grid-cols-3">
        <StatCard label="Pinned" value={status.pinned_tokens.toLocaleString()} unit="tokens" />
        <StatCard label="Rolling" value={status.rolling_tokens.toLocaleString()} unit="tokens" />
        <StatCard label="Compressed" value={status.compressed_tokens.toLocaleString()} unit="tokens" />
      </section>

      <section className="grid gap-3 md:grid-cols-3">
        <StatCard label="Total" value={status.total_tokens.toLocaleString()} unit="tokens" />
        <StatCard label="Used" value={status.used_tokens.toLocaleString()} unit="tokens" />
        <StatCard
          label="Available"
          value={Math.max(status.total_tokens - status.used_tokens, 0).toLocaleString()}
          unit="tokens"
        />
      </section>

      <section
        className="rounded px-4 py-4"
        style={{ background: "var(--surface-1)", border: "1px solid var(--border)" }}
      >
        <div className="text-[11px] uppercase tracking-[0.18em]" style={{ color: "var(--text-2)" }}>
          Compaction Log
        </div>
        <p className="mt-2 text-sm leading-6" style={{ color: "var(--text-1)" }}>
          {status.last_compaction_action ??
            "No compaction action recorded yet. The runtime will emit pressure events once usage crosses strategy thresholds."}
        </p>
      </section>
    </div>
  );
}
