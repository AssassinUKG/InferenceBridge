import type { ContextStatus } from "../../lib/types";

interface Props {
  status: ContextStatus;
}

export function ContextPanel({ status }: Props) {
  const pct = Math.round(status.fill_ratio * 100);
  const barColor =
    pct > 95
      ? "bg-red-500"
      : pct > 80
        ? "bg-yellow-500"
        : pct > 50
          ? "bg-blue-500"
          : "bg-green-500";

  if (status.total_tokens === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        <div className="text-center">
          <p className="text-2xl mb-2">Context</p>
          <p>Load a model to view KV cache status</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <h2 className="text-lg font-semibold text-gray-200">
        KV Cache Utilization
      </h2>

      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">
            {status.used_tokens.toLocaleString()} /{" "}
            {status.total_tokens.toLocaleString()} tokens
          </span>
          <span
            className={`font-mono ${
              pct > 95
                ? "text-red-400"
                : pct > 80
                  ? "text-yellow-400"
                  : "text-gray-300"
            }`}
          >
            {pct}%
          </span>
        </div>
        <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${barColor} rounded-full transition-all duration-500`}
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <StatCard
          label="Total"
          value={status.total_tokens.toLocaleString()}
          unit="tokens"
        />
        <StatCard
          label="Used"
          value={status.used_tokens.toLocaleString()}
          unit="tokens"
        />
        <StatCard
          label="Available"
          value={(status.total_tokens - status.used_tokens).toLocaleString()}
          unit="tokens"
        />
      </div>

      <div className="space-y-2 text-xs text-gray-500">
        <p>
          <span className="inline-block w-2 h-2 rounded bg-yellow-500 mr-2" />
          80% - Rolling compression triggered
        </p>
        <p>
          <span className="inline-block w-2 h-2 rounded bg-red-500 mr-2" />
          95% - Aggressive summarization triggered
        </p>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit: string;
}) {
  return (
    <div className="p-3 bg-gray-800/60 rounded-lg border border-gray-700/50">
      <p className="text-xs text-gray-500 uppercase tracking-wider">{label}</p>
      <p className="text-xl font-mono text-gray-200 mt-1">{value}</p>
      <p className="text-xs text-gray-600">{unit}</p>
    </div>
  );
}
