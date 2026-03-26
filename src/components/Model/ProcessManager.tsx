import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

interface ExternalProcess {
  pid: number;
  name: string;
  command_line: string;
  memory_mb: number;
}

export function ProcessManager() {
  const [processes, setProcesses] = useState<ExternalProcess[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [killingPid, setKillingPid] = useState<number | null>(null);

  const refresh = useCallback(async (showLoading = true) => {
    if (showLoading) {
      setLoading(true);
    }
    setError(null);
    try {
      const procs = await invoke<ExternalProcess[]>("list_llama_processes");
      setProcesses((current) => {
        const same =
          current.length === procs.length &&
          current.every((proc, index) => {
            const next = procs[index];
            return (
              proc.pid === next.pid &&
              proc.name === next.name &&
              proc.command_line === next.command_line &&
              proc.memory_mb === next.memory_mb
            );
          });

        return same ? current : procs;
      });
    } catch (event) {
      setError(String(event));
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(() => {
      refresh(false);
    }, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  const handleKill = async (pid: number) => {
    setKillingPid(pid);
    try {
      await invoke<string>("kill_process", { pid });
      await refresh();
    } catch (event) {
      setError(String(event));
    } finally {
      setKillingPid(null);
    }
  };

  const handleKillAll = async () => {
    try {
      await invoke<string>("kill_all_llama_processes");
      await refresh();
    } catch (event) {
      setError(String(event));
    }
  };

  return (
    <section className="rounded-[28px] border border-white/10 bg-[linear-gradient(180deg,rgba(15,23,42,0.82),rgba(8,15,29,0.94))] p-5 shadow-[0_12px_40px_rgba(2,6,23,0.28)]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
            Running Processes
          </h3>
          <p className="mt-2 text-sm text-slate-400">
            Detect and clean up stray `llama-server` processes before they collide on ports.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => {
              void refresh();
            }}
            disabled={loading}
            className="rounded-2xl border border-white/10 bg-white/6 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-55"
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>

          {processes.length > 1 && (
            <button
              onClick={handleKillAll}
              className="rounded-2xl border border-rose-400/22 bg-rose-500/12 px-4 py-2 text-sm font-medium text-rose-100 transition hover:bg-rose-500/18"
            >
              Kill All
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-2xl border border-rose-400/22 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
          {error}
        </div>
      )}

      {processes.length === 0 ? (
        <div className="mt-4 rounded-2xl border border-dashed border-white/12 bg-white/4 px-4 py-10 text-center text-sm text-slate-500">
          {loading ? "Scanning processes..." : "No llama-server processes found"}
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          {processes.map((proc) => (
            <article
              key={proc.pid}
              className="rounded-2xl border border-white/8 bg-white/5 p-4"
            >
              <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-full border border-white/10 bg-slate-950/60 px-3 py-1 font-mono text-sm text-slate-100">
                      PID {proc.pid}
                    </span>
                    <span className="text-sm font-medium text-slate-200">
                      {proc.name}
                    </span>
                    <span className="rounded-full border border-cyan-400/18 bg-cyan-400/10 px-3 py-1 text-xs text-cyan-100">
                      {proc.memory_mb.toFixed(0)} MB
                    </span>
                  </div>

                  {proc.command_line && (
                    <p
                      className="mt-3 break-all font-mono text-xs text-slate-500"
                      title={proc.command_line}
                    >
                      {proc.command_line}
                    </p>
                  )}
                </div>

                <button
                  onClick={() => handleKill(proc.pid)}
                  disabled={killingPid === proc.pid}
                  className="rounded-2xl border border-rose-400/22 bg-rose-500/12 px-4 py-2 text-sm font-medium text-rose-100 transition hover:bg-rose-500/18 disabled:cursor-not-allowed disabled:opacity-55"
                >
                  {killingPid === proc.pid ? "Killing..." : "Kill"}
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
