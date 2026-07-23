import { useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronRight, Copy, TerminalSquare, Wrench } from "lucide-react";
import type { ToolCallInfo } from "../../lib/types";

function prettyJson(value: string | null) {
  if (!value) return "{}";
  try { return JSON.stringify(JSON.parse(value), null, 2); } catch { return value; }
}

type CapabilityRejection = {
  code: "capability_unavailable";
  message: string;
};

function capabilityRejection(value: string | null): CapabilityRejection | null {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value) as Partial<CapabilityRejection>;
    if (parsed.code === "capability_unavailable" && typeof parsed.message === "string") {
      return parsed as CapabilityRejection;
    }
  } catch {
    // Ordinary tool results can be plain text.
  }
  return null;
}

export function ToolActivityCard({ call }: { call: ToolCallInfo }) {
  const [open, setOpen] = useState(false);
  const argumentsText = useMemo(() => prettyJson(call.arguments), [call.arguments]);
  const rejection = useMemo(() => capabilityRejection(call.result), [call.result]);
  const complete = call.result != null && rejection == null;
  const toolName = call.name.toLowerCase();

  return (
    <section className={`my-2 overflow-hidden rounded-xl border bg-black/10 ${rejection ? "border-amber-400/20" : "border-white/8"}`}>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-white/[0.025]"
      >
        <ChevronRight size={13} className={`shrink-0 text-[var(--text-3)] transition-transform ${open ? "rotate-90" : ""}`} />
        <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-lg ${rejection ? "bg-amber-400/10 text-amber-300" : "bg-white/5 text-[var(--text-2)]"}`}>
          {rejection ? <AlertTriangle size={13} /> : toolName.includes("bash") || toolName.includes("command") ? <TerminalSquare size={13} /> : <Wrench size={13} />}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-xs font-medium text-[var(--text-0)]">{call.name}</span>
          <span className={`block text-[10px] ${rejection ? "text-amber-300/80" : "text-[var(--text-3)]"}`}>
            {rejection ? "Blocked · capability unavailable" : complete ? "Tool result captured" : "Tool selected · not executed by IB chat"}
          </span>
        </span>
        {rejection && <AlertTriangle size={14} className="text-amber-300" />}
        {complete && <CheckCircle2 size={14} className="text-emerald-300" />}
      </button>
      {open && (
        <div className="border-t border-white/[0.06] p-3">
          {rejection && (
            <div className="mb-3 rounded-lg border border-amber-400/15 bg-amber-400/[0.06] px-3 py-2 text-[11px] leading-5 text-amber-100/90">
              {rejection.message}
            </div>
          )}
          <div className="mb-1 flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-3)]">
            Arguments
            <button type="button" onClick={() => void navigator.clipboard.writeText(argumentsText)} className="rounded p-1 hover:bg-white/5" aria-label="Copy tool arguments"><Copy size={12} /></button>
          </div>
          <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-black/20 p-2.5 text-[11px] leading-5 text-[var(--text-1)]">{argumentsText}</pre>
          {call.result != null && rejection == null && (
            <>
              <div className="mb-1 mt-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-3)]">Result</div>
              <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-black/20 p-2.5 text-[11px] leading-5 text-[var(--text-1)]">{call.result}</pre>
            </>
          )}
          {call.call_id && <div className="mt-2 truncate font-mono text-[9px] text-[var(--text-3)]">{call.call_id}</div>}
        </div>
      )}
    </section>
  );
}
