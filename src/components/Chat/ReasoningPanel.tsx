import { useEffect, useState } from "react";
import { Brain, ChevronRight } from "lucide-react";
import { MarkdownContent } from "./MarkdownContent";

export function ReasoningPanel({
  reasoning,
  live = false,
  storageKey,
}: {
  reasoning: string;
  live?: boolean;
  storageKey?: string;
}) {
  const [open, setOpen] = useState(live);

  useEffect(() => {
    if (live) setOpen(true);
  }, [live]);

  useEffect(() => {
    if (!storageKey || live) return;
    const saved = window.sessionStorage.getItem(`ib-reasoning:${storageKey}`);
    if (saved != null) setOpen(saved === "open");
  }, [live, storageKey]);

  if (!reasoning.trim()) return null;

  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (storageKey && !live) {
      window.sessionStorage.setItem(`ib-reasoning:${storageKey}`, next ? "open" : "closed");
    }
  };

  return (
    <section className="my-3 overflow-hidden rounded-xl border border-white/8 bg-white/[0.018]">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-[var(--text-2)] hover:bg-white/[0.025] hover:text-[var(--text-0)]"
      >
        <ChevronRight size={13} className={`shrink-0 transition-transform ${open ? "rotate-90" : ""}`} />
        <Brain size={13} className={live ? "text-sky-300" : "text-[var(--text-3)]"} />
        <span>{live ? "Thinking" : "Reasoned"}</span>
        {live && (
          <span className="ml-0.5 flex gap-1" aria-label="Reasoning in progress">
            <i className="h-1 w-1 animate-bounce rounded-full bg-sky-300" />
            <i className="h-1 w-1 animate-bounce rounded-full bg-sky-300 [animation-delay:120ms]" />
            <i className="h-1 w-1 animate-bounce rounded-full bg-sky-300 [animation-delay:240ms]" />
          </span>
        )}
        <span className="ml-auto text-[10px] text-[var(--text-3)]">{Math.max(1, Math.round(reasoning.length / 4)).toLocaleString()} tokens est.</span>
      </button>
      {open && (
        <div className="max-h-[420px] overflow-y-auto border-t border-white/[0.06] px-3 py-2 text-sm text-[var(--text-1)]">
          <MarkdownContent content={reasoning} />
        </div>
      )}
    </section>
  );
}
