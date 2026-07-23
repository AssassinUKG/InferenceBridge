import { MarkdownContent } from "./MarkdownContent";
import { ReasoningPanel } from "./ReasoningPanel";

interface Props {
  text: string;
  reasoning?: string;
  onOpenHtml?: (html: string) => void;
}

export function StreamingText({ text, reasoning = "", onOpenHtml }: Props) {
  return (
    <article className="flex gap-3 py-4">
      <div className="ib-brand-mark mt-0.5 h-7 w-7 text-[9px]">IB</div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-2 text-[13px] font-semibold text-[var(--text-0)]">
          InferenceBridge
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
        </div>
        {reasoning && <ReasoningPanel reasoning={reasoning} live />}
        {text ? (
          <div className="relative">
            <MarkdownContent content={text} onOpenHtml={onOpenHtml} />
            <span className="ml-1 inline-block h-4 w-1.5 animate-pulse rounded-sm bg-white/70 align-text-bottom" />
          </div>
        ) : !reasoning ? (
          <div className="flex h-7 items-center gap-1.5" aria-label="Generating">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--text-2)]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--text-2)] [animation-delay:150ms]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--text-2)] [animation-delay:300ms]" />
          </div>
        ) : null}
      </div>
    </article>
  );
}
