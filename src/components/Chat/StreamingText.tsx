import { MarkdownContent } from "./MarkdownContent";

interface Props {
  text: string;
  reasoning?: string;
}

export function StreamingText({ text, reasoning = "" }: Props) {
  return (
    <div className="flex gap-3 px-4 py-3">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-purple-600/30 text-xs text-purple-300">
        AI
      </div>

      <div className="min-w-0 flex-1">
        {reasoning && (
          <details
            className="mb-2 rounded border px-3 py-2"
            style={{
              borderColor: "rgba(167,139,250,0.2)",
              background: "rgba(167,139,250,0.05)",
            }}
          >
            <summary className="cursor-pointer text-xs font-medium" style={{ color: "#a78bfa" }}>
              Thinking
            </summary>
            <div className="mt-2 text-xs leading-6" style={{ color: "rgba(196,181,253,0.9)" }}>
              <MarkdownContent content={reasoning} />
            </div>
          </details>
        )}

        {text ? (
          <div className="relative">
            <MarkdownContent content={text} />
            <span
              className="ml-0.5 inline-block h-3.5 w-2 animate-pulse align-text-bottom"
              style={{ background: "#60a5fa", borderRadius: "1px", verticalAlign: "text-bottom" }}
            />
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span className="flex gap-1">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-500" style={{ animationDelay: "0ms" }} />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-500" style={{ animationDelay: "150ms" }} />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-500" style={{ animationDelay: "300ms" }} />
            </span>
            <span>Generating...</span>
          </div>
        )}
      </div>
    </div>
  );
}
