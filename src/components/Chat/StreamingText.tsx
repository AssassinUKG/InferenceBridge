import { MarkdownContent } from "./MarkdownContent";

interface Props {
  text: string;
}

export function StreamingText({ text }: Props) {
  return (
    <div className="flex gap-3 px-4 py-3">
      {/* Avatar */}
      <div className="w-7 h-7 rounded-full bg-purple-600/30 flex items-center justify-center text-xs text-purple-300 shrink-0 mt-0.5">
        AI
      </div>

      <div className="flex-1 min-w-0">
        {text ? (
          <div className="relative">
            <MarkdownContent content={text} />
            {/* Blinking cursor appended after last paragraph */}
            <span
              className="inline-block w-2 h-3.5 animate-pulse align-text-bottom ml-0.5"
              style={{ background: "#60a5fa", borderRadius: "1px", verticalAlign: "text-bottom" }}
            />
          </div>
        ) : (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span className="flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: "300ms" }} />
            </span>
            <span>Generating…</span>
          </div>
        )}
      </div>
    </div>
  );
}
