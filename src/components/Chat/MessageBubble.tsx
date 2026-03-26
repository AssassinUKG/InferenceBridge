import { useState } from "react";
import type { MessageInfo } from "../../lib/types";
import { MarkdownContent, CopyButton } from "./MarkdownContent";

interface Props {
  message: MessageInfo;
}

/** Split a message body into think-tag sections and regular text. */
function parseThinkBlocks(content: string): Array<{ type: "think" | "text"; text: string }> {
  const parts: Array<{ type: "think" | "text"; text: string }> = [];
  const re = /<think>([\s\S]*?)<\/think>/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(content)) !== null) {
    if (match.index > last) {
      parts.push({ type: "text", text: content.slice(last, match.index) });
    }
    parts.push({ type: "think", text: match[1].trim() });
    last = match.index + match[0].length;
  }
  if (last < content.length) {
    parts.push({ type: "text", text: content.slice(last) });
  }
  return parts;
}

function ThinkBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className="my-1.5 rounded"
      style={{
        border: "1px solid rgba(167,139,250,0.2)",
        background: "rgba(167,139,250,0.05)",
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs"
        style={{ color: "#a78bfa", cursor: "pointer", background: "none", border: "none" }}
      >
        <span style={{ fontSize: "10px", transition: "transform 0.15s", transform: open ? "rotate(90deg)" : "none" }}>▶</span>
        💭 Thinking…
      </button>
      {open && (
        <div
          className="px-3 pb-3"
          style={{ borderTop: "1px solid rgba(167,139,250,0.15)" }}
        >
          <div style={{ color: "rgba(167,139,250,0.8)", fontSize: "12px" }}>
            <MarkdownContent content={text} />
          </div>
        </div>
      )}
    </div>
  );
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const content = message.content ?? "";
  const isImage = typeof content === "string" && content.startsWith("data:image/");

  const hasThinkTags = !isUser && content.includes("<think>");
  const parts = hasThinkTags ? parseThinkBlocks(content) : null;

  // The plain text to copy (strip think tags)
  const plainTextForCopy = content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

  return (
    <div className={`flex gap-3 px-4 py-3 group ${isUser ? "" : "bg-gray-800/30"}`}>
      {/* Avatar */}
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center text-xs shrink-0 mt-0.5 ${
          isUser
            ? "bg-blue-600/30 text-blue-300"
            : "bg-purple-600/30 text-purple-300"
        }`}
      >
        {isUser ? "U" : "AI"}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {isImage ? (
          <img
            src={content}
            alt="image"
            className="max-w-xs max-h-64 rounded border border-gray-700"
          />
        ) : hasThinkTags && parts ? (
          <div>
            {parts.map((p, i) =>
              p.type === "think" ? (
                <ThinkBlock key={i} text={p.text} />
              ) : (
                <MarkdownContent key={i} content={p.text} />
              )
            )}
          </div>
        ) : isUser ? (
          // User messages: simple text (preserves newlines, no markdown)
          <p className="text-sm text-gray-300 whitespace-pre-wrap break-words leading-relaxed">
            {content}
          </p>
        ) : (
          <MarkdownContent content={content} />
        )}

        {/* Footer row: token count + copy button */}
        <div className="flex items-center justify-between mt-1 gap-2">
          {message.token_count != null && message.token_count > 0 && (
            <span className="text-xs text-gray-600">
              {message.token_count} tokens
            </span>
          )}
          {/* Copy button — only on AI messages, shown on hover */}
          {!isUser && plainTextForCopy && (
            <span className="opacity-0 group-hover:opacity-100 transition-opacity ml-auto">
              <CopyButton text={plainTextForCopy} small />
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
