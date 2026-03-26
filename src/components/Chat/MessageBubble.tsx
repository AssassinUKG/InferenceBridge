import { useState } from "react";
import type { MessageInfo } from "../../lib/types";

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
          className="px-3 pb-2 text-xs whitespace-pre-wrap"
          style={{ color: "rgba(167,139,250,0.75)", borderTop: "1px solid rgba(167,139,250,0.15)" }}
        >
          {text}
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

  return (
    <div className={`flex gap-3 px-4 py-3 ${isUser ? "" : "bg-gray-800/30"}`}>
      <div
        className={`w-7 h-7 rounded-full flex items-center justify-center text-xs shrink-0 mt-0.5 ${
          isUser
            ? "bg-blue-600/30 text-blue-300"
            : "bg-purple-600/30 text-purple-300"
        }`}
      >
        {isUser ? "U" : "AI"}
      </div>
      <div className="flex-1 min-w-0">
        {isImage ? (
          <img src={content} alt="image" className="max-w-xs max-h-64 rounded border border-gray-700" />
        ) : hasThinkTags && parts ? (
          <div>
            {parts.map((p, i) =>
              p.type === "think" ? (
                <ThinkBlock key={i} text={p.text} />
              ) : (
                <p key={i} className="text-sm text-gray-300 whitespace-pre-wrap break-words">{p.text}</p>
              )
            )}
          </div>
        ) : (
          <p className="text-sm text-gray-300 whitespace-pre-wrap break-words">{content}</p>
        )}
        {message.token_count != null && message.token_count > 0 && (
          <span className="text-xs text-gray-600 mt-1 inline-block">
            {message.token_count} tokens
          </span>
        )}
      </div>
    </div>
  );
}
