import { useState } from "react";
import type { MessageInfo } from "../../lib/types";
import { MarkdownContent, CopyButton } from "./MarkdownContent";

interface Props {
  message: MessageInfo;
}

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
        <span style={{ fontSize: "10px", transition: "transform 0.15s", transform: open ? "rotate(90deg)" : "none" }}>
          ▶
        </span>
        Thinking
      </button>
      {open && (
        <div className="px-3 pb-3" style={{ borderTop: "1px solid rgba(167,139,250,0.15)" }}>
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
  const imageSrc =
    message.image_base64 ??
    (typeof content === "string" && content.startsWith("data:image/") ? content : null);
  const hasImage = !!imageSrc;
  const textContent = imageSrc === content ? "" : content;

  const hasThinkTags = !isUser && textContent.includes("<think>");
  const parts = hasThinkTags ? parseThinkBlocks(textContent) : null;
  const plainTextForCopy = textContent.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

  return (
    <div className={`group flex gap-3 px-4 py-3 ${isUser ? "" : "bg-gray-800/30"}`}>
      <div
        className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs ${
          isUser ? "bg-blue-600/30 text-blue-300" : "bg-purple-600/30 text-purple-300"
        }`}
      >
        {isUser ? "U" : "AI"}
      </div>

      <div className="min-w-0 flex-1">
        {hasImage && (
          <img
            src={imageSrc ?? undefined}
            alt="attachment"
            className="mb-2 max-h-64 max-w-xs rounded border border-gray-700"
          />
        )}

        {hasThinkTags && parts ? (
          <div>
            {parts.map((part, index) =>
              part.type === "think" ? (
                <ThinkBlock key={index} text={part.text} />
              ) : (
                <MarkdownContent key={index} content={part.text} />
              ),
            )}
          </div>
        ) : textContent && isUser ? (
          <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-gray-300">
            {textContent}
          </p>
        ) : textContent ? (
          <MarkdownContent content={textContent} />
        ) : null}

        {!hasImage && !textContent && (
          <p className="text-sm italic text-gray-500">Empty message</p>
        )}

        <div className="mt-1 flex items-center justify-between gap-2">
          {message.token_count != null && message.token_count > 0 && (
            <span className="text-xs text-gray-600">{message.token_count} tokens</span>
          )}
          {!isUser && plainTextForCopy && (
            <span className="ml-auto opacity-0 transition-opacity group-hover:opacity-100">
              <CopyButton text={plainTextForCopy} small />
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
