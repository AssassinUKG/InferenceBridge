import { useState } from "react";
import type { MessageInfo } from "../../lib/types";
import { MarkdownContent, CopyButton } from "./MarkdownContent";

interface Props {
  message: MessageInfo;
}

function parseThinkBlocks(content: string): Array<{ type: "think" | "text"; text: string }> {
  const parts: Array<{ type: "think" | "text"; text: string }> = [];
  const orphanClose = ["</think>", "<|/think|>"]
    .map((tag) => ({ tag, index: content.indexOf(tag) }))
    .filter((item) => item.index >= 0)
    .sort((a, b) => a.index - b.index)[0];

  if (
    orphanClose &&
    !content.includes("<think>") &&
    !content.includes("<|think|>")
  ) {
    const reasoning = content.slice(0, orphanClose.index).trim();
    const text = content.slice(orphanClose.index + orphanClose.tag.length);
    if (reasoning) parts.push({ type: "think", text: reasoning });
    if (text) parts.push({ type: "text", text });
    return parts;
  }

  const re = /<think>([\s\S]*?)<\/think>|<\|think\|>([\s\S]*?)<\|\/think\|>/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(content)) !== null) {
    if (match.index > last) {
      parts.push({ type: "text", text: content.slice(last, match.index) });
    }
    parts.push({ type: "think", text: (match[1] ?? match[2] ?? "").trim() });
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
        <span
          style={{
            fontSize: "10px",
            transition: "transform 0.15s",
            transform: open ? "rotate(90deg)" : "none",
          }}
        >
          {">"}
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
  const isSystem = message.role === "system";
  const content = message.content ?? "";
  const imageSrc =
    message.image_base64 ??
    (typeof content === "string" && content.startsWith("data:image/") ? content : null);
  const hasImage = !!imageSrc;
  const textContent = imageSrc === content ? "" : content;

  const hasThinkTags =
    !isUser &&
    (textContent.includes("<think>") ||
      textContent.includes("</think>") ||
      textContent.includes("<|think|>") ||
      textContent.includes("<|/think|>"));
  const parts = hasThinkTags ? parseThinkBlocks(textContent) : null;
  const plainTextForCopy = (parts ?? [{ type: "text" as const, text: textContent }])
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("")
    .trim();

  const roleLabel = isSystem ? "SYSTEM" : isUser ? "USER" : "ASSISTANT";

  return (
    <div className={`group flex min-w-0 gap-3 px-4 py-3 ${isUser ? "" : isSystem ? "bg-gray-950/20" : "bg-gray-800/30"}`}>
      {!isSystem && (
        <div
          className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs ${
            isUser ? "bg-blue-600/30 text-blue-300" : "bg-purple-600/30 text-purple-300"
          }`}
        >
          {isUser ? "U" : "AI"}
        </div>
      )}

      <div className="min-w-0 flex-1 overflow-visible">
        <div
          className="mb-1 text-[11px] font-semibold uppercase tracking-[0.14em]"
          style={{ color: isSystem ? "#f87171" : isUser ? "#60a5fa" : "#34d399" }}
        >
          {roleLabel}
        </div>
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
              )
            )}
          </div>
        ) : textContent && isUser ? (
          <p className="min-w-0 whitespace-pre-wrap break-words text-sm leading-relaxed text-gray-300">
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
