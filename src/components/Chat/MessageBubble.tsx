import type { MessageInfo, ToolCallInfo } from "../../lib/types";
import { CopyButton, MarkdownContent } from "./MarkdownContent";
import { ReasoningPanel } from "./ReasoningPanel";
import { ToolActivityCard } from "./ToolActivityCard";

interface Props {
  message: MessageInfo;
  onOpenHtml?: (html: string) => void;
}

function splitLegacyReasoning(content: string) {
  const parts: Array<{ type: "think" | "text"; text: string }> = [];
  const orphanClose = ["</think>", "<|/think|>"]
    .map((tag) => ({ tag, index: content.indexOf(tag) }))
    .filter((item) => item.index >= 0)
    .sort((a, b) => a.index - b.index)[0];

  if (orphanClose && !content.includes("<think>") && !content.includes("<|think|>")) {
    const reasoning = content.slice(0, orphanClose.index).trim();
    const text = content.slice(orphanClose.index + orphanClose.tag.length);
    if (reasoning) parts.push({ type: "think", text: reasoning });
    if (text) parts.push({ type: "text", text });
    return parts;
  }

  const expression = /<think>([\s\S]*?)<\/think>|<\|think\|>([\s\S]*?)<\|\/think\|>/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = expression.exec(content)) !== null) {
    if (match.index > last) parts.push({ type: "text", text: content.slice(last, match.index) });
    parts.push({ type: "think", text: (match[1] ?? match[2] ?? "").trim() });
    last = match.index + match[0].length;
  }
  if (last < content.length) parts.push({ type: "text", text: content.slice(last) });
  return parts;
}

function extractLegacyToolCards(content: string) {
  const calls: ToolCallInfo[] = [];
  let visible = content;
  const expression = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/gi;
  visible = visible.replace(expression, (_full, body: string) => {
    try {
      const parsed = JSON.parse(body) as { id?: string; name?: string; arguments?: unknown };
      if (typeof parsed.name === "string" && parsed.name.trim()) {
        calls.push({
          id: -(calls.length + 1),
          call_id: typeof parsed.id === "string" ? parsed.id : null,
          name: parsed.name,
          arguments: typeof parsed.arguments === "string" ? parsed.arguments : JSON.stringify(parsed.arguments ?? {}),
          result: null,
        });
        return "";
      }
    } catch {
      // This is display-only legacy recovery; malformed text stays visible.
    }
    return _full;
  });
  return { visible: visible.trim(), calls };
}

export function MessageBubble({ message, onOpenHtml }: Props) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const rawContent = message.content ?? "";
  const imageSrc = message.image_base64 ?? (rawContent.startsWith("data:image/") ? rawContent : null);
  const contentWithoutImage = imageSrc === rawContent ? "" : rawContent;
  const legacyParts = !isUser && !message.display_content &&
    (contentWithoutImage.includes("<think>") || contentWithoutImage.includes("</think>") || contentWithoutImage.includes("<|think|>") || contentWithoutImage.includes("<|/think|>"))
    ? splitLegacyReasoning(contentWithoutImage)
    : null;
  const legacyVisible = legacyParts?.filter((part) => part.type === "text").map((part) => part.text).join("") ?? contentWithoutImage;
  const legacyReasoning = legacyParts?.filter((part) => part.type === "think").map((part) => part.text).join("\n\n") ?? "";
  const storedCalls = message.tool_calls ?? [];
  const legacyTools = storedCalls.length === 0 ? extractLegacyToolCards(message.display_content ?? legacyVisible) : null;
  const textContent = legacyTools?.visible ?? message.display_content ?? legacyVisible;
  const reasoning = message.reasoning_content ?? legacyReasoning;
  const toolCalls = storedCalls.length > 0 ? storedCalls : legacyTools?.calls ?? [];
  const copyText = textContent.trim();

  if (isSystem) {
    return (
      <div className="my-4 rounded-lg border border-white/8 bg-black/10 px-3 py-2 text-xs leading-5 text-[var(--text-2)]" data-context-copy={textContent || undefined} data-context-label="system message">
        <span className="mr-2 font-semibold text-[var(--text-1)]">System</span>
        {textContent}
      </div>
    );
  }

  if (isUser) {
    return (
      <article className="group flex justify-end py-3" data-context-copy={copyText || undefined} data-context-label="prompt">
        <div className="max-w-[82%] sm:max-w-[72%]">
          <div className="overflow-hidden rounded-[20px] bg-[var(--surface-3)] px-4 py-2.5 text-[15px] leading-6 text-[var(--text-0)]">
            {imageSrc && <img src={imageSrc} alt="User attachment" className={`max-h-[360px] w-auto max-w-full rounded-xl object-contain ${textContent ? "mb-2" : ""}`} />}
            {textContent && <p className="whitespace-pre-wrap break-words">{textContent}</p>}
            {!imageSrc && !textContent && <span className="italic text-[var(--text-2)]">Empty message</span>}
          </div>
          <div className="mt-1 flex h-7 items-center justify-end gap-2 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
            {message.token_count != null && message.token_count > 0 && <span className="text-[10px] text-[var(--text-3)]">{message.token_count} tokens</span>}
            {copyText && <CopyButton text={copyText} small />}
          </div>
        </div>
      </article>
    );
  }

  return (
    <article className="group flex gap-3 py-4" data-context-copy={copyText || undefined} data-context-label="response">
      <div className="ib-brand-mark mt-0.5 h-7 w-7 text-[9px]">IB</div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 text-[13px] font-semibold text-[var(--text-0)]">InferenceBridge</div>
        {imageSrc && <img src={imageSrc} alt="Assistant attachment" className="mb-3 max-h-[360px] max-w-full rounded-xl object-contain" />}
        {reasoning && <ReasoningPanel reasoning={reasoning} storageKey={String(message.id)} />}
        {textContent ? <MarkdownContent content={textContent} onOpenHtml={onOpenHtml} /> : null}
        {toolCalls.length > 0 && (
          <div className="mt-3" aria-label="Tool selections">
            {toolCalls.map((call) => <ToolActivityCard key={`${message.id}-${call.id}`} call={call} />)}
          </div>
        )}
        {!imageSrc && !textContent && toolCalls.length === 0 && <p className="text-sm italic text-[var(--text-2)]">Empty message</p>}
        <div className="mt-1 flex h-8 items-center gap-2 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
          {copyText && <CopyButton text={copyText} small />}
          {message.token_count != null && message.token_count > 0 && <span className="text-[10px] text-[var(--text-3)]">{message.token_count} tokens</span>}
        </div>
      </div>
    </article>
  );
}
