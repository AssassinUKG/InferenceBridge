import { useEffect, useState, type ReactNode } from "react";
import {
  Clock3,
  Download,
  FolderOpen,
  Gauge,
  Hash,
  Image as ImageIcon,
  Maximize2,
} from "lucide-react";
import type { MessageInfo, ToolCallInfo } from "../../lib/types";
import { readGeneratedImageDataUrl, showInFolder } from "../../lib/tauri";
import {
  formatImageDuration,
  formatImageFileSize,
  formatImageSampler,
  imageAspectRatio,
  imageDataUrlByteSize,
  imageModelLabel,
  parseGeneratedImageMetadata,
} from "../../lib/generatedImagePresentation";
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
  const hasImageAttachment = !!imageSrc || !!message.image_path;
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
        {imageSrc && <img src={imageSrc} alt="Assistant attachment" className="mb-3 max-h-[520px] max-w-full rounded-xl object-contain" />}
        {!imageSrc && message.image_path && (
          <GeneratedImageAttachment
            path={message.image_path}
            messageId={message.id}
            metadataJson={message.image_metadata}
            caption={textContent}
          />
        )}
        {reasoning && <ReasoningPanel reasoning={reasoning} storageKey={String(message.id)} />}
        {textContent && !message.image_path
          ? <MarkdownContent content={textContent} onOpenHtml={onOpenHtml} />
          : null}
        {toolCalls.length > 0 && (
          <div className="mt-3" aria-label="Tool selections">
            {toolCalls.map((call) => <ToolActivityCard key={`${message.id}-${call.id}`} call={call} />)}
          </div>
        )}
        {!hasImageAttachment && !textContent && toolCalls.length === 0 && <p className="text-sm italic text-[var(--text-2)]">Empty message</p>}
        <div className="mt-1 flex h-8 items-center gap-2 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
          {copyText && <CopyButton text={copyText} small />}
          {message.token_count != null && message.token_count > 0 && <span className="text-[10px] text-[var(--text-3)]">{message.token_count} tokens</span>}
        </div>
      </div>
    </article>
  );
}

function GeneratedImageAttachment({
  path,
  messageId,
  metadataJson,
  caption,
}: {
  path: string;
  messageId: number;
  metadataJson?: string | null;
  caption: string;
}) {
  const [source, setSource] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const metadata = parseGeneratedImageMetadata(metadataJson);

  useEffect(() => {
    let cancelled = false;
    setSource(null);
    setError(null);
    void readGeneratedImageDataUrl(path).then(
      (dataUrl) => {
        if (!cancelled) setSource(dataUrl);
      },
      (imageError) => {
        if (!cancelled) setError(String(imageError));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [path]);

  if (error) {
    return (
      <div className="mb-3 rounded-xl border border-rose-400/20 bg-rose-950/20 px-3 py-2 text-xs text-rose-200">
        {error}
      </div>
    );
  }
  if (!source) {
    return (
      <div className="mb-3 flex h-48 max-w-xl items-center justify-center rounded-xl border border-white/10 bg-black/15 text-xs text-[var(--text-2)]">
        Loading generated image...
      </div>
    );
  }

  const duration = formatImageDuration(metadata?.elapsed_seconds);
  const fileSize = formatImageFileSize(
    metadata?.file_size_bytes ?? imageDataUrlByteSize(source),
  );
  const aspectRatio = imageAspectRatio(metadata?.width, metadata?.height);
  const dimensions =
    metadata?.width && metadata.height
      ? `${metadata.width}×${metadata.height}${aspectRatio ? ` · ${aspectRatio}` : ""}`
      : "PNG image";
  const quality =
    metadata?.steps != null
      ? `${metadata.steps} steps${metadata.cfg_scale != null ? ` · CFG ${metadata.cfg_scale}` : ""}`
      : "Quality render";
  const sampler = formatImageSampler(metadata?.sampling_method);
  const completedAt = metadata?.completed_at
    ? new Date(metadata.completed_at)
    : null;
  const completedLabel =
    completedAt && Number.isFinite(completedAt.getTime())
      ? completedAt.toLocaleString()
      : null;

  return (
    <div className="mb-4 max-w-4xl overflow-hidden rounded-2xl border border-white/10 bg-[var(--surface-2)] shadow-[0_14px_40px_rgba(0,0,0,0.16)]">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/8 px-3.5 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-violet-400/10 text-violet-300">
            <ImageIcon size={14} />
          </div>
          <div className="min-w-0">
            <div className="truncate text-xs font-semibold text-[var(--text-0)]">
              Generated image
            </div>
            <div className="truncate text-[10px] text-[var(--text-3)]">
              {imageModelLabel(metadata)}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {duration && (
            <span className="inline-flex items-center gap-1 rounded-full bg-black/20 px-2 py-1 text-[10px] text-[var(--text-2)]">
              <Clock3 size={11} />
              {duration}
            </span>
          )}
          {fileSize && (
            <span className="rounded-full bg-black/20 px-2 py-1 text-[10px] text-[var(--text-2)]">
              {fileSize}
            </span>
          )}
        </div>
      </div>

      <div className="bg-black/20">
        <img
          src={source}
          alt="Generated attachment"
          className="mx-auto max-h-[680px] w-auto max-w-full object-contain"
        />
      </div>

      <div className="p-3.5">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <ImageMetric icon={<Maximize2 size={13} />} label="Canvas" value={dimensions} />
          <ImageMetric icon={<Gauge size={13} />} label="Quality" value={quality} />
          <ImageMetric
            icon={<Hash size={13} />}
            label="Seed"
            value={metadata?.seed != null ? String(metadata.seed) : "Not recorded"}
          />
          <ImageMetric
            icon={<Clock3 size={13} />}
            label="Render time"
            value={duration ?? "Not recorded"}
          />
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-white/8 pt-3">
          <div className="flex flex-wrap items-center gap-2">
            <a
              href={source}
              download={`inferencebridge-image-${messageId}.png`}
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-[11px] font-medium text-[var(--text-1)] transition hover:bg-white/5 hover:text-white"
            >
              <Download size={12} />
              Save copy
            </a>
            <button
              type="button"
              onClick={() => { void showInFolder(path); }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-[11px] font-medium text-[var(--text-1)] transition hover:bg-white/5 hover:text-white"
            >
              <FolderOpen size={12} />
              Show in folder
            </button>
          </div>
          {sampler && (
            <span className="text-[10px] text-[var(--text-3)]">
              {sampler} sampler
            </span>
          )}
        </div>

        <details className="group mt-3 rounded-xl border border-white/8 bg-black/10">
          <summary className="cursor-pointer select-none px-3 py-2 text-[11px] font-medium text-[var(--text-2)] hover:text-white">
            Prompt and generation details
          </summary>
          <div className="space-y-3 border-t border-white/8 px-3 py-3 text-xs">
            <div>
              <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-[var(--text-3)]">
                Prompt
              </div>
              <div className="whitespace-pre-wrap leading-5 text-[var(--text-1)]">
                {metadata?.prompt?.trim() || caption || "Prompt not recorded"}
              </div>
            </div>
            {metadata?.negative_prompt?.trim() && (
              <div>
                <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-[var(--text-3)]">
                  Negative prompt
                </div>
                <div className="whitespace-pre-wrap leading-5 text-[var(--text-2)]">
                  {metadata.negative_prompt}
                </div>
              </div>
            )}
            <dl className="grid gap-x-5 gap-y-2 border-t border-white/8 pt-3 sm:grid-cols-2">
              <ImageDetail label="Model" value={imageModelLabel(metadata)} />
              <ImageDetail label="Profile" value={metadata?.profile_id || "Not recorded"} />
              <ImageDetail label="Sampler" value={sampler || "Not recorded"} />
              <ImageDetail
                label="CFG scale"
                value={metadata?.cfg_scale != null ? String(metadata.cfg_scale) : "Not recorded"}
              />
              <ImageDetail label="Completed" value={completedLabel || "Not recorded"} />
              <ImageDetail label="Job ID" value={metadata?.job_id || "Not recorded"} mono />
            </dl>
          </div>
        </details>
      </div>
    </div>
  );
}

function ImageMetric({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-white/8 bg-black/10 px-2.5 py-2">
      <div className="flex items-center gap-1.5 text-[10px] text-[var(--text-3)]">
        {icon}
        {label}
      </div>
      <div className="mt-1 truncate text-[11px] font-medium text-[var(--text-1)]" title={value}>
        {value}
      </div>
    </div>
  );
}

function ImageDetail({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="min-w-0">
      <dt className="text-[10px] text-[var(--text-3)]">{label}</dt>
      <dd
        className={`mt-0.5 truncate text-[11px] text-[var(--text-1)] ${mono ? "font-mono" : ""}`}
        title={value}
      >
        {value}
      </dd>
    </div>
  );
}
