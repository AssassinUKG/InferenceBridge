import { Children, isValidElement, useEffect, useMemo, useState } from "react";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Check, Copy, ExternalLink, Eye, FileCode2 } from "lucide-react";
import { open } from "@tauri-apps/plugin-shell";
import "highlight.js/styles/github-dark-dimmed.css";
import "katex/dist/katex.min.css";

let mermaidRenderId = 0;

function safeUrl(raw: string | undefined, image = false) {
  if (!raw) return null;
  const value = raw.trim();
  if (image && /^data:image\/(png|jpeg|jpg|gif|webp);base64,/i.test(value)) return value;
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return null;
    return parsed.toString();
  } catch {
    return value.startsWith("#") ? value : null;
  }
}

function nodeText(node: React.ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  if (isValidElement<{ children?: React.ReactNode }>(node)) return nodeText(node.props.children);
  return "";
}

function codeLanguage(children: React.ReactNode) {
  const child = Children.toArray(children).find(isValidElement);
  const className = isValidElement<{ className?: string }>(child) ? child.props.className ?? "" : "";
  return /(?:^|\s)language-([\w-]+)/.exec(className)?.[1] ?? "";
}

function looksLikeStandaloneHtml(language: string, code: string) {
  const normalized = code.trimStart().toLocaleLowerCase();
  return language === "html" || language === "htm" || language === "svg" ||
    normalized.startsWith("<!doctype html") || normalized.startsWith("<html") || normalized.startsWith("<svg");
}

function CopyButton({ text, small = false }: { text: string; small?: boolean }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <button
      type="button"
      onClick={() => void handleCopy()}
      aria-label={copied ? "Copied" : "Copy"}
      title={copied ? "Copied" : "Copy"}
      className={`inline-flex h-7 items-center justify-center gap-1.5 rounded-lg px-2 text-[11px] transition ${
        copied ? "text-emerald-300" : "text-[var(--text-2)] hover:bg-white/5 hover:text-white"
      }`}
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {!small && <span>{copied ? "Copied" : "Copy"}</span>}
    </button>
  );
}

function MermaidDiagram({ code }: { code: string }) {
  const [svg, setSvg] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const bounded = code.slice(0, 40_000).trim();
    if (!bounded) return undefined;
    void import("mermaid").then(async ({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        suppressErrorRendering: true,
        theme: document.documentElement.dataset.theme?.includes("light") ? "default" : "dark",
      });
      try {
        const id = `ib-mermaid-${++mermaidRenderId}`;
        const result = await mermaid.render(id, bounded);
        const sanitized = DOMPurify.sanitize(result.svg, {
          USE_PROFILES: { svg: true, svgFilters: true },
          FORBID_TAGS: ["script", "foreignObject", "iframe", "object", "embed"],
          FORBID_ATTR: ["onload", "onclick", "onerror", "style"],
        });
        if (active) {
          setSvg(sanitized);
          setError(null);
        }
      } catch (reason) {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      }
    }).catch((reason) => {
      if (active) setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => { active = false; };
  }, [code]);

  if (svg && !error) {
    return <div className="ib-mermaid-diagram overflow-x-auto rounded-xl border border-white/8 bg-black/10 p-4" dangerouslySetInnerHTML={{ __html: svg }} />;
  }
  return (
    <div className="rounded-xl border border-white/8 bg-black/15 p-3">
      {error && <div className="mb-2 text-xs text-amber-300">Diagram preview unavailable: {error.slice(0, 240)}</div>}
      <pre className="overflow-x-auto whitespace-pre text-xs text-[var(--text-1)]"><code>{code}</code></pre>
    </div>
  );
}

function CodeBlock({
  children,
  onOpenHtml,
}: {
  children: React.ReactNode;
  onOpenHtml?: (html: string) => void;
}) {
  const code = nodeText(children).replace(/\n$/, "");
  const language = codeLanguage(children);

  if (language === "mermaid") return <MermaidDiagram code={code} />;

  const canPreview = !!onOpenHtml && looksLikeStandaloneHtml(language, code) && code.length <= 200_000;
  return (
    <div
      className="my-3 max-w-full overflow-hidden rounded-xl border border-white/8 bg-[#151515]"
      data-context-copy={code}
      data-context-label="code"
    >
      <div className="flex items-center justify-between border-b border-white/8 bg-white/[0.035] px-3 py-1.5">
        <span className="flex items-center gap-1.5 text-[11px] font-mono text-[var(--text-3)]">
          <FileCode2 size={12} />
          {language || "plaintext"}
        </span>
        <div className="flex items-center gap-1">
          {canPreview && (
            <button
              type="button"
              onClick={() => onOpenHtml(code)}
              className="inline-flex h-7 items-center gap-1.5 rounded-lg px-2 text-[11px] text-[var(--text-2)] hover:bg-white/5 hover:text-white"
            >
              <Eye size={13} /> Preview
            </button>
          )}
          <CopyButton text={code} small />
        </div>
      </div>
      <pre className="m-0 max-w-full overflow-x-auto p-3 text-[12px] leading-relaxed text-[#e7e7e7] [font-family:ui-monospace,'Cascadia_Code',Consolas,monospace]">
        {children}
      </pre>
    </div>
  );
}

function SourceLink({ href, children }: { href?: string; children: React.ReactNode }) {
  const [cardOpen, setCardOpen] = useState(false);
  const safe = safeUrl(href);
  const numeric = /^\d{1,3}$/.test(nodeText(children).trim());
  const host = useMemo(() => {
    if (!safe || safe.startsWith("#")) return "";
    try { return new URL(safe).hostname.replace(/^www\./, ""); } catch { return ""; }
  }, [safe]);

  if (!safe) return <span className="text-[var(--text-2)]">{children}</span>;
  if (safe.startsWith("#")) return <a href={safe}>{children}</a>;
  return (
    <span className="relative inline-block" onMouseEnter={() => setCardOpen(true)} onMouseLeave={() => setCardOpen(false)}>
      <a
        href={safe}
        title={host ? `Open source: ${host}` : "Open source"}
        onFocus={() => setCardOpen(true)}
        onBlur={() => setCardOpen(false)}
        onClick={(event) => {
          event.preventDefault();
          void open(safe).catch(() => undefined);
        }}
        className={numeric
          ? "mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full border border-sky-300/20 bg-sky-400/10 px-1.5 align-super text-[10px] font-semibold text-sky-200 no-underline hover:bg-sky-400/20"
          : "inline-flex items-baseline gap-1 text-[#8ab4f8] underline underline-offset-2 hover:text-sky-200"}
      >
        {children}{!numeric && <ExternalLink size={10} className="inline shrink-0" />}
      </a>
      {cardOpen && host && (
        <span role="tooltip" className="absolute bottom-full left-0 z-40 mb-2 block w-72 max-w-[80vw] rounded-xl border border-white/10 bg-[var(--surface-2)] p-3 text-left shadow-2xl">
          <span className="block truncate text-xs font-semibold text-[var(--text-0)]">{host}</span>
          <span className="mt-1 block break-all text-[10px] leading-4 text-[var(--text-3)]">{safe}</span>
          <span className="mt-2 flex items-center gap-1 text-[10px] text-sky-200"><ExternalLink size={10} /> Open in default browser</span>
        </span>
      )}
    </span>
  );
}

interface Props {
  content: string;
  onOpenHtml?: (html: string) => void;
}

export function MarkdownContent({ content, onOpenHtml }: Props) {
  const boundedContent = content.length > 1_000_000
    ? `${content.slice(0, 1_000_000)}\n\n_[Output truncated in the renderer]_`
    : content;
  return (
    <div className="markdown-body min-w-0 max-w-full overflow-visible break-words text-[15px] leading-7 text-[var(--text-0)]">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeHighlight, { detect: false, subset: ["bash", "css", "javascript", "json", "markdown", "python", "rust", "typescript", "xml"] }], rehypeKatex]}
        urlTransform={(url, key) => safeUrl(url, key === "src") ?? ""}
        components={{
          pre: ({ children }) => <CodeBlock onOpenHtml={onOpenHtml}>{children}</CodeBlock>,
          code: ({ className, children }) => <code className={className ?? "rounded border border-white/7 bg-white/8 px-1 py-0.5 font-mono text-[12px] text-[#eeeeee]"}>{children}</code>,
          h1: ({ children }) => <h1 className="mb-2 mt-5 break-words text-xl font-semibold text-[#f4f4f4]">{children}</h1>,
          h2: ({ children }) => <h2 className="mb-2 mt-5 break-words text-lg font-semibold text-[#f4f4f4]">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-1 mt-4 break-words text-base font-semibold text-[#eeeeee]">{children}</h3>,
          h4: ({ children }) => <h4 className="mb-0.5 mt-2 break-words text-sm font-medium text-slate-300">{children}</h4>,
          p: ({ children }) => <p className="my-2 break-words leading-7">{children}</p>,
          ul: ({ children }) => <ul className="my-1.5 list-disc space-y-0.5 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="my-1.5 list-decimal space-y-0.5 pl-5">{children}</ol>,
          li: ({ children }) => <li className="leading-7 text-[#e6e6e6]">{children}</li>,
          blockquote: ({ children }) => <blockquote className="my-2 rounded-r border-l-[3px] border-white/25 bg-white/[0.02] py-0.5 pl-3 text-[#b4b4b4]">{children}</blockquote>,
          hr: () => <hr className="my-3 border-0 border-t border-white/10" />,
          table: ({ children }) => <div className="my-2 overflow-x-auto rounded-lg border border-white/10"><table className="w-full border-collapse text-xs">{children}</table></div>,
          thead: ({ children }) => <thead className="bg-white/5">{children}</thead>,
          th: ({ children }) => <th className="border-b border-white/10 px-3 py-2 text-left font-semibold text-slate-200">{children}</th>,
          td: ({ children }) => <td className="border-b border-white/[0.06] px-3 py-1.5 text-gray-300">{children}</td>,
          a: ({ href, children }) => <SourceLink href={href}>{children}</SourceLink>,
          img: ({ src, alt }) => {
            const safe = safeUrl(typeof src === "string" ? src : undefined, true);
            return safe ? <img src={safe} alt={alt ?? "Generated content"} loading="lazy" className="my-3 max-h-[520px] max-w-full rounded-xl border border-white/8 object-contain" /> : <span className="text-xs text-amber-300">[Blocked unsafe image]</span>;
          },
          strong: ({ children }) => <strong className="font-semibold text-slate-100">{children}</strong>,
          em: ({ children }) => <em className="italic text-slate-300">{children}</em>,
        }}
      >
        {boundedContent}
      </ReactMarkdown>
    </div>
  );
}

export { CopyButton };
