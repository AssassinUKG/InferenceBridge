import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ── Copy button ───────────────────────────────────────────────────────────────

function CopyButton({ text, small = false }: { text: string; small?: boolean }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <button
      onClick={handleCopy}
      title={copied ? "Copied!" : "Copy"}
      className="transition select-none"
      style={{
        background: copied ? "rgba(34,211,238,0.15)" : "rgba(255,255,255,0.07)",
        border: `1px solid ${copied ? "rgba(34,211,238,0.3)" : "rgba(255,255,255,0.1)"}`,
        color: copied ? "#22d3ee" : "#9ca3af",
        borderRadius: "4px",
        cursor: "pointer",
        fontSize: small ? "10px" : "11px",
        padding: small ? "1px 6px" : "2px 8px",
        lineHeight: "1.6",
        fontFamily: "inherit",
      }}
    >
      {copied ? "✓ Copied" : "Copy"}
    </button>
  );
}

// ── Code block with header ─────────────────────────────────────────────────────

function CodeBlock({ language, code }: { language: string; code: string }) {
  return (
    <div
      className="my-2 rounded overflow-hidden"
      style={{
        background: "#0d1117",
        border: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      {/* Header bar */}
      <div
        className="flex items-center justify-between px-3 py-1.5"
        style={{
          background: "rgba(255,255,255,0.04)",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <span
          className="text-[11px] font-mono"
          style={{ color: "#6b7280" }}
        >
          {language || "plaintext"}
        </span>
        <CopyButton text={code} small />
      </div>
      {/* Code body */}
      <pre
        className="overflow-x-auto m-0 p-3 text-[12px] leading-relaxed"
        style={{ color: "#e2e8f0", fontFamily: "ui-monospace, 'Cascadia Code', Consolas, monospace" }}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}

// ── Main markdown renderer ────────────────────────────────────────────────────

interface Props {
  content: string;
}

export function MarkdownContent({ content }: Props) {
  return (
    <div className="markdown-body text-sm text-gray-300 break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // ── Code: block vs inline ─────────────────────────────────────────
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children }) => {
            const match = /language-(\w+)/.exec(className || "");
            const codeStr = String(children).replace(/\n$/, "");
            // Treat as a block if there's a language tag OR it contains newlines
            if (match || codeStr.includes("\n")) {
              return <CodeBlock language={match?.[1] ?? ""} code={codeStr} />;
            }
            // Inline code
            return (
              <code
                className="rounded px-1 py-0.5 font-mono text-[12px]"
                style={{
                  background: "rgba(255,255,255,0.08)",
                  color: "#e2e8f0",
                  border: "1px solid rgba(255,255,255,0.07)",
                }}
              >
                {children}
              </code>
            );
          },

          // ── Headings ──────────────────────────────────────────────────────
          h1: ({ children }) => (
            <h1 className="text-lg font-bold mt-3 mb-1.5" style={{ color: "#f1f5f9", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "6px" }}>
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base font-semibold mt-3 mb-1.5" style={{ color: "#f1f5f9", borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: "4px" }}>
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold mt-2.5 mb-1" style={{ color: "#e2e8f0" }}>
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-sm font-medium mt-2 mb-0.5" style={{ color: "#cbd5e1" }}>
              {children}
            </h4>
          ),

          // ── Paragraph ─────────────────────────────────────────────────────
          p: ({ children }) => (
            <p className="my-1.5 leading-relaxed">{children}</p>
          ),

          // ── Lists ─────────────────────────────────────────────────────────
          ul: ({ children }) => (
            <ul className="my-1.5 pl-5 space-y-0.5" style={{ listStyleType: "disc" }}>
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="my-1.5 pl-5 space-y-0.5" style={{ listStyleType: "decimal" }}>
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed" style={{ color: "#d1d5db" }}>
              {children}
            </li>
          ),

          // ── Blockquote ────────────────────────────────────────────────────
          blockquote: ({ children }) => (
            <blockquote
              className="my-2 pl-3 py-0.5"
              style={{
                borderLeft: "3px solid rgba(34,211,238,0.4)",
                color: "#9ca3af",
                background: "rgba(255,255,255,0.02)",
                borderRadius: "0 4px 4px 0",
              }}
            >
              {children}
            </blockquote>
          ),

          // ── Horizontal rule ───────────────────────────────────────────────
          hr: () => (
            <hr className="my-3" style={{ border: "none", borderTop: "1px solid rgba(255,255,255,0.1)" }} />
          ),

          // ── Table ─────────────────────────────────────────────────────────
          table: ({ children }) => (
            <div className="my-2 overflow-x-auto rounded" style={{ border: "1px solid rgba(255,255,255,0.1)" }}>
              <table className="w-full text-xs border-collapse">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead style={{ background: "rgba(255,255,255,0.05)" }}>{children}</thead>
          ),
          th: ({ children }) => (
            <th
              className="px-3 py-2 text-left font-semibold"
              style={{ color: "#e2e8f0", borderBottom: "1px solid rgba(255,255,255,0.1)" }}
            >
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td
              className="px-3 py-1.5"
              style={{ color: "#d1d5db", borderBottom: "1px solid rgba(255,255,255,0.06)" }}
            >
              {children}
            </td>
          ),

          // ── Links ─────────────────────────────────────────────────────────
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer"
              style={{ color: "#22d3ee", textDecoration: "underline", textUnderlineOffset: "2px" }}
            >
              {children}
            </a>
          ),

          // ── Strong / em ───────────────────────────────────────────────────
          strong: ({ children }) => (
            <strong style={{ color: "#f1f5f9", fontWeight: 600 }}>{children}</strong>
          ),
          em: ({ children }) => (
            <em style={{ color: "#cbd5e1", fontStyle: "italic" }}>{children}</em>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

// Re-export CopyButton for use in MessageBubble
export { CopyButton };
