import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Copy, Download, RefreshCw, X } from "lucide-react";

export interface CanvasVersion {
  html: string;
  label: string;
}

interface CanvasError {
  message: string;
  detail?: string;
}

function injectSafetyShell(html: string, token: string) {
  const csp = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; font-src data:; script-src 'unsafe-inline'; media-src data: blob:; connect-src 'none'; form-action 'none'; base-uri 'none'">`;
  const shim = `<script>(function(){var sent=false;function report(message,detail){if(sent)return;sent=true;try{parent.postMessage({__ibCanvas:true,token:${JSON.stringify(token)},message:String(message).slice(0,400),detail:String(detail||'').slice(0,400)},'*')}catch(_){}}window.addEventListener('error',function(e){report(e.message||'Preview error',(e.filename||'')+':'+(e.lineno||0))});window.addEventListener('unhandledrejection',function(e){report((e.reason&&e.reason.message)||String(e.reason||'Unhandled promise rejection'))});document.addEventListener('click',function(e){var a=e.target&&e.target.closest&&e.target.closest('a');if(a&&a.href){e.preventDefault();report('External navigation blocked',a.getAttribute('href')||'')}},true)})();<\/script>`;
  const shell = csp + shim;
  const head = /<head[^>]*>/i.exec(html);
  if (head?.index != null) {
    const index = head.index + head[0].length;
    return html.slice(0, index) + shell + html.slice(index);
  }
  const documentRoot = /<html[^>]*>/i.exec(html);
  if (documentRoot?.index != null) {
    const index = documentRoot.index + documentRoot[0].length;
    return html.slice(0, index) + `<head>${shell}</head>` + html.slice(index);
  }
  return `<!doctype html><html><head>${shell}</head><body>${html}</body></html>`;
}

export function CanvasPanel({
  versions,
  index,
  onSelect,
  onClose,
}: {
  versions: CanvasVersion[];
  index: number;
  onSelect: (index: number) => void;
  onClose: () => void;
}) {
  const [error, setError] = useState<CanvasError | null>(null);
  const [copied, setCopied] = useState(false);
  const [reload, setReload] = useState(0);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const tokenRef = useRef(crypto.randomUUID());
  const current = versions[index];
  const srcDoc = useMemo(() => current ? injectSafetyShell(current.html.slice(0, 250_000), tokenRef.current) : "", [current, reload]);

  useEffect(() => setError(null), [index, reload]);
  useEffect(() => {
    const receive = (event: MessageEvent) => {
      if (event.source !== iframeRef.current?.contentWindow) return;
      const data = event.data as { __ibCanvas?: boolean; token?: string; message?: string; detail?: string };
      if (!data?.__ibCanvas || data.token !== tokenRef.current || typeof data.message !== "string") return;
      setError({ message: data.message, detail: data.detail });
    };
    window.addEventListener("message", receive);
    return () => window.removeEventListener("message", receive);
  }, []);

  useEffect(() => {
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [onClose]);

  if (!current) return null;

  const save = () => {
    const url = URL.createObjectURL(new Blob([current.html], { type: "text/html;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "inferencebridge-canvas.html";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  return (
    <div className="fixed inset-0 z-[110] flex bg-black/60 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="HTML Canvas preview">
      <section className="m-3 flex min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border border-white/12 bg-[var(--surface-1)] shadow-2xl sm:m-6">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b border-white/8 px-3">
          <div className="min-w-0 flex-1">
            <div className="text-sm font-semibold text-white">Canvas preview</div>
            <div className="truncate text-[10px] text-[var(--text-3)]">Sandboxed · network and external navigation blocked</div>
          </div>
          <button type="button" onClick={() => setReload((value) => value + 1)} className="ib-canvas-action" aria-label="Reload preview"><RefreshCw size={14} /></button>
          <button type="button" onClick={async () => { await navigator.clipboard.writeText(current.html); setCopied(true); window.setTimeout(() => setCopied(false), 1200); }} className="ib-canvas-action" aria-label="Copy HTML">{copied ? <Check size={14} /> : <Copy size={14} />}</button>
          <button type="button" onClick={save} className="ib-canvas-action" aria-label="Save HTML"><Download size={14} /></button>
          <button type="button" onClick={onClose} className="ib-canvas-action" aria-label="Close Canvas"><X size={16} /></button>
        </header>
        <div className="flex min-h-0 flex-1 flex-col sm:flex-row">
          {versions.length > 1 && (
            <aside className="flex max-h-20 shrink-0 gap-1 overflow-auto border-b border-white/8 p-2 sm:max-h-none sm:w-44 sm:flex-col sm:border-b-0 sm:border-r">
              {versions.map((version, versionIndex) => (
                <button key={`${versionIndex}-${version.label}`} type="button" onClick={() => onSelect(versionIndex)} className={`shrink-0 rounded-lg px-2.5 py-2 text-left text-xs ${versionIndex === index ? "bg-white/10 text-white" : "text-[var(--text-2)] hover:bg-white/5"}`}>{version.label}</button>
              ))}
            </aside>
          )}
          <div className="relative min-h-0 flex-1 bg-white">
            <iframe
              ref={iframeRef}
              key={`${index}-${reload}`}
              srcDoc={srcDoc}
              sandbox="allow-scripts"
              title="Generated HTML preview"
              className="h-full w-full border-0"
            />
            {error && (
              <div className="absolute bottom-3 left-3 right-3 rounded-xl border border-amber-400/30 bg-[#211b0d]/95 px-3 py-2 text-xs text-amber-100 shadow-xl">
                <div className="font-semibold">Preview reported: {error.message}</div>
                {error.detail && <div className="mt-1 truncate text-amber-200/70">{error.detail}</div>}
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
