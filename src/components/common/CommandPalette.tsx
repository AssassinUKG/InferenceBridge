import { useEffect, useMemo, useRef, useState } from "react";
import {
  Boxes,
  Braces,
  Gauge,
  Library,
  MessageSquare,
  Plus,
  Search,
  Settings,
  SlidersHorizontal,
} from "lucide-react";
import type { SessionInfo } from "../../lib/types";
import type { AppNavId } from "./AppSidebar";

interface PaletteAction {
  id: string;
  label: string;
  detail: string;
  keywords: string;
  icon: React.ReactNode;
  run: () => void;
}

interface Props {
  sessions: SessionInfo[];
  onCreateChat: () => void;
  onSelectSession: (id: string) => void;
  onNavigate: (tab: AppNavId) => void;
  onChooseModel: () => void;
}

const destinations: Array<{ id: AppNavId; label: string; icon: React.ReactNode }> = [
  { id: "models", label: "Models", icon: <Boxes size={16} /> },
  { id: "browse", label: "Model Hub", icon: <Search size={16} /> },
  { id: "benchmark", label: "Benchmarks", icon: <Gauge size={16} /> },
  { id: "context", label: "Context", icon: <Library size={16} /> },
  { id: "debug", label: "API", icon: <Braces size={16} /> },
  { id: "settings", label: "Settings", icon: <Settings size={16} /> },
];

export function CommandPalette({
  sessions,
  onCreateChat,
  onSelectSession,
  onNavigate,
  onChooseModel,
}: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const openPalette = () => setOpen(true);
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase() === "k") {
        event.preventDefault();
        setOpen((current) => !current);
      }
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("ib-open-command-palette", openPalette);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("ib-open-command-palette", openPalette);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setActiveIndex(0);
    window.requestAnimationFrame(() => inputRef.current?.focus());
  }, [open]);

  const actions = useMemo<PaletteAction[]>(() => {
    const closeThen = (action: () => void) => () => {
      setOpen(false);
      action();
    };
    return [
      {
        id: "new-chat",
        label: "New chat",
        detail: "Start a fresh conversation",
        keywords: "new create conversation",
        icon: <Plus size={16} />,
        run: closeThen(onCreateChat),
      },
      {
        id: "focus-composer",
        label: "Focus message composer",
        detail: "Jump back to chat input",
        keywords: "chat prompt input compose",
        icon: <MessageSquare size={16} />,
        run: closeThen(() => {
          onNavigate("chat");
          window.setTimeout(() => window.dispatchEvent(new CustomEvent("ib-focus-composer")), 0);
        }),
      },
      {
        id: "choose-model",
        label: "Choose or switch model",
        detail: "Open the IB model picker",
        keywords: "load eject swap model gguf",
        icon: <SlidersHorizontal size={16} />,
        run: closeThen(onChooseModel),
      },
      ...destinations.map((destination) => ({
        id: `destination-${destination.id}`,
        label: `Open ${destination.label}`,
        detail: "Workspace",
        keywords: `${destination.label} navigate workspace`,
        icon: destination.icon,
        run: closeThen(() => onNavigate(destination.id)),
      })),
      ...sessions.map((session) => ({
        id: `session-${session.id}`,
        label: session.name || `Chat ${session.id.slice(0, 6)}`,
        detail: session.pinned ? "Pinned conversation" : "Recent conversation",
        keywords: `${session.name ?? "chat"} conversation session`,
        icon: <MessageSquare size={16} />,
        run: closeThen(() => {
          onSelectSession(session.id);
          onNavigate("chat");
        }),
      })),
    ];
  }, [onChooseModel, onCreateChat, onNavigate, onSelectSession, sessions]);

  const matches = useMemo(() => {
    const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    if (terms.length === 0) return actions;
    return actions.filter((action) => {
      const haystack = `${action.label} ${action.detail} ${action.keywords}`.toLocaleLowerCase();
      return terms.every((term) => haystack.includes(term));
    });
  }, [actions, query]);

  useEffect(() => {
    setActiveIndex((current) => Math.min(current, Math.max(0, matches.length - 1)));
  }, [matches.length]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-[120] flex items-start justify-center bg-black/55 px-4 pt-[12vh] backdrop-blur-sm"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) setOpen(false);
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        className="w-full max-w-[620px] overflow-hidden rounded-2xl border border-white/12 bg-[var(--surface-1)] shadow-[0_28px_90px_rgba(0,0,0,0.62)]"
      >
        <label className="flex h-14 items-center gap-3 border-b border-white/8 px-4">
          <Search size={18} className="shrink-0 text-[var(--text-2)]" />
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setActiveIndex(0);
            }}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setActiveIndex((current) => Math.min(matches.length - 1, current + 1));
              } else if (event.key === "ArrowUp") {
                event.preventDefault();
                setActiveIndex((current) => Math.max(0, current - 1));
              } else if (event.key === "Enter" && matches[activeIndex]) {
                event.preventDefault();
                matches[activeIndex].run();
              }
            }}
            placeholder="Search chats, models and actions"
            aria-label="Search commands"
            className="min-w-0 flex-1 bg-transparent text-[15px] text-white outline-none placeholder:text-[var(--text-3)]"
          />
          <kbd className="rounded-md border border-white/10 bg-black/15 px-1.5 py-0.5 text-[10px] text-[var(--text-3)]">ESC</kbd>
        </label>
        <div className="max-h-[55vh] overflow-y-auto p-2" role="listbox">
          {matches.length === 0 ? (
            <div className="px-3 py-10 text-center text-sm text-[var(--text-3)]">No matching action</div>
          ) : matches.map((action, index) => (
            <button
              key={action.id}
              type="button"
              role="option"
              aria-selected={index === activeIndex}
              onMouseEnter={() => setActiveIndex(index)}
              onClick={action.run}
              className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left ${index === activeIndex ? "bg-white/8" : "hover:bg-white/5"}`}
            >
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-black/15 text-[var(--text-1)]">{action.icon}</span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-[var(--text-0)]">{action.label}</span>
                <span className="block truncate text-[11px] text-[var(--text-3)]">{action.detail}</span>
              </span>
            </button>
          ))}
        </div>
        <div className="flex items-center justify-between border-t border-white/8 px-4 py-2 text-[10px] text-[var(--text-3)]">
          <span>↑↓ Navigate · Enter Select</span>
          <span>Ctrl/Cmd K</span>
        </div>
      </section>
    </div>
  );
}
