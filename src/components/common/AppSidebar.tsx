import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Boxes,
  Braces,
  Check,
  Download,
  Gauge,
  Library,
  MessageSquare,
  MoreHorizontal,
  PanelLeftClose,
  Pencil,
  Pin,
  PinOff,
  Plus,
  ScrollText,
  Search,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import type { SessionInfo } from "../../lib/types";
import { IconButton } from "../ui/Controls";

export type AppNavId =
  | "chat"
  | "models"
  | "browse"
  | "benchmark"
  | "context"
  | "logs"
  | "debug"
  | "settings";

interface Props {
  activeTab: AppNavId;
  sessions: SessionInfo[];
  activeSessionId: string | null;
  modelName: string | null;
  apiState: string;
  sessionReady: boolean;
  sessionError?: string | null;
  creatingSession: boolean;
  onNavigate: (tab: AppNavId) => void;
  onSelectSession: (id: string) => void;
  onCreateSession: () => void;
  onDeleteSession: (id: string) => void;
  onRenameSession: (id: string, name: string) => Promise<boolean>;
  onSetSessionPinned: (id: string, pinned: boolean) => Promise<boolean>;
  onExportSession: (session: SessionInfo) => void;
}

const mainNavigation = [
  { id: "chat" as const, label: "Chat", icon: MessageSquare },
  { id: "models" as const, label: "Models", icon: Boxes },
  { id: "browse" as const, label: "Model Hub", icon: Search },
];

const workspaceNavigation = [
  { id: "benchmark" as const, label: "Benchmarks", icon: Gauge },
  { id: "context" as const, label: "Context", icon: Library },
  { id: "logs" as const, label: "Logs", icon: ScrollText },
  { id: "debug" as const, label: "API", icon: Braces },
];

export function AppSidebar({
  activeTab,
  sessions,
  activeSessionId,
  modelName,
  apiState,
  sessionReady,
  sessionError,
  creatingSession,
  onNavigate,
  onSelectSession,
  onCreateSession,
  onDeleteSession,
  onRenameSession,
  onSetSessionPinned,
  onExportSession,
}: Props) {
  const [collapsed, setCollapsed] = useState(() => window.matchMedia("(max-width: 720px)").matches);
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const sidebarHomeRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const apiRunning = apiState === "Running";
  const filteredSessions = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    if (!normalized) return sessions;
    return sessions.filter((session) =>
      (session.name ?? `Chat ${session.id.slice(0, 6)}`).toLocaleLowerCase().includes(normalized)
    );
  }, [query, sessions]);

  useEffect(() => {
    if (!openMenuId) return undefined;
    const close = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setOpenMenuId(null);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [openMenuId]);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 720px)");
    const respond = (event: MediaQueryListEvent) => setCollapsed(event.matches);
    media.addEventListener("change", respond);
    return () => media.removeEventListener("change", respond);
  }, []);

  const collapseSidebar = () => {
    setCollapsed(true);
    window.requestAnimationFrame(() => sidebarHomeRef.current?.focus());
  };

  const selectSession = (id: string) => {
    onSelectSession(id);
    onNavigate("chat");
  };

  const beginRename = (session: SessionInfo) => {
    setEditingId(session.id);
    setEditingName(session.name ?? "");
    setOpenMenuId(null);
  };

  const finishRename = async () => {
    const id = editingId;
    const name = editingName.trim();
    if (!id || !name) return;
    if (await onRenameSession(id, name)) {
      setEditingId(null);
      setEditingName("");
    }
  };

  return (
    <aside
      id="app-sidebar-navigation"
      className={`ib-app-sidebar ${collapsed ? "is-collapsed" : ""}`}
      aria-label="Application navigation"
    >
      <div className={`flex h-14 shrink-0 items-center gap-2 ${collapsed ? "justify-center px-2" : "px-3"}`}>
        <button
          ref={sidebarHomeRef}
          type="button"
          className={`flex min-w-0 items-center gap-2 rounded-lg px-1 py-1 text-left ${collapsed ? "justify-center" : "flex-1"}`}
          onClick={() => collapsed ? setCollapsed(false) : onNavigate("chat")}
          aria-label={collapsed ? "Expand sidebar" : "InferenceBridge home"}
          aria-expanded={collapsed ? false : undefined}
          aria-controls={collapsed ? "app-sidebar-navigation" : undefined}
          title={collapsed ? "Expand sidebar" : undefined}
        >
          <span className="ib-brand-mark">IB</span>
          {!collapsed && (
            <span className="min-w-0">
              <span className="block truncate text-sm font-semibold text-white">InferenceBridge</span>
              <span className="block truncate text-[11px] text-[var(--text-2)]">Local AI workspace</span>
            </span>
          )}
        </button>
        {!collapsed && (
          <IconButton
            label="Collapse sidebar"
            size="sm"
            onClick={collapseSidebar}
            aria-expanded={true}
            aria-controls="app-sidebar-navigation"
          >
            <PanelLeftClose size={16} />
          </IconButton>
        )}
      </div>

      <div className="px-2 pb-2">
        <button
          type="button"
          className="ib-new-chat-button"
          disabled={!sessionReady || creatingSession}
          onClick={() => {
            onCreateSession();
            onNavigate("chat");
          }}
          aria-label={creatingSession ? "Creating chat" : "New chat"}
          aria-busy={creatingSession}
        >
          <Plus size={17} />
          {!collapsed && <span>{creatingSession ? "Creating chat..." : "New chat"}</span>}
        </button>
      </div>

      <nav className="space-y-0.5 px-2" aria-label="Primary">
        <button
          type="button"
          className="ib-nav-item"
          onClick={() => window.dispatchEvent(new CustomEvent("ib-open-command-palette"))}
          title={collapsed ? "Search and commands" : undefined}
        >
          <Search size={17} />
          {!collapsed && <><span>Search</span><kbd className="ml-auto text-[9px] text-[var(--text-3)]">Ctrl K</kbd></>}
        </button>
        {mainNavigation.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              type="button"
              className={`ib-nav-item ${activeTab === item.id ? "is-active" : ""}`}
              onClick={() => onNavigate(item.id)}
              title={collapsed ? item.label : undefined}
            >
              <Icon size={17} />
              {!collapsed && <span>{item.label}</span>}
            </button>
          );
        })}
      </nav>

      {!collapsed && (
        <div className="mt-5 flex min-h-0 flex-1 flex-col overflow-hidden">
          <div className="flex items-center justify-between px-4 pb-1.5">
            <span className="ib-sidebar-label">Recent chats</span>
            <span className="text-[10px] tabular-nums text-[var(--text-3)]">{sessions.length}</span>
          </div>
          {sessions.length > 0 && (
            <label className="relative mx-2 mb-2 block">
              <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--text-3)]" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search chats"
                aria-label="Search conversations"
                className="h-8 w-full rounded-lg border border-white/8 bg-black/10 pl-8 pr-7 text-xs text-[var(--text-0)] outline-none placeholder:text-[var(--text-3)] focus:border-white/20"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  aria-label="Clear conversation search"
                  className="absolute right-1.5 top-1/2 flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded text-[var(--text-3)] hover:bg-white/5 hover:text-white"
                >
                  <X size={12} />
                </button>
              )}
            </label>
          )}
          {sessionError && (
            <div
              role="status"
              title={sessionError}
              className="mx-2 mb-2 rounded-lg border border-rose-400/15 bg-rose-400/5 px-2.5 py-2 text-[11px] leading-4 text-rose-200"
            >
              Conversation update failed. Try again.
            </div>
          )}
          <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
            {sessions.length === 0 ? (
              <div className="px-2 py-3 text-xs leading-5 text-[var(--text-3)]">No chats yet</div>
            ) : filteredSessions.length === 0 ? (
              <div className="px-2 py-3 text-xs leading-5 text-[var(--text-3)]">No matching chats</div>
            ) : (
              filteredSessions.map((session) => {
                const active = session.id === activeSessionId && activeTab === "chat";
                const editing = session.id === editingId;
                return (
                  <div key={session.id} className={`ib-session-row group relative ${active ? "is-active" : ""}`}>
                    {session.pinned && !editing && <Pin size={11} className="shrink-0 text-[var(--text-3)]" fill="currentColor" />}
                    {editing ? (
                      <form
                        className="flex min-w-0 flex-1 items-center gap-1"
                        onSubmit={(event) => {
                          event.preventDefault();
                          void finishRename();
                        }}
                      >
                        <input
                          autoFocus
                          value={editingName}
                          maxLength={120}
                          onChange={(event) => setEditingName(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Escape") setEditingId(null);
                          }}
                          aria-label="Conversation title"
                          className="h-7 min-w-0 flex-1 rounded border border-white/20 bg-black/20 px-2 text-xs text-white outline-none"
                        />
                        <button type="submit" aria-label="Save title" className="ib-session-icon"><Check size={13} /></button>
                        <button type="button" aria-label="Cancel rename" className="ib-session-icon" onClick={() => setEditingId(null)}><X size={13} /></button>
                      </form>
                    ) : (
                      <button
                        type="button"
                        className="min-w-0 flex-1 truncate text-left"
                        onClick={() => selectSession(session.id)}
                        onDoubleClick={() => beginRename(session)}
                        title={session.name ?? undefined}
                      >
                        {session.name || `Chat ${session.id.slice(0, 6)}`}
                      </button>
                    )}
                    {!editing && (
                      <>
                        <button
                          type="button"
                          aria-label={`Actions for ${session.name || "chat"}`}
                          title="Conversation actions"
                          className="ib-session-action ib-session-more"
                          onClick={(event) => {
                            event.stopPropagation();
                            setOpenMenuId((current) => current === session.id ? null : session.id);
                          }}
                        >
                          <MoreHorizontal size={15} />
                        </button>
                        <button
                          type="button"
                          aria-label={`Delete ${session.name || "chat"}`}
                          title="Delete chat"
                          className="ib-session-action ib-session-delete"
                          onClick={(event) => {
                            event.stopPropagation();
                            setOpenMenuId(null);
                            if (window.confirm(`Delete "${session.name || "this chat"}"? This cannot be undone.`)) {
                              onDeleteSession(session.id);
                            }
                          }}
                        >
                          <Trash2 size={14} />
                        </button>
                      </>
                    )}
                    {openMenuId === session.id && (
                      <div ref={menuRef} className="absolute right-1 top-8 z-50 w-40 rounded-xl border border-white/10 bg-[var(--surface-2)] p-1 shadow-2xl">
                        <button type="button" className="ib-session-menu-item" onClick={() => beginRename(session)}><Pencil size={13} />Rename</button>
                        <button type="button" className="ib-session-menu-item" onClick={() => { void onSetSessionPinned(session.id, !session.pinned); setOpenMenuId(null); }}>
                          {session.pinned ? <PinOff size={13} /> : <Pin size={13} />}{session.pinned ? "Unpin" : "Pin"}
                        </button>
                        <button type="button" className="ib-session-menu-item" onClick={() => { onExportSession(session); setOpenMenuId(null); }}><Download size={13} />Export</button>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}

      {collapsed && <div className="min-h-3 flex-1" />}

      <div className="border-t border-[var(--border)] px-2 py-2">
        <div className="space-y-0.5">
          {workspaceNavigation.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                type="button"
                className={`ib-nav-item ${activeTab === item.id ? "is-active" : ""}`}
                onClick={() => onNavigate(item.id)}
                title={collapsed ? item.label : undefined}
              >
                <Icon size={17} />
                {!collapsed && <span>{item.label}</span>}
              </button>
            );
          })}
          <button
            type="button"
            className={`ib-nav-item ${activeTab === "settings" ? "is-active" : ""}`}
            onClick={() => onNavigate("settings")}
            title={collapsed ? "Settings" : undefined}
          >
            <Settings size={17} />
            {!collapsed && <span>Settings</span>}
          </button>
        </div>

        {!collapsed && (
          <div className="mt-2 rounded-xl bg-[var(--surface-2)] px-3 py-2.5">
            <div className="flex items-center gap-2 text-xs text-[var(--text-1)]">
              <Activity size={14} className={apiRunning ? "text-emerald-400" : "text-[var(--text-3)]"} />
              <span className="truncate">API {apiState}</span>
            </div>
            <div className="mt-1.5 flex items-center gap-2 text-[11px] text-[var(--text-2)]">
              <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${modelName ? "bg-emerald-400" : "bg-[var(--text-3)]"}`} />
              <span className="truncate" title={modelName ?? "No model loaded"}>
                {modelName ?? "No model loaded"}
              </span>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
