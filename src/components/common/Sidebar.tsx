import type { SessionInfo } from "../../lib/types";

interface Props {
  sessions: SessionInfo[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
}

export function Sidebar({ sessions, activeId, onSelect, onCreate, onDelete }: Props) {
  return (
    <div
      className="flex flex-col"
      style={{
        width: "220px",
        background: "var(--surface-1)",
        borderRight: "1px solid var(--border)",
        height: "100%",
      }}
    >
      {/* Header */}
      <div
        className="flex shrink-0 items-center justify-between px-3 py-2"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <span
          className="text-[10px] font-semibold uppercase tracking-widest"
          style={{ color: "var(--text-2)" }}
        >
          Sessions
        </span>
        <button
          onClick={onCreate}
          title="New session"
          className="flex h-5 w-5 items-center justify-center rounded transition"
          style={{
            background: "rgba(34,211,238,0.10)",
            border: "1px solid rgba(34,211,238,0.22)",
            color: "#22d3ee",
            fontSize: "14px",
            cursor: "pointer",
            lineHeight: 1,
          }}
          onMouseEnter={(e) =>
            ((e.currentTarget as HTMLButtonElement).style.background =
              "rgba(34,211,238,0.18)")
          }
          onMouseLeave={(e) =>
            ((e.currentTarget as HTMLButtonElement).style.background =
              "rgba(34,211,238,0.10)")
          }
        >
          +
        </button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto py-1">
        {sessions.length === 0 && (
          <p
            className="px-3 py-4 text-center text-xs"
            style={{ color: "var(--text-2)" }}
          >
            No sessions yet.
            <br />
            Click + to create one.
          </p>
        )}

        {sessions.map((s) => {
          const isActive = activeId === s.id;
          return (
            <div
              key={s.id}
              onClick={() => onSelect(s.id)}
              className="group relative flex cursor-pointer items-center px-3 py-2 transition"
              style={{
                background: isActive
                  ? "rgba(34,211,238,0.08)"
                  : "transparent",
                borderLeft: isActive
                  ? "2px solid #22d3ee"
                  : "2px solid transparent",
              }}
              onMouseEnter={(e) => {
                if (!isActive)
                  (e.currentTarget as HTMLDivElement).style.background =
                    "rgba(255,255,255,0.04)";
              }}
              onMouseLeave={(e) => {
                if (!isActive)
                  (e.currentTarget as HTMLDivElement).style.background =
                    "transparent";
              }}
            >
              <span
                className="flex-1 truncate text-sm"
                style={{
                  color: isActive ? "#22d3ee" : "var(--text-1)",
                  fontWeight: isActive ? 500 : 400,
                }}
              >
                {s.name || `Session ${s.id.slice(0, 6)}`}
              </span>

              {/* Delete button — shown on hover */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(s.id);
                }}
                title="Delete session"
                className="ml-1 shrink-0 rounded px-1 py-0.5 text-xs opacity-0 transition group-hover:opacity-100"
                style={{
                  color: "#f87171",
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                }}
                onMouseEnter={(e) =>
                  ((e.currentTarget as HTMLButtonElement).style.background =
                    "rgba(248,113,113,0.12)")
                }
                onMouseLeave={(e) =>
                  ((e.currentTarget as HTMLButtonElement).style.background =
                    "transparent")
                }
              >
                ✕
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
