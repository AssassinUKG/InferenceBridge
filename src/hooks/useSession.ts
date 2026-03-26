import { useState, useCallback, useEffect } from "react";
import type { SessionInfo } from "../lib/types";
import * as api from "../lib/tauri";

export function useSession() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await api.listSessions();
      setSessions(list);
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const createSession = useCallback(
    async (name: string) => {
      try {
        const id = await api.createSession(name);
        await refresh();
        setActiveId(id);
        return id;
      } catch (err) {
        setError(String(err));
      }
    },
    [refresh]
  );

  const deleteSession = useCallback(
    async (id: string) => {
      try {
        await api.deleteSession(id);
        if (activeId === id) setActiveId(null);
        await refresh();
      } catch (err) {
        setError(String(err));
      }
    },
    [activeId, refresh]
  );

  return {
    sessions,
    activeId,
    setActiveId,
    createSession,
    deleteSession,
    error,
    refresh,
  };
}
