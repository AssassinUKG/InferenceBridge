import { useState, useCallback, useEffect, useRef } from "react";
import type { SessionInfo } from "../lib/types";
import * as api from "../lib/tauri";

export function useSession() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const creatingRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const list = await api.listSessions();
      setSessions(list);
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setReady(true);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const createSession = useCallback(
    async (name: string) => {
      if (creatingRef.current) return undefined;
      creatingRef.current = true;
      setIsCreating(true);
      try {
        const id = await api.createSession(name);
        await refresh();
        setActiveId(id);
        return id;
      } catch (err) {
        setError(String(err));
        return undefined;
      } finally {
        creatingRef.current = false;
        setIsCreating(false);
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

  const renameSession = useCallback(
    async (id: string, name: string) => {
      try {
        await api.renameSession(id, name);
        await refresh();
        return true;
      } catch (err) {
        setError(String(err));
        return false;
      }
    },
    [refresh]
  );

  const setSessionPinned = useCallback(
    async (id: string, pinned: boolean) => {
      try {
        await api.setSessionPinned(id, pinned);
        await refresh();
        return true;
      } catch (err) {
        setError(String(err));
        return false;
      }
    },
    [refresh]
  );

  return {
    sessions,
    activeId,
    setActiveId,
    createSession,
    deleteSession,
    renameSession,
    setSessionPinned,
    error,
    ready,
    isCreating,
    refresh,
  };
}
