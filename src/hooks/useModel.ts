import { useState, useCallback, useEffect } from "react";
import { listen } from "@tauri-apps/api/event";
import type { ModelInfo, ProcessStatusInfo, LoadProgress } from "../lib/types";
import * as api from "../lib/tauri";

interface ModelState {
  models: ModelInfo[];
  processStatus: ProcessStatusInfo | null;
  isLoading: boolean;
  loadProgress: LoadProgress | null;
  error: string | null;
}

function sameModels(a: ModelInfo[], b: ModelInfo[]) {
  if (a === b) {
    return true;
  }
  if (a.length !== b.length) {
    return false;
  }
  return a.every((model, index) => {
    const other = b[index];
    return (
      model.filename === other.filename &&
      model.path === other.path &&
      model.family === other.family &&
      model.quant === other.quant
    );
  });
}

function sameProcessStatus(
  a: ProcessStatusInfo | null,
  b: ProcessStatusInfo | null
) {
  if (a === b) {
    return true;
  }

  if (!a || !b) {
    return false;
  }

  return (
    a.state === b.state &&
    a.model === b.model &&
    a.previous_model === b.previous_model &&
    a.crash_count === b.crash_count &&
    a.server_version === b.server_version &&
    a.server_path === b.server_path &&
    a.backend === b.backend &&
    a.api_state === b.api_state &&
    a.api_error === b.api_error &&
    a.api_url === b.api_url &&
    a.api_reachable === b.api_reachable &&
    a.api_port_owner?.pid === b.api_port_owner?.pid &&
    a.api_port_owner?.killable === b.api_port_owner?.killable
  );
}

export function useModel() {
  const [state, setState] = useState<ModelState>({
    models: [],
    processStatus: null,
    isLoading: false,
    loadProgress: null,
    error: null,
  });

  const refresh = useCallback(async () => {
    try {
      const [models, status] = await Promise.all([
        api.listModels(),
        api.getProcessStatus(),
      ]);
      setState((s) => ({ ...s, models, processStatus: status, error: null }));
    } catch (err) {
      setState((s) => ({ ...s, error: String(err) }));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Listen for model load progress events from Rust
  useEffect(() => {
    const unlisten = listen<LoadProgress>("model-load-progress", (event) => {
      const p = event.payload;
      setState((s) => ({
        ...s,
        loadProgress: p,
        error: p.error ?? s.error,
      }));
      // When loading is done, refresh model list and status
      if (p.done && !p.error) {
        setTimeout(() => {
          refresh();
          setState((s) => ({ ...s, isLoading: false, loadProgress: null }));
        }, 500);
      }
      if (p.done && p.error) {
        setState((s) => ({ ...s, isLoading: false }));
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [refresh]);

  // Poll status and registry so API-driven loads appear in the GUI too.
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const [status, models] = await Promise.all([
          api.getProcessStatus(),
          api.listModels(),
        ]);
        setState((s) =>
          sameProcessStatus(s.processStatus, status) && sameModels(s.models, models)
            ? s
            : { ...s, processStatus: status, models }
        );
      } catch {
        // ignore polling errors
      }
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Listen for instant API server state notifications (no poll delay).
  useEffect(() => {
    const unlisten = listen<{ state: string; error: string | null }>(
      "api-server-state-changed",
      () => {
        // Re-read full process status so the GUI reflects the new state immediately.
        refresh();
      }
    );
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [refresh]);

  const scanModels = useCallback(async () => {
    setState((s) => ({ ...s, isLoading: true, error: null }));
    try {
      const count = await api.scanModels();
      const models = await api.listModels();
      setState((s) => ({ ...s, models, isLoading: false, error: null }));
      return count;
    } catch (err) {
      setState((s) => ({ ...s, isLoading: false, error: String(err) }));
    }
  }, []);

  const loadModel = useCallback(
    async (modelName: string, contextSize?: number) => {
      setState((s) => ({
        ...s,
        isLoading: true,
        error: null,
        loadProgress: {
          stage: "resolving",
          message: "Starting...",
          progress: 0,
          done: false,
          error: null,
        },
      }));
      try {
        const result = await api.loadModel(modelName, contextSize);
        await refresh();
        setState((s) => ({ ...s, isLoading: false, loadProgress: null }));
        return result;
      } catch (err) {
        setState((s) => ({
          ...s,
          isLoading: false,
          error: String(err),
          loadProgress: {
            stage: "error",
            message: String(err),
            progress: 0,
            done: true,
            error: String(err),
          },
        }));
      }
    },
    [refresh]
  );

  const unloadModel = useCallback(async () => {
    setState((s) => ({ ...s, isLoading: true }));
    try {
      await api.unloadModel();
      await refresh();
      setState((s) => ({ ...s, isLoading: false, loadProgress: null }));
    } catch (err) {
      setState((s) => ({ ...s, isLoading: false, error: String(err) }));
    }
  }, [refresh]);

  const swapModel = useCallback(
    async (modelName?: string, contextSize?: number) => {
      setState((s) => ({
        ...s,
        isLoading: true,
        error: null,
        loadProgress: {
          stage: "resolving",
          message: modelName ? `Swapping to ${modelName}...` : "Swapping back...",
          progress: 0,
          done: false,
          error: null,
        },
      }));
      try {
        const result = await api.swapModel(modelName, contextSize);
        await refresh();
        setState((s) => ({ ...s, isLoading: false, loadProgress: null }));
        return result;
      } catch (err) {
        setState((s) => ({
          ...s,
          isLoading: false,
          error: String(err),
          loadProgress: {
            stage: "error",
            message: String(err),
            progress: 0,
            done: true,
            error: String(err),
          },
        }));
      }
    },
    [refresh]
  );

  const setApiServerRunning = useCallback(
    async (running: boolean) => {
      try {
        await api.setApiServerRunning(running);
        await refresh();
      } catch (err) {
        setState((s) => ({ ...s, error: String(err) }));
      }
    },
    [refresh]
  );

  return { ...state, scanModels, loadModel, unloadModel, swapModel, setApiServerRunning, refresh };
}
