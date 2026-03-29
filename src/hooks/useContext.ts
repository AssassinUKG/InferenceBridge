import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import type { ContextStatus } from "../lib/types";
import * as api from "../lib/tauri";

const EMPTY_STATUS: ContextStatus = {
  total_tokens: 0,
  used_tokens: 0,
  fill_ratio: 0,
  pinned_tokens: 0,
  rolling_tokens: 0,
  compressed_tokens: 0,
  last_compaction_action: null,
};

function sameContextStatus(a: ContextStatus, b: ContextStatus) {
  return (
    a.total_tokens === b.total_tokens &&
    a.used_tokens === b.used_tokens &&
    a.fill_ratio === b.fill_ratio &&
    a.pinned_tokens === b.pinned_tokens &&
    a.rolling_tokens === b.rolling_tokens &&
    a.compressed_tokens === b.compressed_tokens &&
    a.last_compaction_action === b.last_compaction_action
  );
}

export function useContext(pollInterval = 2000) {
  const [status, setStatus] = useState<ContextStatus>(EMPTY_STATUS);

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const next = await api.getContextStatus();
        if (active) {
          setStatus((current) => (sameContextStatus(current, next) ? current : next));
        }
      } catch {
        // ignore transient poll failures
      }
    };

    poll();
    const intervalId = setInterval(poll, pollInterval);
    let unlisten: (() => void) | null = null;

    listen("context-pressure", () => {
      void poll();
    }).then((dispose) => {
      unlisten = dispose;
    }).catch(() => undefined);

    return () => {
      active = false;
      clearInterval(intervalId);
      unlisten?.();
    };
  }, [pollInterval]);

  return status;
}
