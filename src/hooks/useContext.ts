import { useState, useEffect } from "react";
import type { ContextStatus } from "../lib/types";
import * as api from "../lib/tauri";

function sameContextStatus(a: ContextStatus, b: ContextStatus) {
  return (
    a.total_tokens === b.total_tokens &&
    a.used_tokens === b.used_tokens &&
    a.fill_ratio === b.fill_ratio
  );
}

export function useContext(pollInterval = 2000) {
  const [status, setStatus] = useState<ContextStatus>({
    total_tokens: 0,
    used_tokens: 0,
    fill_ratio: 0,
  });

  useEffect(() => {
    let active = true;

    const poll = async () => {
      try {
        const s = await api.getContextStatus();
        if (active) {
          setStatus((current) => (sameContextStatus(current, s) ? current : s));
        }
      } catch {
        // ignore
      }
    };

    poll();
    const id = setInterval(poll, pollInterval);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [pollInterval]);

  return status;
}
