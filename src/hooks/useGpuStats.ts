import { useEffect, useState } from "react";
import type { GpuStats } from "../lib/types";
import * as api from "../lib/tauri";

const POLL_INTERVAL_MS = 3000;

export function useGpuStats() {
  const [stats, setStats] = useState<GpuStats | null>(null);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const s = await api.getGpuStats();
        if (!cancelled) setStats(s);
      } catch {
        // nvidia-smi not available — leave stats null
        if (!cancelled) setStats(null);
      }
    };

    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return stats;
}
