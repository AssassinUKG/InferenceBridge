export interface DownloadSnapshot {
  id: string;
}

interface ClearableDownload {
  done: boolean;
  resumable: boolean;
  status: string;
}

/**
 * Merge the durable startup snapshot with progress events already received by
 * the UI. Live entries win so a slow list response cannot roll a transfer back
 * to older byte counts or an earlier status.
 */
export function mergeDownloadSnapshots<T extends DownloadSnapshot>(
  restored: readonly T[],
  current: Readonly<Record<string, T>>,
): Record<string, T> {
  return {
    ...Object.fromEntries(restored.map((entry) => [entry.id, entry])),
    ...current,
  };
}

/** A resumable failure still owns useful partial bytes and must be discarded explicitly. */
export function isClearableDownload(entry: ClearableDownload): boolean {
  return entry.done && !entry.resumable && entry.status !== "Cleanup pending";
}
