export interface BenchmarkDataRecord {
  id: string;
}

export interface BenchmarkDataState<
  TResult extends BenchmarkDataRecord,
  THistory extends BenchmarkDataRecord,
> {
  results: ReadonlyArray<TResult>;
  history: ReadonlyArray<THistory>;
}

export type BenchmarkDeletion =
  | { scope: "result"; id: string }
  | { scope: "current-results" }
  | { scope: "history-run"; id: string }
  | { scope: "history" }
  | { scope: "all" };

/**
 * Applies one explicitly scoped benchmark deletion without touching saved
 * presets. Current results and saved run history are intentionally separate;
 * only the `all` scope clears both collections.
 */
export function applyBenchmarkDeletion<
  TResult extends BenchmarkDataRecord,
  THistory extends BenchmarkDataRecord,
>(
  state: BenchmarkDataState<TResult, THistory>,
  deletion: BenchmarkDeletion,
): { results: TResult[]; history: THistory[] } {
  const results = [...state.results];
  const history = [...state.history];

  switch (deletion.scope) {
    case "result":
      return {
        results: results.filter((result) => result.id !== deletion.id),
        history,
      };
    case "current-results":
      return { results: [], history };
    case "history-run":
      return {
        results,
        history: history.filter((run) => run.id !== deletion.id),
      };
    case "history":
      return { results, history: [] };
    case "all":
      return { results: [], history: [] };
  }
}

export function retainExpandedBenchmarkRows(
  expanded: Readonly<Record<string, boolean>>,
  remainingResultIds: ReadonlySet<string>,
) {
  return Object.fromEntries(
    Object.entries(expanded).filter(
      ([id, isExpanded]) => isExpanded && remainingResultIds.has(id),
    ),
  );
}
