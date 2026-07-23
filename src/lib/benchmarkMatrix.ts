export type BenchmarkSamplingId = "deterministic" | "instruct" | "coding";
export type BenchmarkRuntimeId = "auto" | "mtp-1" | "mtp-2" | "mtp-3" | "mtp-4";

export interface BenchmarkSamplingCandidate {
  id: BenchmarkSamplingId;
  label: string;
  description: string;
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  presencePenalty: number;
  repeatPenalty: number;
  seed: number;
}

export interface BenchmarkRuntimeCandidate {
  id: BenchmarkRuntimeId;
  label: string;
  description: string;
  mtpOnly: boolean;
  specType: string | null;
  specDraftNMax: number | null;
}

export const BENCHMARK_SAMPLING_CANDIDATES: BenchmarkSamplingCandidate[] = [
  {
    id: "deterministic",
    label: "Deterministic",
    description: "Stable comparison baseline.",
    temperature: 0,
    topP: 0.9,
    topK: 20,
    minP: 0,
    presencePenalty: 0,
    repeatPenalty: 1,
    seed: 42,
  },
  {
    id: "instruct",
    label: "Instruct",
    description: "Non-thinking instruction/tool candidate.",
    temperature: 0.7,
    topP: 0.8,
    topK: 20,
    minP: 0,
    presencePenalty: 1.5,
    repeatPenalty: 1,
    seed: 42,
  },
  {
    id: "coding",
    label: "Coding",
    description: "Precise coding/repository candidate.",
    temperature: 0.6,
    topP: 0.95,
    topK: 20,
    minP: 0,
    presencePenalty: 0,
    repeatPenalty: 1,
    seed: 42,
  },
];

export const BENCHMARK_RUNTIME_CANDIDATES: BenchmarkRuntimeCandidate[] = [
  {
    id: "auto",
    label: "Runtime auto",
    description: "Reset speculation and use InferenceBridge's detected per-model defaults.",
    mtpOnly: false,
    specType: null,
    specDraftNMax: null,
  },
  {
    id: "mtp-1",
    label: "MTP x1",
    description: "Self-MTP with one draft token.",
    mtpOnly: true,
    specType: "draft-mtp",
    specDraftNMax: 1,
  },
  {
    id: "mtp-2",
    label: "MTP x2",
    description: "Self-MTP with two draft tokens.",
    mtpOnly: true,
    specType: "draft-mtp",
    specDraftNMax: 2,
  },
  {
    id: "mtp-3",
    label: "MTP x3",
    description: "Three-token self-MTP candidate between the detected default and aggressive depth.",
    mtpOnly: true,
    specType: "draft-mtp",
    specDraftNMax: 3,
  },
  {
    id: "mtp-4",
    label: "MTP x4",
    description: "Aggressive self-MTP candidate.",
    mtpOnly: true,
    specType: "draft-mtp",
    specDraftNMax: 4,
  },
];

export function selectedSamplingCandidates(ids: BenchmarkSamplingId[]) {
  const selected = new Set(ids);
  return BENCHMARK_SAMPLING_CANDIDATES.filter((candidate) => selected.has(candidate.id));
}

export function selectedRuntimeCandidates(ids: BenchmarkRuntimeId[], modelName: string) {
  const selected = new Set(ids);
  const supportsMtp = /(?:^|[-_.\s])mtp(?:[-_.\s]|$)/i.test(modelName);
  return BENCHMARK_RUNTIME_CANDIDATES.filter(
    (candidate) => selected.has(candidate.id) && (!candidate.mtpOnly || supportsMtp),
  );
}

export function benchmarkCombinationCount(
  modelNames: string[],
  contextCount: number,
  testCount: number,
  samplingIds: BenchmarkSamplingId[],
  runtimeIds: BenchmarkRuntimeId[],
) {
  const samplingCount = selectedSamplingCandidates(samplingIds).length;
  return modelNames.reduce(
    (total, modelName) => total
      + selectedRuntimeCandidates(runtimeIds, modelName).length
        * contextCount
        * testCount
        * samplingCount,
    0,
  );
}
