// Typed wrappers around Tauri invoke calls.

import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import type {
  AppSettings,
  ApiAccessInfo,
  ContextStatus,
  DebugApiResponse,
  EffectiveProfileInfo,
  GpuStats,
  HubAccessStatus,
  HfSidecarCacheStatus,
  HfSidecarRollbackSummary,
  HfSidecarSyncSummary,
  ImageGenerationCapabilityStatus,
  ImageGenerationPreview,
  ImageGenerationRequest,
  ImageGenerationResult,
  LlamaServerInfo,
  LogEntry,
  MessageInfo,
  ModelInfo,
  ProcessStatusInfo,
  RuntimePackInfo,
  RuntimeDoctorReport,
  ServerInfo,
  SessionInfo,
  TemplateDryRunReport,
} from "./types";

export type { HubAccessStatus, HfSidecarCacheStatus, HfSidecarRollbackSummary, HfSidecarSyncSummary } from "./types";

function isTauriRuntime() {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauriRuntime()) {
    return Promise.reject(
      new Error(
        "Tauri desktop runtime is not available. Start the full app with `npm run tauri dev`; the Vite URL is only the frontend preview."
      )
    );
  }

  return tauriInvoke<T>(command, args);
}

// Image generation

export const getImageGenerationStatus = () =>
  invoke<ImageGenerationCapabilityStatus>("get_image_generation_status");

export const previewImageGeneration = (request: ImageGenerationRequest) =>
  invoke<ImageGenerationPreview>("preview_image_generation", { request });

export const generateImage = (request: ImageGenerationRequest) =>
  invoke<ImageGenerationResult>("generate_image", { request });

export const cancelImageGeneration = () =>
  invoke<void>("cancel_image_generation");

// Models

export const listModels = () => invoke<ModelInfo[]>("list_models");

export const scanModels = () => invoke<number>("scan_models");

export const setModelVisionOverride = (modelName: string, supportsVision: boolean) =>
  invoke<void>("set_model_vision_override", { modelName, supportsVision });

export interface LoadModelOptions {
  contextSize?: number;
  gpuLayers?: number;
  threads?: number;
  threadsBatch?: number;
  batchSize?: number;
  ubatchSize?: number;
  flashAttn?: boolean;
  useMmap?: boolean;
  useMlock?: boolean;
  contBatching?: boolean;
  parallelSlots?: number;
  mainGpu?: number;
  defragThold?: number;
  ropeFreqScale?: number;
  cacheTypeK?: string;
  cacheTypeV?: string;
  kvUnified?: boolean;
  noWarmup?: boolean;
  ctxShift?: boolean;
  hfRepo?: string;
  hfFile?: string;
  fitMode?: string;
  cacheRamMb?: number | null;
  ctxcp?: number | null;
  useJinja?: boolean;
  reasoningMode?: string;
  reasoningPreserve?: boolean;
  templateMode?: string;
  templateName?: string | null;
  customTemplatePath?: string | null;
  chatTemplateKwargsJson?: string | null;
  draftModelPath?: string | null;
  specType?: string | null;
  specDraftNMax?: number;
  draftMaxTokens?: number;
  draftMinTokens?: number;
  draftPMin?: number;
  attachMmproj?: boolean;
  forceReload?: boolean;
  extraArgs?: string[];
}

export const loadModel = (modelName: string, options?: LoadModelOptions) =>
  invoke<string>("load_model", {
    modelName,
    options: {
      contextSize: options?.contextSize,
      gpuLayers: options?.gpuLayers,
      threads: options?.threads,
      threadsBatch: options?.threadsBatch,
      batchSize: options?.batchSize,
      ubatchSize: options?.ubatchSize,
      flashAttn: options?.flashAttn,
      useMmap: options?.useMmap,
      useMlock: options?.useMlock,
      contBatching: options?.contBatching,
      parallelSlots: options?.parallelSlots,
      mainGpu: options?.mainGpu,
      defragThold: options?.defragThold,
      ropeFreqScale: options?.ropeFreqScale,
      cacheTypeK: options?.cacheTypeK,
      cacheTypeV: options?.cacheTypeV,
      kvUnified: options?.kvUnified,
      noWarmup: options?.noWarmup,
      ctxShift: options?.ctxShift,
      hfRepo: options?.hfRepo,
      hfFile: options?.hfFile,
      fitMode: options?.fitMode,
      cacheRamMb: options?.cacheRamMb ?? null,
      ctxcp: options?.ctxcp ?? null,
      useJinja: options?.useJinja,
      reasoningMode: options?.reasoningMode,
      reasoningPreserve: options?.reasoningPreserve,
      templateMode: options?.templateMode,
      templateName: options?.templateName ?? null,
      customTemplatePath: options?.customTemplatePath ?? null,
      chatTemplateKwargsJson: options?.chatTemplateKwargsJson ?? null,
      draftModelPath: options?.draftModelPath ?? null,
      specType: options?.specType ?? null,
      specDraftNMax: options?.specDraftNMax,
      draftMaxTokens: options?.draftMaxTokens,
      draftMinTokens: options?.draftMinTokens,
      draftPMin: options?.draftPMin,
      attachMmproj: options?.attachMmproj,
      forceReload: options?.forceReload,
      extraArgs: options?.extraArgs,
    },
  });

export const unloadModel = () => invoke<string>("unload_model");

export const swapModel = (modelName?: string, options?: LoadModelOptions) =>
  invoke<string>("swap_model", {
    modelName,
    options: {
      contextSize: options?.contextSize,
      gpuLayers: options?.gpuLayers,
      threads: options?.threads,
      threadsBatch: options?.threadsBatch,
      batchSize: options?.batchSize,
      ubatchSize: options?.ubatchSize,
      flashAttn: options?.flashAttn,
      useMmap: options?.useMmap,
      useMlock: options?.useMlock,
      contBatching: options?.contBatching,
      parallelSlots: options?.parallelSlots,
      mainGpu: options?.mainGpu,
      defragThold: options?.defragThold,
      ropeFreqScale: options?.ropeFreqScale,
      cacheTypeK: options?.cacheTypeK,
      cacheTypeV: options?.cacheTypeV,
      kvUnified: options?.kvUnified,
      noWarmup: options?.noWarmup,
      ctxShift: options?.ctxShift,
      hfRepo: options?.hfRepo,
      hfFile: options?.hfFile,
      fitMode: options?.fitMode,
      cacheRamMb: options?.cacheRamMb ?? null,
      ctxcp: options?.ctxcp ?? null,
      useJinja: options?.useJinja,
      reasoningMode: options?.reasoningMode,
      reasoningPreserve: options?.reasoningPreserve,
      templateMode: options?.templateMode,
      templateName: options?.templateName ?? null,
      customTemplatePath: options?.customTemplatePath ?? null,
      chatTemplateKwargsJson: options?.chatTemplateKwargsJson ?? null,
      draftModelPath: options?.draftModelPath ?? null,
      specType: options?.specType ?? null,
      specDraftNMax: options?.specDraftNMax,
      draftMaxTokens: options?.draftMaxTokens,
      draftMinTokens: options?.draftMinTokens,
      draftPMin: options?.draftPMin,
      attachMmproj: options?.attachMmproj,
      forceReload: options?.forceReload,
      extraArgs: options?.extraArgs,
    },
  });

export const getProcessStatus = () =>
  invoke<ProcessStatusInfo>("get_process_status");

export const killProcess = (pid: number) =>
  invoke<string>("kill_process", { pid });

export const recoverApiPort = (pid: number, port: number, expectedKind: string) =>
  invoke<string>("recover_api_port", { pid, port, expectedKind });

export const checkLlamaServer = () =>
  invoke<ServerInfo>("check_llama_server");

export const updateLlamaServer = () =>
  invoke<string>("update_llama_server");

export const getLlamaInfo = () =>
  invoke<LlamaServerInfo>("get_llama_info");

export const listRuntimePacks = () =>
  invoke<RuntimePackInfo[]>("list_runtime_packs");

export const downloadLlamaBuild = (backend: string) =>
  invoke<string>("download_llama_build", { backend });

// Settings

export const getSettings = () => invoke<AppSettings>("get_settings");

export const getApiAccessInfo = () => invoke<ApiAccessInfo>("get_api_access_info");

export const getConfigFilePath = () => invoke<string>("get_config_file_path");

export const updateSettings = (settings: AppSettings) =>
  invoke<void>("update_settings", { settings });

export const setApiServerRunning = (running: boolean) =>
  invoke<string>("set_api_server_running", { running });

// Chat

export interface SamplingParams {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  presence_penalty?: number;
  repeat_penalty?: number;
  max_tokens?: number;
  seed?: number;
}

export const sendMessage = (
  sessionId: string,
  content: string,
  sampling?: SamplingParams,
  imageBase64?: string | null,
  showThinking?: boolean | null
) =>
  invoke<string>("send_message", {
    sessionId,
    content,
    temperature: sampling?.temperature,
    topP: sampling?.top_p,
    topK: sampling?.top_k,
    minP: sampling?.min_p,
    presencePenalty: sampling?.presence_penalty,
    repeatPenalty: sampling?.repeat_penalty,
    maxTokens: sampling?.max_tokens,
    seed: sampling?.seed,
    imageBase64,
    showThinking,
  });

export const stopGeneration = () => invoke<void>("stop_generation");

// Sessions

export const createSession = (name: string) =>
  invoke<string>("create_session", { name });

export const listSessions = () => invoke<SessionInfo[]>("list_sessions");

export const deleteSession = (sessionId: string) =>
  invoke<void>("delete_session", { sessionId });

export const renameSession = (sessionId: string, name: string) =>
  invoke<void>("rename_session", { sessionId, name });

export const setSessionPinned = (sessionId: string, pinned: boolean) =>
  invoke<void>("set_session_pinned", { sessionId, pinned });

export const getSessionMessages = (sessionId: string) =>
  invoke<MessageInfo[]>("get_session_messages", { sessionId });

// Context

export const getContextStatus = () =>
  invoke<ContextStatus>("get_context_status");

// Debug

export const getRawPrompt = () => invoke<string>("get_raw_prompt");

export const getParseTrace = () => invoke<string>("get_parse_trace");

export const getLaunchPreview = () => invoke<string>("get_launch_preview");

export const getEffectiveProfile = (modelName?: string) =>
  invoke<EffectiveProfileInfo>("get_effective_profile", { modelName });

export const getRuntimeDoctor = () =>
  invoke<RuntimeDoctorReport>("get_runtime_doctor");

export const templateDryRun = (request: {
  modelName?: string | null;
  useJinja: boolean;
  templateMode: string;
  templateName?: string | null;
  customTemplatePath?: string | null;
  chatTemplateKwargsJson?: string | null;
  reasoningMode: string;
  parallelSlots: number;
}) => invoke<TemplateDryRunReport>("template_dry_run", { request });

export const getLogs = (limit?: number) =>
  invoke<LogEntry[]>("get_logs", { limit });

export const clearLogs = () => invoke<void>("clear_logs");

export const debugApiRequest = (request: {
  method: string;
  path: string;
  body?: string | null;
}) => invoke<DebugApiResponse>("debug_api_request", { request });

// GPU Stats

export const getGpuStats = () => invoke<GpuStats>("get_gpu_stats");

// Model Browser

export interface HubQuant {
  quant: string;
  size_bytes?: number | null;
  size_gb: number;
  url: string;
  filename: string;
}

export interface HubModel {
  id: string;
  name: string;
  family: string;
  author: string | null;
  params: string;
  description: string;
  hf_url: string;
  readme?: string | null;
  license?: string | null;
  base_model?: string | null;
  pipeline_tag?: string | null;
  library_name: string | null;
  tags: string[];
  supports_vision: boolean;
  downloads: number;
  likes: number;
  created_at: string | null;
  last_modified: string | null;
  gguf_total: number | null;
  gguf_architecture: string | null;
  gguf_context_length: number | null;
  quants: HubQuant[];
}

export interface DownloadProgress {
  id: string;
  filename: string;
  dest_path: string | null;
  partial_path: string | null;
  supports_vision?: boolean | null;
  repo_id?: string | null;
  downloaded_bytes: number;
  total_bytes: number;
  percent: number;
  speed_bps?: number | null;
  eta_seconds?: number | null;
  resumable: boolean;
  attempt: number;
  done: boolean;
  status: string;
  error: string | null;
}

export interface MetadataSyncSummary {
  scanned_models: number;
  matched_models: number;
  updated_models: number;
}

export const searchHubModels = (query: string, offset: number = 0, sort?: string, tag?: string | null) =>
  invoke<HubModel[]>("search_hub_models", { query, offset, sort, tag: tag ?? null });

export const getHubModelDetails = (repoId: string, includeReadme = false) =>
  invoke<HubModel | null>("get_hub_model_details", { repoId, includeReadme });

export const openExternalUrl = (url: string) =>
  invoke<void>("open_external_url", { url });

export const showInFolder = (path: string) =>
  invoke<void>("show_in_folder", { path });

export const downloadHubModel = (
  url: string,
  filename: string,
  supportsVision?: boolean,
  repoId?: string
) => invoke<string>("download_hub_model", {
  url,
  filename,
  supportsVision,
  repoId,
});

export const listDownloads = () =>
  invoke<DownloadProgress[]>("list_downloads");

export const cancelDownload = (id: string) =>
  invoke<void>("cancel_download", { id });

export const pauseDownload = (id: string) =>
  invoke<void>("pause_download", { id });

export const clearCompletedDownloads = () =>
  invoke<void>("clear_completed_downloads");

export const getHubAccessStatus = () =>
  invoke<HubAccessStatus>("get_hub_access_status");

export const syncLocalModelMetadata = () =>
  invoke<MetadataSyncSummary>("sync_local_model_metadata");

export const getHfSidecarCacheStatus = () =>
  invoke<HfSidecarCacheStatus[]>("get_hf_sidecar_cache_status");

export const checkHfSidecarUpdates = (modelNames?: string[]) =>
  invoke<HfSidecarSyncSummary>("check_hf_sidecar_updates", {
    modelNames: modelNames ?? null,
  });

export const syncHfSidecarCache = (modelNames?: string[]) =>
  invoke<HfSidecarSyncSummary>("sync_hf_sidecar_cache", {
    modelNames: modelNames ?? null,
  });

export const rollbackHfSidecarCache = (modelName: string) =>
  invoke<HfSidecarRollbackSummary>("rollback_hf_sidecar_cache", { modelName });

export const deleteModelFile = (path: string) =>
  invoke<string>("delete_model_file", { path });

// Benchmarks

export interface ModelTestStats {
  model: string;
  context_size: number;
  prompt: string;
  response: string;
  tool_calls: Array<{
    id: string;
    name: string;
    arguments: unknown;
    raw_text: string | null;
  }>;
  tool_remaining_text: string;
  load_ms: number | null;
  load_reused: boolean;
  ttft_ms: number | null;
  elapsed_ms: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  prompt_tokens_per_second: number | null;
  decode_tokens_per_second: number | null;
  end_to_end_tokens_per_second: number | null;
  prefill_ms: number | null;
  decode_ms: number | null;
  sampling: {
    temperature: number | null;
    top_p: number | null;
    top_k: number | null;
    min_p: number | null;
    presence_penalty: number | null;
    repeat_penalty: number | null;
    seed: number | null;
  };
  runtime: {
    spec_type: string;
    spec_draft_n_max: number;
    launch_args: string[];
  };
  agent_steps: Array<{
    turn: number;
    tool: string;
    arguments: unknown;
    ok: boolean;
    result: unknown;
  }>;
  agent_success: boolean | null;
  agent_failure: string | null;
}

export interface BenchmarkRunRequest {
  modelName: string;
  contextSize: number;
  temperature?: number | null;
  topP?: number | null;
  topK?: number | null;
  minP?: number | null;
  presencePenalty?: number | null;
  repeatPenalty?: number | null;
  seed?: number | null;
  specType?: string | null;
  specDraftNMax?: number | null;
}

export const runModelTest = (request: BenchmarkRunRequest & {
  prompt: string;
  maxTokens: number;
}) =>
  invoke<ModelTestStats>("run_model_test", {
    modelName: request.modelName,
    contextSize: request.contextSize,
    prompt: request.prompt,
    maxTokens: request.maxTokens,
    temperature: request.temperature ?? null,
    topP: request.topP ?? null,
    topK: request.topK ?? null,
    minP: request.minP ?? null,
    presencePenalty: request.presencePenalty ?? null,
    repeatPenalty: request.repeatPenalty ?? null,
    seed: request.seed ?? null,
    specType: request.specType ?? null,
    specDraftNMax: request.specDraftNMax ?? null,
  });

export const runAgentToolLoop = (request: BenchmarkRunRequest) =>
  invoke<ModelTestStats>("run_agent_tool_loop", {
    modelName: request.modelName,
    contextSize: request.contextSize,
    temperature: request.temperature ?? null,
    topP: request.topP ?? null,
    topK: request.topK ?? null,
    minP: request.minP ?? null,
    presencePenalty: request.presencePenalty ?? null,
    repeatPenalty: request.repeatPenalty ?? null,
    seed: request.seed ?? null,
    specType: request.specType ?? null,
    specDraftNMax: request.specDraftNMax ?? null,
  });
