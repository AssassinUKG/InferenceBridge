// Typed wrappers around Tauri invoke calls.

import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import type {
  AppSettings,
  ApiAccessInfo,
  ContextStatus,
  DebugApiResponse,
  EffectiveProfileInfo,
  GpuStats,
  LlamaServerInfo,
  LogEntry,
  MessageInfo,
  ModelInfo,
  ProcessStatusInfo,
  RuntimeDoctorReport,
  ServerInfo,
  SessionInfo,
} from "./types";

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

// Models

export const listModels = () => invoke<ModelInfo[]>("list_models");

export const scanModels = () => invoke<number>("scan_models");

export const setModelVisionOverride = (modelName: string, supportsVision: boolean) =>
  invoke<void>("set_model_vision_override", { modelName, supportsVision });

export interface LoadModelOptions {
  contextSize?: number;
  hfRepo?: string;
  hfFile?: string;
  fitMode?: string;
  cacheRamMb?: number | null;
  ctxcp?: number | null;
  useJinja?: boolean;
  reasoningMode?: string;
  templateMode?: string;
  templateName?: string | null;
  customTemplatePath?: string | null;
  chatTemplateKwargsJson?: string | null;
  extraArgs?: string[];
}

export const loadModel = (modelName: string, options?: LoadModelOptions) =>
  invoke<string>("load_model", {
    modelName,
    options: {
      contextSize: options?.contextSize,
      hfRepo: options?.hfRepo,
      hfFile: options?.hfFile,
      fitMode: options?.fitMode,
      cacheRamMb: options?.cacheRamMb ?? null,
      ctxcp: options?.ctxcp ?? null,
      useJinja: options?.useJinja,
      reasoningMode: options?.reasoningMode,
      templateMode: options?.templateMode,
      templateName: options?.templateName ?? null,
      customTemplatePath: options?.customTemplatePath ?? null,
      chatTemplateKwargsJson: options?.chatTemplateKwargsJson ?? null,
      extraArgs: options?.extraArgs,
    },
  });

export const unloadModel = () => invoke<string>("unload_model");

export const swapModel = (modelName?: string, options?: LoadModelOptions) =>
  invoke<string>("swap_model", {
    modelName,
    options: {
      contextSize: options?.contextSize,
      hfRepo: options?.hfRepo,
      hfFile: options?.hfFile,
      fitMode: options?.fitMode,
      cacheRamMb: options?.cacheRamMb ?? null,
      ctxcp: options?.ctxcp ?? null,
      useJinja: options?.useJinja,
      reasoningMode: options?.reasoningMode,
      templateMode: options?.templateMode,
      templateName: options?.templateName ?? null,
      customTemplatePath: options?.customTemplatePath ?? null,
      chatTemplateKwargsJson: options?.chatTemplateKwargsJson ?? null,
      extraArgs: options?.extraArgs,
    },
  });

export const getProcessStatus = () =>
  invoke<ProcessStatusInfo>("get_process_status");

export const killProcess = (pid: number) =>
  invoke<string>("kill_process", { pid });

export const checkLlamaServer = () =>
  invoke<ServerInfo>("check_llama_server");

export const updateLlamaServer = () =>
  invoke<string>("update_llama_server");

export const getLlamaInfo = () =>
  invoke<LlamaServerInfo>("get_llama_info");

export const downloadLlamaBuild = (backend: string) =>
  invoke<string>("download_llama_build", { backend });

// Settings

export const getSettings = () => invoke<AppSettings>("get_settings");

export const getApiAccessInfo = () => invoke<ApiAccessInfo>("get_api_access_info");

export const updateSettings = (settings: AppSettings) =>
  invoke<void>("update_settings", { settings });

export const setApiServerRunning = (running: boolean) =>
  invoke<string>("set_api_server_running", { running });

// Chat

export interface SamplingParams {
  temperature?: number;
  top_p?: number;
  top_k?: number;
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
  size_gb: number;
  url: string;
  filename: string;
}

export interface HubModel {
  id: string;
  name: string;
  family: string;
  params: string;
  description: string;
  tags: string[];
  supports_vision: boolean;
  quants: HubQuant[];
}

export interface DownloadProgress {
  id: string;
  filename: string;
  dest_path: string | null;
  downloaded_bytes: number;
  total_bytes: number;
  percent: number;
  done: boolean;
  status: string;
  error: string | null;
}

export interface MetadataSyncSummary {
  scanned_models: number;
  matched_models: number;
  updated_models: number;
}

export const searchHubModels = (query: string, offset: number = 0) =>
  invoke<HubModel[]>("search_hub_models", { query, offset });

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

export const clearCompletedDownloads = () =>
  invoke<void>("clear_completed_downloads");

export const syncLocalModelMetadata = () =>
  invoke<MetadataSyncSummary>("sync_local_model_metadata");

export const deleteModelFile = (path: string) =>
  invoke<void>("delete_model_file", { path });
