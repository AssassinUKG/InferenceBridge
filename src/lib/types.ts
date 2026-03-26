// Shared TypeScript types that mirror Rust serde structs.

export interface ModelInfo {
  filename: string;
  path: string;
  size_gb: number;
  family: string;
  supports_tools: boolean;
  supports_reasoning: boolean;
  supports_vision: boolean;
  context_window: number | null;
  /** True training context ceiling — used as slider maximum. */
  max_context_window: number | null;
  max_output_tokens: number | null;
  quant: string | null;
  tool_call_format: string;
  think_tag_style: string;
}

export interface ProcessStatusInfo {
  state: string;
  model: string | null;
  previous_model: string | null;
  crash_count: number;
  server_version: string | null;
  server_path: string | null;
  backend: string | null;
  api_state: string;
  api_error: string | null;
  api_url: string;
  api_port_owner: ApiPortOwnerInfo | null;
}

export interface ApiPortOwnerInfo {
  pid: number;
  name: string | null;
  kind: string;
  killable: boolean;
}

export interface ServerInfo {
  found: boolean;
  path: string | null;
  version: string | null;
}

export interface LoadProgress {
  stage: string; // resolving | downloading | launching | starting | loading | ready | error
  message: string;
  progress: number;
  done: boolean;
  error: string | null;
}

export interface SessionInfo {
  id: string;
  name: string | null;
  model_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface MessageInfo {
  id: number;
  role: string;
  content: string | null;
  token_count: number | null;
  created_at: string;
}

export interface ContextStatus {
  total_tokens: number;
  used_tokens: number;
  fill_ratio: number;
}

export interface AppSettings {
  api_autostart: boolean;
  kill_on_exit: boolean;
  gpu_layers: number;
  threads: number;
  threads_batch: number;
  theme: string;
  backend_preference: string;
  server_host: string;
  server_port: number;
  /** Directories scanned for .gguf model files. */
  scan_dirs: string[];
  // llama.cpp inference settings
  batch_size: number;
  ubatch_size: number;
  flash_attn: boolean;
  use_mmap: boolean;
  use_mlock: boolean;
  cont_batching: boolean;
  parallel_slots: number;
  main_gpu: number;
  defrag_thold: number;
  rope_freq_scale: number;
  /** Bearer token required by the public API. Null / empty = no auth. */
  api_key: string | null;
}

export interface GpuStats {
  name: string;
  used_mb: number;
  /** Dedicated on-board VRAM (fast, from nvidia-smi) */
  dedicated_mb: number;
  free_mb: number;
  /** Total system RAM — shown as the overflow/spill zone */
  system_ram_mb: number;
}

export interface LlamaServerInfo {
  version: string | null;
  binary_path: string | null;
  has_managed_binary: boolean;
  managed_dir: string;
  latest_version: string | null;
  update_available: boolean;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  target: string;
  message: string;
}

export interface DebugApiResponse {
  status: number;
  headers: [string, string][];
  body: string;
  transport: string;
}
