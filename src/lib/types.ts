export interface ModelInfo {
  filename: string;
  path: string;
  size_gb: number;
  family: string;
  supports_tools: boolean;
  supports_reasoning: boolean;
  supports_vision: boolean;
  context_window: number | null;
  max_context_window: number | null;
  max_output_tokens: number | null;
  quant: string | null;
  tool_call_format: string;
  think_tag_style: string;
}

export interface LaunchPreview {
  server_path: string;
  model_path: string;
  mmproj_path: string | null;
  backend_preference: string;
  context_size: number;
  port: number;
  parallel_slots: number;
  args: string[];
}

export interface GenerationRequest {
  id: string;
  source: string;
  session_id: string | null;
  model: string;
  started_at: string;
  status: string;
}

export interface RuntimePerformanceMetrics {
  source: string;
  model: string;
  started_at: string;
  finished_at: string;
  elapsed_ms: number;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  prompt_tokens_per_second: number | null;
  decode_tokens_per_second: number | null;
  end_to_end_tokens_per_second: number | null;
}

export interface ProcessStatusInfo {
  state: string;
  model: string | null;
  previous_model: string | null;
  model_load_state: string;
  model_load_progress: LoadProgress | null;
  active_generation: GenerationRequest | null;
  crash_count: number;
  server_version: string | null;
  server_path: string | null;
  backend: string | null;
  api_state: string;
  api_error: string | null;
  api_url: string;
  api_reachable: boolean;
  api_port_owner: ApiPortOwnerInfo | null;
  startup_duration_ms: number | null;
  parallel_slots: number | null;
  slot_count: number | null;
  active_requests: number;
  queued_requests: number;
  scheduler_limit: number | null;
  last_launch_preview: LaunchPreview | null;
  last_generation_metrics: RuntimePerformanceMetrics | null;
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
  stage: string;
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
  image_base64?: string | null;
  token_count: number | null;
  tokens_evaluated?: number | null;
  tokens_predicted?: number | null;
  created_at: string;
}

export interface ContextStatus {
  total_tokens: number;
  used_tokens: number;
  fill_ratio: number;
  pinned_tokens: number;
  rolling_tokens: number;
  compressed_tokens: number;
  last_compaction_action: string | null;
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
  scan_dirs: string[];
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
  api_key: string | null;
}

export interface ApiAccessInfo {
  bind_host: string;
  loopback_url: string;
  lan_host: string | null;
  lan_url: string | null;
}

export interface GpuStats {
  name: string;
  used_mb: number;
  dedicated_mb: number;
  free_mb: number;
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

export interface ModelProfile {
  family: string;
  tool_call_format: string;
  think_tag_style: string;
  interleaved_think_tool: boolean;
  supports_parallel_tools: boolean;
  supports_vision: boolean;
  default_max_output_tokens: number | null;
  default_context_window: number | null;
  max_context_window: number | null;
  parser_type: string;
  renderer_type: string;
  stop_markers: string[];
  allow_fallback_extraction: boolean;
  default_presence_penalty: number | null;
  default_temperature: number | null;
  default_top_p: number | null;
  default_top_k: number | null;
  default_min_p: number | null;
  disable_thinking_for_tools: boolean;
  split_tool_calling: boolean;
}

export interface ModelProfileOverride {
  supports_vision?: boolean;
  tool_call_format?: string;
  think_tag_style?: string;
  interleaved_think_tool?: boolean;
  supports_parallel_tools?: boolean;
  default_max_output_tokens?: number | null;
  default_context_window?: number | null;
  max_context_window?: number | null;
  parser_type?: string;
  renderer_type?: string;
  stop_markers?: string[];
  allow_fallback_extraction?: boolean;
  default_presence_penalty?: number | null;
  default_temperature?: number | null;
  default_top_p?: number | null;
  default_top_k?: number | null;
  default_min_p?: number | null;
  disable_thinking_for_tools?: boolean;
  split_tool_calling?: boolean;
}

export interface EffectiveProfileInfo {
  requested_model: string | null;
  resolved_model: string | null;
  profile: ModelProfile;
  override_entry: ModelProfileOverride | null;
}
