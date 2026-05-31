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
  default_temperature: number | null;
  default_top_p: number | null;
  default_top_k: number | null;
  default_min_p: number | null;
  default_presence_penalty: number | null;
  quant: string | null;
  tool_call_format: string;
  think_tag_style: string;
  hf_repo: string | null;
  hf_file: string | null;
  template_mode: string | null;
  template_source: string | null;
  vision_runtime_ready: boolean;
  vision_status: string;
  provider_type: string;
  provider_name: string;
  provider_base_url: string | null;
  provider_managed: boolean;
  // GGUF architecture metadata (null for external providers)
  n_layers: number | null;
  n_kv_heads: number | null;
  head_dim: number | null;
  gguf_architecture: string | null;
}

export interface LaunchPreview {
  server_path: string;
  model_path: string;
  hf_repo: string | null;
  hf_file: string | null;
  mmproj_path: string | null;
  backend_preference: string;
  context_size: number | null;
  port: number;
  parallel_slots: number;
  fit_mode: string | null;
  cache_ram_mb: number | null;
  ctxcp: number | null;
  use_jinja: boolean;
  reasoning_mode: string | null;
  template_mode: string;
  template_source: string | null;
  template_path: string | null;
  template_name: string | null;
  chat_template_kwargs_json: string | null;
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
  request_id: string;
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

export interface LiveStreamEvent {
  timestamp: string;
  kind: string;
  text: string;
}

export interface LiveStreamSnapshot {
  request_id: string;
  source: string;
  model: string;
  started_at: string;
  status: string;
  raw_output: string;
  visible_output: string;
  reasoning_output: string;
  events: LiveStreamEvent[];
}

export interface LiveStreamDelta extends LiveStreamEvent {
  request_id: string;
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
  live_stream: LiveStreamSnapshot | null;
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
  fit_mode: string;
  cache_ram_mb: number | null;
  ctxcp: number | null;
  use_jinja: boolean;
  reasoning_mode: string;
  template_mode: string;
  template_name: string | null;
  custom_template_path: string | null;
  chat_template_kwargs_json: string | null;
  extra_args: string[];
  api_key: string | null;
  active_provider: string;
  lm_studio_enabled: boolean;
  lm_studio_base_url: string;
  lm_studio_api_key: string | null;
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

export interface RuntimeDoctorReport {
  checked_at: string;
  app_api: AppApiDoctor;
  active_runtime: ActiveRuntimeDoctor;
  providers: ProviderProbe[];
  summary: RuntimeDoctorSummary;
}

export interface RuntimeDoctorSummary {
  reachable_providers: number;
  total_providers: number;
  loaded_model: string | null;
  preferred_next_step: string;
}

export interface AppApiDoctor {
  state: string;
  url: string;
  reachable: boolean;
  error: string | null;
}

export interface ActiveRuntimeDoctor {
  managed: boolean;
  state: string;
  model: string | null;
  port: number | null;
  backend: string | null;
  launch_context_size: number | null;
}

export type ProviderType =
  | "managed_llama_cpp"
  | "external_llama_cpp"
  | "lm_studio"
  | "ollama"
  | "open_ai_compatible";

export interface ProviderProbe {
  id: string;
  provider_type: ProviderType;
  name: string;
  base_url: string;
  managed: boolean;
  reachable: boolean;
  status: string;
  models: ProviderModelInfo[];
  model_count: number;
  context_limit: number | null;
  output_limit: number | null;
  endpoints: ProviderEndpointSupport;
  build_info: string | null;
  error: string | null;
  hints: string[];
}

export interface ProviderEndpointSupport {
  health: boolean;
  props: boolean;
  slots: boolean;
  openai_models: boolean;
  ollama_tags: boolean;
}

export interface ProviderModelInfo {
  id: string;
  name: string | null;
  context_limit: number | null;
  output_limit: number | null;
}
