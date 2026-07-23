export interface ModelInfo {
  filename: string;
  path: string;
  size_gb: number;
  family: string;
  supports_tools: boolean;
  supports_parallel_tools: boolean;
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
  mmproj_available: boolean;
  mmproj_candidate_path: string | null;
  provider_type: string;
  provider_name: string;
  provider_base_url: string | null;
  provider_managed: boolean;
  // GGUF architecture metadata (null for external providers)
  n_layers: number | null;
  n_kv_heads: number | null;
  head_dim: number | null;
  gguf_architecture: string | null;
  // Whether the GGUF ships its own embedded chat template (loaded with --jinja).
  has_chat_template: boolean;
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
  reasoning_preserve: boolean;
  template_mode: string;
  template_source: string | null;
  template_path: string | null;
  template_name: string | null;
  chat_template_kwargs_json: string | null;
  sampling_defaults?: {
    temperature: number | null;
    top_p: number | null;
    top_k: number | null;
    min_p: number | null;
    presence_penalty: number | null;
    repeat_penalty: number | null;
  };
  draft_model_path: string;
  spec_type: string;
  spec_draft_n_max: number;
  draft_max_tokens: number;
  draft_min_tokens: number;
  draft_p_min: number;
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
  live_streams: LiveStreamSnapshot[];
}

export type ApiServerAction = "starting" | "stopping" | null;

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

export interface ImageGenerationRequest {
  prompt: string;
  session_id?: string | null;
  bundle_id?: string | null;
  profile_id?: string | null;
  seed?: number | null;
  width?: number | null;
  height?: number | null;
  steps?: number | null;
  cfg_scale?: number | null;
  sampling_method?: string | null;
  negative_prompt?: string | null;
}

export interface ImageGenerationProgress {
  job_id: string;
  status: string;
  stage: "loading" | "generating" | "saving" | "completed" | "cancelled" | "failed" | string;
  message: string;
  bundle_id: string;
  profile_id: string;
  current_step: number;
  total_steps: number;
  progress: number;
  elapsed_seconds: number;
  eta_seconds: number | null;
  started_at: string;
  updated_at: string;
  done: boolean;
  error: string | null;
  output_path: string | null;
}

export interface ImageBundleStatus {
  id: string;
  name: string;
  architecture: string;
  quantization: string;
  ready: boolean;
  reasons: string[];
}

export interface ImageProfileStatus {
  id: string;
  name: string;
  width: number;
  height: number;
  steps: number;
  ready: boolean;
  reason: string | null;
}

export interface ImageSizePreset {
  id: string;
  name: string;
  aspect_ratio: string;
  width: number;
  height: number;
  tier: string;
  note: string;
}

export interface ImageGenerationCapabilityStatus {
  enabled: boolean;
  ready: boolean;
  automatic_model_swap_enabled: boolean;
  runner_path: string | null;
  output_dir: string;
  default_bundle: string;
  default_profile: string;
  warn_temperature_c: number;
  cooldown_temperature_c: number;
  reasons: string[];
  bundles: ImageBundleStatus[];
  profiles: ImageProfileStatus[];
  size_presets: ImageSizePreset[];
  active_job: ImageGenerationProgress | null;
}

export interface DetectedImageLabSetup {
  runner_path: string;
  transformer_path: string;
  text_encoder_path: string;
  vae_path: string;
  output_dir: string;
}

export interface ImageGenerationPreview {
  bundle_id: string;
  bundle_name: string;
  profile_id: string;
  profile_name: string;
  width: number;
  height: number;
  steps: number;
  seed: number;
  output_path: string;
  arguments: string[];
}

export interface ImageGenerationResult {
  job_id: string;
  status: string;
  bundle_id: string;
  bundle_name: string;
  quantization: string;
  profile_id: string;
  prompt: string;
  negative_prompt: string | null;
  seed: number;
  width: number;
  height: number;
  steps: number;
  cfg_scale: number;
  sampling_method: string;
  elapsed_seconds: number;
  file_size_bytes: number | null;
  output_path: string | null;
  error: string | null;
}

export interface SessionInfo {
  id: string;
  name: string | null;
  model_id: string | null;
  pinned: boolean;
  created_at: string;
  updated_at: string;
}

export interface MessageInfo {
  id: number;
  role: string;
  content: string | null;
  display_content?: string | null;
  reasoning_content?: string | null;
  image_base64?: string | null;
  image_path?: string | null;
  image_metadata?: string | null;
  token_count: number | null;
  tokens_evaluated?: number | null;
  tokens_predicted?: number | null;
  created_at: string;
  tool_calls?: ToolCallInfo[];
}

export interface ToolCallInfo {
  id: number;
  call_id: string | null;
  name: string;
  arguments: string | null;
  result: string | null;
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
  reasoning_preserve: boolean;
  template_mode: string;
  template_name: string | null;
  custom_template_path: string | null;
  chat_template_kwargs_json: string | null;
  draft_model_path: string;
  spec_type: string;
  spec_draft_n_max: number;
  draft_max_tokens: number;
  draft_min_tokens: number;
  draft_p_min: number;
  extra_args: string[];
  llama_diffusion_cli_path: string;
  diffusion_n_predict: number;
  diffusion_kv_cache: string;
  diffusion_visual: boolean;
  diffusion_extra_args: string[];
  image_generation_enabled: boolean;
  image_runner_path: string;
  image_output_dir: string;
  image_default_profile: string;
  image_transformer_path: string;
  image_text_encoder_path: string;
  image_vae_path: string;
  image_warn_temperature_c: number;
  image_cooldown_temperature_c: number;
  api_key: string | null;
  active_provider: string;
  lm_studio_enabled: boolean;
  lm_studio_base_url: string;
  lm_studio_api_key: string | null;
  sglang_enabled: boolean;
  sglang_base_url: string;
  sglang_api_key: string | null;
  openai_enabled: boolean;
  openai_base_url: string;
  openai_api_key: string | null;
  hf_api_key: string | null;
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

export interface LlamaFlagSupport {
  checked: boolean;
  binary_path: string | null;
  supported_flags: string[];
  missing_critical_flags: string[];
  error: string | null;
}

export interface LlamaServerInfo {
  version: string | null;
  binary_path: string | null;
  has_managed_binary: boolean;
  managed_dir: string;
  latest_version: string | null;
  update_available: boolean;
  flag_support: LlamaFlagSupport;
}

export interface RuntimePackInfo {
  id: string;
  name: string;
  description: string;
  backend: string;
  installed_version: string | null;
  latest_version: string | null;
  update_available: boolean;
  size_bytes: number | null;
  available: boolean;
  error: string | null;
}

export interface HubAccessStatus {
  configured: boolean;
  reachable: boolean;
  user: string | null;
  error: string | null;
}

export interface HfSidecarSyncFile {
  repo_id: string;
  path: string;
  cached_path: string | null;
  status: string;
  message: string | null;
}

export interface HfSidecarSyncSummary {
  mode: "check" | "apply";
  models_checked: number;
  repos_checked: number;
  repos_with_updates: number;
  repos_updated: number;
  files_cached: number;
  files_updated: number;
  files_unchanged: number;
  files_skipped: number;
  files_failed: number;
  hf_token_configured: boolean;
  cache_root: string;
  results: HfSidecarSyncFile[];
}

export interface HfSidecarCacheStatus {
  filename: string;
  repo_id: string | null;
  source_repo_id: string | null;
  template_path: string | null;
  template_cached: boolean;
  template_cache_path: string | null;
  template_source: string | null;
  sidecar_cached_count: number;
  sidecar_expected_count: number;
  sidecar_cache_dir: string | null;
  active_revision: string | null;
  remote_revision: string | null;
  update_available: boolean;
  last_checked_at: string | null;
  rollback_available: boolean;
}

export interface HfSidecarRollbackSummary {
  repo_id: string;
  restored_revision: string | null;
  replaced_revision: string | null;
  files_restored: number;
  template_restored: boolean;
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

export interface TemplateDryRunReport {
  model_name: string;
  family: string;
  renderer: string;
  tool_format: string;
  prompt: string;
  checks: string[];
  warnings: string[];
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
  | "sg_lang"
  | "open_ai"
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
