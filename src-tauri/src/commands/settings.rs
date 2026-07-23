use crate::engine::download;
use crate::engine::process::{detect_llama_flag_support, LlamaFlagSupport, LlamaProcess};
use crate::engine::scheduler::RequestScheduler;
use crate::state::SharedState;
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, UdpSocket};
use tauri::Emitter;

/// Settings exposed to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub api_autostart: bool,
    pub kill_on_exit: bool,
    pub gpu_layers: i32,
    pub threads: u32,
    pub threads_batch: u32,
    pub theme: String,
    pub backend_preference: String,
    pub server_host: String,
    pub server_port: u16,
    /// Directories scanned for .gguf model files (absolute paths as strings).
    pub scan_dirs: Vec<String>,
    // llama.cpp inference settings
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub flash_attn: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub cont_batching: bool,
    pub parallel_slots: u32,
    pub main_gpu: i32,
    pub defrag_thold: f32,
    pub rope_freq_scale: f32,
    pub fit_mode: String,
    pub cache_ram_mb: Option<u32>,
    pub ctxcp: Option<u32>,
    pub use_jinja: bool,
    pub reasoning_mode: String,
    pub reasoning_preserve: bool,
    pub template_mode: String,
    pub template_name: Option<String>,
    pub custom_template_path: Option<String>,
    pub chat_template_kwargs_json: Option<String>,
    pub draft_model_path: String,
    pub spec_type: String,
    pub spec_draft_n_max: u32,
    pub draft_max_tokens: u32,
    pub draft_min_tokens: u32,
    pub draft_p_min: f32,
    pub extra_args: Vec<String>,
    pub llama_diffusion_cli_path: String,
    pub diffusion_n_predict: u32,
    pub diffusion_kv_cache: String,
    pub diffusion_visual: bool,
    pub diffusion_extra_args: Vec<String>,
    /// API key for Bearer token auth. None / empty string = no auth required.
    pub api_key: Option<String>,
    pub active_provider: String,
    pub lm_studio_enabled: bool,
    pub lm_studio_base_url: String,
    pub lm_studio_api_key: Option<String>,
    pub sglang_enabled: bool,
    pub sglang_base_url: String,
    pub sglang_api_key: Option<String>,
    pub openai_enabled: bool,
    pub openai_base_url: String,
    pub openai_api_key: Option<String>,
    pub hf_api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiAccessInfo {
    pub bind_host: String,
    pub loopback_url: String,
    pub lan_host: Option<String>,
    pub lan_url: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimePackInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub backend: String,
    pub installed_version: Option<String>,
    pub latest_version: Option<String>,
    pub update_available: bool,
    pub size_bytes: Option<u64>,
    pub available: bool,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn get_settings(state: tauri::State<'_, SharedState>) -> Result<AppSettings, String> {
    let s = state.read().await;
    Ok(AppSettings {
        api_autostart: s.config.server.autostart,
        kill_on_exit: s.config.process.kill_on_exit,
        gpu_layers: s.config.process.gpu_layers,
        threads: s.config.process.threads,
        threads_batch: s.config.process.threads_batch,
        theme: s.config.ui.theme.clone(),
        backend_preference: s.config.process.backend_preference.clone(),
        server_host: s.config.server.host.clone(),
        server_port: s.config.server.port,
        scan_dirs: s
            .config
            .models
            .scan_dirs
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect(),
        batch_size: s.config.process.batch_size,
        ubatch_size: s.config.process.ubatch_size,
        flash_attn: s.config.process.flash_attn,
        use_mmap: s.config.process.use_mmap,
        use_mlock: s.config.process.use_mlock,
        cont_batching: s.config.process.cont_batching,
        parallel_slots: s.config.process.parallel_slots,
        main_gpu: s.config.process.main_gpu,
        defrag_thold: s.config.process.defrag_thold,
        rope_freq_scale: s.config.process.rope_freq_scale,
        fit_mode: s.config.process.fit_mode.clone(),
        cache_ram_mb: s.config.process.cache_ram_mb,
        ctxcp: s.config.process.ctxcp,
        use_jinja: s.config.process.use_jinja,
        reasoning_mode: s.config.process.reasoning_mode.clone(),
        reasoning_preserve: s.config.process.reasoning_preserve,
        template_mode: s.config.process.template_mode.clone(),
        template_name: s.config.process.template_name.clone(),
        custom_template_path: s.config.process.custom_template_path.clone(),
        chat_template_kwargs_json: s.config.process.chat_template_kwargs_json.clone(),
        draft_model_path: s.config.process.draft_model_path.clone(),
        spec_type: s.config.process.spec_type.clone(),
        spec_draft_n_max: s.config.process.spec_draft_n_max,
        draft_max_tokens: s.config.process.draft_max_tokens,
        draft_min_tokens: s.config.process.draft_min_tokens,
        draft_p_min: s.config.process.draft_p_min,
        extra_args: s.config.process.extra_args.clone(),
        llama_diffusion_cli_path: s.config.process.llama_diffusion_cli_path.clone(),
        diffusion_n_predict: s.config.process.diffusion_n_predict,
        diffusion_kv_cache: s.config.process.diffusion_kv_cache.clone(),
        diffusion_visual: s.config.process.diffusion_visual,
        diffusion_extra_args: s.config.process.diffusion_extra_args.clone(),
        api_key: s.config.server.api_key.clone(),
        active_provider: s.config.providers.active.clone(),
        lm_studio_enabled: s.config.providers.lm_studio.enabled,
        lm_studio_base_url: s.config.providers.lm_studio.base_url.clone(),
        lm_studio_api_key: s.config.providers.lm_studio.api_key.clone(),
        sglang_enabled: s.config.providers.sglang.enabled,
        sglang_base_url: s.config.providers.sglang.base_url.clone(),
        sglang_api_key: s.config.providers.sglang.api_key.clone(),
        openai_enabled: s.config.providers.openai.enabled,
        openai_base_url: s.config.providers.openai.base_url.clone(),
        openai_api_key: s.config.providers.openai.api_key.clone(),
        hf_api_key: s.config.hub.hf_api_key.clone(),
    })
}

#[tauri::command]
pub async fn get_api_access_info(
    state: tauri::State<'_, SharedState>,
) -> Result<ApiAccessInfo, String> {
    let s = state.read().await;
    let bind_host = s.config.server.host.clone();
    let port = s.config.server.port;
    let loopback_url = format!("http://127.0.0.1:{port}/v1");
    let lan_host = detect_primary_lan_ipv4();
    let lan_url = lan_host
        .as_ref()
        .map(|host| format!("http://{host}:{port}/v1"));

    Ok(ApiAccessInfo {
        bind_host,
        loopback_url,
        lan_host,
        lan_url,
    })
}

#[tauri::command]
pub async fn get_config_file_path() -> Result<String, String> {
    Ok(crate::config::AppConfig::config_file_path()
        .to_string_lossy()
        .to_string())
}

#[tauri::command]
pub async fn update_settings(
    state: tauri::State<'_, SharedState>,
    settings: AppSettings,
) -> Result<(), String> {
    let shared = state.inner().clone();
    let host = settings.server_host.clone();
    let normalized_api_key = settings.api_key.clone().filter(|k| !k.trim().is_empty());
    let active_provider = match settings.active_provider.trim() {
        "lm_studio" if settings.lm_studio_enabled => "lm_studio".to_string(),
        "sglang" if settings.sglang_enabled => "sglang".to_string(),
        "openai" if settings.openai_enabled => "openai".to_string(),
        _ => "managed_llamacpp".to_string(),
    };
    let lm_studio_base_url =
        crate::providers::normalize_openai_base_url(&settings.lm_studio_base_url);
    let lm_studio_api_key = settings
        .lm_studio_api_key
        .clone()
        .filter(|key| !key.trim().is_empty());
    let sglang_base_url = crate::providers::normalize_openai_base_url(&settings.sglang_base_url);
    let sglang_api_key = settings
        .sglang_api_key
        .clone()
        .filter(|key| !key.trim().is_empty());
    let openai_base_url = crate::providers::normalize_openai_base_url(&settings.openai_base_url);
    let openai_api_key = settings
        .openai_api_key
        .clone()
        .filter(|key| !key.trim().is_empty());
    let hf_api_key = settings
        .hf_api_key
        .clone()
        .filter(|key| !key.trim().is_empty());
    let (
        should_restart_api,
        should_start_api,
        previous_host,
        previous_port,
        previous_api_key,
        previous_lm_studio_api_key,
        previous_sglang_api_key,
        previous_openai_api_key,
        previous_hf_api_key,
    ) = {
        let mut s = shared.write().await;
        let previous_host = s.config.server.host.clone();
        let previous_port = s.config.server.port;
        let previous_api_key = s.config.server.api_key.clone();
        let previous_lm_studio_api_key = s.config.providers.lm_studio.api_key.clone();
        let previous_sglang_api_key = s.config.providers.sglang.api_key.clone();
        let previous_openai_api_key = s.config.providers.openai.api_key.clone();
        let previous_hf_api_key = s.config.hub.hf_api_key.clone();
        let previous_autostart = s.config.server.autostart;
        let previous_api_state = s.api_server_state.clone();

        s.config.server.autostart = settings.api_autostart;
        s.config.process.kill_on_exit = settings.kill_on_exit;
        s.config.process.gpu_layers = settings.gpu_layers;
        s.config.process.threads = settings.threads;
        s.config.process.threads_batch = settings.threads_batch;
        s.config.ui.theme = settings.theme;
        s.config.process.backend_preference = settings.backend_preference;
        s.config.server.host = host.clone();
        s.config.server.port = settings.server_port;
        // Update model scan directories (deduplicate, filter non-empty paths).
        s.config.models.scan_dirs = settings
            .scan_dirs
            .iter()
            .filter(|p| !p.trim().is_empty())
            .map(std::path::PathBuf::from)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        // Force a re-scan so the new dirs are reflected immediately.
        let rescan_dirs = s.config.models.scan_dirs.clone();
        let scanned = crate::models::scanner::scan_all(&rescan_dirs);
        let count = scanned.len();
        s.model_registry.update(scanned);
        tracing::info!(count, "Re-scanned models after scan_dirs update");
        s.config.process.batch_size = settings.batch_size;
        s.config.process.ubatch_size = settings.ubatch_size;
        s.config.process.flash_attn = settings.flash_attn;
        s.config.process.use_mmap = settings.use_mmap;
        s.config.process.use_mlock = settings.use_mlock;
        s.config.process.cont_batching = settings.cont_batching;
        s.config.process.parallel_slots = settings.parallel_slots;
        s.request_scheduler = std::sync::Arc::new(RequestScheduler::new(settings.parallel_slots));
        s.config.process.main_gpu = settings.main_gpu;
        s.config.process.defrag_thold = settings.defrag_thold;
        s.config.process.rope_freq_scale = settings.rope_freq_scale;
        s.config.process.fit_mode = settings.fit_mode.trim().to_string();
        s.config.process.cache_ram_mb = settings.cache_ram_mb.filter(|value| *value > 0);
        s.config.process.ctxcp = settings.ctxcp.filter(|value| *value > 0);
        s.config.process.use_jinja = settings.use_jinja;
        s.config.process.reasoning_mode = settings.reasoning_mode.trim().to_string();
        s.config.process.reasoning_preserve = settings.reasoning_preserve;
        s.config.process.template_mode = settings.template_mode.trim().to_lowercase();
        s.config.process.template_name = settings
            .template_name
            .clone()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        s.config.process.custom_template_path = settings
            .custom_template_path
            .clone()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        s.config.process.chat_template_kwargs_json = settings
            .chat_template_kwargs_json
            .clone()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        s.config.process.draft_model_path = settings.draft_model_path.trim().to_string();
        s.config.process.spec_type = settings.spec_type.trim().to_string();
        s.config.process.spec_draft_n_max = settings.spec_draft_n_max;
        s.config.process.draft_max_tokens = settings.draft_max_tokens;
        s.config.process.draft_min_tokens = settings.draft_min_tokens;
        s.config.process.draft_p_min = settings.draft_p_min.max(0.0);
        s.config.process.extra_args = settings
            .extra_args
            .iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect();
        s.config.process.llama_diffusion_cli_path =
            settings.llama_diffusion_cli_path.trim().to_string();
        let diffusion_cli_path = s.config.process.llama_diffusion_cli_path.clone();
        if !diffusion_cli_path.is_empty() {
            s.process.set_diffusion_cli_path(diffusion_cli_path.into());
        }
        s.config.process.diffusion_n_predict = settings.diffusion_n_predict.max(1);
        let kv_cache = settings.diffusion_kv_cache.trim().to_ascii_lowercase();
        s.config.process.diffusion_kv_cache = match kv_cache.as_str() {
            "on" | "off" => kv_cache,
            _ => "auto".to_string(),
        };
        s.config.process.diffusion_visual = settings.diffusion_visual;
        s.config.process.diffusion_extra_args = settings
            .diffusion_extra_args
            .iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect();
        s.config.server.api_key = normalized_api_key.clone();
        s.config.providers.active = active_provider.clone();
        s.config.providers.lm_studio.enabled = settings.lm_studio_enabled;
        s.config.providers.lm_studio.base_url = lm_studio_base_url.clone();
        s.config.providers.lm_studio.api_key = lm_studio_api_key.clone();
        s.config.providers.sglang.enabled = settings.sglang_enabled;
        s.config.providers.sglang.base_url = sglang_base_url.clone();
        s.config.providers.sglang.api_key = sglang_api_key.clone();
        s.config.providers.openai.enabled = settings.openai_enabled;
        s.config.providers.openai.base_url = openai_base_url.clone();
        s.config.providers.openai.api_key = openai_api_key.clone();
        s.config.hub.hf_api_key = hf_api_key.clone();

        // Persist to disk.
        s.config
            .save()
            .map_err(|e| format!("Failed to save settings: {e}"))?;

        let api_settings_changed = previous_host != host
            || previous_port != settings.server_port
            || previous_api_key != normalized_api_key;
        let api_was_active = !matches!(previous_api_state, crate::state::ApiServerState::Idle);
        let should_restart_api = api_settings_changed && (api_was_active || previous_autostart);
        let should_start_api = settings.api_autostart || api_was_active;

        (
            should_restart_api,
            should_start_api,
            previous_host,
            previous_port,
            previous_api_key,
            previous_lm_studio_api_key,
            previous_sglang_api_key,
            previous_openai_api_key,
            previous_hf_api_key,
        )
    };

    if should_restart_api {
        tracing::info!(
            previous_host = %previous_host,
            previous_port,
            host = %host,
            port = settings.server_port,
            "API settings changed; restarting managed API server"
        );
        let _ = crate::api::runtime::stop_managed(shared.clone()).await;
        if should_start_api {
            crate::api::runtime::start_managed(
                shared.clone(),
                host.clone(),
                settings.server_port,
                "settings-save",
            );
        }
    }

    tracing::info!(
        kill_on_exit = settings.kill_on_exit,
        api_autostart = settings.api_autostart,
        host = %host,
        port = settings.server_port,
        previous_host = %previous_host,
        previous_port,
        api_key_changed = previous_api_key != normalized_api_key,
        lm_studio_api_key_changed = previous_lm_studio_api_key != lm_studio_api_key,
        sglang_api_key_changed = previous_sglang_api_key != sglang_api_key,
        openai_api_key_changed = previous_openai_api_key != openai_api_key,
        hf_api_key_changed = previous_hf_api_key != hf_api_key,
        "Settings updated"
    );
    Ok(())
}

#[tauri::command]
pub async fn set_api_server_running(
    state: tauri::State<'_, SharedState>,
    running: bool,
) -> Result<String, String> {
    let shared = state.inner().clone();
    let (host, port) = {
        let s = shared.read().await;
        (s.config.server.host.clone(), s.config.server.port)
    };

    if running {
        let started = crate::api::runtime::start_managed(shared, host.clone(), port, "gui-toggle");
        if started {
            Ok(format!("Starting API server on http://{host}:{port}/v1"))
        } else {
            Ok(format!(
                "API server already running on http://{host}:{port}/v1"
            ))
        }
    } else {
        let stopped = crate::api::runtime::stop_managed(shared).await;
        if stopped {
            Ok("API server stopped".to_string())
        } else {
            Ok("API server already stopped".to_string())
        }
    }
}

fn detect_primary_lan_ipv4() -> Option<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    match socket.local_addr().ok()?.ip() {
        IpAddr::V4(ip) if !ip.is_loopback() => Some(ip.to_string()),
        _ => None,
    }
}

/// Info about the current llama-server installation.
#[derive(Debug, Clone, Serialize)]
pub struct LlamaServerInfo {
    /// Currently installed version tag (e.g. "b8502").
    pub version: Option<String>,
    /// Path to the binary that would be used.
    pub binary_path: Option<String>,
    /// Whether our managed CUDA binary exists.
    pub has_managed_binary: bool,
    /// Directory where managed binaries are stored.
    pub managed_dir: String,
    /// Latest version available on GitHub (None if check failed / not checked).
    pub latest_version: Option<String>,
    /// Whether an update is available.
    pub update_available: bool,
    /// llama-server CLI flag support detected from --help / -h.
    pub flag_support: LlamaFlagSupport,
}

#[tauri::command]
pub async fn get_llama_info(
    state: tauri::State<'_, SharedState>,
) -> Result<LlamaServerInfo, String> {
    let s = state.read().await;
    let binary_path = s.process.find_server_binary();
    let flag_support = detect_llama_flag_support(binary_path.as_deref());
    let managed_dir = LlamaProcess::managed_binary_dir();
    let has_managed = managed_dir.join("llama-server.exe").exists();
    let version = download::current_version();

    // Try to check latest version (quick timeout).
    let (latest_version, update_available) = match download::check_for_update().await {
        Ok(Some((tag, _, _))) => (Some(tag), true),
        Ok(None) => (version.clone(), false),
        Err(_) => (None, false),
    };

    Ok(LlamaServerInfo {
        version,
        binary_path: binary_path.map(|p| p.to_string_lossy().to_string()),
        has_managed_binary: has_managed,
        managed_dir: managed_dir.to_string_lossy().to_string(),
        latest_version,
        update_available,
        flag_support,
    })
}

#[tauri::command]
pub async fn list_runtime_packs() -> Result<Vec<RuntimePackInfo>, String> {
    let installed_version = download::current_version();
    let mut packs = Vec::new();

    for (backend, name, description) in [
        (
            "cuda",
            "CUDA llama.cpp (Windows)",
            "NVIDIA CUDA accelerated llama.cpp engine",
        ),
        (
            "cpu",
            "CPU llama.cpp (Windows)",
            "CPU-only llama.cpp engine",
        ),
    ] {
        let pattern = download::asset_pattern_for(backend);
        let result = download::find_release_asset(&pattern).await;
        match result {
            Ok((tag, _url, size)) => {
                let update_available =
                    download::release_is_newer(installed_version.as_deref(), &tag);
                packs.push(RuntimePackInfo {
                    id: backend.to_string(),
                    name: name.to_string(),
                    description: description.to_string(),
                    backend: backend.to_string(),
                    installed_version: installed_version.clone(),
                    latest_version: Some(tag),
                    update_available,
                    size_bytes: Some(size),
                    available: true,
                    error: None,
                });
            }
            Err(error) => packs.push(RuntimePackInfo {
                id: backend.to_string(),
                name: name.to_string(),
                description: description.to_string(),
                backend: backend.to_string(),
                installed_version: installed_version.clone(),
                latest_version: None,
                update_available: false,
                size_bytes: None,
                available: false,
                error: Some(error.to_string()),
            }),
        }
    }

    Ok(packs)
}

/// Download a specific backend build of llama-server.
/// `backend` should be "cuda" or "cpu".
/// For CUDA: downloads the server binary and the CUDA runtime DLLs (if available separately).
#[tauri::command]
pub async fn download_llama_build(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
    backend: String,
) -> Result<String, String> {
    let pattern = match backend.to_lowercase().as_str() {
        "cuda" => download::asset_pattern_for("cuda"),
        "cpu" | "avx2" => download::asset_pattern_for("cpu"),
        other => return Err(format!("Unknown backend: {other}. Use 'cuda' or 'cpu'.")),
    };

    // Find server binary (may scan multiple releases).
    let (tag, url, size) = download::find_release_asset(&pattern)
        .await
        .map_err(|e| format!("Failed to find {backend} build: {e}"))?;

    tracing::info!(
        backend = %backend,
        version = %tag,
        "Downloading llama-server build"
    );

    let load_mutex = {
        let s = state.read().await;
        s.model_load_mutex.clone()
    };
    let _load_guard = load_mutex.lock().await;

    let stopped_model =
        crate::commands::model::stop_model_for_binary_update(state.inner().clone()).await?;
    if let Some(model) = stopped_model {
        let _ = app.emit(
            "model-load-progress",
            crate::state::LoadProgress {
                stage: "downloading".to_string(),
                message: format!("Stopped {model} before installing llama-server {tag}"),
                progress: 0.02,
                done: false,
                error: None,
            },
        );
    }

    let server_path = download::download_llama_server(&app, &url, &tag, size)
        .await
        .map_err(|e| format!("Download failed: {e}"))?;

    // For CUDA: also download CUDA runtime DLLs if available (cudart-* packages).
    if backend.to_lowercase() == "cuda" {
        match download::find_cuda_runtime_asset().await {
            Ok(Some((rt_tag, rt_url, rt_size))) => {
                tracing::info!(
                    version = %rt_tag,
                    "Also downloading CUDA runtime DLLs"
                );
                // Download and extract into the same directory (DLLs alongside llama-server.exe).
                match download::download_llama_server(&app, &rt_url, &rt_tag, rt_size).await {
                    Ok(_) => tracing::info!("CUDA runtime DLLs installed"),
                    Err(e) => tracing::warn!(
                        error = %e,
                        "Failed to download CUDA runtime DLLs (server binary is still usable)"
                    ),
                }
            }
            Ok(None) => tracing::debug!("No separate CUDA runtime package found"),
            Err(e) => tracing::warn!(error = %e, "Could not check for CUDA runtime package"),
        }
    }

    // Update the process to use the new binary.
    let mut s = state.write().await;
    s.process.set_server_path(server_path.clone());
    let _ = app.emit(
        "model-load-progress",
        crate::state::LoadProgress {
            stage: "ready".to_string(),
            message: format!("Installed {backend} llama-server {tag}"),
            progress: 1.0,
            done: true,
            error: None,
        },
    );
    Ok(format!(
        "Installed {backend} build {} at {}",
        tag,
        server_path.display()
    ))
}
