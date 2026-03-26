use crate::engine::download;
use crate::engine::process::LlamaProcess;
use crate::state::SharedState;
use serde::{Deserialize, Serialize};

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
    /// API key for Bearer token auth. None / empty string = no auth required.
    pub api_key: Option<String>,
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
        api_key: s.config.server.api_key.clone(),
    })
}

#[tauri::command]
pub async fn update_settings(
    state: tauri::State<'_, SharedState>,
    settings: AppSettings,
) -> Result<(), String> {
    let shared = state.inner().clone();
    let host = settings.server_host.clone();
    let normalized_api_key = settings.api_key.clone().filter(|k| !k.trim().is_empty());
    let (
        should_restart_api,
        should_start_api,
        previous_host,
        previous_port,
        previous_api_key,
    ) = {
        let mut s = shared.write().await;
        let previous_host = s.config.server.host.clone();
        let previous_port = s.config.server.port;
        let previous_api_key = s.config.server.api_key.clone();
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
        s.config.process.main_gpu = settings.main_gpu;
        s.config.process.defrag_thold = settings.defrag_thold;
        s.config.process.rope_freq_scale = settings.rope_freq_scale;
        s.config.server.api_key = normalized_api_key.clone();

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
}

#[tauri::command]
pub async fn get_llama_info(
    state: tauri::State<'_, SharedState>,
) -> Result<LlamaServerInfo, String> {
    let s = state.read().await;
    let binary_path = s.process.find_server_binary();
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
    })
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
    Ok(format!(
        "Installed {backend} build {} at {}",
        tag,
        server_path.display()
    ))
}
