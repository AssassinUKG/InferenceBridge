pub mod api;
pub mod commands;
pub mod config;
pub mod context;
pub mod engine;
pub mod logging;
pub mod models;
pub mod normalize;
pub mod session;
pub mod state;
pub mod templates;

use std::sync::Arc;
use tauri::Manager;
use tauri::RunEvent;
use tokio::sync::RwLock;

fn command_no_window(program: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

#[cfg(windows)]
struct SingleInstanceGuard {
    handle: *mut std::ffi::c_void,
}

#[cfg(windows)]
impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = release_mutex(self.handle);
            let _ = close_handle(self.handle);
        }
    }
}

#[cfg(windows)]
fn try_acquire_single_instance() -> Option<SingleInstanceGuard> {
    const ERROR_ALREADY_EXISTS: u32 = 183;
    let name = wide_null("Local\\InferenceBridge.Gui.SingleInstance");

    unsafe {
        let handle = create_mutex_w(std::ptr::null_mut(), 0, name.as_ptr());
        if handle.is_null() {
            show_single_instance_message();
            return None;
        }

        let error = get_last_error();
        if error == ERROR_ALREADY_EXISTS {
            let _ = close_handle(handle);
            tracing_single_instance_conflict();
            show_single_instance_message();
            return None;
        }

        Some(SingleInstanceGuard { handle })
    }
}

#[cfg(not(windows))]
fn try_acquire_single_instance() -> Option<()> {
    Some(())
}

#[cfg(windows)]
fn tracing_single_instance_conflict() {
    eprintln!("InferenceBridge is already running. Reusing the existing GUI instance.");
}

#[cfg(windows)]
fn show_single_instance_message() {
    const MB_OK: u32 = 0x0000_0000;
    const MB_ICONINFORMATION: u32 = 0x0000_0040;
    let title = wide_null("InferenceBridge");
    let body = wide_null(
        "InferenceBridge is already running.\n\nUse the existing window. The embedded API stays on the first app instance.",
    );

    unsafe {
        let _ = message_box_w(
            std::ptr::null_mut(),
            body.as_ptr(),
            title.as_ptr(),
            MB_OK | MB_ICONINFORMATION,
        );
    }
}

#[cfg(windows)]
fn wide_null(value: &str) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    std::ffi::OsStr::new(value)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

#[cfg(windows)]
unsafe extern "system" {
    fn CreateMutexW(
        lp_mutex_attributes: *mut std::ffi::c_void,
        b_initial_owner: i32,
        lp_name: *const u16,
    ) -> *mut std::ffi::c_void;
    fn ReleaseMutex(h_mutex: *mut std::ffi::c_void) -> i32;
    fn CloseHandle(h_object: *mut std::ffi::c_void) -> i32;
    fn GetLastError() -> u32;
    fn MessageBoxW(
        h_wnd: *mut std::ffi::c_void,
        lp_text: *const u16,
        lp_caption: *const u16,
        u_type: u32,
    ) -> i32;
}

#[cfg(windows)]
unsafe fn create_mutex_w(
    attrs: *mut std::ffi::c_void,
    initial_owner: i32,
    name: *const u16,
) -> *mut std::ffi::c_void {
    CreateMutexW(attrs, initial_owner, name)
}

#[cfg(windows)]
unsafe fn release_mutex(handle: *mut std::ffi::c_void) -> i32 {
    ReleaseMutex(handle)
}

#[cfg(windows)]
unsafe fn close_handle(handle: *mut std::ffi::c_void) -> i32 {
    CloseHandle(handle)
}

#[cfg(windows)]
unsafe fn get_last_error() -> u32 {
    GetLastError()
}

#[cfg(windows)]
unsafe fn message_box_w(
    hwnd: *mut std::ffi::c_void,
    text: *const u16,
    caption: *const u16,
    ty: u32,
) -> i32 {
    MessageBoxW(hwnd, text, caption, ty)
}

/// Run InferenceBridge in GUI mode (Tauri window + API server).
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _single_instance = match try_acquire_single_instance() {
        Some(guard) => guard,
        None => return,
    };

    logging::init("inference_bridge_lib=info", false);

    // Load config (this is sync, safe outside tokio)
    let app_config = config::AppConfig::load();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            commands::model::list_models,
            commands::model::scan_models,
            commands::model::load_model,
            commands::model::unload_model,
            commands::model::swap_model,
            commands::model::get_process_status,
            commands::model::get_effective_profile,
            commands::model::reload_last_known_good,
            commands::model::check_llama_server,
            commands::model::update_llama_server,
            commands::model::list_llama_processes,
            commands::model::kill_process,
            commands::model::kill_all_llama_processes,
            commands::model::get_gpu_stats,
            commands::chat::send_message,
            commands::chat::stop_generation,
            commands::session::create_session,
            commands::session::list_sessions,
            commands::session::delete_session,
            commands::session::get_session_messages,
            commands::context::get_context_status,
            commands::debug::get_raw_prompt,
            commands::debug::get_parse_trace,
            commands::debug::get_launch_preview,
            commands::debug::get_logs,
            commands::debug::clear_logs,
            commands::debug::debug_api_request,
            commands::settings::get_settings,
            commands::settings::get_api_access_info,
            commands::settings::update_settings,
            commands::settings::set_api_server_running,
            commands::settings::get_llama_info,
            commands::settings::download_llama_build,
            commands::benchmark::run_model_test,
            commands::browse::list_hub_models,
            commands::browse::download_hub_model,
            commands::browse::list_downloads,
            commands::browse::cancel_download,
            commands::browse::clear_completed_downloads,
            commands::browse::delete_model_file,
            commands::browse::search_hub_models,
            commands::browse::show_in_folder,
        ])
        .setup(move |app| {
            // Warn if existing llama-server processes may cause port conflicts
            warn_existing_processes();

            // Now inside the tokio runtime — safe to create watch channels etc.
            let api_autostart = app_config.server.autostart;
            let api_port = app_config.server.port;
            let api_host = app_config.server.host.clone();
            let mut app_state =
                state::AppState::new(app_config).expect("Failed to initialize app state");
            // Store the app handle so API/backend paths can emit GUI events.
            app_state.app_handle = Some(app.handle().clone());
            let shared_state: state::SharedState = Arc::new(RwLock::new(app_state));

            // Register state so Tauri commands can access it
            app.manage(shared_state.clone());

            // Mirror headless mode: populate the model registry on GUI launch so
            // the public API can answer /v1/models immediately.
            let scan_state = shared_state.clone();
            tauri::async_runtime::spawn(async move {
                let scan_dirs = {
                    let s = scan_state.read().await;
                    s.config.models.scan_dirs.clone()
                };

                let models = tauri::async_runtime::spawn_blocking(move || {
                    models::scanner::scan_all(&scan_dirs)
                })
                .await
                .unwrap_or_default();
                let count = models.len();

                let mut s = scan_state.write().await;
                s.model_registry.update(models);
                tracing::info!(count, "GUI startup auto-scan completed");
            });

            if api_autostart {
                tracing::info!(
                    pid = std::process::id(),
                    host = %api_host,
                    port = api_port,
                    "GUI startup scheduling embedded API server"
                );
                crate::api::runtime::start_managed(
                    shared_state.clone(),
                    api_host.clone(),
                    api_port,
                    "gui",
                );
            } else {
                tracing::info!(
                    pid = std::process::id(),
                    host = %api_host,
                    port = api_port,
                    "API autostart disabled for GUI launch"
                );
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                // Kill managed llama-server on exit if configured
                let state: Option<tauri::State<'_, state::SharedState>> = app.try_state();
                if let Some(shared) = state {
                    let shared = shared.inner().clone();
                    tauri::async_runtime::block_on(async {
                        let _ = crate::api::runtime::stop_managed(shared.clone()).await;
                        let mut s = shared.write().await;
                        if s.config.process.kill_on_exit {
                            tracing::info!("kill_on_exit enabled — shutting down llama-server");
                            if let Err(e) = s.process.shutdown().await {
                                tracing::warn!(error = %e, "Error shutting down llama-server");
                            }
                        }
                    });
                }
            }
        });
}

/// Run InferenceBridge in headless mode (API server only, no GUI).
/// Similar to `lms server start` in LM Studio.
pub fn run_headless(
    port_override: Option<u16>,
    host_override: Option<String>,
    auto_model: Option<String>,
    ctx_size: Option<u32>,
    gpu_layers: Option<i32>,
    threads: Option<u32>,
    backend_preference: Option<String>,
    extra_scan_dirs: Vec<std::path::PathBuf>,
    verbose: bool,
    default_temperature: Option<f32>,
    default_top_p: Option<f32>,
    default_top_k: Option<i32>,
    default_max_tokens: Option<u32>,
) {
    // Re-attach console on Windows release builds (since we have windows_subsystem = "windows")
    #[cfg(windows)]
    {
        unsafe {
            let _ = windows_attach_console();
        }
    }

    let log_filter = if verbose {
        "inference_bridge_lib=debug,tower_http=debug"
    } else {
        "inference_bridge_lib=info"
    };
    logging::init(log_filter, verbose);

    let mut app_config = config::AppConfig::load();
    if let Some(port) = port_override {
        app_config.server.port = port;
    }
    if let Some(host) = host_override {
        app_config.server.host = host;
    }
    if let Some(gl) = gpu_layers {
        app_config.process.gpu_layers = gl;
    }
    if let Some(th) = threads {
        app_config.process.threads = th;
    }
    if let Some(bp) = backend_preference {
        app_config.process.backend_preference = bp;
    }
    // Merge any CLI --scan-dir paths with those from the config file.
    for dir in extra_scan_dirs {
        if !app_config.models.scan_dirs.contains(&dir) {
            app_config.models.scan_dirs.push(dir);
        }
    }
    // Apply CLI sampling/context defaults (override config file values when specified).
    if let Some(t) = default_temperature {
        app_config.server.default_temperature = Some(t);
    }
    if let Some(p) = default_top_p {
        app_config.server.default_top_p = Some(p);
    }
    if let Some(k) = default_top_k {
        app_config.server.default_top_k = Some(k);
    }
    if let Some(m) = default_max_tokens {
        app_config.server.default_max_tokens = Some(m);
    }
    if let Some(c) = ctx_size {
        app_config.server.default_ctx_size = Some(c);
    }

    let api_port = app_config.server.port;
    let api_host = app_config.server.host.clone();

    // Build and run a tokio runtime
    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    rt.block_on(async move {
        let app_state = state::AppState::new(app_config).expect("Failed to initialize app state");
        let shared_state: state::SharedState = Arc::new(RwLock::new(app_state));

        // Auto-scan models
        {
            let mut s = shared_state.write().await;
            let models = models::scanner::scan_all(&s.config.models.scan_dirs);
            let count = models.len();
            s.model_registry.update(models);
            tracing::info!(count, "Scanned models");
        }

        // Auto-load model if specified. CLI --ctx-size takes priority; fall back to
        // server.default_ctx_size from config/CLI flags.
        if let Some(ref model_name) = auto_model {
            let effective_ctx = {
                let s = shared_state.read().await;
                ctx_size.or(s.config.server.default_ctx_size)
            };
            headless_load_model(&shared_state, model_name, effective_ctx).await;
        }

        // Print status banner
        {
            let s = shared_state.read().await;
            let model_count = s.model_registry.list().len();
            let loaded = s.loaded_model.as_deref().unwrap_or("none");
            eprintln!();
            eprintln!("  InferenceBridge (headless mode)");
            eprintln!("  ==============================");
            eprintln!("  API:    http://{api_host}:{api_port}/v1");
            eprintln!("  Models: {model_count} available");
            eprintln!("  Loaded: {loaded}");
            eprintln!("  GPU:    {} layers", s.config.process.gpu_layers);
            eprintln!("  CPU:    {} threads", s.config.process.threads);
            eprintln!(
                "  Backend preference: {}",
                s.config.process.backend_preference
            );
            eprintln!();
            eprintln!("  Compatible with OpenAI API clients.");
            eprintln!("  POST /v1/chat/completions");
            eprintln!("  POST /v1/completions");
            eprintln!("  GET  /v1/models");
            eprintln!("  GET  /v1/models/{{name}}");
            eprintln!("  POST /v1/models/load | /v1/models/unload");
            eprintln!("  GET  /v1/models/stats");
            eprintln!("  GET  /v1/context/status");
            eprintln!("  GET  /v1/sessions");
            eprintln!("  GET  /v1/health");
            if let Some(t) = s.config.server.default_temperature {
                eprintln!("  Default temperature: {t}");
            }
            if let Some(m) = s.config.server.default_max_tokens {
                eprintln!("  Default max tokens: {m}");
            }
            if let Some(c) = s.config.server.default_ctx_size {
                eprintln!("  Default ctx size:   {c}");
            }
            eprintln!();
            eprintln!("  Press Ctrl+C to stop.");
            eprintln!();
        }

        // Set up Ctrl+C handler
        let shutdown_state = shared_state.clone();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown_tx = std::sync::Mutex::new(Some(shutdown_tx));

        ctrlc::set_handler(move || {
            eprintln!("\nShutting down...");
            if let Ok(mut guard) = shutdown_tx.lock() {
                if let Some(tx) = guard.take() {
                    let _ = tx.send(());
                }
            }
        })
        .expect("Failed to set Ctrl+C handler");

        // Start the API server
        let api_state = shared_state.clone();
        let api_host = api_host.clone();
        tracing::info!(
            pid = std::process::id(),
            host = %api_host,
            port = api_port,
            "Headless startup scheduling API server"
        );
        let server_handle = tokio::spawn(async move {
            if let Err(e) =
                api::server::start_api_server(api_state, &api_host, api_port, "headless").await
            {
                tracing::error!(error = %e, "API server failed");
            }
        });

        // Wait for shutdown signal
        let _ = shutdown_rx.await;

        // Clean up: shutdown llama-server process
        {
            let mut s = shutdown_state.write().await;
            if let Err(e) = s.process.shutdown().await {
                tracing::warn!(error = %e, "Error shutting down llama-server");
            }
        }

        server_handle.abort();
        tracing::info!("InferenceBridge stopped.");
    });
}

/// Load a model by name in headless mode (no Tauri app handle for events).
async fn headless_load_model(state: &state::SharedState, model_name: &str, ctx_size: Option<u32>) {
    use engine::process::LaunchConfig;

    // Resolve model
    let config = {
        let s = state.read().await;
        let model = match s.model_registry.find_by_name(model_name) {
            Some(m) => m.clone(),
            None => {
                tracing::error!(model = model_name, "Model not found. Available models:");
                for m in s.model_registry.list() {
                    tracing::info!("  - {} ({:.1} GB)", m.filename, m.size_bytes as f64 / 1e9);
                }
                return;
            }
        };

        let ctx = ctx_size
            .or(model.profile.default_context_window)
            .unwrap_or(s.config.models.default_context);

        LaunchConfig {
            model_path: model.path.clone(),
            context_size: ctx,
            gpu_layers: s.config.process.gpu_layers,
            threads: s.config.process.threads,
            threads_batch: s.config.process.threads_batch,
            port: 8801,
            backend_preference: s.config.process.backend_preference.clone(),
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
        }
    };

    let model_display = config
        .model_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| model_name.to_string());

    tracing::info!(
        model = %model_display,
        ctx = config.context_size,
        gpu_layers = config.gpu_layers,
        "Loading model..."
    );

    // Check for llama-server binary, auto-download if needed
    {
        let s = state.read().await;
        if s.process.find_server_binary().is_none() {
            drop(s);
            tracing::info!("llama-server not found, attempting auto-download...");
            match engine::download::check_for_update().await {
                Ok(Some((tag, url, size))) => {
                    tracing::info!(version = %tag, "Downloading llama-server...");
                    // No app handle in headless mode — just log progress
                    let dir = engine::process::LlamaProcess::managed_binary_dir();
                    std::fs::create_dir_all(&dir).ok();
                    match headless_download(&url, &tag, size).await {
                        Ok(path) => {
                            let mut s = state.write().await;
                            s.process.set_server_path(path);
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Failed to download llama-server");
                            tracing::error!("Install manually: winget install ggml.llamacpp");
                            return;
                        }
                    }
                }
                Ok(None) => {
                    tracing::error!("llama-server not found and no update available");
                    tracing::error!("Install manually: winget install ggml.llamacpp");
                    return;
                }
                Err(e) => {
                    tracing::error!(error = %e, "Failed to check for llama-server");
                    tracing::error!("Install manually: winget install ggml.llamacpp");
                    return;
                }
            }
        }
    }

    // Launch
    {
        let mut s = state.write().await;
        if let Err(e) = s.process.launch(config).await {
            tracing::error!(error = %e, "Failed to launch llama-server");
            return;
        }
    }

    // Wait for healthy
    tracing::info!("Waiting for llama-server to become healthy...");
    let client = reqwest::Client::new();
    let health_url = "http://127.0.0.1:8801/health";
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(300);

    loop {
        if start.elapsed() > timeout {
            tracing::error!("llama-server did not become healthy within 5 minutes");
            let mut s = state.write().await;
            let _ = s.process.shutdown().await;
            return;
        }

        match client.get(health_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(body) = resp.text().await {
                    if body.contains("ok") {
                        break;
                    }
                    if body.contains("loading") {
                        let elapsed = start.elapsed().as_secs();
                        if elapsed % 5 == 0 {
                            tracing::info!("Loading model... ({elapsed}s)");
                        }
                    }
                } else {
                    break;
                }
            }
            Ok(resp) if resp.status().as_u16() == 503 => {
                let elapsed = start.elapsed().as_secs();
                if elapsed % 5 == 0 {
                    tracing::info!("Loading model weights... ({elapsed}s)");
                }
            }
            _ => {}
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    // Mark loaded
    {
        let mut s = state.write().await;
        s.loaded_model = Some(model_display.clone());
        s.process.set_state_running();
    }

    tracing::info!(model = %model_display, "Model loaded and ready");
}

/// Download llama-server in headless mode (no Tauri events, just logging).
async fn headless_download(
    url: &str,
    tag: &str,
    total_size: u64,
) -> anyhow::Result<std::path::PathBuf> {
    use futures_util::StreamExt;

    let dir = engine::process::LlamaProcess::managed_binary_dir();
    std::fs::create_dir_all(&dir)?;
    let temp_file = dir.join("download.tmp");

    let client = reqwest::Client::builder()
        .user_agent("InferenceBridge/0.1")
        .redirect(reqwest::redirect::Policy::limited(10))
        .timeout(std::time::Duration::from_secs(600))
        .build()?;

    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        return Err(anyhow::anyhow!("Download failed: {}", response.status()));
    }

    let total = response.content_length().unwrap_or(total_size);
    let mut stream = response.bytes_stream();
    let mut file = std::fs::File::create(&temp_file)?;
    let mut downloaded: u64 = 0;
    let mut last_log: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        use std::io::Write;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;

        // Log every 10MB
        if downloaded - last_log > 10 * 1024 * 1024 {
            let mb = downloaded / (1024 * 1024);
            let total_mb = total / (1024 * 1024);
            tracing::info!("Downloading llama-server {tag}: {mb}/{total_mb} MB");
            last_log = downloaded;
        }
    }
    drop(file);

    tracing::info!("Extracting...");

    // Reuse the zip extraction from the download module
    let server_path = engine::download::extract_archive(&temp_file, &dir)?;
    let _ = std::fs::remove_file(&temp_file);

    // Write version file
    std::fs::write(dir.join("llama-server.version"), tag)?;

    tracing::info!(path = %server_path.display(), "llama-server installed");
    Ok(server_path)
}

/// Log a warning if existing llama-server processes are running.
/// Does NOT kill them — the user can manage them via the Process Manager UI.
fn warn_existing_processes() {
    let output = command_no_window("tasklist")
        .args(["/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"])
        .output();

    let output = match output {
        Ok(o) => o,
        Err(_) => return,
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let mut count = 0u32;
    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 2 {
            let pid_str = parts[1].trim_matches('"').trim();
            if pid_str.parse::<u32>().unwrap_or(0) > 0 {
                count += 1;
            }
        }
    }
    if count > 0 {
        tracing::warn!(
            count = count,
            "Existing llama-server processes detected — may cause port conflicts. \
             Use the Process Manager (Models tab) to manage them."
        );
    }
}

/// Try to re-attach to the parent console on Windows.
/// Needed because release builds use `windows_subsystem = "windows"` which hides the console.
#[cfg(windows)]
unsafe fn windows_attach_console() -> bool {
    // AttachConsole(-1) = ATTACH_PARENT_PROCESS
    extern "system" {
        fn AttachConsole(dw_process_id: u32) -> i32;
    }
    unsafe { AttachConsole(u32::MAX) != 0 }
}

// ── CLI subcommand implementations ──────────────────────────────────────

/// `inference-bridge models` — list available models and exit.
pub fn run_list_models() {
    #[cfg(windows)]
    {
        unsafe {
            let _ = windows_attach_console();
        }
    }

    let app_config = config::AppConfig::load();
    let models = models::scanner::scan_all(&app_config.models.scan_dirs);

    if models.is_empty() {
        eprintln!("No models found.");
        eprintln!("Configure scan_dirs in your inference-bridge.toml");
        return;
    }

    eprintln!();
    eprintln!("  Available Models");
    eprintln!("  ────────────────────────────────────────────────────");
    for m in &models {
        use models::profiles::{ThinkTagStyle, ToolCallFormat};
        let size = m.size_bytes as f64 / 1e9;
        let supports_tools = !matches!(m.profile.tool_call_format, ToolCallFormat::NativeApi)
            || m.profile.supports_parallel_tools;
        let supports_reasoning = !matches!(m.profile.think_tag_style, ThinkTagStyle::None);
        let caps = [
            if supports_tools { "tools" } else { "" },
            if supports_reasoning { "reasoning" } else { "" },
        ]
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
        let cap_str = if caps.is_empty() {
            String::new()
        } else {
            format!(" [{caps}]")
        };
        let ctx = m
            .profile
            .default_context_window
            .map(|c| format!(" ctx:{c}"))
            .unwrap_or_default();
        eprintln!("  {:<50} {:>6.1} GB{ctx}{cap_str}", m.filename, size);
    }
    eprintln!();
    eprintln!("  {} model(s) found", models.len());
    eprintln!();
}

/// `inference-bridge run` — one-shot inference: load model, run prompt, print output, exit.
pub fn run_one_shot(
    model_name: String,
    prompt: String,
    ctx_size: Option<u32>,
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<i64>,
    _port_override: Option<u16>,
    gpu_layers: Option<i32>,
    threads: Option<u32>,
    backend_preference: Option<String>,
) {
    #[cfg(windows)]
    {
        unsafe {
            let _ = windows_attach_console();
        }
    }

    // Minimal logging — only errors to stderr so stdout is clean for piping
    logging::init("inference_bridge_lib=warn", true);

    let mut app_config = config::AppConfig::load();
    if let Some(gl) = gpu_layers {
        app_config.process.gpu_layers = gl;
    }
    if let Some(th) = threads {
        app_config.process.threads = th;
    }
    if let Some(bp) = backend_preference {
        app_config.process.backend_preference = bp;
    }

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(async move {
        let app_state = state::AppState::new(app_config).expect("Failed to initialize app state");
        let shared_state: state::SharedState = Arc::new(RwLock::new(app_state));

        // Scan models
        {
            let mut s = shared_state.write().await;
            let models = models::scanner::scan_all(&s.config.models.scan_dirs);
            s.model_registry.update(models);
        }

        // Load and wait for model
        headless_load_model(&shared_state, &model_name, ctx_size).await;

        // Check model loaded
        {
            let s = shared_state.read().await;
            if s.loaded_model.is_none() {
                eprintln!("Failed to load model.");
                return;
            }
        }

        // Build request
        let s = shared_state.read().await;
        let profile =
            models::profiles::ModelProfile::detect(s.loaded_model.as_deref().unwrap_or(""));

        let messages = vec![templates::engine::ChatMessage {
            role: "user".to_string(),
            content: prompt,
        }];
        let rendered = templates::engine::render_prompt(&messages, &profile);

        let request = engine::client::CompletionRequest {
            prompt: rendered,
            n_predict: Some(max_tokens as i32),
            temperature: temperature.or(profile.default_temperature),
            top_p: top_p.or(profile.default_top_p),
            top_k: top_k.or(profile.default_top_k),
            min_p: profile.default_min_p,
            presence_penalty: profile.default_presence_penalty,
            frequency_penalty: None,
            repeat_penalty: None,
            seed,
            stream: false,
            stop: profile.stop_markers.clone(),
            special: true,
            image_data: vec![],
        };

        let client = engine::client::LlamaClient::new(s.process.port());
        drop(s);

        match client.complete(&request).await {
            Ok(resp) => {
                let text = normalize::think_strip::strip_think_tags_with_style(
                    &resp.content,
                    profile.think_tag_style,
                );
                // Print to stdout — clean for piping
                print!("{text}");
                if let Some(timings) = &resp.timings {
                    if let Some(tps) = timings.predicted_per_second {
                        eprintln!("\n\n({:.1} tok/s)", tps);
                    }
                }
            }
            Err(e) => {
                eprintln!("Generation failed: {e}");
            }
        }

        // Shutdown
        let mut s = shared_state.write().await;
        let _ = s.process.shutdown().await;
    });
}

/// `inference-bridge status` — query a running server's health and display info.
pub fn run_status(port: u16) {
    #[cfg(windows)]
    {
        unsafe {
            let _ = windows_attach_console();
        }
    }

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap_or_default();

        // Query /v1/health
        let health_url = format!("http://127.0.0.1:{port}/v1/health");
        match client.get(&health_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(json) => {
                        let status = json["status"].as_str().unwrap_or("unknown");
                        let model = json["model"].as_str().unwrap_or("none");
                        eprintln!();
                        eprintln!("  InferenceBridge Status");
                        eprintln!("  ──────────────────────────");
                        eprintln!("  API:    http://127.0.0.1:{port}/v1");
                        eprintln!("  Status: {status}");
                        eprintln!("  Model:  {model}");
                        if let Some(kv) = json.get("kv_cache") {
                            let used = kv["used_tokens"].as_u64().unwrap_or(0);
                            let total = kv["total_tokens"].as_u64().unwrap_or(0);
                            let ratio = kv["fill_ratio"].as_f64().unwrap_or(0.0);
                            eprintln!("  KV:     {used}/{total} tokens ({:.0}%)", ratio * 100.0);
                        }
                        eprintln!();
                    }
                    Err(_) => {
                        eprintln!("Server responded but returned invalid JSON.");
                    }
                }
            }
            Ok(resp) => {
                eprintln!("Server returned status: {}", resp.status());
            }
            Err(_) => {
                eprintln!("No InferenceBridge server running on port {port}.");
                eprintln!("Start one with: inference-bridge serve");
            }
        }
    });
}

/// `inference-bridge update` — check for llama-server updates and download.
pub fn run_update() {
    #[cfg(windows)]
    {
        unsafe {
            let _ = windows_attach_console();
        }
    }

    logging::init("inference_bridge_lib=info", false);

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(async move {
        match engine::download::check_for_update().await {
            Ok(Some((tag, url, size))) => {
                let size_mb = size / (1024 * 1024);
                eprintln!("Update available: {tag} ({size_mb} MB)");
                eprintln!("Downloading...");
                match headless_download(&url, &tag, size).await {
                    Ok(path) => {
                        eprintln!("Installed: {}", path.display());
                    }
                    Err(e) => {
                        eprintln!("Download failed: {e}");
                    }
                }
            }
            Ok(None) => {
                let current =
                    engine::download::current_version().unwrap_or_else(|| "unknown".to_string());
                eprintln!("llama-server is up-to-date ({current}).");
            }
            Err(e) => {
                eprintln!("Failed to check for updates: {e}");
            }
        }
    });
}

pub fn run_model_test_cli(
    model: String,
    prompt: String,
    ctx_size: Option<u32>,
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<i64>,
    port_override: Option<u16>,
    gpu_layers: Option<i32>,
    threads: Option<u32>,
    backend_preference: Option<String>,
    verbose: bool,
) {
    use crate::engine::benchmark::test_model;
    use std::sync::Arc;
    use tokio::runtime::Runtime;
    use tokio::sync::RwLock;

    let rt = Runtime::new().expect("Failed to create tokio runtime");
    if verbose {
        eprintln!("[DEBUG] Starting run_model_test_cli");
        eprintln!("[DEBUG] Args: model={model}, ctx_size={ctx_size:?}, max_tokens={max_tokens}, temperature={temperature:?}, top_p={top_p:?}, top_k={top_k:?}, seed={seed:?}, port={port_override:?}, gpu_layers={gpu_layers:?}, threads={threads:?}, backend_preference={backend_preference:?}");
    }
    rt.block_on(async move {
        let mut app_config = config::AppConfig::load();
        if verbose {
            eprintln!("[DEBUG] Loaded config");
        }
        if let Some(port) = port_override {
            app_config.server.port = port;
            if verbose {
                eprintln!("[DEBUG] Set port to {port}");
            }
        }
        if let Some(gl) = gpu_layers {
            app_config.process.gpu_layers = gl;
            if verbose {
                eprintln!("[DEBUG] Set gpu_layers to {gl}");
            }
        }
        if let Some(th) = threads {
            app_config.process.threads = th;
            if verbose {
                eprintln!("[DEBUG] Set threads to {th}");
            }
        }
        if let Some(bp) = backend_preference {
            app_config.process.backend_preference = bp;
            if verbose {
                eprintln!("[DEBUG] Set backend_preference");
            }
        }
        let app_state = state::AppState::new(app_config).expect("Failed to initialize app state");
        if verbose {
            eprintln!("[DEBUG] Created app_state");
        }
        let shared_state: state::SharedState = Arc::new(RwLock::new(app_state));
        // Scan models
        {
            let mut s = shared_state.write().await;
            let models = models::scanner::scan_all(&s.config.models.scan_dirs);
            s.model_registry.update(models);
            if verbose {
                eprintln!("[DEBUG] Scanned and updated model registry");
            }
        }
        if verbose {
            eprintln!("[DEBUG] Calling test_model");
        }
        match test_model(
            shared_state,
            &model,
            ctx_size.unwrap_or(8192),
            &prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
        )
        .await
        {
            Ok(stats) => {
                if verbose {
                    eprintln!("[DEBUG] test_model returned Ok");
                }
                println!("Model: {}", stats.model);
                println!("Context size: {}", stats.context_size);
                println!("Prompt: {}", stats.prompt);
                println!("Response: {}", stats.response);
                if let Some(t) = stats.timings {
                    if let Some(tps) = t.predicted_per_second {
                        println!("Tokens/sec: {:.2}", tps);
                    }
                    if let Some(pps) = t.prompt_per_second {
                        println!("Prompt tokens/sec: {:.2}", pps);
                    }
                }
                println!("Elapsed: {} ms", stats.elapsed_ms);
            }
            Err(e) => {
                if verbose {
                    eprintln!("[DEBUG] test_model returned Err");
                }
                eprintln!("Model test failed: {e}");
            }
        }
        if verbose {
            eprintln!("[DEBUG] run_model_test_cli finished");
        }
    });
}
