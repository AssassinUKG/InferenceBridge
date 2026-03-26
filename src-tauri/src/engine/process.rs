//! llama-server process lifecycle management.
//!
//! Manages a single llama-server process: spawn, health-check, restart, shutdown.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::sync::{watch, Mutex as TokioMutex};
use tokio::task::JoinHandle;

fn system_command(program: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

fn filename_supports_vision(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    name.contains("vision")
        || name.contains("llava")
        || name.contains("multimodal")
        || name.contains("qwen2.5-vl")
        || name.contains("-vl")
        || name.contains("_vl")
}

fn is_mmproj_candidate(path: &Path) -> bool {
    let Some(filename) = path
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
    else {
        return false;
    };
    path.extension()
        .map(|ext| ext.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
        && (filename.starts_with("mmproj")
            || filename.contains("-mmproj")
            || filename.contains("_mmproj")
            || filename.contains("mmproj-model"))
}

fn shared_token_score(model_path: &Path, candidate: &Path) -> usize {
    let model_tokens = model_path
        .file_stem()
        .map(|stem| {
            stem.to_string_lossy()
                .to_lowercase()
                .split(|ch: char| !ch.is_ascii_alphanumeric())
                .filter(|token| token.len() > 2)
                .map(|token| token.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let candidate_name = candidate
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    model_tokens
        .iter()
        .filter(|token| candidate_name.contains(token.as_str()))
        .count()
}

fn find_mmproj_for_model(model_path: &Path) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    let mut candidates = std::fs::read_dir(parent)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| is_mmproj_candidate(path))
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return None;
    }

    candidates
        .sort_by_key(|candidate| std::cmp::Reverse(shared_token_score(model_path, candidate)));
    candidates.into_iter().next()
}

/// Process state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ProcessState {
    Idle,
    Starting,
    Running,
    Stopping,
    Error,
}

/// Configuration for launching llama-server.
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub model_path: PathBuf,
    pub context_size: u32,
    pub gpu_layers: i32,
    pub threads: u32,
    pub threads_batch: u32,
    pub port: u16,
    pub backend_preference: String,
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
}

/// Manages the llama-server child process.
pub struct LlamaProcess {
    child: Option<Child>,
    state: ProcessState,
    llama_server_path: Option<PathBuf>,
    current_model: Option<String>,
    current_port: u16,
    crash_count: u32,
    state_tx: watch::Sender<ProcessState>,
    state_rx: watch::Receiver<ProcessState>,
    /// GPU backend detected from server stderr (e.g. "CUDA", "Vulkan", "CPU").
    detected_backend: Arc<TokioMutex<Option<String>>>,
    /// Recent stderr lines captured from llama-server (ring buffer for crash diagnostics).
    stderr_lines: Arc<TokioMutex<VecDeque<String>>>,
    /// Handles for background I/O reader tasks — aborted on shutdown to prevent leaks.
    io_tasks: Vec<JoinHandle<()>>,
}

impl LlamaProcess {
    fn path_looks_cuda(path: &Path) -> bool {
        let path_lower = path.to_string_lossy().to_lowercase();
        if path_lower.contains("cuda") {
            return true;
        }
        let dir = path.parent().unwrap_or(path);
        let cuda_indicators = [
            "ggml-cuda.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudart64_12.dll",
        ];
        cuda_indicators.iter().any(|dll| dir.join(dll).exists())
    }

    fn matches_backend_preference(path: &Path, backend_preference: &str) -> bool {
        match backend_preference {
            "cuda" => Self::path_looks_cuda(path),
            "cpu" | "avx2" => !Self::path_looks_cuda(path),
            _ => true,
        }
    }

    pub fn find_server_binary_with_preference(&self, backend_preference: &str) -> Option<PathBuf> {
        // Explicit path is user-provided, always honor it.
        if let Some(ref path) = self.llama_server_path {
            if Path::new(path).exists() {
                return Some(path.clone());
            }
        }

        // Our managed install location (usually CUDA build).
        let our_dir = Self::managed_binary_dir();
        let our_exe = our_dir.join("llama-server.exe");
        if our_exe.exists() && Self::matches_backend_preference(&our_exe, backend_preference) {
            tracing::debug!(
                path = %our_exe.display(),
                backend_preference,
                "Using managed llama-server"
            );
            return Some(our_exe);
        }

        let mut candidates: Vec<PathBuf> = Vec::new();

        // Check PATH
        if let Ok(output) = system_command("where").arg("llama-server").output() {
            if output.status.success() {
                for line in String::from_utf8_lossy(&output.stdout).lines() {
                    let p = PathBuf::from(line.trim());
                    if p.exists() {
                        candidates.push(p);
                    }
                }
            }
        }

        // Also check for llama-server.exe on PATH
        if let Ok(output) = system_command("where").arg("llama-server.exe").output() {
            if output.status.success() {
                for line in String::from_utf8_lossy(&output.stdout).lines() {
                    let p = PathBuf::from(line.trim());
                    if p.exists() {
                        candidates.push(p);
                    }
                }
            }
        }

        // WinGet install location
        if let Some(local_app_data) = dirs::data_local_dir() {
            let winget_base = local_app_data
                .join("Microsoft")
                .join("WinGet")
                .join("Packages");
            if winget_base.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&winget_base) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if name.contains("llamacpp")
                            || name.contains("llama.cpp")
                            || name.contains("ggml")
                        {
                            let exe = entry.path().join("llama-server.exe");
                            if exe.exists() {
                                candidates.push(exe);
                            }
                            for sub in &["bin", "build/bin/Release"] {
                                let exe = entry.path().join(sub).join("llama-server.exe");
                                if exe.exists() {
                                    candidates.push(exe);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Common Windows locations
        let common = [
            dirs::home_dir().map(|h| h.join(".local/bin/llama-server.exe")),
            dirs::home_dir().map(|h| h.join("llama.cpp/build/bin/Release/llama-server.exe")),
            Some(PathBuf::from("C:/llama.cpp/llama-server.exe")),
            Some(PathBuf::from(
                "C:/llama.cpp/build/bin/Release/llama-server.exe",
            )),
        ];
        for candidate in common.into_iter().flatten() {
            if candidate.exists() {
                candidates.push(candidate);
            }
        }

        candidates
            .into_iter()
            .find(|p| Self::matches_backend_preference(p, backend_preference))
    }

    pub fn new() -> Self {
        let (state_tx, state_rx) = watch::channel(ProcessState::Idle);
        Self {
            child: None,
            state: ProcessState::Idle,
            llama_server_path: None,
            current_model: None,
            current_port: 8801,
            crash_count: 0,
            state_tx,
            state_rx,
            detected_backend: Arc::new(TokioMutex::new(None)),
            stderr_lines: Arc::new(TokioMutex::new(VecDeque::new())),
            io_tasks: Vec::new(),
        }
    }

    /// Returns the GPU backend detected from the server's startup logs.
    pub fn detected_backend(&self) -> Arc<TokioMutex<Option<String>>> {
        self.detected_backend.clone()
    }

    /// Get a receiver for state change notifications.
    pub fn state_watch(&self) -> watch::Receiver<ProcessState> {
        self.state_rx.clone()
    }

    pub fn state(&self) -> ProcessState {
        self.state
    }

    pub fn current_model(&self) -> Option<&str> {
        self.current_model.as_deref()
    }

    /// The port the llama-server is listening on.
    pub fn port(&self) -> u16 {
        self.current_port
    }

    /// Set the path to the llama-server binary.
    pub fn set_server_path(&mut self, path: PathBuf) {
        self.llama_server_path = Some(path);
    }

    fn set_state(&mut self, state: ProcessState) {
        self.state = state;
        let _ = self.state_tx.send(state);
    }

    /// Externally mark the process as running (called after health check passes).
    pub fn set_state_running(&mut self) {
        self.set_state(ProcessState::Running);
    }

    /// Find llama-server binary — checks explicit path, then our managed CUDA build,
    /// then PATH, then common locations.
    pub fn find_server_binary(&self) -> Option<PathBuf> {
        self.find_server_binary_with_preference("auto")
    }

    /// Directory where we store our own managed llama-server binary.
    pub fn managed_binary_dir() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("InferenceBridge")
            .join("bin")
    }

    /// Launch llama-server with the given configuration.
    pub async fn launch(&mut self, config: LaunchConfig) -> anyhow::Result<()> {
        // Shutdown any existing process first
        self.shutdown().await?;

        let server_path = self
            .find_server_binary_with_preference(&config.backend_preference)
            .ok_or_else(|| anyhow::anyhow!("llama-server binary not found"))?;

        self.current_port = config.port;
        self.set_state(ProcessState::Starting);

        let mut cmd = Command::new(&server_path);
        cmd.arg("--model")
            .arg(&config.model_path)
            .arg("--ctx-size")
            .arg(config.context_size.to_string())
            .arg("--port")
            .arg(config.port.to_string())
            .arg("--parallel")
            .arg(config.parallel_slots.max(1).to_string())
            .arg("--slots"); // Enable /slots endpoint for KV cache monitoring

        if filename_supports_vision(&config.model_path) {
            if let Some(mmproj_path) = find_mmproj_for_model(&config.model_path) {
                tracing::info!(
                    model = %config.model_path.display(),
                    mmproj = %mmproj_path.display(),
                    "Using multimodal projection sidecar for vision model"
                );
                cmd.arg("--mmproj").arg(mmproj_path);
            } else {
                tracing::warn!(
                    model = %config.model_path.display(),
                    "Vision-capable model detected but no mmproj sidecar was found nearby; image understanding may fail"
                );
            }
        }

        if config.gpu_layers != 0 {
            cmd.arg("--n-gpu-layers").arg(if config.gpu_layers < 0 {
                "999".to_string()
            } else {
                config.gpu_layers.to_string()
            });
        }

        if config.threads > 0 {
            cmd.arg("--threads").arg(config.threads.to_string());
        }

        if config.threads_batch > 0 {
            cmd.arg("--threads-batch")
                .arg(config.threads_batch.to_string());
        }

        if config.batch_size > 0 {
            cmd.arg("--batch-size").arg(config.batch_size.to_string());
        }

        if config.ubatch_size > 0 {
            cmd.arg("--ubatch-size").arg(config.ubatch_size.to_string());
        }

        if config.flash_attn {
            cmd.arg("--flash-attn");
        }

        if !config.use_mmap {
            cmd.arg("--no-mmap");
        }

        if config.use_mlock {
            cmd.arg("--mlock");
        }

        if config.cont_batching {
            cmd.arg("--cont-batching");
        }

        if config.main_gpu != 0 {
            cmd.arg("--main-gpu").arg(config.main_gpu.to_string());
        }

        if config.defrag_thold > 0.0 {
            cmd.arg("--defrag-thold")
                .arg(format!("{:.4}", config.defrag_thold));
        }

        if config.rope_freq_scale > 0.0 {
            cmd.arg("--rope-freq-scale")
                .arg(format!("{:.6}", config.rope_freq_scale));
        }

        // Suppress console window on Windows
        #[cfg(windows)]
        {
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        // Add the server binary's directory to PATH so CUDA runtime DLLs are found
        if let Some(bin_dir) = server_path.parent() {
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = format!("{};{}", bin_dir.display(), current_path);
            cmd.env("PATH", new_path);
        }

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        tracing::info!(
            server = %server_path.display(),
            model = %config.model_path.display(),
            ctx = config.context_size,
            port = config.port,
            gpu_layers = config.gpu_layers,
            "Launching llama-server"
        );

        let mut child = cmd.spawn()?;

        // Abort any leftover I/O tasks from a previous launch
        for handle in self.io_tasks.drain(..) {
            handle.abort();
        }

        // Spawn background tasks to stream stdout/stderr to tracing
        if let Some(stdout) = child.stdout.take() {
            let handle = tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    tracing::info!(target: "llama_server", "{}", line);
                }
            });
            self.io_tasks.push(handle);
        }
        if let Some(stderr) = child.stderr.take() {
            let backend_handle = self.detected_backend.clone();
            let stderr_buf = self.stderr_lines.clone();
            // Clear previous stderr
            stderr_buf.lock().await.clear();
            let handle = tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    // Detect GPU backend from startup logs
                    // CUDA takes priority — once detected, don't overwrite with Vulkan
                    let lower = line.to_lowercase();
                    {
                        let mut guard = backend_handle.lock().await;
                        let is_cuda = guard.as_deref() == Some("CUDA");
                        if lower.contains("ggml_cuda_init") || lower.contains("using cuda") {
                            *guard = Some("CUDA".to_string());
                        } else if !is_cuda
                            && (lower.contains("ggml_vulkan_init")
                                || lower.contains("using vulkan"))
                        {
                            *guard = Some("Vulkan".to_string());
                        } else if !is_cuda && guard.is_none() && lower.contains("ggml_metal_init") {
                            *guard = Some("Metal".to_string());
                        }
                    }
                    // Keep last 50 lines for crash diagnostics (O(1) with VecDeque)
                    {
                        let mut buf = stderr_buf.lock().await;
                        buf.push_back(line.clone());
                        if buf.len() > 50 {
                            buf.pop_front();
                        }
                    }
                    // llama-server logs almost everything to stderr
                    tracing::info!(target: "llama_server", "{}", line);
                }
            });
            self.io_tasks.push(handle);
        }

        self.child = Some(child);
        self.current_model = config
            .model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string());
        self.crash_count = 0;

        Ok(())
    }

    /// Wait for the server to become healthy (responds to /health).
    pub async fn wait_for_healthy(&self, timeout: Duration) -> anyhow::Result<()> {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/health", self.current_port);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                return Err(anyhow::anyhow!(
                    "llama-server did not become healthy within {:?}",
                    timeout
                ));
            }
            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    }

    /// Check if the server is currently healthy.
    pub async fn check_health(&self) -> bool {
        if self.child.is_none() {
            return false;
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_default();
        let url = format!("http://127.0.0.1:{}/health", self.current_port);
        matches!(client.get(&url).send().await, Ok(r) if r.status().is_success())
    }

    /// Gracefully shutdown the llama-server process.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        if let Some(mut child) = self.child.take() {
            self.set_state(ProcessState::Stopping);
            tracing::info!(
                model = ?self.current_model,
                port = self.current_port,
                "Shutting down llama-server"
            );

            // Try graceful shutdown via /shutdown endpoint
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap_or_default();
            let shutdown_url = format!("http://127.0.0.1:{}/shutdown", self.current_port);
            match client.post(&shutdown_url).send().await {
                Ok(_) => tracing::debug!("Graceful shutdown request sent"),
                Err(e) => {
                    tracing::debug!(error = %e, "Graceful shutdown request failed (process may already be stopped)")
                }
            }

            // Wait up to 5 seconds for the process to exit
            let exit = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;

            if exit.is_err() {
                tracing::warn!("llama-server did not exit gracefully, killing");
                let _ = child.kill().await;
            } else {
                tracing::info!("llama-server exited gracefully");
            }

            self.current_model = None;
            *self.detected_backend.lock().await = None;
            self.set_state(ProcessState::Idle);
        }
        // Abort background I/O reader tasks
        for handle in self.io_tasks.drain(..) {
            handle.abort();
        }
        Ok(())
    }

    /// Check if the process has crashed and record it.
    pub async fn check_crashed(&mut self) -> bool {
        if let Some(ref mut child) = self.child {
            match child.try_wait() {
                Ok(Some(status)) => {
                    // Give stderr reader a moment to flush
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    let last_lines = {
                        let lines = self.stderr_lines.lock().await;
                        lines
                            .iter()
                            .rev()
                            .take(20)
                            .rev()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("\n")
                    };
                    tracing::error!(
                        ?status,
                        stderr = %last_lines,
                        "llama-server process exited unexpectedly"
                    );
                    self.child = None;
                    self.crash_count += 1;
                    self.set_state(ProcessState::Error);
                    true
                }
                Ok(None) => false, // Still running
                Err(e) => {
                    tracing::error!(error = %e, "Failed to check llama-server status");
                    false
                }
            }
        } else {
            false
        }
    }

    /// Get captured stderr lines (for crash diagnostics).
    pub async fn last_stderr(&self) -> Vec<String> {
        self.stderr_lines.lock().await.iter().cloned().collect()
    }

    pub fn crash_count(&self) -> u32 {
        self.crash_count
    }
}
