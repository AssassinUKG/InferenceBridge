use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// InferenceBridge configuration loaded from `inference-bridge.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub models: ModelsConfig,
    pub process: ProcessConfig,
    pub ui: UiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
    pub autostart: bool,
    /// Server-level default temperature (overridden by per-request value).
    pub default_temperature: Option<f32>,
    /// Server-level default top-p.
    pub default_top_p: Option<f32>,
    /// Server-level default top-k.
    pub default_top_k: Option<i32>,
    /// Server-level default max output tokens.
    pub default_max_tokens: Option<u32>,
    /// Server-level default context size for model loading (0 = use model profile default).
    pub default_ctx_size: Option<u32>,
    /// Optional API key required on all public API requests (Bearer token).
    /// Empty string or None = no authentication required.
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    /// Directories to scan for .gguf model files.
    pub scan_dirs: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProcessConfig {
    /// Path to llama-server binary. Empty = auto-detect.
    pub llama_server_path: String,
    /// Number of GPU layers. -1 = all layers on GPU.
    pub gpu_layers: i32,
    /// Number of threads for generation. 0 = auto-detect.
    pub threads: u32,
    /// Number of threads for batch processing. 0 = same as threads.
    pub threads_batch: u32,
    /// Kill managed llama-server processes when the app exits.
    pub kill_on_exit: bool,
    /// Backend preference: "auto", "cuda", "cpu".
    pub backend_preference: String,
    /// Logical batch size for prompt processing (-b). 0 = default (2048).
    pub batch_size: u32,
    /// Physical micro-batch size (-ub). 0 = default (512).
    pub ubatch_size: u32,
    /// Enable Flash Attention (-fa).
    pub flash_attn: bool,
    /// Use memory-mapped model files (--mmap). Default true.
    pub use_mmap: bool,
    /// Force model to stay in RAM with mlock (--mlock).
    pub use_mlock: bool,
    /// Enable continuous batching (-cb). Default true.
    pub cont_batching: bool,
    /// Number of parallel inference slots (--parallel). Default 1.
    pub parallel_slots: u32,
    /// Primary GPU device index for multi-GPU (--main-gpu). Default 0.
    pub main_gpu: i32,
    /// KV cache defragmentation threshold (--defrag-thold). 0 = disabled.
    pub defrag_thold: f32,
    /// RoPE frequency scaling factor (--rope-freq-scale). 0 = auto.
    pub rope_freq_scale: f32,
    /// Port for the internal llama-server backend process. Default 8801.
    pub backend_port: u16,
    /// Maximum time (seconds) to wait for a model to load. Default 300 (5 min).
    pub model_load_timeout_secs: u64,
    /// Maximum time (seconds) to wait for the first token during inference. Default 300.
    pub first_token_timeout_secs: u64,
    /// Maximum time (seconds) to wait between tokens during inference. Default 120.
    pub inter_token_timeout_secs: u64,
    /// Health check polling interval (milliseconds) during model load. Default 150.
    pub health_poll_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub theme: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            models: ModelsConfig::default(),
            process: ProcessConfig::default(),
            ui: UiConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8800,
            host: "127.0.0.1".to_string(),
            autostart: true,
            default_temperature: None,
            default_top_p: None,
            default_top_k: None,
            default_max_tokens: None,
            default_ctx_size: None,
            api_key: None,
        }
    }
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            scan_dirs: vec![],
        }
    }
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            llama_server_path: String::new(),
            gpu_layers: -1,
            threads: 0,
            threads_batch: 0,
            kill_on_exit: true,
            backend_preference: "auto".to_string(),
            batch_size: 0,
            ubatch_size: 0,
            flash_attn: false,
            use_mmap: true,
            use_mlock: false,
            cont_batching: true,
            parallel_slots: 1,
            main_gpu: 0,
            defrag_thold: 0.1,
            rope_freq_scale: 0.0,
            backend_port: 8801,
            model_load_timeout_secs: 300,
            first_token_timeout_secs: 300,
            inter_token_timeout_secs: 120,
            health_poll_interval_ms: 150,
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
        }
    }
}

pub fn app_support_dir() -> PathBuf {
    let mut candidates = Vec::new();

    if let Some(local) = dirs::data_local_dir() {
        candidates.push(local.join("InferenceBridge"));
    }
    if let Some(data) = dirs::data_dir() {
        candidates.push(data.join("InferenceBridge"));
    }
    if let Some(config) = dirs::config_dir() {
        candidates.push(config.join("InferenceBridge"));
    }
    candidates.push(PathBuf::from(".inference-bridge"));

    for candidate in &candidates {
        if directory_is_writable(candidate) {
            return candidate.clone();
        }
    }

    PathBuf::from(".")
}

fn directory_is_writable(path: &PathBuf) -> bool {
    if std::fs::create_dir_all(path).is_err() {
        return false;
    }

    let probe = path.join(".write-test");
    match std::fs::write(&probe, b"ok") {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

impl AppConfig {
    /// Load config, checking multiple locations in priority order:
    /// 1. `./inference-bridge.toml` (current working directory / project root)
    /// 2. app support directory config file
    ///
    /// If no config file is found anywhere, falls back to defaults with
    /// auto-detected LM Studio model cache as a scan directory.
    pub fn load() -> Self {
        let candidates = Self::config_candidates();
        for path in &candidates {
            if path.exists() {
                tracing::info!("Loading config from: {}", path.display());
                match std::fs::read_to_string(path) {
                    Ok(contents) => match toml::from_str::<AppConfig>(&contents) {
                        Ok(mut config) => {
                            if config.models.scan_dirs.is_empty() {
                                config.models.scan_dirs = Self::detect_model_dirs();
                                if !config.models.scan_dirs.is_empty() {
                                    tracing::info!(
                                        "Auto-detected model directories: {:?}",
                                        config.models.scan_dirs
                                    );
                                }
                            }
                            return config;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse config at {}: {e}", path.display());
                        }
                    },
                    Err(e) => {
                        tracing::warn!("Failed to read config at {}: {e}", path.display());
                    }
                }
            }
        }

        tracing::warn!(
            "No config file found. Searched locations:\n{}",
            candidates
                .iter()
                .map(|p| format!("  - {}", p.display()))
                .collect::<Vec<_>>()
                .join("\n")
        );
        tracing::warn!(
            "Create one at {} or place inference-bridge.toml in the working directory.",
            Self::appdata_config_path().display()
        );

        let mut config = Self::default();
        config.models.scan_dirs = Self::detect_model_dirs();
        if config.models.scan_dirs.is_empty() {
            tracing::warn!(
                "No model directories detected. Configure scan_dirs in your config file."
            );
        } else {
            tracing::info!(
                "Auto-detected model directories: {:?}",
                config.models.scan_dirs
            );
        }
        config
    }

    /// Save config to disk.
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::appdata_config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&path, contents)?;
        Ok(())
    }

    fn config_candidates() -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        candidates.push(PathBuf::from("inference-bridge.toml"));
        if let Ok(cwd) = std::env::current_dir() {
            let cwd_path = cwd.join("inference-bridge.toml");
            if !candidates.contains(&cwd_path) {
                candidates.push(cwd_path);
            }
        }
        candidates.push(Self::appdata_config_path());
        candidates
    }

    fn appdata_config_path() -> PathBuf {
        app_support_dir().join("inference-bridge.toml")
    }

    fn detect_model_dirs() -> Vec<PathBuf> {
        let mut dirs_found = BTreeSet::new();

        for candidate in Self::candidate_model_dirs() {
            if candidate.is_dir() {
                tracing::info!("Found model directory candidate: {}", candidate.display());
                dirs_found.insert(candidate);
            }
        }

        for discovered in Self::discover_model_dirs_with_gguf() {
            tracing::info!("Found GGUF model directory: {}", discovered.display());
            dirs_found.insert(discovered);
        }

        dirs_found.into_iter().collect()
    }

    fn candidate_model_dirs() -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        if let Some(home) = dirs::home_dir() {
            candidates.push(home.join(".cache").join("lm-studio").join("models"));
            candidates.push(home.join(".lmstudio").join("models"));
            candidates.push(home.join("models"));
            candidates.push(home.join("Models"));
            candidates.push(home.join("gguf-models"));
            candidates.push(home.join("Documents").join("models"));
            candidates.push(home.join("Documents").join("Models"));
            candidates.push(home.join("Documents").join("gguf-models"));
            candidates.push(home.join("Downloads").join("models"));
            candidates.push(home.join("Downloads").join("Models"));
            candidates.push(home.join("Downloads").join("gguf-models"));
            candidates.push(home.join("Desktop").join("models"));
            candidates.push(home.join("Desktop").join("Models"));
        }

        if let Some(local) = dirs::data_local_dir() {
            candidates.push(local.join("LM Studio").join("models"));
            candidates.push(local.join("lm-studio").join("models"));
            candidates.push(local.join("nomic.ai").join("GPT4All").join("models"));
            candidates.push(local.join("Ollama").join("models"));
        }

        if let Some(data) = dirs::data_dir() {
            candidates.push(data.join("LM Studio").join("models"));
            candidates.push(data.join("lm-studio").join("models"));
        }

        candidates
    }

    fn discover_model_dirs_with_gguf() -> Vec<PathBuf> {
        let mut roots = Vec::new();

        if let Some(home) = dirs::home_dir() {
            roots.push(home.join("Documents"));
            roots.push(home.join("Downloads"));
            roots.push(home.join("Desktop"));
            roots.push(home.join(".cache").join("lm-studio"));
        }

        if let Some(local) = dirs::data_local_dir() {
            roots.push(local.join("LM Studio"));
            roots.push(local.join("lm-studio"));
            roots.push(local.join("nomic.ai").join("GPT4All"));
        }

        let mut found = BTreeSet::new();
        for root in roots {
            Self::collect_gguf_dirs(&root, 0, 3, &mut found);
        }

        found.into_iter().collect()
    }

    fn collect_gguf_dirs(
        dir: &PathBuf,
        depth: usize,
        max_depth: usize,
        found: &mut BTreeSet<PathBuf>,
    ) {
        if depth > max_depth || !dir.is_dir() {
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return,
        };

        let mut has_gguf = false;
        let mut child_dirs = Vec::new();

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                child_dirs.push(path);
                continue;
            }

            if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
            {
                has_gguf = true;
            }
        }

        if has_gguf {
            found.insert(dir.clone());
        }

        for child in child_dirs {
            Self::collect_gguf_dirs(&child, depth + 1, max_depth, found);
        }
    }
}
