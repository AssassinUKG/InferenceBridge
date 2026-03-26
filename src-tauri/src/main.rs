use clap::{Parser, Subcommand};

/// InferenceBridge — Local LLM inference bridge
///
/// Run with no arguments to launch the GUI.
/// Use subcommands for headless operation (like Fox / Ollama).
#[derive(Parser, Debug)]
#[command(name = "inference-bridge", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    // Keep legacy --headless flag working for backwards compatibility
    /// (Deprecated) Use `serve` subcommand instead.
    #[arg(long, hide = true)]
    headless: bool,

    /// Port for the API server
    #[arg(long, short = 'p', global = true)]
    port: Option<u16>,

    /// GPU layers (-1 = all, 0 = CPU only)
    #[arg(long, global = true)]
    gpu_layers: Option<i32>,

    /// CPU threads (0 = auto-detect)
    #[arg(long, global = true)]
    threads: Option<u32>,

    /// Backend preference (auto, cuda, cpu)
    #[arg(long, global = true)]
    backend_preference: Option<String>,

    /// Verbose output (debug logging)
    #[arg(long, short = 'v', global = true)]
    verbose: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the API server (headless, no GUI).
    ///
    /// Example: inference-bridge serve --model qwen3 --port 8800
    Serve {
        /// Model to auto-load on startup (filename or partial match).
        #[arg(long, short = 'm')]
        model: Option<String>,

        /// Context size override (tokens).
        #[arg(long, short = 'c')]
        ctx_size: Option<u32>,

        /// Host to bind the API server on (default: from config, usually 127.0.0.1).
        /// Use 0.0.0.0 to expose on all interfaces.
        #[arg(long, short = 'H')]
        host: Option<String>,

        /// Additional directories to scan for .gguf model files (repeatable).
        /// Merged with directories from the config file.
        #[arg(long = "scan-dir", value_name = "DIR")]
        scan_dirs: Vec<std::path::PathBuf>,

        /// Default temperature for completions (0.0–2.0).
        #[arg(long, short = 't')]
        temperature: Option<f32>,

        /// Default top-p nucleus sampling (0.0–1.0).
        #[arg(long)]
        top_p: Option<f32>,

        /// Default top-k sampling (0 = disabled).
        #[arg(long)]
        top_k: Option<i32>,

        /// Default max tokens per response.
        #[arg(long)]
        max_tokens: Option<u32>,
    },

    /// List available models from configured scan directories.
    ///
    /// Example: inference-bridge models
    Models,

    /// One-shot inference: load a model, run a prompt, print the result, and exit.
    ///
    /// Example: inference-bridge run --model qwen3 "Explain what Rust is"
    Run {
        /// Model to load (filename or partial match).
        #[arg(long, short = 'm')]
        model: String,

        /// The prompt text.
        prompt: String,

        /// Context size override.
        #[arg(long, short = 'c')]
        ctx_size: Option<u32>,

        /// Maximum tokens to generate.
        #[arg(long, default_value = "2048")]
        max_tokens: u32,

        /// Temperature for sampling.
        #[arg(long, short = 't')]
        temperature: Option<f32>,

        /// Top-p nucleus sampling (0.0–1.0).
        #[arg(long)]
        top_p: Option<f32>,

        /// Top-k sampling (0 = disabled).
        #[arg(long)]
        top_k: Option<i32>,

        /// Random seed (-1 = random).
        #[arg(long)]
        seed: Option<i64>,
    },

    /// Show status of the running InferenceBridge API server.
    ///
    /// Example: inference-bridge status
    Status {
        /// API server port to query (default: 8800).
        #[arg(long, short = 'p', default_value = "8800")]
        port: u16,
    },

    /// Check for llama-server updates and download the latest version.
    ///
    /// Example: inference-bridge update
    Update,

    /// Benchmark a model: load, run a prompt, and print stats.
    ///
    /// Example: inference-bridge test-model --model qwen3.5-9b --ctx 8192 --prompt "What is 2 + 2?" --max-tokens 64 --temperature 0.1
    TestModel {
        /// Model to load (filename or partial match).
        #[arg(long, short = 'm')]
        model: String,

        /// The prompt text.
        #[arg(long)]
        prompt: String,

        /// Context size override.
        #[arg(long, short = 'c')]
        ctx_size: Option<u32>,

        /// Maximum tokens to generate.
        #[arg(long, default_value = "2048")]
        max_tokens: u32,

        /// Temperature for sampling.
        #[arg(long, short = 't')]
        temperature: Option<f32>,

        /// Top-p nucleus sampling (0.0–1.0).
        #[arg(long)]
        top_p: Option<f32>,

        /// Top-k sampling (0 = disabled).
        #[arg(long)]
        top_k: Option<i32>,

        /// Random seed (-1 = random).
        #[arg(long)]
        seed: Option<i64>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Serve {
            model,
            ctx_size,
            host,
            scan_dirs,
            temperature,
            top_p,
            top_k,
            max_tokens,
        }) => {
            inference_bridge_lib::run_headless(
                cli.port,
                host,
                model,
                ctx_size,
                cli.gpu_layers,
                cli.threads,
                cli.backend_preference.clone(),
                scan_dirs,
                cli.verbose,
                temperature,
                top_p,
                top_k,
                max_tokens,
            );
        }
        Some(Commands::Models) => {
            inference_bridge_lib::run_list_models();
        }
        Some(Commands::Run {
            model,
            prompt,
            ctx_size,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
        }) => {
            inference_bridge_lib::run_one_shot(
                model,
                prompt,
                ctx_size,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                cli.port,
                cli.gpu_layers,
                cli.threads,
                cli.backend_preference.clone(),
            );
        }
        Some(Commands::Status { port }) => {
            inference_bridge_lib::run_status(port);
        }
        Some(Commands::Update) => {
            inference_bridge_lib::run_update();
        }
        Some(Commands::TestModel {
            model,
            prompt,
            ctx_size,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
        }) => {
            inference_bridge_lib::run_model_test_cli(
                model,
                prompt,
                ctx_size,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                cli.port,
                cli.gpu_layers,
                cli.threads,
                cli.backend_preference.clone(),
                cli.verbose,
            );
        }
        None => {
            // Legacy --headless flag support
            if cli.headless {
                inference_bridge_lib::run_headless(
                    cli.port,
                    None,
                    None,
                    None,
                    cli.gpu_layers,
                    cli.threads,
                    cli.backend_preference.clone(),
                    vec![],
                    cli.verbose,
                    None,
                    None,
                    None,
                    None,
                );
            } else {
                // No command provided, launch GUI (no message)
                inference_bridge_lib::run();
            }
        }
    }
}
