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

use crate::models::profiles::{ModelFamily, ModelProfile};

fn system_command(program: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

fn binary_command(program: &Path) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    cmd
}

pub const LLAMA_FLAG_DETECTION_TARGETS: &[&str] = &[
    "--jinja",
    "--reasoning",
    "--reasoning-preserve",
    "--chat-template",
    "--chat-template-file",
    "--chat-template-kwargs",
    "--parallel",
    "--slots",
    "--ctx-size",
    "--mmproj",
    "--fit",
    "--cache-ram",
    "-ctxcp",
    "--flash-attn",
    "--cont-batching",
    "--cache-type-k",
    "--cache-type-v",
    "--kv-unified",
    "--no-warmup",
    "--ctx-shift",
    "--tensor-split",
    "-md",
    "--spec-type",
    "--spec-draft-n-max",
    "--draft-max",
    "--draft-min",
    "--draft-p-min",
];

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LlamaFlagSupport {
    pub checked: bool,
    pub binary_path: Option<String>,
    pub supported_flags: Vec<String>,
    pub missing_critical_flags: Vec<String>,
    pub error: Option<String>,
}

pub fn detect_llama_flag_support(binary_path: Option<&Path>) -> LlamaFlagSupport {
    let Some(binary_path) = binary_path else {
        return LlamaFlagSupport {
            checked: false,
            binary_path: None,
            supported_flags: Vec::new(),
            missing_critical_flags: LLAMA_FLAG_DETECTION_TARGETS
                .iter()
                .map(|flag| (*flag).to_string())
                .collect(),
            error: Some("llama-server binary not found".to_string()),
        };
    };

    let mut last_error = None;
    let help_text = ["--help", "-h"].into_iter().find_map(|help_arg| {
        match binary_command(binary_path).arg(help_arg).output() {
            Ok(output)
                if output.status.success()
                    || !output.stdout.is_empty()
                    || !output.stderr.is_empty() =>
            {
                let mut text = String::from_utf8_lossy(&output.stdout).to_string();
                text.push_str(&String::from_utf8_lossy(&output.stderr));
                Some(text)
            }
            Ok(output) => {
                last_error = Some(format!("{help_arg} exited with {}", output.status));
                None
            }
            Err(error) => {
                last_error = Some(error.to_string());
                None
            }
        }
    });

    match help_text {
        Some(text) => {
            let (supported_flags, missing_critical_flags) =
                llama_flags_from_help_text(&text, LLAMA_FLAG_DETECTION_TARGETS);
            LlamaFlagSupport {
                checked: true,
                binary_path: Some(binary_path.to_string_lossy().to_string()),
                supported_flags,
                missing_critical_flags,
                error: None,
            }
        }
        None => LlamaFlagSupport {
            checked: false,
            binary_path: Some(binary_path.to_string_lossy().to_string()),
            supported_flags: Vec::new(),
            missing_critical_flags: Vec::new(),
            error: Some(format!(
                "Unable to inspect llama-server flags{}",
                last_error
                    .map(|error| format!(": {error}"))
                    .unwrap_or_default()
            )),
        },
    }
}

pub fn unsupported_detected_launch_flags(
    args: &[String],
    support: &LlamaFlagSupport,
) -> Vec<String> {
    if !support.checked || support.missing_critical_flags.is_empty() {
        return Vec::new();
    }
    let missing = support
        .missing_critical_flags
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let mut unsupported = args
        .iter()
        .filter(|arg| missing.contains(arg.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    unsupported.sort();
    unsupported.dedup();
    unsupported
}

fn llama_flags_from_help_text(help_text: &str, flags: &[&str]) -> (Vec<String>, Vec<String>) {
    let mut supported = Vec::new();
    let mut missing = Vec::new();
    for flag in flags {
        if help_text.contains(flag) {
            supported.push((*flag).to_string());
        } else {
            missing.push((*flag).to_string());
        }
    }
    (supported, missing)
}

fn filename_supports_vision(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    crate::models::overrides::detect_effective_profile(&name).supports_vision
}

fn reasoning_mode_disables_thinking(reasoning_mode: Option<&str>) -> bool {
    matches!(
        reasoning_mode.map(|value| value.trim().to_ascii_lowercase()),
        Some(value)
            if matches!(
                value.as_str(),
                "off" | "false" | "none" | "disabled" | "disable" | "0" | "no"
            )
    )
}

fn normalize_chat_template_kwargs_for_profile(
    kwargs_json: Option<&str>,
    reasoning_mode: Option<&str>,
    profile: &ModelProfile,
) -> anyhow::Result<Option<String>> {
    let trimmed = kwargs_json.map(str::trim).filter(|value| !value.is_empty());
    let should_disable_template_thinking =
        matches!(profile.family, ModelFamily::Gemma4 | ModelFamily::Qwen3)
            && reasoning_mode_disables_thinking(reasoning_mode);

    if !should_disable_template_thinking {
        return Ok(trimmed.map(ToOwned::to_owned));
    }

    let mut value = match trimmed {
        Some(raw) => serde_json::from_str::<serde_json::Value>(raw)
            .map_err(|error| anyhow::anyhow!("Invalid chat_template_kwargs_json: {error}"))?,
        None => serde_json::json!({}),
    };

    let Some(object) = value.as_object_mut() else {
        anyhow::bail!("chat_template_kwargs_json must be a JSON object");
    };
    object
        .entry("enable_thinking".to_string())
        .or_insert(serde_json::Value::Bool(false));

    Ok(Some(serde_json::to_string(&value)?))
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

fn model_family_token(path: &Path) -> Option<&'static str> {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    [
        "gemma", "qwen", "llava", "mistral", "phi", "internvl", "minicpm",
    ]
    .into_iter()
    .find(|family| name.contains(family))
}

fn find_mmproj_for_model(model_path: &Path) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    let model_family = model_family_token(model_path);
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
    candidates.into_iter().find(|candidate| {
        let score = shared_token_score(model_path, candidate);
        if score == 0 {
            return false;
        }
        model_family.map_or(true, |family| {
            candidate
                .file_name()
                .map(|name| name.to_string_lossy().to_lowercase().contains(family))
                .unwrap_or(false)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::{
        filename_supports_vision, find_mmproj_for_model, llama_flags_from_help_text,
        normalize_chat_template_kwargs_for_profile, parse_llama_load_progress, LaunchConfig,
        LlamaProcess,
    };
    use crate::models::profiles::ModelProfile;

    #[test]
    fn gemma4_12b_filename_is_vision_capable() {
        assert!(filename_supports_vision(std::path::Path::new(
            "gemma-4-12B-it-Q4_K_M.gguf"
        )));
    }

    #[test]
    fn detects_missing_llama_flags_from_help_text() {
        let help = r#"
            --jinja
            --reasoning MODE
            --chat-template-kwargs JSON
            --cache-type-k TYPE
        "#;
        let (supported, missing) =
            llama_flags_from_help_text(help, &["--jinja", "--reasoning", "--kv-unified"]);

        assert_eq!(supported, vec!["--jinja", "--reasoning"]);
        assert_eq!(missing, vec!["--kv-unified"]);
    }

    #[test]
    fn gemma4_reasoning_off_sets_enable_thinking_false() {
        let profile = ModelProfile::detect("gemma-4-26B-A4B-it-QAT-Q4_0.gguf");
        let kwargs =
            normalize_chat_template_kwargs_for_profile(None, Some("off"), &profile).unwrap();

        assert_eq!(kwargs.as_deref(), Some(r#"{"enable_thinking":false}"#));
    }

    #[test]
    fn qwen36_reasoning_off_sets_enable_thinking_false_without_clobbering_kwargs() {
        let profile = ModelProfile::detect("Qwen3.6-27B-Q4_K_M.gguf");
        let kwargs = normalize_chat_template_kwargs_for_profile(
            Some(r#"{"preserve_thinking":true}"#),
            Some("none"),
            &profile,
        )
        .unwrap()
        .expect("kwargs should exist");
        let parsed: serde_json::Value = serde_json::from_str(&kwargs).unwrap();

        assert_eq!(parsed["enable_thinking"], false);
        assert_eq!(parsed["preserve_thinking"], true);
    }

    #[test]
    fn explicit_enable_thinking_kwargs_are_preserved() {
        let profile = ModelProfile::detect("Qwen3.6-27B-Q4_K_M.gguf");
        let kwargs = normalize_chat_template_kwargs_for_profile(
            Some(r#"{"enable_thinking":true}"#),
            Some("off"),
            &profile,
        )
        .unwrap();

        assert_eq!(kwargs.as_deref(), Some(r#"{"enable_thinking":true}"#));
    }

    #[test]
    fn finds_gemma4_mmproj_sidecar() {
        let dir = std::env::temp_dir().join(format!(
            "inference-bridge-mmproj-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp model dir");
        let model = dir.join("gemma-4-12B-it-Q4_K_M.gguf");
        let mmproj = dir.join("mmproj-gemma-4-12B-it-BF16.gguf");
        std::fs::write(&model, b"").expect("write model placeholder");
        std::fs::write(&mmproj, b"").expect("write mmproj placeholder");

        let found = find_mmproj_for_model(&model).expect("mmproj should be found");
        assert_eq!(found.file_name(), mmproj.file_name());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_real_load_progress_from_stderr() {
        // GPU-offload counter → real fraction in the 0.20–0.65 band.
        let p = parse_llama_load_progress("load_tensors: offloaded 49/49 layers to GPU").unwrap();
        assert!((p - 0.65).abs() < 1e-4, "got {p}");
        let half = parse_llama_load_progress("offloaded 24/48 layers to GPU").unwrap();
        assert!((half - 0.425).abs() < 1e-3, "got {half}");

        // Milestones are monotonic checkpoints.
        assert_eq!(
            parse_llama_load_progress(
                "load_tensors: loading model tensors, this can take a while..."
            ),
            Some(0.15)
        );
        assert_eq!(
            parse_llama_load_progress("main: server is listening on 127.0.0.1:8080"),
            Some(0.95)
        );

        // Explicit percentage if a build prints one.
        let pct = parse_llama_load_progress("loading: 50%").unwrap();
        assert!((pct - 0.525).abs() < 1e-3, "got {pct}");

        // Unrelated lines carry no signal.
        assert_eq!(
            parse_llama_load_progress("ggml_cuda_init: found 1 CUDA device"),
            None
        );
    }

    #[test]
    fn ignores_unrelated_mmproj_sidecar() {
        let dir = std::env::temp_dir().join(format!(
            "inference-bridge-mmproj-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp model dir");
        let model = dir.join("gemma-4-12B-it-Q4_K_M.gguf");
        let mmproj = dir.join("mmproj-qwen2-vl-7B-BF16.gguf");
        std::fs::write(&model, b"").expect("write model placeholder");
        std::fs::write(&mmproj, b"").expect("write mmproj placeholder");

        assert!(find_mmproj_for_model(&model).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn diffusion_gemma_preview_uses_diffusion_cli_runner() {
        let dir = std::env::temp_dir().join(format!(
            "inference-bridge-diffusion-preview-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("diffusiongemma-26B-A4B-it-Q4_K_M.gguf");
        let runner = dir.join(if cfg!(windows) {
            "llama-diffusion-cli.exe"
        } else {
            "llama-diffusion-cli"
        });
        std::fs::write(&model, b"").expect("write model placeholder");
        std::fs::write(&runner, b"").expect("write runner placeholder");

        let mut process = LlamaProcess::new();
        process.set_diffusion_cli_path(runner.clone());
        let preview = process
            .build_args_preview(&LaunchConfig {
                model_path: model.clone(),
                hf_repo: None,
                hf_file: None,
                context_size: None,
                gpu_layers: -1,
                threads: 0,
                threads_batch: 0,
                port: 8800,
                backend_preference: "auto".to_string(),
                batch_size: 0,
                ubatch_size: 0,
                flash_attn: true,
                use_mmap: true,
                use_mlock: false,
                cont_batching: true,
                parallel_slots: 1,
                main_gpu: 0,
                defrag_thold: 0.1,
                rope_freq_scale: 0.0,
                fit_mode: None,
                cache_ram_mb: None,
                ctxcp: None,
                use_jinja: false,
                reasoning_mode: None,
                reasoning_preserve: false,
                template_mode: "repo".to_string(),
                template_source: None,
                template_file: None,
                template_name: None,
                chat_template_kwargs_json: None,
                extra_args: vec![],
                cache_type_k: "q8_0".to_string(),
                cache_type_v: "q8_0".to_string(),
                kv_unified: true,
                no_warmup: false,
                ctx_shift: false,
                tensor_split: vec![],
                draft_model_path: String::new(),
                spec_type: String::new(),
                spec_draft_n_max: 0,
                draft_max_tokens: 0,
                draft_min_tokens: 0,
                draft_p_min: 0.0,
                diffusion_n_predict: 2048,
                diffusion_kv_cache: "auto".to_string(),
                diffusion_visual: true,
                diffusion_extra_args: vec!["--diffusion-max-steps".to_string(), "48".to_string()],
                attach_mmproj: true,
            })
            .expect("build diffusion preview");

        assert_eq!(preview.runtime, "diffusion-cli");
        assert_eq!(preview.server_path, runner.to_string_lossy());
        assert_eq!(preview.port, 0);
        assert!(preview
            .args
            .windows(2)
            .any(|pair| pair[0] == "-m" && pair[1] == model.to_string_lossy()));
        assert!(preview
            .args
            .windows(2)
            .any(|pair| pair[0] == "-ngl" && pair[1] == "999"));
        assert!(preview.args.contains(&"-cnv".to_string()));
        assert!(preview
            .args
            .windows(2)
            .any(|pair| pair[0] == "-n" && pair[1] == "2048"));
        assert!(preview
            .args
            .windows(2)
            .any(|pair| { pair[0] == "--diffusion-kv-cache" && pair[1] == "auto" }));
        assert!(preview.args.contains(&"--diffusion-visual".to_string()));
        assert!(preview
            .args
            .windows(2)
            .any(|pair| pair[0] == "--diffusion-max-steps" && pair[1] == "48"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn test_launch_config(model_path: std::path::PathBuf) -> LaunchConfig {
        LaunchConfig {
            model_path,
            hf_repo: None,
            hf_file: None,
            context_size: None,
            gpu_layers: -1,
            threads: 0,
            threads_batch: 0,
            port: 8800,
            backend_preference: "auto".to_string(),
            batch_size: 0,
            ubatch_size: 0,
            flash_attn: true,
            use_mmap: true,
            use_mlock: false,
            cont_batching: true,
            parallel_slots: 1,
            main_gpu: 0,
            defrag_thold: 0.1,
            rope_freq_scale: 0.0,
            fit_mode: None,
            cache_ram_mb: None,
            ctxcp: None,
            use_jinja: false,
            reasoning_mode: None,
            reasoning_preserve: false,
            template_mode: "repo".to_string(),
            template_source: None,
            template_file: None,
            template_name: None,
            chat_template_kwargs_json: None,
            extra_args: vec![],
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "q8_0".to_string(),
            kv_unified: true,
            no_warmup: false,
            ctx_shift: false,
            tensor_split: vec![],
            draft_model_path: String::new(),
            spec_type: String::new(),
            spec_draft_n_max: 0,
            draft_max_tokens: 0,
            draft_min_tokens: 0,
            draft_p_min: 0.0,
            diffusion_n_predict: 2048,
            diffusion_kv_cache: "auto".to_string(),
            diffusion_visual: true,
            diffusion_extra_args: vec![],
            attach_mmproj: true,
        }
    }

    #[test]
    fn validates_missing_draft_model_before_launch() {
        let dir = std::env::temp_dir().join(format!(
            "inference-bridge-missing-draft-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("gemma-4-26B-A4B-it-QAT-Q4_0.gguf");
        std::fs::write(&model, b"").expect("write model placeholder");

        let missing_draft = dir.join("missing-draft.gguf");
        let mut config = test_launch_config(model);
        config.draft_model_path = missing_draft.to_string_lossy().to_string();
        config.spec_type = "draft-mtp".to_string();

        let error = LlamaProcess::validate_launch_config(&config)
            .expect_err("missing draft model should fail validation")
            .to_string();

        assert!(error.contains("Draft model file does not exist"));
        assert!(error.contains("missing-draft.gguf"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn self_mtp_emits_spec_type_without_draft_model() {
        // A single MTP GGUF (no separate -md draft model) must still produce
        // --spec-type / --spec-draft-n-max, otherwise self-MTP never activates.
        let dir = std::env::temp_dir().join(format!(
            "inference-bridge-self-mtp-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("Qwen3.6-27B-MTP-UD-Q4_K_XL.gguf");
        std::fs::write(&model, b"").expect("write model placeholder");
        let server = dir.join("llama-server.exe");
        std::fs::write(&server, b"").expect("write server placeholder");

        let mut process = LlamaProcess::new();
        process.set_server_path(server.clone());

        let mut config = test_launch_config(model);
        config.spec_type = "draft-mtp".to_string();
        config.spec_draft_n_max = 3;
        // draft_model_path intentionally left empty (self-MTP).

        let preview = process
            .build_args_preview(&config)
            .expect("build self-MTP preview");

        assert!(
            preview
                .args
                .windows(2)
                .any(|pair| pair[0] == "--spec-type" && pair[1] == "draft-mtp"),
            "expected --spec-type draft-mtp, got {:?}",
            preview.args
        );
        assert!(
            preview
                .args
                .windows(2)
                .any(|pair| pair[0] == "--spec-draft-n-max" && pair[1] == "3"),
            "expected --spec-draft-n-max 3, got {:?}",
            preview.args
        );
        assert!(
            !preview.args.iter().any(|arg| arg == "-md"),
            "self-MTP must not emit a -md draft model, got {:?}",
            preview.args
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reasoning_preserve_emits_llama_server_flag() {
        let dir = std::env::temp_dir().join(format!(
            "inference-bridge-reasoning-preserve-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("Qwen3.6-27B-Q4_K_M.gguf");
        std::fs::write(&model, b"").expect("write model placeholder");
        let server = dir.join("llama-server.exe");
        std::fs::write(&server, b"").expect("write server placeholder");

        let mut process = LlamaProcess::new();
        process.set_server_path(server);

        let mut config = test_launch_config(model);
        config.reasoning_preserve = true;

        let preview = process
            .build_args_preview(&config)
            .expect("build reasoning preserve preview");

        assert!(
            preview.args.contains(&"--reasoning-preserve".to_string()),
            "expected --reasoning-preserve, got {:?}",
            preview.args
        );
        assert!(preview.reasoning_preserve);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dflash_emits_spec_type_with_draft_model() {
        let dir = std::env::temp_dir().join(format!(
            "inference-bridge-dflash-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("Qwen3.6-27B-Q4_K_M.gguf");
        let draft = dir.join("Qwen3.6-27B-DFlash-draft.gguf");
        std::fs::write(&model, b"").expect("write model placeholder");
        std::fs::write(&draft, b"").expect("write draft placeholder");
        let server = dir.join("llama-server.exe");
        std::fs::write(&server, b"").expect("write server placeholder");

        let mut process = LlamaProcess::new();
        process.set_server_path(server);

        let mut config = test_launch_config(model);
        config.draft_model_path = draft.to_string_lossy().to_string();
        config.spec_type = "draft-dflash".to_string();
        config.spec_draft_n_max = 8;

        let preview = process
            .build_args_preview(&config)
            .expect("build DFlash preview");

        assert!(
            preview
                .args
                .windows(2)
                .any(|pair| pair[0] == "-md" && pair[1] == draft.to_string_lossy()),
            "expected -md draft model, got {:?}",
            preview.args
        );
        assert!(
            preview
                .args
                .windows(2)
                .any(|pair| pair[0] == "--spec-type" && pair[1] == "draft-dflash"),
            "expected --spec-type draft-dflash, got {:?}",
            preview.args
        );
        assert!(
            preview
                .args
                .windows(2)
                .any(|pair| pair[0] == "--spec-draft-n-max" && pair[1] == "8"),
            "expected --spec-draft-n-max 8, got {:?}",
            preview.args
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dflash_requires_draft_model_path() {
        let dir = std::env::temp_dir().join(format!(
            "inference-bridge-dflash-missing-draft-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let model = dir.join("Qwen3.6-27B-Q4_K_M.gguf");
        std::fs::write(&model, b"").expect("write model placeholder");

        let mut config = test_launch_config(model);
        config.spec_type = "draft-dflash".to_string();

        let error = LlamaProcess::validate_launch_config(&config)
            .expect_err("DFlash without a draft model should fail")
            .to_string();

        assert!(error.contains("DFlash speculative decoding requires a draft model path"));

        let _ = std::fs::remove_dir_all(&dir);
    }
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
    pub hf_repo: Option<String>,
    pub hf_file: Option<String>,
    /// When `None`, llama-server uses the model's native context window.
    pub context_size: Option<u32>,
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
    pub fit_mode: Option<String>,
    pub cache_ram_mb: Option<u32>,
    pub ctxcp: Option<u32>,
    pub use_jinja: bool,
    pub reasoning_mode: Option<String>,
    pub reasoning_preserve: bool,
    pub template_mode: String,
    pub template_source: Option<String>,
    pub template_file: Option<PathBuf>,
    pub template_name: Option<String>,
    pub chat_template_kwargs_json: Option<String>,
    pub extra_args: Vec<String>,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub kv_unified: bool,
    pub no_warmup: bool,
    pub ctx_shift: bool,
    pub tensor_split: Vec<f32>,
    /// Draft model path for speculative decoding (-md). Empty = disabled.
    pub draft_model_path: String,
    pub spec_type: String,
    pub spec_draft_n_max: u32,
    pub draft_max_tokens: u32,
    pub draft_min_tokens: u32,
    pub draft_p_min: f32,
    pub diffusion_n_predict: u32,
    pub diffusion_kv_cache: String,
    pub diffusion_visual: bool,
    pub diffusion_extra_args: Vec<String>,
    pub attach_mmproj: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LaunchPreview {
    pub runtime: String,
    pub server_path: String,
    pub model_path: String,
    pub hf_repo: Option<String>,
    pub hf_file: Option<String>,
    pub mmproj_path: Option<String>,
    pub backend_preference: String,
    /// Actual context size: either explicitly requested or discovered from server after launch.
    pub context_size: Option<u32>,
    pub port: u16,
    pub parallel_slots: u32,
    pub fit_mode: Option<String>,
    pub cache_ram_mb: Option<u32>,
    pub ctxcp: Option<u32>,
    pub use_jinja: bool,
    pub reasoning_mode: Option<String>,
    pub reasoning_preserve: bool,
    pub template_mode: String,
    pub template_source: Option<String>,
    pub template_path: Option<String>,
    pub template_name: Option<String>,
    pub chat_template_kwargs_json: Option<String>,
    pub draft_model_path: String,
    pub spec_type: String,
    pub spec_draft_n_max: u32,
    pub draft_max_tokens: u32,
    pub draft_min_tokens: u32,
    pub draft_p_min: f32,
    pub args: Vec<String>,
}

/// Resources extracted from a live LlamaProcess for async shutdown outside the AppState lock.
/// Produced by `LlamaProcess::begin_shutdown()`; consumed by `PendingShutdown::complete()`.
pub struct PendingShutdown {
    child: Child,
    port: u16,
    pid: Option<u32>,
    io_tasks: Vec<JoinHandle<()>>,
    detected_backend: Arc<TokioMutex<Option<String>>>,
    stderr_lines: Arc<TokioMutex<VecDeque<String>>>,
    load_progress: Arc<TokioMutex<Option<f32>>>,
}

impl PendingShutdown {
    /// Perform the slow async portion of shutdown (HTTP graceful stop, kill, port-release
    /// wait) without holding the AppState write lock.
    pub async fn complete(mut self) {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap_or_default();
        let shutdown_url = format!("http://127.0.0.1:{}/shutdown", self.port);
        let _ = client.post(&shutdown_url).send().await;

        let exit = tokio::time::timeout(Duration::from_secs(2), self.child.wait()).await;
        if exit.is_err() {
            tracing::warn!("llama-server did not exit gracefully, killing");
            let _ = self.child.kill().await;
            let _ = self.child.wait().await;
        } else {
            tracing::info!("llama-server exited gracefully");
        }

        if let Some(pid) = self.pid {
            let _ = LlamaProcess::force_kill_process_tree(pid);
        }

        *self.detected_backend.lock().await = None;
        self.stderr_lines.lock().await.clear();
        *self.load_progress.lock().await = None;

        for handle in self.io_tasks.drain(..) {
            handle.abort();
        }

        LlamaProcess::wait_for_port_release(self.port, Duration::from_millis(1500)).await;
    }
}

/// Manages the llama-server child process.
pub struct LlamaProcess {
    child: Option<Child>,
    state: ProcessState,
    llama_server_path: Option<PathBuf>,
    llama_diffusion_cli_path: Option<PathBuf>,
    current_model: Option<String>,
    current_pid: Option<u32>,
    current_port: u16,
    crash_count: u32,
    last_exit_status: Option<String>,
    state_tx: watch::Sender<ProcessState>,
    state_rx: watch::Receiver<ProcessState>,
    /// GPU backend detected from server stderr (e.g. "CUDA", "Vulkan", "CPU").
    detected_backend: Arc<TokioMutex<Option<String>>>,
    /// Recent stderr lines captured from llama-server (ring buffer for crash diagnostics).
    stderr_lines: Arc<TokioMutex<VecDeque<String>>>,
    /// Handles for background I/O reader tasks — aborted on shutdown to prevent leaks.
    io_tasks: Vec<JoinHandle<()>>,
    /// App handle for pushing live GUI events (llama-server output + state
    /// changes) so the UI updates in real time instead of polling.
    app_handle: Option<tauri::AppHandle>,
    /// Real load progress (0.0–1.0) parsed from llama-server stderr during the
    /// current launch. Monotonic; reset to `None` on each launch/shutdown.
    load_progress: Arc<TokioMutex<Option<f32>>>,
}

/// Parse a coarse load-progress fraction (0.0–1.0) from a single llama-server
/// stderr line, or `None` if the line carries no progress signal.
///
/// llama.cpp doesn't print a continuous percentage during mmap loads, so we map
/// its stable milestone lines to checkpoints and extract a real fraction from
/// the GPU-offload counter (`offloaded X/Y layers`) — the slow part of a load.
/// Callers keep the maximum so the reported progress is monotonic.
pub fn parse_llama_load_progress(line: &str) -> Option<f32> {
    let lower = line.to_lowercase();

    // Real sub-progress during GPU offload, e.g. "offloaded 33/49 layers to GPU".
    if let Some(rest) = lower.split("offloaded ").nth(1) {
        if let Some(frac) = rest.split(" layers").next() {
            if let Some((done, total)) = frac.split_once('/') {
                if let (Ok(done), Ok(total)) =
                    (done.trim().parse::<f32>(), total.trim().parse::<f32>())
                {
                    if total > 0.0 {
                        return Some(0.20 + 0.45 * (done / total).clamp(0.0, 1.0));
                    }
                }
            }
        }
    }

    // Explicit percentage if a build happens to print one (e.g. "... 42%").
    if let Some(pct) = extract_trailing_percent(&lower) {
        return Some(0.10 + 0.85 * (pct / 100.0).clamp(0.0, 1.0));
    }

    // Stable milestone lines → coarse checkpoints.
    let checkpoint = if lower.contains("loading model tensors") {
        0.15
    } else if lower.contains("model buffer size") {
        0.68
    } else if lower.contains("kv cache") || lower.contains("kv self size") {
        0.82
    } else if lower.contains("warming up") || lower.contains("warmup") {
        0.90
    } else if lower.contains("model loaded")
        || lower.contains("server is listening")
        || lower.contains("starting the main loop")
    {
        0.95
    } else {
        return None;
    };
    Some(checkpoint)
}

/// Extract a number immediately preceding a `%` sign, e.g. `"42%"` → `42.0`.
fn extract_trailing_percent(s: &str) -> Option<f32> {
    let pos = s.find('%')?;
    let bytes = s.as_bytes();
    let mut start = pos;
    while start > 0 {
        let c = bytes[start - 1];
        if c.is_ascii_digit() || c == b'.' {
            start -= 1;
        } else {
            break;
        }
    }
    let token = s[start..pos].trim_matches('.');
    if token.is_empty() {
        return None;
    }
    token.parse::<f32>().ok()
}

impl LlamaProcess {
    async fn wait_for_port_release(port: u16, timeout: Duration) {
        let started = std::time::Instant::now();
        while started.elapsed() < timeout {
            if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub fn validate_launch_config(config: &LaunchConfig) -> anyhow::Result<()> {
        if config.hf_repo.as_deref().unwrap_or("").trim().is_empty() && !config.model_path.exists()
        {
            anyhow::bail!("Model file does not exist: {}", config.model_path.display());
        }
        let draft_model_path = config.draft_model_path.trim();
        if config.spec_type.trim() == "draft-dflash" && draft_model_path.is_empty() {
            anyhow::bail!("DFlash speculative decoding requires a draft model path (-md)");
        }
        if !draft_model_path.is_empty() {
            let draft_path = Path::new(draft_model_path);
            if !draft_path.exists() {
                anyhow::bail!("Draft model file does not exist: {}", draft_path.display());
            }
            if !draft_path.is_file() {
                anyhow::bail!("Draft model path is not a file: {}", draft_path.display());
            }
        }
        if config.context_size == Some(0) {
            anyhow::bail!("Context size must be greater than 0 when specified");
        }
        if config.parallel_slots == 0 {
            anyhow::bail!("Parallel slots must be at least 1");
        }
        // port == 0 is valid: means "auto-assign a free ephemeral port at launch time"
        if config.main_gpu < 0 {
            anyhow::bail!("Main GPU index cannot be negative");
        }
        if config.defrag_thold < 0.0 {
            anyhow::bail!("Defrag threshold cannot be negative");
        }
        if config.rope_freq_scale < 0.0 {
            anyhow::bail!("RoPE frequency scale cannot be negative");
        }
        if let Some(cache_ram_mb) = config.cache_ram_mb {
            if cache_ram_mb == 0 {
                anyhow::bail!("cache_ram_mb must be greater than 0 when specified");
            }
        }
        if let Some(ctxcp) = config.ctxcp {
            if ctxcp == 0 {
                anyhow::bail!("ctxcp must be greater than 0 when specified");
            }
        }
        Ok(())
    }

    pub fn build_args_preview(&self, config: &LaunchConfig) -> anyhow::Result<LaunchPreview> {
        Self::validate_launch_config(config)?;

        let profile = ModelProfile::detect(&format!(
            "{} {} {}",
            config.model_path.to_string_lossy(),
            config.hf_repo.as_deref().unwrap_or_default(),
            config.hf_file.as_deref().unwrap_or_default()
        ));

        if matches!(profile.family, ModelFamily::DiffusionGemma) {
            return self.build_diffusion_args_preview(config);
        }

        let server_path = self
            .find_server_binary_with_preference(&config.backend_preference)
            .ok_or_else(|| anyhow::anyhow!("llama-server binary not found"))?;

        let mmproj_path = if config.attach_mmproj && filename_supports_vision(&config.model_path) {
            find_mmproj_for_model(&config.model_path)
        } else {
            None
        };

        let chat_template_kwargs_json = normalize_chat_template_kwargs_for_profile(
            config.chat_template_kwargs_json.as_deref(),
            config.reasoning_mode.as_deref(),
            &profile,
        )?;

        let mut args = vec![];

        if let Some(hf_repo) = config
            .hf_repo
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            args.push("-hf".to_string());
            let hf_ref = match config
                .hf_file
                .as_ref()
                .filter(|value| !value.trim().is_empty())
            {
                Some(file) => format!("{hf_repo}:{file}"),
                None => hf_repo.clone(),
            };
            args.push(hf_ref);
        } else {
            args.push("--model".to_string());
            args.push(config.model_path.to_string_lossy().to_string());
        }

        args.extend([
            "--port".to_string(),
            config.port.to_string(),
            "--parallel".to_string(),
            config.parallel_slots.max(1).to_string(),
            "--slots".to_string(),
        ]);

        // Only pass --ctx-size when explicitly specified; otherwise let
        // llama-server use the model's native context window.
        if let Some(ctx) = config.context_size {
            args.push("--ctx-size".to_string());
            args.push(ctx.to_string());
        }

        if let Some(mmproj) = &mmproj_path {
            args.push("--mmproj".to_string());
            args.push(mmproj.to_string_lossy().to_string());
        }
        if let Some(fit_mode) = config
            .fit_mode
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            args.push("--fit".to_string());
            args.push(fit_mode.clone());
        }
        if let Some(cache_ram_mb) = config.cache_ram_mb {
            args.push("--cache-ram".to_string());
            args.push(cache_ram_mb.to_string());
        }
        if let Some(ctxcp) = config.ctxcp {
            args.push("-ctxcp".to_string());
            args.push(ctxcp.to_string());
        }
        if config.use_jinja {
            args.push("--jinja".to_string());
        }
        if let Some(reasoning_mode) = config
            .reasoning_mode
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            args.push("--reasoning".to_string());
            args.push(reasoning_mode.clone());
        }
        if config.reasoning_preserve {
            args.push("--reasoning-preserve".to_string());
        }
        if let Some(template_name) = config
            .template_name
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            args.push("--chat-template".to_string());
            args.push(template_name.clone());
        }
        if let Some(template_file) = config.template_file.as_ref() {
            args.push("--chat-template-file".to_string());
            args.push(template_file.to_string_lossy().to_string());
        }
        if let Some(kwargs_json) = chat_template_kwargs_json
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            args.push("--chat-template-kwargs".to_string());
            args.push(kwargs_json.clone());
        }

        if config.gpu_layers != 0 {
            args.push("--n-gpu-layers".to_string());
            args.push(if config.gpu_layers < 0 {
                "999".to_string()
            } else {
                config.gpu_layers.to_string()
            });
        }
        if config.threads > 0 {
            args.push("--threads".to_string());
            args.push(config.threads.to_string());
        }
        if config.threads_batch > 0 {
            args.push("--threads-batch".to_string());
            args.push(config.threads_batch.to_string());
        }
        if config.batch_size > 0 {
            args.push("--batch-size".to_string());
            args.push(config.batch_size.to_string());
        }
        if config.ubatch_size > 0 {
            args.push("--ubatch-size".to_string());
            args.push(config.ubatch_size.to_string());
        }
        if config.flash_attn {
            args.push("--flash-attn".to_string());
            args.push("on".to_string());
        }
        if !config.use_mmap {
            args.push("--no-mmap".to_string());
        }
        if config.use_mlock {
            args.push("--mlock".to_string());
        }
        if config.cont_batching {
            args.push("--cont-batching".to_string());
        }
        if config.main_gpu != 0 {
            args.push("--main-gpu".to_string());
            args.push(config.main_gpu.to_string());
        }
        if config.defrag_thold > 0.0 {
            args.push("--defrag-thold".to_string());
            args.push(format!("{:.4}", config.defrag_thold));
        }
        if config.rope_freq_scale > 0.0 {
            args.push("--rope-freq-scale".to_string());
            args.push(format!("{:.6}", config.rope_freq_scale));
        }
        if !config.cache_type_k.is_empty() {
            args.push("--cache-type-k".to_string());
            args.push(config.cache_type_k.clone());
        }
        if !config.cache_type_v.is_empty() {
            args.push("--cache-type-v".to_string());
            args.push(config.cache_type_v.clone());
        }
        if config.kv_unified {
            args.push("--kv-unified".to_string());
        }
        if config.no_warmup {
            args.push("--no-warmup".to_string());
        }
        if config.ctx_shift {
            args.push("--ctx-shift".to_string());
        }
        if !config.tensor_split.is_empty() {
            args.push("--tensor-split".to_string());
            args.push(
                config
                    .tensor_split
                    .iter()
                    .map(|v| format!("{v}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        // A separate draft model (-md) enables traditional, draft-model-based
        // speculative decoding. This is OPTIONAL: self-MTP (e.g. draft-mtp on a
        // Qwen3.6-27B-MTP GGUF) drafts from the main model's own MTP heads and
        // needs no -md at all.
        let has_draft_model = !config.draft_model_path.is_empty();
        if has_draft_model {
            args.push("-md".to_string());
            args.push(config.draft_model_path.clone());
        }
        // --spec-type drives BOTH self-MTP (no draft model) and draft-model modes.
        // It must be emitted whenever a spec type is configured, independent of -md,
        // otherwise self-MTP never activates. --spec-draft-n-max pairs with it.
        if !config.spec_type.trim().is_empty() {
            args.push("--spec-type".to_string());
            args.push(config.spec_type.clone());
            if config.spec_draft_n_max > 0 {
                args.push("--spec-draft-n-max".to_string());
                args.push(config.spec_draft_n_max.to_string());
            }
        }
        // The remaining --draft-* knobs only apply to a separate draft model.
        if has_draft_model {
            if config.draft_max_tokens > 0 {
                args.push("--draft-max".to_string());
                args.push(config.draft_max_tokens.to_string());
            }
            if config.draft_min_tokens > 0 {
                args.push("--draft-min".to_string());
                args.push(config.draft_min_tokens.to_string());
            }
            if config.draft_p_min > 0.0 {
                args.push("--draft-p-min".to_string());
                args.push(format!("{:.4}", config.draft_p_min));
            }
        }
        args.extend(config.extra_args.iter().cloned());

        Ok(LaunchPreview {
            runtime: "llama-server".to_string(),
            server_path: server_path.to_string_lossy().to_string(),
            model_path: config.model_path.to_string_lossy().to_string(),
            hf_repo: config.hf_repo.clone(),
            hf_file: config.hf_file.clone(),
            mmproj_path: mmproj_path.map(|path| path.to_string_lossy().to_string()),
            backend_preference: config.backend_preference.clone(),
            context_size: config.context_size,
            port: config.port,
            parallel_slots: config.parallel_slots.max(1),
            fit_mode: config.fit_mode.clone(),
            cache_ram_mb: config.cache_ram_mb,
            ctxcp: config.ctxcp,
            use_jinja: config.use_jinja,
            reasoning_mode: config.reasoning_mode.clone(),
            reasoning_preserve: config.reasoning_preserve,
            template_mode: config.template_mode.clone(),
            template_source: config.template_source.clone(),
            template_path: config
                .template_file
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            template_name: config.template_name.clone(),
            chat_template_kwargs_json,
            draft_model_path: config.draft_model_path.clone(),
            spec_type: config.spec_type.clone(),
            spec_draft_n_max: config.spec_draft_n_max,
            draft_max_tokens: config.draft_max_tokens,
            draft_min_tokens: config.draft_min_tokens,
            draft_p_min: config.draft_p_min,
            args,
        })
    }

    fn build_diffusion_args_preview(&self, config: &LaunchConfig) -> anyhow::Result<LaunchPreview> {
        if config
            .hf_repo
            .as_ref()
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
        {
            anyhow::bail!(
                "DiffusionGemma currently requires a local GGUF file for llama-diffusion-cli; HF repo loading is not supported by the bridge yet"
            );
        }

        let runner_path = self
            .find_diffusion_cli_binary_with_preference(&config.backend_preference)
            .ok_or_else(|| anyhow::anyhow!("llama-diffusion-cli binary not found"))?;

        let n_predict = config.diffusion_n_predict.max(1);
        let kv_cache = config.diffusion_kv_cache.trim();
        let mut args = vec![
            "-m".to_string(),
            config.model_path.to_string_lossy().to_string(),
            "-ngl".to_string(),
            if config.gpu_layers < 0 {
                "999".to_string()
            } else {
                config.gpu_layers.to_string()
            },
            "-cnv".to_string(),
            "-n".to_string(),
            n_predict.to_string(),
        ];

        if !kv_cache.is_empty() {
            args.push("--diffusion-kv-cache".to_string());
            args.push(kv_cache.to_string());
        }
        if config.diffusion_visual {
            args.push("--diffusion-visual".to_string());
        }
        args.extend(config.diffusion_extra_args.iter().cloned());

        Ok(LaunchPreview {
            runtime: "diffusion-cli".to_string(),
            server_path: runner_path.to_string_lossy().to_string(),
            model_path: config.model_path.to_string_lossy().to_string(),
            hf_repo: config.hf_repo.clone(),
            hf_file: config.hf_file.clone(),
            mmproj_path: None,
            backend_preference: config.backend_preference.clone(),
            context_size: config.context_size,
            port: 0,
            parallel_slots: 1,
            fit_mode: config.fit_mode.clone(),
            cache_ram_mb: config.cache_ram_mb,
            ctxcp: config.ctxcp,
            use_jinja: false,
            reasoning_mode: config.reasoning_mode.clone(),
            reasoning_preserve: false,
            template_mode: "diffusion-cli".to_string(),
            template_source: Some("llama-diffusion-cli:-cnv".to_string()),
            template_path: None,
            template_name: None,
            chat_template_kwargs_json: None,
            draft_model_path: String::new(),
            spec_type: String::new(),
            spec_draft_n_max: 0,
            draft_max_tokens: 0,
            draft_min_tokens: 0,
            draft_p_min: 0.0,
            args,
        })
    }

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

    pub fn find_diffusion_cli_binary_with_preference(
        &self,
        backend_preference: &str,
    ) -> Option<PathBuf> {
        if let Some(ref path) = self.llama_diffusion_cli_path {
            if Path::new(path).exists() {
                return Some(path.clone());
            }
        }

        let our_dir = Self::managed_binary_dir();
        let our_exe = our_dir.join("llama-diffusion-cli.exe");
        if our_exe.exists() && Self::matches_backend_preference(&our_exe, backend_preference) {
            return Some(our_exe);
        }

        let mut candidates: Vec<PathBuf> = Vec::new();
        for binary in ["llama-diffusion-cli", "llama-diffusion-cli.exe"] {
            if let Ok(output) = system_command("where").arg(binary).output() {
                if output.status.success() {
                    for line in String::from_utf8_lossy(&output.stdout).lines() {
                        let p = PathBuf::from(line.trim());
                        if p.exists() {
                            candidates.push(p);
                        }
                    }
                }
            }
        }

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
                            for sub in ["", "bin", "build/bin", "build/bin/Release"] {
                                let exe = entry.path().join(sub).join("llama-diffusion-cli.exe");
                                if exe.exists() {
                                    candidates.push(exe);
                                }
                            }
                        }
                    }
                }
            }
        }

        let common = [
            dirs::home_dir().map(|h| h.join(".local/bin/llama-diffusion-cli.exe")),
            dirs::home_dir().map(|h| h.join("llama.cpp/build/bin/llama-diffusion-cli.exe")),
            dirs::home_dir().map(|h| h.join("llama.cpp/build/bin/Release/llama-diffusion-cli.exe")),
            Some(PathBuf::from("C:/llama.cpp/llama-diffusion-cli.exe")),
            Some(PathBuf::from(
                "C:/llama.cpp/build/bin/llama-diffusion-cli.exe",
            )),
            Some(PathBuf::from(
                "C:/llama.cpp/build/bin/Release/llama-diffusion-cli.exe",
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

    /// Reserve a free TCP port by binding to port 0 and holding the listener
    /// until immediately before spawning llama-server. This narrows the race
    /// window compared with dropping the listener before process launch.
    fn reserve_free_port() -> anyhow::Result<(std::net::TcpListener, u16)> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        Ok((listener, port))
    }

    pub fn new() -> Self {
        let (state_tx, state_rx) = watch::channel(ProcessState::Idle);
        Self {
            child: None,
            state: ProcessState::Idle,
            llama_server_path: None,
            llama_diffusion_cli_path: None,
            current_model: None,
            current_pid: None,
            current_port: 0,
            crash_count: 0,
            last_exit_status: None,
            state_tx,
            state_rx,
            detected_backend: Arc::new(TokioMutex::new(None)),
            stderr_lines: Arc::new(TokioMutex::new(VecDeque::new())),
            io_tasks: Vec::new(),
            app_handle: None,
            load_progress: Arc::new(TokioMutex::new(None)),
        }
    }

    /// Returns the GPU backend detected from the server's startup logs.
    pub fn detected_backend(&self) -> Arc<TokioMutex<Option<String>>> {
        self.detected_backend.clone()
    }

    /// Shared handle to the live load-progress fraction parsed from stderr.
    /// Callers can read `*handle.lock().await` each poll without touching the
    /// state lock.
    pub fn load_progress_handle(&self) -> Arc<TokioMutex<Option<f32>>> {
        self.load_progress.clone()
    }

    /// Get a receiver for state change notifications.
    pub fn state_watch(&self) -> watch::Receiver<ProcessState> {
        self.state_rx.clone()
    }

    pub fn state(&self) -> ProcessState {
        self.state
    }

    pub fn has_child(&self) -> bool {
        self.child.is_some()
    }

    pub fn mark_idle_if_no_child(&mut self) -> bool {
        if self.child.is_some() || self.state == ProcessState::Idle {
            return false;
        }

        tracing::warn!(
            state = ?self.state,
            pid = self.current_pid,
            port = self.current_port,
            "Repairing stale llama-server process state with no child process"
        );
        self.current_model = None;
        self.current_pid = None;
        self.current_port = 0;
        self.set_state(ProcessState::Idle);
        true
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

    pub fn set_diffusion_cli_path(&mut self, path: PathBuf) {
        self.llama_diffusion_cli_path = Some(path);
    }

    fn set_state(&mut self, state: ProcessState) {
        self.state = state;
        // Push the transition to the GUI immediately so status panels don't
        // wait for the next background poll.
        if let Some(handle) = &self.app_handle {
            use tauri::Emitter;
            let _ = handle.emit("process-state-changed", format!("{state:?}"));
        }
        let _ = self.state_tx.send(state);
    }

    /// Store the app handle used to push live `llama-server-log` and
    /// `process-state-changed` events to the GUI. Set once at startup.
    pub fn set_app_handle(&mut self, handle: tauri::AppHandle) {
        self.app_handle = Some(handle);
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

    /// Kill any stale managed llama-server processes before launching a new one.
    ///
    /// **Call this BEFORE acquiring the state write lock.**  The underlying Windows
    /// process query can take 1-3 seconds; running it while holding the write lock
    /// would block every concurrent reader (including in-flight API requests).
    #[cfg(windows)]
    pub fn clear_stale_managed_processes() {
        match Self::kill_all_managed_processes() {
            Ok(killed) if killed > 0 => {
                tracing::warn!(
                    killed,
                    "Pre-launch: cleared stale managed llama-server processes"
                );
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "Pre-launch managed process cleanup failed");
            }
        }
    }

    /// Launch llama-server with the given configuration.
    ///
    /// When `config.port` is 0, an ephemeral port is auto-assigned by the OS.
    /// Stale-process cleanup should be done **before** calling this
    /// via [`LlamaProcess::clear_stale_managed_processes`] so the write lock is
    /// not held during a slow system scan.
    pub async fn launch(&mut self, mut config: LaunchConfig) -> anyhow::Result<()> {
        Self::validate_launch_config(&config)?;

        // Auto-assign a free ephemeral port when port is 0, and keep the
        // reservation open until just before the child is spawned.
        let mut port_reservation = None;
        if config.port == 0 {
            let (listener, port) = Self::reserve_free_port()?;
            config.port = port;
            port_reservation = Some(listener);
            tracing::info!(
                port = config.port,
                "Auto-assigned free port for llama-server"
            );
        }

        let preview = self.build_args_preview(&config)?;
        if preview.runtime == "diffusion-cli" {
            anyhow::bail!(
                "DiffusionGemma GGUFs require llama-diffusion-cli and do not expose a llama-server HTTP API yet. Launch preview is ready, but OpenAI-compatible /v1 chat proxying is disabled until llama.cpp provides server support for DiffusionGemma. Runner: {} Args: {}",
                preview.server_path,
                preview.args.join(" ")
            );
        }
        let server_path = PathBuf::from(&preview.server_path);

        // Shutdown any existing process only after the new launch is known to be valid.
        self.shutdown().await?;
        // NOTE: stale-process cleanup (kill_all_managed_processes) was moved
        // to `clear_stale_managed_processes()` and must be called BEFORE the write lock.

        self.current_port = config.port;
        self.set_state(ProcessState::Starting);

        let mut cmd = Command::new(&server_path);
        cmd.args(&preview.args);

        if let Some(mmproj_path) = &preview.mmproj_path {
            tracing::info!(
                model = %config.model_path.display(),
                mmproj = %mmproj_path,
                "Using multimodal projection sidecar for vision model"
            );
        } else if config.attach_mmproj && filename_supports_vision(&config.model_path) {
            tracing::warn!(
                model = %config.model_path.display(),
                "Vision-capable model detected but no mmproj sidecar was found nearby; image understanding may fail"
            );
        } else if !config.attach_mmproj && filename_supports_vision(&config.model_path) {
            tracing::info!(
                model = %config.model_path.display(),
                "Skipping mmproj attachment for this text-only launch"
            );
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
            ctx = ?config.context_size,
            port = config.port,
            gpu_layers = config.gpu_layers,
            args = ?preview.args,
            "Launching llama-server"
        );
        // Explicit, greppable record of the speculative-decoding mode so MTP
        // activation (self-MTP or draft-model) is obvious in the bridge logs.
        if !config.spec_type.trim().is_empty() {
            tracing::info!(
                target: "speculative",
                spec_type = %config.spec_type,
                spec_draft_n_max = config.spec_draft_n_max,
                draft_model = %config.draft_model_path,
                self_mtp = config.draft_model_path.trim().is_empty(),
                "Speculative decoding enabled"
            );
        }

        drop(port_reservation);
        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => {
                self.current_port = 0;
                self.current_model = None;
                self.current_pid = None;
                self.set_state(ProcessState::Error);
                return Err(error.into());
            }
        };
        let child_pid = child.id();

        // Abort any leftover I/O tasks from a previous launch
        for handle in self.io_tasks.drain(..) {
            handle.abort();
        }

        // Spawn background tasks to stream stdout/stderr to tracing
        if let Some(stdout) = child.stdout.take() {
            let app_handle = self.app_handle.clone();
            let handle = tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if let Some(h) = &app_handle {
                        use tauri::Emitter;
                        let _ = h.emit("llama-server-log", &line);
                    }
                    tracing::info!(target: "llama_server", "{}", line);
                }
            });
            self.io_tasks.push(handle);
        }
        if let Some(stderr) = child.stderr.take() {
            let backend_handle = self.detected_backend.clone();
            let stderr_buf = self.stderr_lines.clone();
            let app_handle = self.app_handle.clone();
            let load_progress = self.load_progress.clone();
            // Clear previous stderr
            stderr_buf.lock().await.clear();
            // Fresh load — reset parsed progress to unknown.
            *load_progress.lock().await = None;
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
                    // Keep a generous stderr tail for crash diagnostics; llama
                    // startup can emit many metadata lines before the actual error.
                    {
                        let mut buf = stderr_buf.lock().await;
                        buf.push_back(line.clone());
                        if buf.len() > 500 {
                            buf.pop_front();
                        }
                    }
                    // Track real load progress (monotonic) for the progress bar.
                    if let Some(p) = parse_llama_load_progress(&line) {
                        let mut guard = load_progress.lock().await;
                        if guard.map_or(true, |current| p > current) {
                            *guard = Some(p);
                        }
                    }
                    // Push the line to the GUI live so the console is 1-1 with
                    // llama-server instead of waiting for the log poll.
                    if let Some(h) = &app_handle {
                        use tauri::Emitter;
                        let _ = h.emit("llama-server-log", &line);
                    }
                    // llama-server logs almost everything to stderr
                    tracing::info!(target: "llama_server", "{}", line);
                }
            });
            self.io_tasks.push(handle);
        }

        self.child = Some(child);
        self.current_pid = child_pid;
        self.current_model = config
            .model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string());
        self.crash_count = 0;
        self.last_exit_status = None;

        Ok(())
    }

    /// Wait for the server to become healthy (responds to /health).
    pub async fn wait_for_healthy(&mut self, timeout: Duration) -> anyhow::Result<()> {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/health", self.current_port);
        let start = std::time::Instant::now();

        loop {
            if self.poll_exited() {
                return Err(anyhow::anyhow!(
                    "llama-server exited before becoming healthy"
                ));
            }
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
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
        {
            Ok(client) => client,
            Err(error) => {
                tracing::warn!(%error, "Falling back to default HTTP client for health check");
                reqwest::Client::new()
            }
        };
        let url = format!("http://127.0.0.1:{}/health", self.current_port);
        matches!(client.get(&url).send().await, Ok(r) if r.status().is_success())
    }

    /// Extract live process resources for shutdown without holding the AppState write lock.
    ///
    /// Call this while holding the write lock to atomically take the child, PID, port, and
    /// IO tasks out of LlamaProcess, then release the lock. Pass the returned
    /// `PendingShutdown` to `complete_shutdown()` which does the slow async work (HTTP
    /// graceful stop, kill, port-release wait) without blocking readers.
    pub fn begin_shutdown(&mut self) -> Option<PendingShutdown> {
        let child = self.child.take()?;
        let port = self.current_port;
        let pid = self.current_pid.take();
        self.current_model = None;
        self.current_port = 0;
        self.set_state(ProcessState::Stopping);
        let io_tasks = self.io_tasks.drain(..).collect();
        let detected_backend = self.detected_backend.clone();
        let stderr_lines = self.stderr_lines.clone();
        let load_progress = self.load_progress.clone();
        Some(PendingShutdown {
            child,
            port,
            pid,
            io_tasks,
            detected_backend,
            stderr_lines,
            load_progress,
        })
    }

    /// Gracefully shutdown the llama-server process.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        if let Some(mut child) = self.child.take() {
            self.set_state(ProcessState::Stopping);
            tracing::info!(
                model = ?self.current_model,
                pid = self.current_pid,
                port = self.current_port,
                "Shutting down llama-server"
            );

            // Try graceful shutdown via /shutdown endpoint
            let client = match reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
            {
                Ok(client) => client,
                Err(error) => {
                    tracing::warn!(%error, "Falling back to default HTTP client for shutdown");
                    reqwest::Client::new()
                }
            };
            let shutdown_url = format!("http://127.0.0.1:{}/shutdown", self.current_port);
            match client.post(&shutdown_url).send().await {
                Ok(_) => tracing::debug!("Graceful shutdown request sent"),
                Err(e) => {
                    tracing::debug!(error = %e, "Graceful shutdown request failed (process may already be stopped)")
                }
            }

            // Give graceful shutdown a short head start, then force-kill to keep swaps snappy.
            let exit = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;

            if exit.is_err() {
                tracing::warn!("llama-server did not exit gracefully, killing");
                let _ = child.kill().await;
                let _ = child.wait().await;
            } else {
                tracing::info!("llama-server exited gracefully");
            }

            if let Some(pid) = self.current_pid {
                let _ = Self::force_kill_process_tree(pid);
            }

            let released_port = self.current_port;
            self.current_model = None;
            self.current_pid = None;
            *self.detected_backend.lock().await = None;
            self.stderr_lines.lock().await.clear();
            *self.load_progress.lock().await = None;

            // Abort background I/O reader tasks before waiting on the port so
            // lingering pipes/handles do not slow teardown.
            for handle in self.io_tasks.drain(..) {
                handle.abort();
            }

            Self::wait_for_port_release(released_port, Duration::from_millis(1500)).await;
            self.current_port = 0;
            self.set_state(ProcessState::Idle);
        }
        // Abort background I/O reader tasks for the no-child path too.
        for handle in self.io_tasks.drain(..) {
            handle.abort();
        }
        self.mark_idle_if_no_child();
        Ok(())
    }

    /// Check if the process has crashed and record it.
    /// Non-blocking crash check: calls `try_wait()` and updates internal state if the
    /// process has exited, but does **not** sleep.  Returns `true` if the process is gone.
    ///
    /// Use this inside a write-lock critical section so the lock is not held during any
    /// subsequent sleep (see health-poll loop in `commands/model.rs`).
    pub fn poll_exited(&mut self) -> bool {
        if let Some(ref mut child) = self.child {
            match child.try_wait() {
                Ok(Some(status)) => {
                    self.child = None;
                    self.current_pid = None;
                    self.crash_count += 1;
                    self.last_exit_status = Some(status.to_string());
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

    /// Crash check with a stderr-flush wait.  Prefer `poll_exited` when inside a lock.
    pub async fn check_crashed(&mut self) -> bool {
        if !self.poll_exited() {
            return false;
        }
        // Give the background stderr-reader task a moment to drain its buffer.
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
        tracing::error!(stderr = %last_lines, "llama-server process exited unexpectedly");
        true
    }

    /// Get captured stderr lines (for crash diagnostics).
    pub async fn last_stderr(&self) -> Vec<String> {
        self.stderr_lines.lock().await.iter().cloned().collect()
    }

    pub fn last_exit_status(&self) -> Option<String> {
        self.last_exit_status.clone()
    }

    pub fn crash_count(&self) -> u32 {
        self.crash_count
    }

    #[cfg(windows)]
    fn normalize_windows_path(path: &str) -> String {
        path.trim_matches('"')
            .trim()
            .replace('/', "\\")
            .to_lowercase()
    }

    #[cfg(windows)]
    fn force_kill_process_tree(pid: u32) -> anyhow::Result<()> {
        let output = system_command("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .output()?;
        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
        if stderr.contains("not found") || stderr.contains("there is no running instance") {
            Ok(())
        } else {
            anyhow::bail!("Failed to kill PID {pid}: {}", stderr.trim())
        }
    }

    #[cfg(not(windows))]
    fn force_kill_process_tree(pid: u32) -> anyhow::Result<()> {
        use std::process::Command;
        // Kill the process group (negative PID) to get all children.
        let _ = Command::new("kill")
            .args(["-9", &format!("-{pid}")])
            .output();
        // Also kill the PID directly in case it's not a process group leader.
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).output();
        Ok(())
    }

    #[cfg(windows)]
    pub fn kill_all_managed_processes() -> anyhow::Result<u32> {
        #[derive(serde::Deserialize)]
        struct CimProcess {
            #[serde(rename = "ProcessId")]
            process_id: Option<u32>,
            #[serde(rename = "ExecutablePath")]
            executable_path: Option<String>,
        }

        let managed_exe = Self::normalize_windows_path(
            &Self::managed_binary_dir()
                .join("llama-server.exe")
                .to_string_lossy(),
        );

        let output = system_command("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name = 'llama-server.exe'\" | Select-Object ProcessId,ExecutablePath | ConvertTo-Json -Compress",
            ])
            .output()?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to query llama-server processes: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let text = String::from_utf8_lossy(&output.stdout);
        let text = text.trim();
        if text.is_empty() {
            return Ok(0);
        }

        let value: serde_json::Value = serde_json::from_str(text)?;
        let rows: Vec<CimProcess> = if value.is_array() {
            serde_json::from_value(value)?
        } else {
            vec![serde_json::from_value(value)?]
        };

        let mut killed = 0u32;
        for row in rows {
            let Some(pid) = row.process_id else {
                continue;
            };
            let executable_path =
                Self::normalize_windows_path(row.executable_path.as_deref().unwrap_or_default());

            if executable_path.is_empty() {
                continue;
            }

            if executable_path == managed_exe {
                match Self::force_kill_process_tree(pid) {
                    Ok(_) => killed += 1,
                    Err(error) => {
                        tracing::warn!(pid, error = %error, "Failed to kill managed llama-server process");
                    }
                }
            }
        }

        Ok(killed)
    }

    #[cfg(not(windows))]
    pub fn kill_all_managed_processes() -> anyhow::Result<u32> {
        Ok(0)
    }
}
