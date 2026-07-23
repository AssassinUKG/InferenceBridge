use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::config::{
    app_support_dir, ImageGenerationConfig, ImageGenerationProfileConfig, ImageModelBundleConfig,
};

const MAX_PROMPT_CHARS: usize = 8_000;
const MAX_IMAGE_DIMENSION: u32 = 2_048;

static STEP_PROGRESS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<step>\d{1,4})/(?P<total>\d{1,4})\s*-\s*(?P<seconds>\d+(?:\.\d+)?)s/it")
        .expect("image progress regex must compile")
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub prompt: String,
    pub session_id: Option<String>,
    pub bundle_id: Option<String>,
    pub profile_id: Option<String>,
    pub seed: Option<i64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub sampling_method: Option<String>,
    pub negative_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationPreview {
    pub bundle_id: String,
    pub bundle_name: String,
    pub profile_id: String,
    pub profile_name: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub seed: i64,
    pub output_path: String,
    pub arguments: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationResult {
    pub job_id: String,
    pub status: String,
    pub bundle_id: String,
    pub bundle_name: String,
    pub quantization: String,
    pub profile_id: String,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub seed: i64,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg_scale: f32,
    pub sampling_method: String,
    pub elapsed_seconds: f64,
    pub file_size_bytes: Option<u64>,
    pub output_path: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationProgress {
    pub job_id: String,
    pub status: String,
    pub stage: String,
    pub message: String,
    pub bundle_id: String,
    pub profile_id: String,
    pub current_step: u32,
    pub total_steps: u32,
    /// 0.0 to 1.0. Loading/saving stages use small reserved ranges around step progress.
    pub progress: f32,
    pub elapsed_seconds: f64,
    pub eta_seconds: Option<f64>,
    pub started_at: String,
    pub updated_at: String,
    pub done: bool,
    pub error: Option<String>,
    pub output_path: Option<String>,
}

impl ImageGenerationProgress {
    pub fn starting(
        job_id: String,
        bundle_id: String,
        profile_id: String,
        total_steps: u32,
    ) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            job_id,
            status: "running".to_string(),
            stage: "loading".to_string(),
            message: "Loading image model".to_string(),
            bundle_id,
            profile_id,
            current_step: 0,
            total_steps,
            progress: 0.0,
            elapsed_seconds: 0.0,
            eta_seconds: None,
            started_at: now.clone(),
            updated_at: now,
            done: false,
            error: None,
            output_path: None,
        }
    }

    pub fn apply_step(&mut self, parsed: ParsedImageProgress, elapsed_seconds: f64) {
        self.status = "running".to_string();
        self.stage = "generating".to_string();
        self.current_step = parsed.current_step;
        self.total_steps = parsed.total_steps;
        self.progress = if parsed.total_steps == 0 {
            0.05
        } else {
            0.05 + 0.90 * (parsed.current_step as f32 / parsed.total_steps as f32).clamp(0.0, 1.0)
        };
        self.elapsed_seconds = elapsed_seconds;
        self.eta_seconds = Some(
            parsed.seconds_per_step * parsed.total_steps.saturating_sub(parsed.current_step) as f64,
        );
        self.message = format!(
            "Generating image - step {} of {}",
            parsed.current_step, parsed.total_steps
        );
        self.updated_at = chrono::Utc::now().to_rfc3339();
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationCapabilityStatus {
    pub enabled: bool,
    pub ready: bool,
    pub automatic_model_swap_enabled: bool,
    pub runner_path: Option<String>,
    pub output_dir: String,
    pub default_bundle: String,
    pub default_profile: String,
    pub warn_temperature_c: f32,
    pub cooldown_temperature_c: f32,
    pub reasons: Vec<String>,
    pub bundles: Vec<ImageBundleStatus>,
    pub profiles: Vec<ImageProfileStatus>,
    pub size_presets: Vec<ImageSizePreset>,
    pub active_job: Option<ImageGenerationProgress>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageSizePreset {
    pub id: String,
    pub name: String,
    pub aspect_ratio: String,
    pub width: u32,
    pub height: u32,
    pub tier: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageBundleStatus {
    pub id: String,
    pub name: String,
    pub architecture: String,
    pub quantization: String,
    pub ready: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageProfileStatus {
    pub id: String,
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub ready: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedImageJob {
    pub runner_path: PathBuf,
    pub bundle: ImageModelBundleConfig,
    pub profile: ImageGenerationProfileConfig,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub seed: i64,
    pub output_path: PathBuf,
    pub arguments: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParsedImageProgress {
    pub current_step: u32,
    pub total_steps: u32,
    pub seconds_per_step: f64,
}

#[derive(Default)]
pub struct NativeProgressParser {
    tail: String,
    last_step: Option<(u32, u32)>,
}

impl NativeProgressParser {
    pub fn push(&mut self, chunk: &str) -> Option<ParsedImageProgress> {
        self.tail.push_str(chunk);
        if self.tail.len() > 4_096 {
            let keep_from = self
                .tail
                .char_indices()
                .rev()
                .nth(2_047)
                .map(|(index, _)| index)
                .unwrap_or(0);
            self.tail = self.tail[keep_from..].to_string();
        }

        let parsed = STEP_PROGRESS_RE
            .captures_iter(&self.tail)
            .filter_map(|captures| {
                Some(ParsedImageProgress {
                    current_step: captures.name("step")?.as_str().parse().ok()?,
                    total_steps: captures.name("total")?.as_str().parse().ok()?,
                    seconds_per_step: captures.name("seconds")?.as_str().parse().ok()?,
                })
            })
            .last()?;

        if parsed.current_step > parsed.total_steps
            || self.last_step == Some((parsed.current_step, parsed.total_steps))
        {
            return None;
        }
        self.last_step = Some((parsed.current_step, parsed.total_steps));
        Some(parsed)
    }
}

pub fn capability_status(
    config: &ImageGenerationConfig,
    active_job: Option<ImageGenerationProgress>,
) -> ImageGenerationCapabilityStatus {
    let output_dir = configured_output_dir(config);
    let mut reasons = Vec::new();
    let runner = trimmed_path(&config.runner_path);
    if !config.enabled {
        reasons.push("Image generation is disabled in configuration".to_string());
    }
    match runner.as_deref() {
        None => reasons.push("Image runner path is not configured".to_string()),
        Some(path) if !path.is_absolute() => {
            reasons.push("Image runner path must be absolute".to_string())
        }
        Some(path) if !path.is_file() => {
            reasons.push(format!("Image runner does not exist: {}", path.display()))
        }
        Some(_) => {}
    }

    if let Some(path) = trimmed_path(&config.output_dir) {
        if !path.is_absolute() {
            reasons.push("Image output directory must be absolute".to_string());
        } else if path.exists() && !path.is_dir() {
            reasons.push(format!(
                "Image output path is not a directory: {}",
                path.display()
            ));
        }
    }
    if !config.warn_temperature_c.is_finite()
        || !(40.0..=100.0).contains(&config.warn_temperature_c)
    {
        reasons.push("Image warning temperature must be between 40 and 100 C".to_string());
    }
    if !config.cooldown_temperature_c.is_finite()
        || !(20.0..=90.0).contains(&config.cooldown_temperature_c)
    {
        reasons.push("Image cooldown temperature must be between 20 and 90 C".to_string());
    } else if config.cooldown_temperature_c >= config.warn_temperature_c {
        reasons.push("Cooldown temperature must be below warning temperature".to_string());
    }

    let bundles = config.bundles.iter().map(bundle_status).collect::<Vec<_>>();
    let profiles = config
        .profiles
        .iter()
        .map(profile_status)
        .collect::<Vec<_>>();
    let mut bundle_ids = HashSet::new();
    for bundle in &bundles {
        if !bundle_ids.insert(bundle.id.as_str()) {
            reasons.push(format!("Duplicate image bundle id: {}", bundle.id));
        }
    }
    let mut profile_ids = HashSet::new();
    for profile in &profiles {
        if !profile_ids.insert(profile.id.as_str()) {
            reasons.push(format!("Duplicate image profile id: {}", profile.id));
        }
    }

    let default_bundle_ready = bundles
        .iter()
        .any(|bundle| bundle.id == config.default_bundle && bundle.ready);
    if !default_bundle_ready {
        reasons.push(format!(
            "Default image bundle `{}` is missing or invalid",
            config.default_bundle
        ));
    }
    let default_profile_ready = profiles
        .iter()
        .any(|profile| profile.id == config.default_profile && profile.ready);
    if !default_profile_ready {
        reasons.push(format!(
            "Default image profile `{}` is missing or invalid",
            config.default_profile
        ));
    }

    ImageGenerationCapabilityStatus {
        enabled: config.enabled,
        ready: config.enabled && reasons.is_empty(),
        automatic_model_swap_enabled: config.automatic_model_swap_enabled,
        runner_path: runner.map(|path| path.to_string_lossy().to_string()),
        output_dir: output_dir.to_string_lossy().to_string(),
        default_bundle: config.default_bundle.clone(),
        default_profile: config.default_profile.clone(),
        warn_temperature_c: config.warn_temperature_c,
        cooldown_temperature_c: config.cooldown_temperature_c,
        reasons,
        bundles,
        profiles,
        size_presets: image_size_presets(),
        active_job,
    }
}

pub fn image_size_presets() -> Vec<ImageSizePreset> {
    [
        (
            "recommended_square",
            "Recommended square",
            "1:1",
            1024,
            1024,
            "recommended",
            "Measured Q6 default: best balance of quality, heat, and generation time.",
        ),
        (
            "official_square",
            "Maximum square",
            "1:1",
            1328,
            1328,
            "max",
            "Official Qwen square size; slower and thermally demanding on RTX 3090.",
        ),
        (
            "widescreen",
            "Widescreen",
            "16:9",
            1664,
            928,
            "official",
            "Official Qwen landscape preset.",
        ),
        (
            "portrait",
            "Tall portrait",
            "9:16",
            928,
            1664,
            "official",
            "Official Qwen portrait preset.",
        ),
        (
            "landscape_4_3",
            "Landscape",
            "4:3",
            1472,
            1104,
            "official",
            "Official Qwen landscape preset.",
        ),
        (
            "portrait_3_4",
            "Portrait",
            "3:4",
            1104,
            1472,
            "official",
            "Official Qwen portrait preset.",
        ),
        (
            "photo_3_2",
            "Photo landscape",
            "3:2",
            1584,
            1056,
            "official",
            "Official Qwen photographic landscape preset.",
        ),
        (
            "photo_2_3",
            "Photo portrait",
            "2:3",
            1056,
            1584,
            "official",
            "Official Qwen photographic portrait preset.",
        ),
    ]
    .into_iter()
    .map(
        |(id, name, aspect_ratio, width, height, tier, note)| ImageSizePreset {
            id: id.to_string(),
            name: name.to_string(),
            aspect_ratio: aspect_ratio.to_string(),
            width,
            height,
            tier: tier.to_string(),
            note: note.to_string(),
        },
    )
    .collect()
}

pub fn resolve_job(
    config: &ImageGenerationConfig,
    request: &ImageGenerationRequest,
    output_path: PathBuf,
) -> Result<ResolvedImageJob, String> {
    let status = capability_status(config, None);
    if !status.ready {
        return Err(status.reasons.join("; "));
    }
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err("Image prompt cannot be empty".to_string());
    }
    if prompt.chars().count() > MAX_PROMPT_CHARS {
        return Err(format!(
            "Image prompt exceeds the {MAX_PROMPT_CHARS} character limit"
        ));
    }
    if prompt.contains('\0') {
        return Err("Image prompt contains an invalid null character".to_string());
    }

    let bundle_id = request
        .bundle_id
        .as_deref()
        .unwrap_or(&config.default_bundle);
    let profile_id = request
        .profile_id
        .as_deref()
        .unwrap_or(&config.default_profile);
    let bundle = config
        .bundles
        .iter()
        .find(|bundle| bundle.id == bundle_id)
        .cloned()
        .ok_or_else(|| format!("Unknown image bundle: {bundle_id}"))?;
    let bundle_check = bundle_status(&bundle);
    if !bundle_check.ready {
        return Err(bundle_check.reasons.join("; "));
    }
    let mut profile = config
        .profiles
        .iter()
        .find(|profile| profile.id == profile_id)
        .cloned()
        .ok_or_else(|| format!("Unknown image profile: {profile_id}"))?;
    let profile_check = profile_status(&profile);
    if !profile_check.ready {
        return Err(profile_check
            .reason
            .unwrap_or_else(|| format!("Invalid image profile: {profile_id}")));
    }
    if let Some(width) = request.width {
        profile.width = width;
    }
    if let Some(height) = request.height {
        profile.height = height;
    }
    if let Some(steps) = request.steps {
        profile.steps = steps;
    }
    if let Some(cfg_scale) = request.cfg_scale {
        profile.cfg_scale = cfg_scale;
    }
    if let Some(sampling_method) = request.sampling_method.as_ref() {
        profile.sampling_method = sampling_method.trim().to_ascii_lowercase();
    }
    validate_profile(&profile)?;
    let negative_prompt = request
        .negative_prompt
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if negative_prompt.is_some_and(|value| value.chars().count() > 4_000) {
        return Err("Negative image prompt exceeds the 4000 character limit".to_string());
    }
    if negative_prompt.is_some_and(|value| value.contains('\0')) {
        return Err("Negative image prompt contains an invalid null character".to_string());
    }
    let runner_path = trimmed_path(&config.runner_path)
        .ok_or_else(|| "Image runner path is not configured".to_string())?;
    let seed = request.seed.unwrap_or_else(|| {
        let bytes = *uuid::Uuid::new_v4().as_bytes();
        i64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    });
    let arguments = build_native_arguments(
        &bundle,
        &profile,
        prompt,
        negative_prompt,
        seed,
        &output_path,
    );
    Ok(ResolvedImageJob {
        runner_path,
        bundle,
        profile,
        prompt: prompt.to_string(),
        negative_prompt: negative_prompt.map(str::to_string),
        seed,
        output_path,
        arguments,
    })
}

pub fn configured_output_dir(config: &ImageGenerationConfig) -> PathBuf {
    trimmed_path(&config.output_dir).unwrap_or_else(|| app_support_dir().join("images"))
}

pub fn preview_output_path(config: &ImageGenerationConfig) -> PathBuf {
    configured_output_dir(config).join("image-preview.png")
}

pub fn build_preview(
    config: &ImageGenerationConfig,
    request: &ImageGenerationRequest,
) -> Result<ImageGenerationPreview, String> {
    let resolved = resolve_job(config, request, preview_output_path(config))?;
    Ok(ImageGenerationPreview {
        bundle_id: resolved.bundle.id,
        bundle_name: resolved.bundle.name,
        profile_id: resolved.profile.id,
        profile_name: resolved.profile.name,
        width: resolved.profile.width,
        height: resolved.profile.height,
        steps: resolved.profile.steps,
        seed: resolved.seed,
        output_path: resolved.output_path.to_string_lossy().to_string(),
        arguments: resolved.arguments,
    })
}

pub fn output_png_dimensions(path: &Path) -> Result<(u32, u32), String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("Failed to open generated image: {error}"))?;
    let mut header = [0_u8; 24];
    file.read_exact(&mut header)
        .map_err(|error| format!("Generated image is incomplete: {error}"))?;
    if &header[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err("Generated output is not a valid PNG".to_string());
    }
    let width = u32::from_be_bytes(header[16..20].try_into().expect("four bytes"));
    let height = u32::from_be_bytes(header[20..24].try_into().expect("four bytes"));
    Ok((width, height))
}

fn build_native_arguments(
    bundle: &ImageModelBundleConfig,
    profile: &ImageGenerationProfileConfig,
    prompt: &str,
    negative_prompt: Option<&str>,
    seed: i64,
    output_path: &Path,
) -> Vec<String> {
    let mut arguments = vec![
        "--diffusion-model".to_string(),
        bundle.transformer_path.clone(),
        "--vae".to_string(),
        bundle.vae_path.clone(),
        "--llm".to_string(),
        bundle.text_encoder_path.clone(),
        "-p".to_string(),
        prompt.to_string(),
        "--cfg-scale".to_string(),
        profile.cfg_scale.to_string(),
        "--sampling-method".to_string(),
        profile.sampling_method.clone(),
        "--flow-shift".to_string(),
        profile.flow_shift.to_string(),
        "-W".to_string(),
        profile.width.to_string(),
        "-H".to_string(),
        profile.height.to_string(),
        "--steps".to_string(),
        profile.steps.to_string(),
        "--seed".to_string(),
        seed.to_string(),
        "--rng".to_string(),
        "cuda".to_string(),
        "--output".to_string(),
        output_path.to_string_lossy().to_string(),
        "--max-vram".to_string(),
        profile.max_vram_gib.to_string(),
        "--verbose".to_string(),
    ];
    if profile.auto_fit {
        arguments.push("--auto-fit".to_string());
    }
    if profile.diffusion_flash_attention {
        arguments.push("--diffusion-fa".to_string());
    }
    if let Some(negative_prompt) = negative_prompt {
        arguments.push("-n".to_string());
        arguments.push(negative_prompt.to_string());
    }
    arguments
}

fn trimmed_path(value: &str) -> Option<PathBuf> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
}

fn bundle_status(bundle: &ImageModelBundleConfig) -> ImageBundleStatus {
    let mut reasons = Vec::new();
    if bundle.id.trim().is_empty() {
        reasons.push("Bundle id is empty".to_string());
    }
    if bundle.architecture.trim() != "qwen_image" {
        reasons.push(format!(
            "Unsupported image architecture: {}",
            bundle.architecture
        ));
    }
    for (role, value, extensions) in [
        (
            "transformer",
            bundle.transformer_path.as_str(),
            &["gguf"][..],
        ),
        (
            "text encoder",
            bundle.text_encoder_path.as_str(),
            &["gguf"][..],
        ),
        ("VAE", bundle.vae_path.as_str(), &["safetensors"][..]),
    ] {
        let Some(path) = trimmed_path(value) else {
            reasons.push(format!("{role} path is not configured"));
            continue;
        };
        if !path.is_absolute() {
            reasons.push(format!("{role} path must be absolute"));
            continue;
        }
        if !path.is_file() {
            reasons.push(format!("{role} file does not exist: {}", path.display()));
            continue;
        }
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !extensions.contains(&extension.as_str()) {
            reasons.push(format!(
                "{role} has unsupported file type: {}",
                path.display()
            ));
        }
    }
    ImageBundleStatus {
        id: bundle.id.clone(),
        name: bundle.name.clone(),
        architecture: bundle.architecture.clone(),
        quantization: bundle.quantization.clone(),
        ready: reasons.is_empty(),
        reasons,
    }
}

fn profile_status(profile: &ImageGenerationProfileConfig) -> ImageProfileStatus {
    let reason = validate_profile(profile).err();
    ImageProfileStatus {
        id: profile.id.clone(),
        name: profile.name.clone(),
        width: profile.width,
        height: profile.height,
        steps: profile.steps,
        ready: reason.is_none(),
        reason,
    }
}

fn validate_profile(profile: &ImageGenerationProfileConfig) -> Result<(), String> {
    if profile.id.trim().is_empty() {
        return Err("Image profile id is empty".to_string());
    }
    if !(64..=MAX_IMAGE_DIMENSION).contains(&profile.width)
        || !(64..=MAX_IMAGE_DIMENSION).contains(&profile.height)
        || profile.width % 16 != 0
        || profile.height % 16 != 0
    {
        return Err(format!(
            "Image dimensions must be multiples of 16 between 64 and {MAX_IMAGE_DIMENSION}"
        ));
    }
    if !(1..=150).contains(&profile.steps) {
        return Err("Image steps must be between 1 and 150".to_string());
    }
    if !["euler", "euler_a", "heun", "dpm2", "dpm++2m"].contains(&profile.sampling_method.as_str())
    {
        return Err(format!(
            "Unsupported image sampler: {}",
            profile.sampling_method
        ));
    }
    if !profile.cfg_scale.is_finite() || !(0.0..=20.0).contains(&profile.cfg_scale) {
        return Err("Image CFG scale must be between 0 and 20".to_string());
    }
    if !profile.flow_shift.is_finite() || !(-20.0..=20.0).contains(&profile.flow_shift) {
        return Err("Image flow shift must be between -20 and 20".to_string());
    }
    if !profile.max_vram_gib.is_finite() || !(-16.0..=128.0).contains(&profile.max_vram_gib) {
        return Err("Image max VRAM must be between -16 and 128 GiB".to_string());
    }
    if profile.timeout_seconds < 30 || profile.timeout_seconds > 14_400 {
        return Err("Image timeout must be between 30 seconds and 4 hours".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_native_arguments, image_size_presets, output_png_dimensions,
        ImageGenerationProfileConfig, ImageModelBundleConfig, NativeProgressParser,
    };
    use crate::config::ImageGenerationConfig;
    use std::path::Path;

    fn bundle() -> ImageModelBundleConfig {
        ImageModelBundleConfig {
            id: "qwen".to_string(),
            name: "Qwen".to_string(),
            architecture: "qwen_image".to_string(),
            quantization: "Q6_K".to_string(),
            transformer_path: "C:\\models\\transformer.gguf".to_string(),
            text_encoder_path: "C:\\models\\encoder.gguf".to_string(),
            vae_path: "C:\\models\\vae.safetensors".to_string(),
        }
    }

    #[test]
    fn native_arguments_are_allowlisted_and_prompt_is_one_argument() {
        let profile = ImageGenerationProfileConfig::default();
        let prompt = "A cat beside a sign reading \"EXACT TEXT\"";
        let arguments = build_native_arguments(
            &bundle(),
            &profile,
            prompt,
            Some("blurry, distorted"),
            42,
            Path::new("output.png"),
        );
        assert_eq!(
            arguments[arguments.iter().position(|arg| arg == "-p").unwrap() + 1],
            prompt
        );
        assert!(arguments.contains(&"--auto-fit".to_string()));
        assert!(arguments.contains(&"--diffusion-fa".to_string()));
        assert!(!arguments.contains(&"--taesd".to_string()));
        assert_eq!(
            arguments[arguments.iter().position(|arg| arg == "-n").unwrap() + 1],
            "blurry, distorted"
        );
        assert_eq!(
            arguments[arguments.iter().position(|arg| arg == "--steps").unwrap() + 1],
            "50"
        );
    }

    #[test]
    fn quality_defaults_and_official_sizes_are_stable() {
        let config = ImageGenerationConfig::default();
        assert_eq!(config.default_bundle, "qwen-image-2512-q6");
        assert_eq!(config.default_profile, "quality");
        let quality = config
            .profiles
            .iter()
            .find(|profile| profile.id == config.default_profile)
            .expect("quality profile should exist");
        assert_eq!(
            (quality.width, quality.height, quality.steps),
            (1024, 1024, 50)
        );
        assert_eq!(quality.sampling_method, "euler");
        assert!((quality.cfg_scale - 2.5).abs() < f32::EPSILON);

        let presets = image_size_presets();
        let recommended = presets
            .first()
            .expect("recommended square preset should exist");
        assert_eq!(recommended.id, "recommended_square");
        assert_eq!((recommended.width, recommended.height), (1024, 1024));
        for aspect_ratio in ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"] {
            assert!(
                presets
                    .iter()
                    .any(|preset| preset.aspect_ratio == aspect_ratio),
                "missing {aspect_ratio} preset"
            );
        }
    }

    #[test]
    fn native_progress_parser_handles_split_and_repeated_chunks() {
        let mut parser = NativeProgressParser::default();
        assert_eq!(parser.push("loading model\r  |===="), None);
        let parsed = parser
            .push("====> | 11/50 - 3.54s/it\r")
            .expect("step should parse");
        assert_eq!(parsed.current_step, 11);
        assert_eq!(parsed.total_steps, 50);
        assert!((parsed.seconds_per_step - 3.54).abs() < f64::EPSILON);
        assert_eq!(parser.push("  |====> | 11/50 - 3.54s/it\r"), None);
    }

    #[test]
    fn native_progress_parser_matches_measured_qwen_runner_output_only() {
        let mut parser = NativeProgressParser::default();
        let parsed = parser
            .push("\u{1b}[DEBUG]\r  |=======================> | 23/50 - 3.58s/it\u{1b}[K")
            .expect("measured runner progress should parse");
        assert_eq!(parsed.current_step, 23);
        assert_eq!(parsed.total_steps, 50);
        assert_eq!(
            parser.push("  |########| 104/104 - 662.24MB/s\u{1b}[K"),
            None
        );
    }

    #[test]
    fn png_dimensions_reject_non_png_data() {
        let path = std::env::temp_dir().join(format!("ib-image-test-{}.bin", uuid::Uuid::new_v4()));
        std::fs::write(&path, b"not a PNG").expect("fixture should write");
        let result = output_png_dimensions(&path);
        let _ = std::fs::remove_file(path);
        assert!(result.is_err());
    }
}
