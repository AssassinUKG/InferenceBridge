use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::profiles::{ModelProfile, ParserType, RendererType, ThinkTagStyle, ToolCallFormat};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelProfileOverride {
    pub supports_vision: Option<bool>,
    pub hf_repo_id: Option<String>,
    pub hf_file: Option<String>,
    pub hf_template_path: Option<String>,
    pub hf_has_repo_template: Option<bool>,
    pub tool_call_format: Option<ToolCallFormat>,
    pub think_tag_style: Option<ThinkTagStyle>,
    pub interleaved_think_tool: Option<bool>,
    pub supports_parallel_tools: Option<bool>,
    pub default_max_output_tokens: Option<Option<u32>>,
    pub default_context_window: Option<Option<u32>>,
    pub max_context_window: Option<Option<u32>>,
    pub parser_type: Option<ParserType>,
    pub renderer_type: Option<RendererType>,
    pub stop_markers: Option<Vec<String>>,
    pub allow_fallback_extraction: Option<bool>,
    pub default_presence_penalty: Option<Option<f32>>,
    pub default_temperature: Option<Option<f32>>,
    pub default_top_p: Option<Option<f32>>,
    pub default_top_k: Option<Option<i32>>,
    pub default_min_p: Option<Option<f32>>,
    pub disable_thinking_for_tools: Option<bool>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelOverrideStore {
    #[serde(default)]
    pub models: HashMap<String, ModelProfileOverride>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HfModelMetadata {
    pub repo_id: Option<String>,
    pub file: Option<String>,
    pub template_path: Option<String>,
    pub has_repo_template: bool,
    pub supports_vision: Option<bool>,
}

impl ModelProfileOverride {
    pub fn apply(&self, profile: &mut ModelProfile) {
        if let Some(value) = self.supports_vision {
            profile.supports_vision = value;
        }
        if let Some(value) = self.tool_call_format {
            profile.tool_call_format = value;
        }
        if let Some(value) = self.think_tag_style {
            profile.think_tag_style = value;
        }
        if let Some(value) = self.interleaved_think_tool {
            profile.interleaved_think_tool = value;
        }
        if let Some(value) = self.supports_parallel_tools {
            profile.supports_parallel_tools = value;
        }
        if let Some(value) = self.default_max_output_tokens.clone() {
            profile.default_max_output_tokens = value;
        }
        if let Some(value) = self.default_context_window.clone() {
            profile.default_context_window = value;
        }
        if let Some(value) = self.max_context_window.clone() {
            profile.max_context_window = value;
        }
        if let Some(value) = self.parser_type {
            profile.parser_type = value;
        }
        if let Some(value) = self.renderer_type {
            profile.renderer_type = value;
        }
        if let Some(value) = self.stop_markers.clone() {
            profile.stop_markers = value;
        }
        if let Some(value) = self.allow_fallback_extraction {
            profile.allow_fallback_extraction = value;
        }
        if let Some(value) = self.default_presence_penalty {
            profile.default_presence_penalty = value;
        }
        if let Some(value) = self.default_temperature {
            profile.default_temperature = value;
        }
        if let Some(value) = self.default_top_p {
            profile.default_top_p = value;
        }
        if let Some(value) = self.default_top_k {
            profile.default_top_k = value;
        }
        if let Some(value) = self.default_min_p {
            profile.default_min_p = value;
        }
        if let Some(value) = self.disable_thinking_for_tools {
            profile.disable_thinking_for_tools = value;
        }
    }
}

impl ModelOverrideStore {
    pub fn matching_override(&self, model_name: &str) -> Option<&ModelProfileOverride> {
        let lower = model_name.to_lowercase();

        self.models.get(&lower).or_else(|| {
            self.models.iter().find_map(|(key, value)| {
                let key_lower = key.to_lowercase();
                if lower == key_lower
                    || lower.trim_end_matches(".gguf") == key_lower.trim_end_matches(".gguf")
                    || lower.contains(&key_lower)
                {
                    Some(value)
                } else {
                    None
                }
            })
        })
    }
}

/// Recommended-settings defaults keyed by GGUF `general.architecture`.
///
/// This is the runtime-editable surface for keeping sampler/profile defaults
/// current with what model authors publish — edit `model-defaults.json` and
/// reload, no recompile. Applied *after* the base profile but *before* the
/// per-model overrides in `model-overrides.json`, so a per-model entry always
/// wins.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArchDefaultsStore {
    #[serde(default)]
    pub architectures: HashMap<String, ModelProfileOverride>,
}

impl ArchDefaultsStore {
    pub fn matching(&self, architecture: &str) -> Option<&ModelProfileOverride> {
        self.architectures.get(&architecture.to_lowercase())
    }
}

fn overrides_path() -> PathBuf {
    crate::config::app_support_dir().join("model-overrides.json")
}

fn arch_defaults_path() -> PathBuf {
    crate::config::app_support_dir().join("model-defaults.json")
}

/// Load architecture-keyed recommended defaults. Missing file = no defaults.
pub fn load_arch_defaults() -> ArchDefaultsStore {
    let path = arch_defaults_path();
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(_) => return ArchDefaultsStore::default(),
    };

    match serde_json::from_str::<ArchDefaultsStore>(&contents) {
        Ok(store) => store,
        Err(error) => {
            tracing::warn!(path = %path.display(), error = %error, "Failed to parse model defaults");
            ArchDefaultsStore::default()
        }
    }
}

fn save_overrides(store: &ModelOverrideStore) -> Result<(), String> {
    let path = overrides_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create {}: {error}", parent.display()))?;
    }

    let contents = serde_json::to_string_pretty(store)
        .map_err(|error| format!("Failed to serialize model overrides: {error}"))?;
    std::fs::write(&path, contents)
        .map_err(|error| format!("Failed to write {}: {error}", path.display()))?;
    Ok(())
}

pub fn load_overrides() -> ModelOverrideStore {
    let path = overrides_path();
    let contents = match std::fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(_) => return ModelOverrideStore::default(),
    };

    match serde_json::from_str::<ModelOverrideStore>(&contents) {
        Ok(store) => store,
        Err(error) => {
            tracing::warn!(path = %path.display(), error = %error, "Failed to parse model overrides");
            ModelOverrideStore::default()
        }
    }
}

pub fn detect_effective_profile(model_name: &str) -> ModelProfile {
    detect_effective_profile_with_arch(model_name, None)
}

/// Build the effective profile, layering (highest precedence last):
/// base profile (name + architecture) → architecture defaults
/// (`model-defaults.json`) → per-model overrides (`model-overrides.json`).
pub fn detect_effective_profile_with_arch(
    model_name: &str,
    architecture: Option<&str>,
) -> ModelProfile {
    let mut profile = ModelProfile::detect_with_arch(model_name, architecture);
    if let Some(arch) = architecture {
        if let Some(defaults) = load_arch_defaults().matching(arch) {
            defaults.apply(&mut profile);
        }
    }
    if let Some(override_entry) = load_overrides().matching_override(model_name) {
        override_entry.apply(&mut profile);
    }
    profile
}

pub fn effective_override(model_name: &str) -> Option<ModelProfileOverride> {
    load_overrides().matching_override(model_name).cloned()
}

pub fn effective_hf_metadata(model_name: &str) -> Option<HfModelMetadata> {
    let override_entry = load_overrides().matching_override(model_name)?.clone();
    if override_entry.hf_repo_id.is_none()
        && override_entry.hf_file.is_none()
        && override_entry.hf_template_path.is_none()
        && override_entry.hf_has_repo_template.is_none()
        && override_entry.supports_vision.is_none()
    {
        return None;
    }

    Some(HfModelMetadata {
        repo_id: override_entry.hf_repo_id,
        file: override_entry.hf_file,
        template_path: override_entry.hf_template_path,
        has_repo_template: override_entry.hf_has_repo_template.unwrap_or(false),
        supports_vision: override_entry.supports_vision,
    })
}

pub fn set_model_supports_vision_override(
    model_name: &str,
    supports_vision: bool,
) -> Result<(), String> {
    let normalized_name = model_name.trim().to_lowercase();
    if normalized_name.is_empty() {
        return Err("Model name cannot be empty when saving overrides".to_string());
    }

    let mut store = load_overrides();
    let entry = store.models.entry(normalized_name).or_default();
    entry.supports_vision = Some(supports_vision);
    save_overrides(&store)
}

pub fn set_model_hf_metadata_override(
    model_name: &str,
    metadata: HfModelMetadata,
) -> Result<(), String> {
    let normalized_name = model_name.trim().to_lowercase();
    if normalized_name.is_empty() {
        return Err("Model name cannot be empty when saving overrides".to_string());
    }

    let mut store = load_overrides();
    let entry = store.models.entry(normalized_name).or_default();
    entry.hf_repo_id = metadata.repo_id;
    entry.hf_file = metadata.file;
    entry.hf_template_path = metadata.template_path;
    entry.hf_has_repo_template = Some(metadata.has_repo_template);
    if let Some(supports_vision) = metadata.supports_vision {
        entry.supports_vision = Some(supports_vision);
    }
    save_overrides(&store)
}
