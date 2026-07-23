//! In-memory model registry — tracks scanned models.

use super::scanner::ScannedModel;
use super::{overrides::detect_effective_profile_with_arch, profiles::ModelProfile};

/// Registry of known model files.
pub struct ModelRegistry {
    models: Vec<ScannedModel>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Replace the registry contents with freshly scanned models.
    pub fn update(&mut self, models: Vec<ScannedModel>) {
        self.models = models;
    }

    /// Get all known models.
    pub fn list(&self) -> &[ScannedModel] {
        &self.models
    }

    /// Find a model by filename (case-insensitive).
    /// Handles path-style names like "qwen/qwen3.5-4b" by extracting the
    /// last segment after `/` for matching.
    pub fn find_by_name(&self, name: &str) -> Option<&ScannedModel> {
        let lower = name.to_lowercase();
        // 1. Exact match (with or without .gguf extension)
        if let Some(m) = self.models.iter().find(|m| {
            let fname = m.filename.to_lowercase();
            fname == lower || fname.trim_end_matches(".gguf") == lower
        }) {
            return Some(m);
        }
        // 2. Path-style name: "org/model-name" → match on "model-name" part
        let search_term = if lower.contains('/') {
            lower.rsplit('/').next().unwrap_or(&lower)
        } else {
            &lower
        };
        // 3. Substring match against the search term
        // Also try matching against the model's parent directory path
        // for "org/model" style lookups.
        self.models.iter().find(|m| {
            let fname = m.filename.to_lowercase();
            fname.contains(search_term)
                || fname.trim_end_matches(".gguf").starts_with(search_term)
                || m.path.to_string_lossy().to_lowercase().contains(&lower)
        })
    }

    /// Find a model by exact path.
    pub fn find_by_path(&self, path: &std::path::Path) -> Option<&ScannedModel> {
        self.models.iter().find(|m| m.path == path)
    }

    /// Resolve the effective runtime profile for a model name using the same
    /// GGUF architecture metadata captured by the scanner.
    ///
    /// Request-time callers must use this instead of filename-only detection;
    /// author names such as `Tess-4-27B` do not contain the underlying Qwen
    /// family even though their GGUF advertises `general.architecture=qwen35`.
    pub fn effective_profile_for_name(&self, name: &str) -> ModelProfile {
        if let Some(model) = self.find_by_name(name) {
            let architecture = model
                .gguf_meta
                .as_ref()
                .and_then(|meta| meta.architecture.as_deref());
            return detect_effective_profile_with_arch(&model.filename, architecture);
        }

        detect_effective_profile_with_arch(name, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{gguf::GgufMeta, profiles::ModelFamily};

    #[test]
    fn registry_uses_qwen35_architecture_for_tess_aliases() {
        let filename = "Tess-4-27B-Q4_K_M.gguf";
        let mut meta = GgufMeta::default();
        meta.architecture = Some("qwen35".to_string());
        let mut registry = ModelRegistry::new();
        registry.update(vec![ScannedModel {
            path: std::path::PathBuf::from(filename),
            filename: filename.to_string(),
            size_bytes: 1,
            profile: detect_effective_profile_with_arch(filename, Some("qwen35")),
            hf_metadata: None,
            gguf_meta: Some(meta),
        }]);

        assert_eq!(
            registry
                .effective_profile_for_name("Tess-4-27B-Q4_K_M")
                .family,
            ModelFamily::Qwen3_5
        );
    }

    #[test]
    fn unscanned_names_still_receive_filename_profile() {
        let registry = ModelRegistry::new();
        assert_eq!(
            registry
                .effective_profile_for_name("Qwen3-8B-Q4_K_M.gguf")
                .family,
            ModelFamily::Qwen3
        );
    }
}
