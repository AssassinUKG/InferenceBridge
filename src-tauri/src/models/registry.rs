//! In-memory model registry — tracks scanned models.

use super::scanner::ScannedModel;

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
}
