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
    pub fn find_by_name(&self, name: &str) -> Option<&ScannedModel> {
        let lower = name.to_lowercase();
        // 1. Try exact match (with or without extension)
        if let Some(m) = self.models.iter().find(|m| {
            let fname = m.filename.to_lowercase();
            fname == lower || fname.trim_end_matches(".gguf") == lower
        }) {
            return Some(m);
        }
        // 2. Fallback: substring match
        self.models
            .iter()
            .find(|m| m.filename.to_lowercase().contains(&lower))
    }

    /// Find a model by exact path.
    pub fn find_by_path(&self, path: &std::path::Path) -> Option<&ScannedModel> {
        self.models.iter().find(|m| m.path == path)
    }
}
