//! Scan directories for .gguf model files and extract metadata.

use std::path::{Path, PathBuf};

use super::profiles::ModelProfile;

/// Information about a discovered model file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScannedModel {
    pub path: PathBuf,
    pub filename: String,
    pub size_bytes: u64,
    pub profile: ModelProfile,
}

/// Scan a directory recursively for .gguf files.
pub fn scan_directory(dir: &Path) -> Vec<ScannedModel> {
    let mut models = Vec::new();
    if !dir.is_dir() {
        tracing::warn!(?dir, "Model scan directory does not exist");
        return models;
    }
    scan_recursive(dir, &mut models);
    models.sort_by(|a, b| a.filename.cmp(&b.filename));
    models
}

fn scan_recursive(dir: &Path, models: &mut Vec<ScannedModel>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(?dir, error = %e, "Failed to read model directory");
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_recursive(&path, models);
        } else if let Some(ext) = path.extension() {
            if ext.eq_ignore_ascii_case("gguf") {
                if let Some(model) = parse_model_file(&path) {
                    // Skip non-model files (multimodal projections, tokenizers, etc.)
                    if !is_auxiliary_file(&model.filename) {
                        models.push(model);
                    }
                }
            }
        }
    }
}

fn parse_model_file(path: &Path) -> Option<ScannedModel> {
    let filename = path.file_name()?.to_string_lossy().to_string();
    let metadata = std::fs::metadata(path).ok()?;
    let profile = ModelProfile::detect(&filename);
    Some(ScannedModel {
        path: path.to_path_buf(),
        filename,
        size_bytes: metadata.len(),
        profile,
    })
}

/// Check if a GGUF file is an auxiliary file (not a standalone model).
/// Filters out multimodal projection files, tokenizers, etc.
fn is_auxiliary_file(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    // Multimodal projection files (vision adapters)
    if lower.starts_with("mmproj") || lower.contains("-mmproj") || lower.contains("_mmproj") {
        return true;
    }
    // Tokenizer / vocab-only files
    if lower.contains("tokenizer") || lower.contains("vocab-only") {
        return true;
    }
    // Control vectors / LoRA adapters
    if lower.contains("control-vector") || lower.contains("control_vector") {
        return true;
    }
    false
}

/// Scan all configured directories and return all found models.
pub fn scan_all(dirs: &[PathBuf]) -> Vec<ScannedModel> {
    let mut all = Vec::new();
    for dir in dirs {
        all.extend(scan_directory(dir));
    }
    // Deduplicate by path
    all.sort_by(|a, b| a.path.cmp(&b.path));
    all.dedup_by(|a, b| a.path == b.path);
    all
}
