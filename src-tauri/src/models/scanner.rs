//! Scan directories for .gguf model files and extract metadata.

use std::path::{Path, PathBuf};

use super::{gguf::GgufMeta, overrides::HfModelMetadata, profiles::ModelProfile};

/// Information about a discovered model file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScannedModel {
    pub path: PathBuf,
    pub filename: String,
    pub size_bytes: u64,
    pub profile: ModelProfile,
    pub hf_metadata: Option<HfModelMetadata>,
    /// Architecture metadata read directly from the GGUF header.
    /// `None` if the file could not be parsed (e.g. during a fast rescan).
    pub gguf_meta: Option<GgufMeta>,
}

/// Scan a directory recursively for .gguf files.
pub fn scan_directory(dir: &Path) -> Vec<ScannedModel> {
    if !dir.is_dir() {
        tracing::warn!(?dir, "Model scan directory does not exist");
        return Vec::new();
    }
    let mut paths = Vec::new();
    collect_gguf_paths(dir, &mut paths);
    let mut models = parse_paths(&paths);
    models.sort_by(|a, b| a.filename.cmp(&b.filename));
    models
}

/// Walk `dir` recursively and collect paths to standalone `.gguf` model files.
///
/// Auxiliary files (mmproj sidecars, tokenizers, control vectors) are filtered
/// out here — by filename alone — so their headers are never parsed.
fn collect_gguf_paths(dir: &Path, out: &mut Vec<PathBuf>) {
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
            collect_gguf_paths(&path, out);
        } else if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            let is_aux = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(is_auxiliary_file);
            if !is_aux {
                out.push(path);
            }
        }
    }
}

/// Parse a batch of GGUF paths in parallel, then flush the metadata cache once.
///
/// Header parsing is I/O-bound and independent per file, so it fans out across
/// the rayon thread pool. The `(size, mtime) → GgufMeta` cache means unchanged
/// files on a rescan skip parsing entirely.
fn parse_paths(paths: &[PathBuf]) -> Vec<ScannedModel> {
    use rayon::prelude::*;
    let models: Vec<ScannedModel> = paths
        .par_iter()
        .filter_map(|p| parse_model_file(p))
        .collect();
    super::gguf::flush_gguf_cache();
    models
}

fn parse_model_file(path: &Path) -> Option<ScannedModel> {
    let filename = path.file_name()?.to_string_lossy().to_string();
    let metadata = std::fs::metadata(path).ok()?;
    let gguf_meta = super::gguf::read_gguf_meta_cached(path);
    let architecture = gguf_meta
        .as_ref()
        .and_then(|meta| meta.architecture.as_deref());
    let profile = super::overrides::detect_effective_profile_with_arch(&filename, architecture);
    let hf_metadata = super::overrides::effective_hf_metadata(&filename);
    Some(ScannedModel {
        path: path.to_path_buf(),
        filename,
        size_bytes: metadata.len(),
        profile,
        hf_metadata,
        gguf_meta,
    })
}

/// Check if a GGUF file is an auxiliary file (not a standalone model).
/// Filters out multimodal projection files, tokenizers, etc.
fn is_auxiliary_file(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    // Multimodal projection files (vision adapters)
    if lower.starts_with("mmproj")
        || lower.contains("-mmproj")
        || lower.contains("_mmproj")
        || lower.contains(".mmproj")
    {
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

#[cfg(test)]
mod tests {
    use super::is_auxiliary_file;

    #[test]
    fn treats_dot_mmproj_files_as_auxiliary() {
        assert!(is_auxiliary_file(
            "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved.mmproj-f16.gguf"
        ));
    }

    #[test]
    fn keeps_regular_models_visible() {
        assert!(!is_auxiliary_file(
            "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved.Q4_K_S.gguf"
        ));
    }
}

/// Scan all configured directories and return all found models.
///
/// Paths are collected across every directory and de-duplicated *before*
/// parsing, so overlapping scan dirs never parse the same file twice and the
/// whole batch is parsed in one parallel pass with a single cache flush.
pub fn scan_all(dirs: &[PathBuf]) -> Vec<ScannedModel> {
    let mut paths = Vec::new();
    for dir in dirs {
        if dir.is_dir() {
            collect_gguf_paths(dir, &mut paths);
        } else {
            tracing::warn!(?dir, "Model scan directory does not exist");
        }
    }
    paths.sort();
    paths.dedup();

    let mut all = parse_paths(&paths);
    all.sort_by(|a, b| a.path.cmp(&b.path));
    all.dedup_by(|a, b| a.path == b.path);
    all
}
