//! Revision-aware Hugging Face metadata and chat-template updates.
//!
//! Model weights are deliberately outside this module. Only a short allowlist
//! of small tokenizer/config files can be fetched, each with a hard size cap.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: u32 = 1;
pub const HF_SIDECAR_MAX_BYTES: usize = 2 * 1024 * 1024;
pub const HF_SIDECAR_DEFAULT_FILES: &[&str] = &[
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
];
const CANONICAL_TEMPLATE_FILE: &str = "chat_template.jinja";

#[derive(Debug, Clone)]
pub struct SidecarTarget {
    pub filename: String,
    pub repo_id: Option<String>,
    pub preferred_template_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfSidecarSyncFile {
    pub repo_id: String,
    pub path: String,
    pub cached_path: Option<String>,
    pub status: String,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfSidecarSyncSummary {
    pub mode: String,
    pub models_checked: usize,
    pub repos_checked: usize,
    pub repos_with_updates: usize,
    pub repos_updated: usize,
    pub files_cached: usize,
    pub files_updated: usize,
    pub files_unchanged: usize,
    pub files_skipped: usize,
    pub files_failed: usize,
    pub hf_token_configured: bool,
    pub cache_root: String,
    pub results: Vec<HfSidecarSyncFile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfSidecarCacheStatus {
    pub filename: String,
    pub repo_id: Option<String>,
    pub source_repo_id: Option<String>,
    pub template_path: Option<String>,
    pub template_cached: bool,
    pub template_cache_path: Option<String>,
    pub template_source: Option<String>,
    pub sidecar_cached_count: usize,
    pub sidecar_expected_count: usize,
    pub sidecar_cache_dir: Option<String>,
    pub active_revision: Option<String>,
    pub remote_revision: Option<String>,
    pub update_available: bool,
    pub last_checked_at: Option<String>,
    pub rollback_available: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HfSidecarRollbackSummary {
    pub repo_id: String,
    pub restored_revision: Option<String>,
    pub replaced_revision: Option<String>,
    pub files_restored: usize,
    pub template_restored: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct SidecarManifest {
    #[serde(default = "manifest_schema_version")]
    schema_version: u32,
    #[serde(default)]
    repo_id: String,
    #[serde(default)]
    source_repo_id: Option<String>,
    #[serde(default)]
    active_snapshot: Option<String>,
    #[serde(default)]
    active_revision: Option<String>,
    #[serde(default)]
    previous_snapshot: Option<String>,
    #[serde(default)]
    previous_revision: Option<String>,
    #[serde(default)]
    remote_revision: Option<String>,
    #[serde(default)]
    update_available: bool,
    #[serde(default)]
    last_checked_at: Option<String>,
    #[serde(default)]
    template_source: Option<String>,
}

fn manifest_schema_version() -> u32 {
    SCHEMA_VERSION
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotRecord {
    schema_version: u32,
    repo_id: String,
    #[serde(default)]
    source_repo_id: Option<String>,
    revision: String,
    created_at: String,
    #[serde(default)]
    files: Vec<SnapshotFile>,
    #[serde(default)]
    template: Option<SnapshotTemplate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotFile {
    source_path: String,
    snapshot_path: String,
    size_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotTemplate {
    source: String,
    snapshot_path: String,
    size_bytes: usize,
}

#[derive(Debug, Deserialize)]
struct HfRepoDescriptor {
    #[serde(default)]
    sha: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    siblings: Vec<HfRepoSibling>,
}

#[derive(Debug, Deserialize)]
struct HfRepoSibling {
    #[serde(default)]
    rfilename: String,
}

#[derive(Debug, Clone)]
struct RemoteAsset {
    source_path: String,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct RemoteTemplate {
    source: String,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct RemoteBundle {
    source_repo_id: String,
    revision: String,
    assets: Vec<RemoteAsset>,
    template: Option<RemoteTemplate>,
    missing: Vec<String>,
}

pub fn sanitize_hf_cache_segment(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub fn is_allowed_hf_sidecar_path(path: &str) -> bool {
    let trimmed = path.trim().trim_start_matches('/');
    if trimmed.is_empty()
        || trimmed.contains('\\')
        || trimmed
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return false;
    }
    let filename = Path::new(trimmed)
        .file_name()
        .map(|value| value.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    matches!(
        filename.as_str(),
        "chat_template.jinja"
            | "tokenizer_config.json"
            | "config.json"
            | "generation_config.json"
            | "special_tokens_map.json"
    )
}

fn repo_sidecar_dir(repo_id: &str) -> PathBuf {
    crate::config::app_support_dir()
        .join("hf-sidecars")
        .join(sanitize_hf_cache_segment(repo_id))
}

pub fn hf_sidecar_cache_path(repo_id: &str, relative_path: &str) -> PathBuf {
    repo_sidecar_dir(repo_id).join(relative_path.trim().trim_start_matches('/'))
}

fn canonical_template_cache_path(repo_id: &str) -> PathBuf {
    crate::config::app_support_dir()
        .join("hf-templates")
        .join(sanitize_hf_cache_segment(repo_id))
        .join(CANONICAL_TEMPLATE_FILE)
}

pub fn hf_template_cache_path(repo_id: &str, relative_path: &str) -> PathBuf {
    crate::config::app_support_dir()
        .join("hf-templates")
        .join(sanitize_hf_cache_segment(repo_id))
        .join(relative_path.trim().trim_start_matches('/'))
}

fn manifest_path(repo_id: &str) -> PathBuf {
    repo_sidecar_dir(repo_id).join("update-manifest.json")
}

fn snapshots_dir(repo_id: &str) -> PathBuf {
    repo_sidecar_dir(repo_id).join("snapshots")
}

fn snapshot_dir(repo_id: &str, snapshot_id: &str) -> PathBuf {
    snapshots_dir(repo_id).join(sanitize_hf_cache_segment(snapshot_id))
}

fn snapshot_record_path(repo_id: &str, snapshot_id: &str) -> PathBuf {
    snapshot_dir(repo_id, snapshot_id).join("snapshot.json")
}

fn read_manifest(repo_id: &str) -> Result<SidecarManifest, String> {
    let path = manifest_path(repo_id);
    if !path.exists() {
        return Ok(SidecarManifest {
            schema_version: SCHEMA_VERSION,
            repo_id: repo_id.to_string(),
            ..SidecarManifest::default()
        });
    }
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    let mut manifest = serde_json::from_slice::<SidecarManifest>(&bytes)
        .map_err(|error| format!("Failed to parse {}: {error}", path.display()))?;
    if manifest.repo_id.is_empty() {
        manifest.repo_id = repo_id.to_string();
    }
    Ok(manifest)
}

fn read_snapshot(repo_id: &str, snapshot_id: &str) -> Result<SnapshotRecord, String> {
    let path = snapshot_record_path(repo_id, snapshot_id);
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("Failed to parse {}: {error}", path.display()))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("Failed to create {}: {error}", parent.display()))?;
    let nonce = uuid::Uuid::new_v4().simple().to_string();
    let next = parent.join(format!(".ib-next-{nonce}"));
    let old = parent.join(format!(".ib-old-{nonce}"));
    std::fs::write(&next, bytes)
        .map_err(|error| format!("Failed to write {}: {error}", next.display()))?;

    let had_existing = path.exists();
    if had_existing {
        std::fs::rename(path, &old).map_err(|error| {
            let _ = std::fs::remove_file(&next);
            format!(
                "Failed to stage replacement for {}: {error}",
                path.display()
            )
        })?;
    }
    if let Err(error) = std::fs::rename(&next, path) {
        if had_existing {
            let _ = std::fs::rename(&old, path);
        }
        let _ = std::fs::remove_file(&next);
        return Err(format!("Failed to replace {}: {error}", path.display()));
    }
    if had_existing {
        let _ = std::fs::remove_file(&old);
    }
    Ok(())
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("Failed to serialize {}: {error}", path.display()))?;
    atomic_write(path, &bytes)
}

fn hf_api_url(repo_id: &str) -> Result<reqwest::Url, String> {
    let mut url = reqwest::Url::parse("https://huggingface.co/api/models/")
        .map_err(|error| format!("Failed to build Hugging Face API URL: {error}"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| "Hugging Face API URL cannot contain path segments".to_string())?;
        segments.pop_if_empty();
        for segment in repo_id.trim().split('/') {
            segments.push(segment);
        }
    }
    Ok(url)
}

fn hf_resolve_url(
    repo_id: &str,
    revision: &str,
    relative_path: &str,
) -> Result<reqwest::Url, String> {
    let mut url = reqwest::Url::parse("https://huggingface.co/")
        .map_err(|error| format!("Failed to build Hugging Face download URL: {error}"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| "Hugging Face download URL cannot contain path segments".to_string())?;
        segments.pop_if_empty();
        for segment in repo_id.trim().split('/') {
            segments.push(segment);
        }
        segments.push("resolve");
        segments.push(revision);
        for segment in relative_path.trim().trim_start_matches('/').split('/') {
            segments.push(segment);
        }
    }
    Ok(url)
}

fn authenticated_request(
    client: &reqwest::Client,
    url: reqwest::Url,
    hf_api_key: Option<&str>,
) -> reqwest::RequestBuilder {
    let request = client.get(url);
    if let Some(key) = hf_api_key.map(str::trim).filter(|value| !value.is_empty()) {
        request.bearer_auth(key)
    } else {
        request
    }
}

async fn fetch_repo_descriptor(
    client: &reqwest::Client,
    repo_id: &str,
    hf_api_key: Option<&str>,
) -> Result<HfRepoDescriptor, String> {
    let url = hf_api_url(repo_id)?;
    let response = authenticated_request(client, url.clone(), hf_api_key)
        .send()
        .await
        .map_err(|error| format!("Failed to check {repo_id}: {error}"))?;
    if response.status() == reqwest::StatusCode::UNAUTHORIZED
        || response.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err(format!(
            "Hugging Face access to {repo_id} was denied (HTTP {}). Add or refresh the token in Settings.",
            response.status()
        ));
    }
    if !response.status().is_success() {
        return Err(format!(
            "Failed to check {repo_id}: Hugging Face returned HTTP {}",
            response.status()
        ));
    }
    let descriptor = response
        .json::<HfRepoDescriptor>()
        .await
        .map_err(|error| format!("Failed to parse Hugging Face metadata for {repo_id}: {error}"))?;
    if descriptor.sha.trim().is_empty() {
        return Err(format!(
            "Hugging Face did not return a commit revision for {repo_id}"
        ));
    }
    Ok(descriptor)
}

fn basename_lower(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_else(|| path.to_ascii_lowercase())
}

fn choose_repo_file(
    siblings: &[HfRepoSibling],
    filename: &str,
    preferred: Option<&str>,
) -> Option<String> {
    let mut matches = siblings
        .iter()
        .map(|sibling| sibling.rfilename.trim().trim_start_matches('/'))
        .filter(|path| is_allowed_hf_sidecar_path(path))
        .filter(|path| basename_lower(path) == filename.to_ascii_lowercase())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if let Some(preferred) = preferred.map(str::trim).filter(|value| !value.is_empty()) {
        if let Some(found) = matches
            .iter()
            .find(|path| path.eq_ignore_ascii_case(preferred))
        {
            return Some(found.clone());
        }
    }
    matches.sort_by_key(|path| {
        (
            path.matches('/').count(),
            path.len(),
            path.to_ascii_lowercase(),
        )
    });
    matches.into_iter().next()
}

fn valid_hf_repo_id(value: &str) -> bool {
    let mut parts = value.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    !owner.is_empty()
        && !name.is_empty()
        && parts.next().is_none()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '-' | '_' | '.'))
}

fn upstream_base_repo(tags: &[String], linked_has_template_metadata: bool) -> Option<String> {
    let prefixes: &[&str] = if linked_has_template_metadata {
        &["base_model:quantized:"]
    } else {
        &["base_model:quantized:", "base_model:"]
    };
    for prefix in prefixes {
        if let Some(repo_id) = tags
            .iter()
            .find_map(|tag| tag.strip_prefix(prefix))
            .map(str::trim)
            .filter(|value| valid_hf_repo_id(value))
        {
            return Some(repo_id.to_string());
        }
    }
    None
}

fn descriptor_has_template_metadata(descriptor: &HfRepoDescriptor) -> bool {
    choose_repo_file(&descriptor.siblings, CANONICAL_TEMPLATE_FILE, None).is_some()
        || choose_repo_file(&descriptor.siblings, "tokenizer_config.json", None).is_some()
}

async fn fetch_small_file(
    client: &reqwest::Client,
    repo_id: &str,
    revision: &str,
    relative_path: &str,
    hf_api_key: Option<&str>,
) -> Result<Vec<u8>, String> {
    if !is_allowed_hf_sidecar_path(relative_path) {
        return Err(format!(
            "Blocked non-sidecar Hugging Face path: {relative_path}"
        ));
    }
    let url = hf_resolve_url(repo_id, revision, relative_path)?;
    let response = authenticated_request(client, url, hf_api_key)
        .send()
        .await
        .map_err(|error| format!("Failed to fetch {relative_path} from {repo_id}: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Failed to fetch {relative_path} from {repo_id}: HTTP {}",
            response.status()
        ));
    }
    if response
        .content_length()
        .is_some_and(|length| length > HF_SIDECAR_MAX_BYTES as u64)
    {
        return Err(format!(
            "Blocked {relative_path} from {repo_id}: file exceeds the 2 MiB sidecar limit"
        ));
    }
    let bytes = response
        .bytes()
        .await
        .map_err(|error| format!("Failed to read {relative_path} from {repo_id}: {error}"))?;
    if bytes.len() > HF_SIDECAR_MAX_BYTES {
        return Err(format!(
            "Blocked {relative_path} from {repo_id}: {} bytes exceeds the 2 MiB sidecar limit",
            bytes.len()
        ));
    }
    Ok(bytes.to_vec())
}

fn validate_json_sidecar(path: &str, bytes: &[u8]) -> Result<(), String> {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .map(|_| ())
        .map_err(|error| format!("Rejected invalid {path}: {error}"))
}

pub(crate) fn validate_template(source: &str, bytes: &[u8]) -> Result<(), String> {
    let template = std::str::from_utf8(bytes)
        .map_err(|error| format!("Rejected non-UTF-8 chat template from {source}: {error}"))?;
    if template.trim().is_empty() {
        return Err(format!("Rejected empty chat template from {source}"));
    }
    if template.contains('\0') {
        return Err(format!(
            "Rejected chat template containing NUL bytes from {source}"
        ));
    }
    if !template.contains("messages") || !(template.contains("{%") || template.contains("{{")) {
        return Err(format!(
            "Rejected chat template from {source}: it does not look like a messages-based Jinja template"
        ));
    }
    Ok(())
}

fn template_from_named_collection(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(template) => Some(template.clone()),
        serde_json::Value::Array(items) => {
            for wanted in ["default", "tool_use"] {
                if let Some(template) = items.iter().find_map(|item| {
                    let object = item.as_object()?;
                    let name = object.get("name")?.as_str()?;
                    name.eq_ignore_ascii_case(wanted)
                        .then(|| object.get("template")?.as_str().map(str::to_string))
                        .flatten()
                }) {
                    return Some(template);
                }
            }
            items.iter().find_map(|item| {
                item.as_object()?
                    .get("template")?
                    .as_str()
                    .map(str::to_string)
            })
        }
        serde_json::Value::Object(map) => {
            for wanted in ["default", "tool_use"] {
                if let Some(template) = map.get(wanted).and_then(serde_json::Value::as_str) {
                    return Some(template.to_string());
                }
            }
            map.values()
                .find_map(serde_json::Value::as_str)
                .map(str::to_string)
        }
        _ => None,
    }
}

fn extract_tokenizer_chat_template(bytes: &[u8]) -> Result<Option<Vec<u8>>, String> {
    let config = serde_json::from_slice::<serde_json::Value>(bytes)
        .map_err(|error| format!("Rejected invalid tokenizer_config.json: {error}"))?;
    let Some(value) = config.get("chat_template") else {
        return Ok(None);
    };
    let Some(template) = template_from_named_collection(value) else {
        return Err(
            "tokenizer_config.json contains an unsupported chat_template shape".to_string(),
        );
    };
    let bytes = template.into_bytes();
    validate_template("tokenizer_config.json#chat_template", &bytes)?;
    Ok(Some(bytes))
}

async fn fetch_remote_bundle(
    client: &reqwest::Client,
    repo_id: &str,
    preferred_template_path: Option<&str>,
    hf_api_key: Option<&str>,
) -> Result<RemoteBundle, String> {
    let linked_descriptor = fetch_repo_descriptor(client, repo_id, hf_api_key).await?;
    let (source_repo_id, descriptor) = if let Some(base_repo) = upstream_base_repo(
        &linked_descriptor.tags,
        descriptor_has_template_metadata(&linked_descriptor),
    ) {
        match fetch_repo_descriptor(client, &base_repo, hf_api_key).await {
            Ok(base_descriptor) if descriptor_has_template_metadata(&base_descriptor) => {
                (base_repo, base_descriptor)
            }
            _ => (repo_id.to_string(), linked_descriptor),
        }
    } else {
        (repo_id.to_string(), linked_descriptor)
    };
    let mut selected = Vec::new();
    let template_path = choose_repo_file(
        &descriptor.siblings,
        CANONICAL_TEMPLATE_FILE,
        preferred_template_path,
    );
    if let Some(path) = template_path.as_ref() {
        selected.push(path.clone());
    }
    for filename in HF_SIDECAR_DEFAULT_FILES {
        if let Some(path) = choose_repo_file(&descriptor.siblings, filename, None) {
            selected.push(path);
        }
    }
    selected.sort();
    selected.dedup();

    let mut assets = Vec::new();
    for path in selected {
        let bytes =
            fetch_small_file(client, &source_repo_id, &descriptor.sha, &path, hf_api_key).await?;
        if path.to_ascii_lowercase().ends_with(".json") {
            validate_json_sidecar(&path, &bytes)?;
        } else {
            validate_template(&path, &bytes)?;
        }
        assets.push(RemoteAsset {
            source_path: path,
            bytes,
        });
    }

    let standalone_template = template_path.as_ref().and_then(|path| {
        assets
            .iter()
            .find(|asset| asset.source_path.eq_ignore_ascii_case(path))
            .map(|asset| RemoteTemplate {
                source: asset.source_path.clone(),
                bytes: asset.bytes.clone(),
            })
    });
    let template = if standalone_template.is_some() {
        standalone_template
    } else if let Some(tokenizer) = assets
        .iter()
        .find(|asset| basename_lower(&asset.source_path) == "tokenizer_config.json")
    {
        extract_tokenizer_chat_template(&tokenizer.bytes)?.map(|bytes| RemoteTemplate {
            source: "tokenizer_config.json#chat_template".to_string(),
            bytes,
        })
    } else {
        None
    };

    let found_names = assets
        .iter()
        .map(|asset| basename_lower(&asset.source_path))
        .collect::<HashSet<_>>();
    let mut missing = HF_SIDECAR_DEFAULT_FILES
        .iter()
        .filter(|filename| !found_names.contains(**filename))
        .map(|filename| filename.to_string())
        .collect::<Vec<_>>();
    if template.is_none() {
        missing.push(CANONICAL_TEMPLATE_FILE.to_string());
    }

    Ok(RemoteBundle {
        source_repo_id,
        revision: descriptor.sha,
        assets,
        template,
        missing,
    })
}

fn asset_active_path(repo_id: &str, asset: &RemoteAsset) -> PathBuf {
    hf_sidecar_cache_path(repo_id, &asset.source_path)
}

fn file_differs(path: &Path, bytes: &[u8]) -> bool {
    std::fs::read(path).map_or(true, |current| current != bytes)
}

fn bundle_change_count(repo_id: &str, bundle: &RemoteBundle) -> usize {
    let files = bundle
        .assets
        .iter()
        .filter(|asset| {
            !asset
                .source_path
                .to_ascii_lowercase()
                .ends_with("chat_template.jinja")
        })
        .filter(|asset| file_differs(&asset_active_path(repo_id, asset), &asset.bytes))
        .count();
    let template = bundle.template.as_ref().map_or(0, |template| {
        usize::from(file_differs(
            &canonical_template_cache_path(repo_id),
            &template.bytes,
        ))
    });
    files + template
}

fn snapshot_asset_path(snapshot_root: &Path, source_path: &str) -> PathBuf {
    snapshot_root.join("files").join(source_path)
}

fn store_remote_snapshot(
    repo_id: &str,
    bundle: &RemoteBundle,
) -> Result<(String, SnapshotRecord), String> {
    let snapshot_id = format!("remote-{}", bundle.revision);
    let root = snapshot_dir(repo_id, &snapshot_id);
    let record_path = snapshot_record_path(repo_id, &snapshot_id);
    if record_path.exists() {
        return Ok((snapshot_id.clone(), read_snapshot(repo_id, &snapshot_id)?));
    }

    let mut files = Vec::new();
    for asset in bundle.assets.iter().filter(|asset| {
        !asset
            .source_path
            .to_ascii_lowercase()
            .ends_with("chat_template.jinja")
    }) {
        let destination = snapshot_asset_path(&root, &asset.source_path);
        atomic_write(&destination, &asset.bytes)?;
        files.push(SnapshotFile {
            source_path: asset.source_path.clone(),
            snapshot_path: destination
                .strip_prefix(&root)
                .unwrap_or(&destination)
                .to_string_lossy()
                .to_string(),
            size_bytes: asset.bytes.len(),
        });
    }
    let template = if let Some(template) = bundle.template.as_ref() {
        let destination = root.join("effective").join(CANONICAL_TEMPLATE_FILE);
        atomic_write(&destination, &template.bytes)?;
        Some(SnapshotTemplate {
            source: template.source.clone(),
            snapshot_path: destination
                .strip_prefix(&root)
                .unwrap_or(&destination)
                .to_string_lossy()
                .to_string(),
            size_bytes: template.bytes.len(),
        })
    } else {
        None
    };
    let record = SnapshotRecord {
        schema_version: SCHEMA_VERSION,
        repo_id: repo_id.to_string(),
        source_repo_id: Some(bundle.source_repo_id.clone()),
        revision: bundle.revision.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        files,
        template,
    };
    write_json_atomic(&record_path, &record)?;
    Ok((snapshot_id, record))
}

fn capture_legacy_snapshot(
    repo_id: &str,
    preferred_template_path: Option<&str>,
) -> Result<Option<(String, SnapshotRecord)>, String> {
    let snapshot_id = format!("local-{}", uuid::Uuid::new_v4().simple());
    let root = snapshot_dir(repo_id, &snapshot_id);
    let mut files = Vec::new();
    for filename in HF_SIDECAR_DEFAULT_FILES {
        let source = hf_sidecar_cache_path(repo_id, filename);
        if !source.exists() {
            continue;
        }
        let bytes = std::fs::read(&source)
            .map_err(|error| format!("Failed to back up {}: {error}", source.display()))?;
        let destination = snapshot_asset_path(&root, filename);
        atomic_write(&destination, &bytes)?;
        files.push(SnapshotFile {
            source_path: filename.to_string(),
            snapshot_path: destination
                .strip_prefix(&root)
                .unwrap_or(&destination)
                .to_string_lossy()
                .to_string(),
            size_bytes: bytes.len(),
        });
    }

    let mut template_candidates = vec![canonical_template_cache_path(repo_id)];
    if let Some(path) = preferred_template_path
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        template_candidates.push(hf_template_cache_path(repo_id, path));
    }
    template_candidates.dedup();
    let template = template_candidates
        .into_iter()
        .find(|path| path.exists())
        .map(|source| -> Result<SnapshotTemplate, String> {
            let bytes = std::fs::read(&source)
                .map_err(|error| format!("Failed to back up {}: {error}", source.display()))?;
            let destination = root.join("effective").join(CANONICAL_TEMPLATE_FILE);
            atomic_write(&destination, &bytes)?;
            Ok(SnapshotTemplate {
                source: "local-cache-before-managed-updates".to_string(),
                snapshot_path: destination
                    .strip_prefix(&root)
                    .unwrap_or(&destination)
                    .to_string_lossy()
                    .to_string(),
                size_bytes: bytes.len(),
            })
        })
        .transpose()?;

    if files.is_empty() && template.is_none() {
        return Ok(None);
    }
    let record = SnapshotRecord {
        schema_version: SCHEMA_VERSION,
        repo_id: repo_id.to_string(),
        source_repo_id: None,
        revision: "local-cache-before-managed-updates".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        files,
        template,
    };
    write_json_atomic(&snapshot_record_path(repo_id, &snapshot_id), &record)?;
    Ok(Some((snapshot_id, record)))
}

fn activate_snapshot(
    repo_id: &str,
    snapshot_id: &str,
    record: &SnapshotRecord,
) -> Result<(), String> {
    let root = snapshot_dir(repo_id, snapshot_id);
    for file in &record.files {
        let source = root.join(&file.snapshot_path);
        let bytes = std::fs::read(&source)
            .map_err(|error| format!("Failed to read snapshot {}: {error}", source.display()))?;
        atomic_write(&hf_sidecar_cache_path(repo_id, &file.source_path), &bytes)?;
    }
    if let Some(template) = record.template.as_ref() {
        let source = root.join(&template.snapshot_path);
        let bytes = std::fs::read(&source)
            .map_err(|error| format!("Failed to read snapshot {}: {error}", source.display()))?;
        validate_template(&template.source, &bytes)?;
        atomic_write(&canonical_template_cache_path(repo_id), &bytes)?;
    }
    Ok(())
}

fn result_rows(repo_id: &str, bundle: &RemoteBundle, apply: bool) -> Vec<HfSidecarSyncFile> {
    let mut rows = Vec::new();
    for asset in bundle.assets.iter().filter(|asset| {
        !asset
            .source_path
            .to_ascii_lowercase()
            .ends_with("chat_template.jinja")
    }) {
        let destination = asset_active_path(repo_id, asset);
        let changed = file_differs(&destination, &asset.bytes);
        rows.push(HfSidecarSyncFile {
            repo_id: bundle.source_repo_id.clone(),
            path: asset.source_path.clone(),
            cached_path: Some(destination.display().to_string()),
            status: if changed {
                if apply {
                    "updated"
                } else {
                    "available"
                }
            } else {
                "current"
            }
            .to_string(),
            message: None,
        });
    }
    if let Some(template) = bundle.template.as_ref() {
        let destination = canonical_template_cache_path(repo_id);
        let changed = file_differs(&destination, &template.bytes);
        rows.push(HfSidecarSyncFile {
            repo_id: bundle.source_repo_id.clone(),
            path: template.source.clone(),
            cached_path: Some(destination.display().to_string()),
            status: if changed {
                if apply {
                    "updated"
                } else {
                    "available"
                }
            } else {
                "current"
            }
            .to_string(),
            message: Some("Effective llama.cpp Jinja template".to_string()),
        });
    }
    for path in &bundle.missing {
        rows.push(HfSidecarSyncFile {
            repo_id: bundle.source_repo_id.clone(),
            path: path.clone(),
            cached_path: None,
            status: "missing".to_string(),
            message: Some(
                "Not published by this repository; no model weights were requested.".to_string(),
            ),
        });
    }
    rows
}

pub async fn update_targets(
    targets: Vec<SidecarTarget>,
    hf_api_key: Option<String>,
    apply: bool,
) -> Result<HfSidecarSyncSummary, String> {
    let hf_token_configured = hf_api_key
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty());
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("InferenceBridge/1.0")
        .build()
        .map_err(|error| format!("Failed to create Hugging Face client: {error}"))?;

    let mut repos = BTreeMap::<String, Option<String>>::new();
    for target in &targets {
        let Some(repo_id) = target
            .repo_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        repos
            .entry(repo_id.to_string())
            .or_insert_with(|| target.preferred_template_path.clone());
    }

    let mut results = Vec::new();
    let mut repos_with_updates = 0usize;
    let mut repos_updated = 0usize;
    for (repo_id, preferred_template_path) in &repos {
        let bundle = match fetch_remote_bundle(
            &client,
            repo_id,
            preferred_template_path.as_deref(),
            hf_api_key.as_deref(),
        )
        .await
        {
            Ok(bundle) => bundle,
            Err(error) => {
                results.push(HfSidecarSyncFile {
                    repo_id: repo_id.clone(),
                    path: "repository metadata".to_string(),
                    cached_path: None,
                    status: "failed".to_string(),
                    message: Some(error),
                });
                continue;
            }
        };
        let changed = bundle_change_count(repo_id, &bundle);
        if changed > 0 {
            repos_with_updates += 1;
        }
        let mut manifest = match read_manifest(repo_id) {
            Ok(manifest) => manifest,
            Err(error) => {
                results.push(HfSidecarSyncFile {
                    repo_id: repo_id.clone(),
                    path: "update-manifest.json".to_string(),
                    cached_path: None,
                    status: "failed".to_string(),
                    message: Some(error),
                });
                continue;
            }
        };
        manifest.remote_revision = Some(bundle.revision.clone());
        manifest.source_repo_id = Some(bundle.source_repo_id.clone());
        manifest.last_checked_at = Some(chrono::Utc::now().to_rfc3339());
        manifest.update_available = changed > 0;
        if changed == 0 {
            manifest.active_revision = Some(bundle.revision.clone());
            manifest.template_source = bundle
                .template
                .as_ref()
                .map(|template| template.source.clone());
        }

        let rows = result_rows(repo_id, &bundle, apply && changed > 0);
        if apply && changed > 0 {
            if manifest.active_snapshot.is_none() {
                match capture_legacy_snapshot(repo_id, preferred_template_path.as_deref()) {
                    Ok(Some((snapshot_id, record))) => {
                        manifest.active_snapshot = Some(snapshot_id);
                        manifest.active_revision = Some(record.revision);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        results.push(HfSidecarSyncFile {
                            repo_id: repo_id.clone(),
                            path: "local rollback snapshot".to_string(),
                            cached_path: None,
                            status: "failed".to_string(),
                            message: Some(error),
                        });
                        continue;
                    }
                }
            }
            let old_snapshot = manifest.active_snapshot.clone();
            let old_revision = manifest.active_revision.clone();
            let (new_snapshot, record) = match store_remote_snapshot(repo_id, &bundle) {
                Ok(value) => value,
                Err(error) => {
                    results.push(HfSidecarSyncFile {
                        repo_id: repo_id.clone(),
                        path: "remote snapshot".to_string(),
                        cached_path: None,
                        status: "failed".to_string(),
                        message: Some(error),
                    });
                    continue;
                }
            };
            if let Err(error) = activate_snapshot(repo_id, &new_snapshot, &record) {
                if let Some(old_snapshot) = old_snapshot.as_deref() {
                    if let Ok(old_record) = read_snapshot(repo_id, old_snapshot) {
                        let _ = activate_snapshot(repo_id, old_snapshot, &old_record);
                    }
                }
                results.push(HfSidecarSyncFile {
                    repo_id: repo_id.clone(),
                    path: "activate update".to_string(),
                    cached_path: None,
                    status: "failed".to_string(),
                    message: Some(error),
                });
                continue;
            }
            manifest.previous_snapshot = old_snapshot.clone();
            manifest.previous_revision = old_revision.clone();
            manifest.active_snapshot = Some(new_snapshot.clone());
            manifest.active_revision = Some(record.revision.clone());
            manifest.template_source = record.template.as_ref().map(|item| item.source.clone());
            manifest.update_available = false;
            if let Err(error) = write_json_atomic(&manifest_path(repo_id), &manifest) {
                if let Some(old_snapshot) = old_snapshot.as_deref() {
                    if let Ok(old_record) = read_snapshot(repo_id, old_snapshot) {
                        let _ = activate_snapshot(repo_id, old_snapshot, &old_record);
                    }
                }
                results.push(HfSidecarSyncFile {
                    repo_id: repo_id.clone(),
                    path: "update-manifest.json".to_string(),
                    cached_path: None,
                    status: "failed".to_string(),
                    message: Some(error),
                });
                continue;
            }
            repos_updated += 1;
        } else if let Err(error) = write_json_atomic(&manifest_path(repo_id), &manifest) {
            results.push(HfSidecarSyncFile {
                repo_id: repo_id.clone(),
                path: "update-manifest.json".to_string(),
                cached_path: None,
                status: "failed".to_string(),
                message: Some(error),
            });
            continue;
        }
        results.extend(rows);
    }

    let files_updated = results
        .iter()
        .filter(|result| result.status == "updated")
        .count();
    let files_unchanged = results
        .iter()
        .filter(|result| result.status == "current")
        .count();
    let files_skipped = results
        .iter()
        .filter(|result| matches!(result.status.as_str(), "missing" | "blocked"))
        .count();
    let files_failed = results
        .iter()
        .filter(|result| result.status == "failed")
        .count();
    let files_cached = files_updated + files_unchanged;

    Ok(HfSidecarSyncSummary {
        mode: if apply { "apply" } else { "check" }.to_string(),
        models_checked: targets.len(),
        repos_checked: repos.len(),
        repos_with_updates,
        repos_updated,
        files_cached,
        files_updated,
        files_unchanged,
        files_skipped,
        files_failed,
        hf_token_configured,
        cache_root: crate::config::app_support_dir()
            .join("hf-sidecars")
            .display()
            .to_string(),
        results,
    })
}

pub fn get_cache_status(targets: Vec<SidecarTarget>) -> Result<Vec<HfSidecarCacheStatus>, String> {
    let mut statuses = Vec::with_capacity(targets.len());
    for target in targets {
        let repo_id = target
            .repo_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string);
        let Some(repo_id_value) = repo_id.as_deref() else {
            statuses.push(HfSidecarCacheStatus {
                filename: target.filename,
                repo_id: None,
                source_repo_id: None,
                template_path: None,
                template_cached: false,
                template_cache_path: None,
                template_source: None,
                sidecar_cached_count: 0,
                sidecar_expected_count: HF_SIDECAR_DEFAULT_FILES.len(),
                sidecar_cache_dir: None,
                active_revision: None,
                remote_revision: None,
                update_available: false,
                last_checked_at: None,
                rollback_available: false,
            });
            continue;
        };
        let manifest = read_manifest(repo_id_value)?;
        let managed_template = canonical_template_cache_path(repo_id_value);
        let legacy_template = target
            .preferred_template_path
            .as_deref()
            .map(|path| hf_template_cache_path(repo_id_value, path));
        let template_cache = if managed_template.exists() {
            Some(managed_template)
        } else {
            legacy_template.filter(|path| path.exists())
        };
        let sidecar_cached_count = HF_SIDECAR_DEFAULT_FILES
            .iter()
            .filter(|path| hf_sidecar_cache_path(repo_id_value, path).exists())
            .count();
        statuses.push(HfSidecarCacheStatus {
            filename: target.filename,
            repo_id: repo_id.clone(),
            source_repo_id: manifest.source_repo_id.clone(),
            template_path: manifest
                .template_source
                .clone()
                .or(target.preferred_template_path),
            template_cached: template_cache.is_some(),
            template_cache_path: template_cache.map(|path| path.display().to_string()),
            template_source: manifest.template_source,
            sidecar_cached_count,
            sidecar_expected_count: HF_SIDECAR_DEFAULT_FILES.len(),
            sidecar_cache_dir: Some(repo_sidecar_dir(repo_id_value).display().to_string()),
            active_revision: manifest.active_revision,
            remote_revision: manifest.remote_revision,
            update_available: manifest.update_available,
            last_checked_at: manifest.last_checked_at,
            rollback_available: manifest.previous_snapshot.is_some(),
        });
    }
    Ok(statuses)
}

pub fn effective_template_cache(
    repo_id: &str,
    fallback_template_path: Option<&str>,
) -> Option<(PathBuf, String, String)> {
    let managed = canonical_template_cache_path(repo_id);
    if managed.exists() {
        let manifest = read_manifest(repo_id).ok();
        let source = manifest
            .as_ref()
            .and_then(|manifest| manifest.template_source.clone())
            .unwrap_or_else(|| CANONICAL_TEMPLATE_FILE.to_string());
        let source_repo = manifest
            .and_then(|manifest| manifest.source_repo_id)
            .unwrap_or_else(|| repo_id.to_string());
        return Some((managed, source_repo, source));
    }
    fallback_template_path
        .map(|path| hf_template_cache_path(repo_id, path))
        .filter(|path| path.exists())
        .map(|path| {
            (
                path,
                repo_id.to_string(),
                fallback_template_path
                    .unwrap_or(CANONICAL_TEMPLATE_FILE)
                    .to_string(),
            )
        })
}

pub fn rollback_target(target: SidecarTarget) -> Result<HfSidecarRollbackSummary, String> {
    let repo_id = target
        .repo_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            format!(
                "{} is not linked to a Hugging Face repository",
                target.filename
            )
        })?;
    let mut manifest = read_manifest(repo_id)?;
    let previous_snapshot = manifest
        .previous_snapshot
        .clone()
        .ok_or_else(|| format!("No previous metadata/template snapshot exists for {repo_id}"))?;
    let record = read_snapshot(repo_id, &previous_snapshot)?;
    activate_snapshot(repo_id, &previous_snapshot, &record)?;

    let replaced_snapshot = manifest.active_snapshot.clone();
    let replaced_revision = manifest.active_revision.clone();
    manifest.active_snapshot = Some(previous_snapshot);
    manifest.active_revision = Some(record.revision.clone());
    manifest.previous_snapshot = replaced_snapshot;
    manifest.previous_revision = replaced_revision.clone();
    manifest.source_repo_id = record.source_repo_id.clone();
    manifest.template_source = record.template.as_ref().map(|item| item.source.clone());
    manifest.update_available =
        manifest.remote_revision.as_deref() != manifest.active_revision.as_deref();
    write_json_atomic(&manifest_path(repo_id), &manifest)?;

    Ok(HfSidecarRollbackSummary {
        repo_id: repo_id.to_string(),
        restored_revision: Some(record.revision),
        replaced_revision,
        files_restored: record.files.len(),
        template_restored: record.template.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        choose_repo_file, extract_tokenizer_chat_template, is_allowed_hf_sidecar_path,
        template_from_named_collection, upstream_base_repo, HfRepoSibling,
    };

    #[test]
    fn blocks_weights_and_traversal_but_allows_nested_templates() {
        assert!(is_allowed_hf_sidecar_path("chat_template.jinja"));
        assert!(is_allowed_hf_sidecar_path("tokenizer/chat_template.jinja"));
        assert!(is_allowed_hf_sidecar_path("config.json"));
        assert!(!is_allowed_hf_sidecar_path("model-Q4_K_M.gguf"));
        assert!(!is_allowed_hf_sidecar_path("../chat_template.jinja"));
        assert!(!is_allowed_hf_sidecar_path("folder\\config.json"));
    }

    #[test]
    fn repo_file_selection_prefers_explicit_then_root() {
        let siblings = vec![
            HfRepoSibling {
                rfilename: "nested/chat_template.jinja".to_string(),
            },
            HfRepoSibling {
                rfilename: "chat_template.jinja".to_string(),
            },
        ];
        assert_eq!(
            choose_repo_file(
                &siblings,
                "chat_template.jinja",
                Some("nested/chat_template.jinja")
            ),
            Some("nested/chat_template.jinja".to_string())
        );
        assert_eq!(
            choose_repo_file(&siblings, "chat_template.jinja", None),
            Some("chat_template.jinja".to_string())
        );
    }

    #[test]
    fn extracts_string_and_named_tokenizer_templates() {
        let direct = br#"{"chat_template":"{% for message in messages %}{{ message.content }}{% endfor %}"}"#;
        assert!(extract_tokenizer_chat_template(direct).unwrap().is_some());

        let named = serde_json::json!([
            {"name": "tool_use", "template": "tools"},
            {"name": "default", "template": "default"}
        ]);
        assert_eq!(
            template_from_named_collection(&named).as_deref(),
            Some("default")
        );
    }

    #[test]
    fn follows_explicit_quantized_sources_without_overriding_finetune_templates() {
        let quantized = vec!["base_model:quantized:google/gemma-4-26B-A4B-it".to_string()];
        assert_eq!(
            upstream_base_repo(&quantized, true).as_deref(),
            Some("google/gemma-4-26B-A4B-it")
        );

        let generic = vec!["base_model:google/gemma-4-26B-A4B-it".to_string()];
        assert_eq!(upstream_base_repo(&generic, true), None);
        assert_eq!(
            upstream_base_repo(&generic, false).as_deref(),
            Some("google/gemma-4-26B-A4B-it")
        );
    }
}
