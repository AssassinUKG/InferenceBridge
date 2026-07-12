//! Lightweight GGUF metadata reader.
//! Only reads the KV header — never touches tensor data.
//!
//! Parses enough to derive accurate VRAM estimates and expose the model's
//! true training context length, layer count, and GQA configuration.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::UNIX_EPOCH;

/// `GGUF` as a little-endian u32
const GGUF_MAGIC: u32 = 0x46554747;

// GGUF value-type constants
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

/// Architecture metadata extracted from a GGUF model file header.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GgufMeta {
    /// Model training context length (`{arch}.context_length`)
    pub context_length: Option<u32>,
    /// Number of transformer blocks/layers (`{arch}.block_count`)
    pub n_layers: Option<u32>,
    /// Hidden/embedding dimension (`{arch}.embedding_length`)
    pub embedding_length: Option<u32>,
    /// Number of query attention heads (`{arch}.attention.head_count`)
    pub n_heads: Option<u32>,
    /// Number of KV heads — less than n_heads when GQA is in use
    /// (`{arch}.attention.head_count_kv`)
    pub n_kv_heads: Option<u32>,
    /// Architecture identifier string (`general.architecture`, e.g. `"qwen2"`)
    pub architecture: Option<String>,
    /// Human-readable model name (`general.name`).
    pub general_name: Option<String>,
    /// Whether the file carries an embedded chat template
    /// (`tokenizer.chat_template`). When present, llama-server can be run with
    /// `--jinja` to use the model author's own template verbatim.
    pub has_chat_template: bool,
}

impl GgufMeta {
    /// Per-head attention dimension: `embedding_length / n_heads`.
    pub fn head_dim(&self) -> Option<u32> {
        let emb = self.embedding_length?;
        let heads = self.n_heads.filter(|&h| h > 0)?;
        Some(emb / heads)
    }

    /// KV-cache bytes consumed per token across all layers.
    ///
    /// `bytes_per_element`: `2.0` = f16, `1.0` = q8_0, `0.5` = q4_0.
    ///
    /// Formula: `n_layers × 2 (K+V) × n_kv_heads × head_dim × bpe`
    pub fn kv_bytes_per_token(&self, bytes_per_element: f32) -> Option<u64> {
        let layers = self.n_layers? as f32;
        let kv = self.n_kv_heads? as f32;
        let dim = self.head_dim()? as f32;
        Some((layers * 2.0 * kv * dim * bytes_per_element).round() as u64)
    }

    /// KV-cache MB for a given context size and cache element type.
    pub fn kv_cache_mb(&self, n_ctx: u32, bytes_per_element: f32) -> Option<f64> {
        let bpt = self.kv_bytes_per_token(bytes_per_element)? as f64;
        Some((n_ctx as f64 * bpt) / (1024.0 * 1024.0))
    }
}

// ── I/O helpers ──────────────────────────────────────────────────────────────

fn read_u32<R: Read>(r: &mut R) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

fn seek_fwd<R: Seek>(r: &mut R, n: u64) -> Option<()> {
    if n > 0 {
        r.seek(SeekFrom::Current(n as i64)).ok()?;
    }
    Some(())
}

fn read_str<R: Read + Seek>(r: &mut R, v1: bool) -> Option<String> {
    let len = if v1 {
        read_u32(r)? as u64
    } else {
        read_u64(r)?
    };
    if len > 1 << 20 {
        return None; // sanity: no metadata string > 1 MB
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

/// Skip a single GGUF value without decoding it.
fn skip_val<R: Read + Seek>(r: &mut R, ty: u32, v1: bool) -> Option<()> {
    match ty {
        T_U8 | T_I8 | T_BOOL => seek_fwd(r, 1),
        T_U16 | T_I16 => seek_fwd(r, 2),
        T_U32 | T_I32 | T_F32 => seek_fwd(r, 4),
        T_U64 | T_I64 | T_F64 => seek_fwd(r, 8),
        T_STRING => {
            let len = if v1 {
                read_u32(r)? as u64
            } else {
                read_u64(r)?
            };
            seek_fwd(r, len)
        }
        T_ARRAY => {
            let elem_ty = read_u32(r)?;
            let count = if v1 {
                read_u32(r)? as u64
            } else {
                read_u64(r)?
            };
            // Fixed-size elements: one seek covers the whole array
            let fixed = match elem_ty {
                T_U8 | T_I8 | T_BOOL => Some(count),
                T_U16 | T_I16 => Some(count * 2),
                T_U32 | T_I32 | T_F32 => Some(count * 4),
                T_U64 | T_I64 | T_F64 => Some(count * 8),
                _ => None,
            };
            if let Some(n) = fixed {
                seek_fwd(r, n)
            } else {
                // Variable-length (e.g. tokenizer vocab strings): iterate
                for _ in 0..count {
                    skip_val(r, elem_ty, v1)?;
                }
                Some(())
            }
        }
        _ => None, // unknown type — abort cleanly
    }
}

// ── Public entry point ───────────────────────────────────────────────────────

/// Parse architecture metadata from a GGUF file header.
///
/// Memory-maps the file and parses from the mapped bytes, so header skips
/// (including large tokenizer arrays that precede `chat_template`) are pure
/// in-memory seeks rather than buffered reads + syscalls. mmap is lazy, so
/// only the touched header pages are actually read from disk — never the
/// multi-GB tensor payload.
///
/// Reads only as far as needed (exits once all target keys are found).
/// Returns `None` on any I/O or format error — never panics.
pub fn read_gguf_meta(path: &Path) -> Option<GgufMeta> {
    let file = File::open(path).ok()?;
    // SAFETY: the file is only read through the returned slice; we never keep
    // the mapping past this function and tolerate concurrent truncation by
    // treating any parse failure as `None`.
    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
    let mut r = Cursor::new(&mmap[..]);
    parse_gguf_meta(&mut r)
}

/// Core GGUF header parser, generic over any seekable byte source.
fn parse_gguf_meta<R: Read + Seek>(r: &mut R) -> Option<GgufMeta> {
    if read_u32(r)? != GGUF_MAGIC {
        return None;
    }

    let version = read_u32(r)?;
    if version == 0 || version > 3 {
        return None;
    }
    let v1 = version == 1;

    // n_tensors (skip), then n_kv
    let n_kv = if v1 {
        let _ = read_u32(r)?;
        read_u32(r)? as u64
    } else {
        let _ = read_u64(r)?;
        read_u64(r)?
    };

    let mut meta = GgufMeta::default();
    let mut found: u8 = 0;

    for _ in 0..n_kv {
        if found >= 8 {
            break; // all target fields collected
        }

        let key = read_str(r, v1)?;
        let ty = read_u32(r)?;

        // Match on key suffix so it works regardless of architecture prefix
        // (e.g. "llama.", "qwen2.", "qwen3.", "mistral.", etc.)
        match (key.as_str(), ty) {
            ("general.architecture", T_STRING) => {
                meta.architecture = Some(read_str(r, v1)?);
                found += 1;
            }
            ("general.name", T_STRING) => {
                meta.general_name = Some(read_str(r, v1)?);
                found += 1;
            }
            ("tokenizer.chat_template", T_STRING) => {
                // Only record presence — the template itself can be many KB and
                // we don't want it bloating the in-memory model registry.
                meta.has_chat_template = true;
                skip_val(r, ty, v1)?;
                found += 1;
            }
            (k, T_U32) if k.ends_with(".context_length") => {
                meta.context_length = Some(read_u32(r)?);
                found += 1;
            }
            (k, T_U32) if k.ends_with(".block_count") => {
                meta.n_layers = Some(read_u32(r)?);
                found += 1;
            }
            (k, T_U32) if k.ends_with(".embedding_length") => {
                meta.embedding_length = Some(read_u32(r)?);
                found += 1;
            }
            (k, T_U32) if k.ends_with(".attention.head_count_kv") => {
                meta.n_kv_heads = Some(read_u32(r)?);
                found += 1;
            }
            (k, T_U32) if k.ends_with(".attention.head_count") => {
                // head_count (no _kv suffix) — checked after head_count_kv
                meta.n_heads = Some(read_u32(r)?);
                found += 1;
            }
            _ => {
                skip_val(r, ty, v1)?;
            }
        }
    }

    Some(meta)
}

// ── Persistent metadata cache ────────────────────────────────────────────────
//
// Parsing a GGUF header still touches disk. During a rescan the vast majority
// of files are unchanged, so we keep a `(size, mtime) → GgufMeta` cache on disk
// keyed by absolute path. A warm rescan then avoids re-mapping every file and
// becomes near-instant. Failed parses are cached as `None` so broken/partial
// files aren't retried on every scan (they re-parse only when size/mtime moves).

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct GgufCacheEntry {
    size: u64,
    mtime: i64,
    meta: Option<GgufMeta>,
}

struct GgufCache {
    entries: HashMap<String, GgufCacheEntry>,
    dirty: bool,
}

static CACHE: OnceLock<Mutex<GgufCache>> = OnceLock::new();

fn cache_file_path() -> PathBuf {
    crate::config::app_support_dir().join("gguf-meta-cache.json")
}

fn cache() -> &'static Mutex<GgufCache> {
    CACHE.get_or_init(|| {
        let entries = std::fs::read_to_string(cache_file_path())
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, GgufCacheEntry>>(&s).ok())
            .unwrap_or_default();
        Mutex::new(GgufCache {
            entries,
            dirty: false,
        })
    })
}

/// `(size, mtime_secs)` for cache validation. `None` if the file can't be stat'd.
fn file_stat(path: &Path) -> Option<(u64, i64)> {
    let md = std::fs::metadata(path).ok()?;
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Some((md.len(), mtime))
}

/// Cache-backed wrapper around [`read_gguf_meta`]. Returns the cached metadata
/// when the file's size and mtime are unchanged; otherwise parses and records
/// the result. Falls back to a direct parse if the file can't be stat'd.
pub fn read_gguf_meta_cached(path: &Path) -> Option<GgufMeta> {
    let Some((size, mtime)) = file_stat(path) else {
        return read_gguf_meta(path);
    };
    let key = path.to_string_lossy().into_owned();

    if let Ok(guard) = cache().lock() {
        if let Some(entry) = guard.entries.get(&key) {
            if entry.size == size && entry.mtime == mtime {
                return entry.meta.clone();
            }
        }
    }

    // Parse outside the lock so parallel scans don't serialize on disk I/O.
    let meta = read_gguf_meta(path);
    if let Ok(mut guard) = cache().lock() {
        guard.entries.insert(
            key,
            GgufCacheEntry {
                size,
                mtime,
                meta: meta.clone(),
            },
        );
        guard.dirty = true;
    }
    meta
}

/// Flush the in-memory cache to disk if it changed. Call once at the end of a
/// scan pass. Best-effort: persistence failures are logged, not fatal.
pub fn flush_gguf_cache() {
    let Some(lock) = CACHE.get() else {
        return;
    };
    let Ok(mut guard) = lock.lock() else {
        return;
    };
    if !guard.dirty {
        return;
    }
    let path = cache_file_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string(&guard.entries) {
        Ok(json) => match std::fs::write(&path, json) {
            Ok(()) => guard.dirty = false,
            Err(e) => tracing::warn!(?path, error = %e, "Failed to persist GGUF metadata cache"),
        },
        Err(e) => tracing::warn!(error = %e, "Failed to serialize GGUF metadata cache"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_dim_gqa() {
        let meta = GgufMeta {
            n_layers: Some(46),
            embedding_length: Some(5120),
            n_heads: Some(40),
            n_kv_heads: Some(8),
            ..Default::default()
        };
        assert_eq!(meta.head_dim(), Some(128));
        // q8_0: 46 × 2 × 8 × 128 × 1 = 94208 bytes/token
        assert_eq!(meta.kv_bytes_per_token(1.0), Some(94208));
    }

    #[test]
    fn kv_cache_mb_32k_q8() {
        let meta = GgufMeta {
            n_layers: Some(46),
            embedding_length: Some(5120),
            n_heads: Some(40),
            n_kv_heads: Some(8),
            ..Default::default()
        };
        // 32768 tokens × 94208 bytes ÷ 1MB ≈ 2944 MB
        let mb = meta.kv_cache_mb(32768, 1.0).unwrap();
        assert!((mb - 2944.0).abs() < 2.0, "got {mb}");
    }
}
