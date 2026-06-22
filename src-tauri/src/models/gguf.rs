//! Lightweight GGUF metadata reader.
//! Only reads the KV header — never touches tensor data.
//!
//! Parses enough to derive accurate VRAM estimates and expose the model's
//! true training context length, layer count, and GQA configuration.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

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
/// Reads only as far as needed (exits once all six target keys are found).
/// Returns `None` on any I/O or format error — never panics.
pub fn read_gguf_meta(path: &Path) -> Option<GgufMeta> {
    let file = File::open(path).ok()?;
    let mut r = BufReader::new(file);

    if read_u32(&mut r)? != GGUF_MAGIC {
        return None;
    }

    let version = read_u32(&mut r)?;
    if version == 0 || version > 3 {
        return None;
    }
    let v1 = version == 1;

    // n_tensors (skip), then n_kv
    let n_kv = if v1 {
        let _ = read_u32(&mut r)?;
        read_u32(&mut r)? as u64
    } else {
        let _ = read_u64(&mut r)?;
        read_u64(&mut r)?
    };

    let mut meta = GgufMeta::default();
    let mut found: u8 = 0;

    for _ in 0..n_kv {
        if found >= 8 {
            break; // all target fields collected
        }

        let key = read_str(&mut r, v1)?;
        let ty = read_u32(&mut r)?;

        // Match on key suffix so it works regardless of architecture prefix
        // (e.g. "llama.", "qwen2.", "qwen3.", "mistral.", etc.)
        match (key.as_str(), ty) {
            ("general.architecture", T_STRING) => {
                meta.architecture = Some(read_str(&mut r, v1)?);
                found += 1;
            }
            ("general.name", T_STRING) => {
                meta.general_name = Some(read_str(&mut r, v1)?);
                found += 1;
            }
            ("tokenizer.chat_template", T_STRING) => {
                // Only record presence — the template itself can be many KB and
                // we don't want it bloating the in-memory model registry.
                meta.has_chat_template = true;
                skip_val(&mut r, ty, v1)?;
                found += 1;
            }
            (k, T_U32) if k.ends_with(".context_length") => {
                meta.context_length = Some(read_u32(&mut r)?);
                found += 1;
            }
            (k, T_U32) if k.ends_with(".block_count") => {
                meta.n_layers = Some(read_u32(&mut r)?);
                found += 1;
            }
            (k, T_U32) if k.ends_with(".embedding_length") => {
                meta.embedding_length = Some(read_u32(&mut r)?);
                found += 1;
            }
            (k, T_U32) if k.ends_with(".attention.head_count_kv") => {
                meta.n_kv_heads = Some(read_u32(&mut r)?);
                found += 1;
            }
            (k, T_U32) if k.ends_with(".attention.head_count") => {
                // head_count (no _kv suffix) — checked after head_count_kv
                meta.n_heads = Some(read_u32(&mut r)?);
                found += 1;
            }
            _ => {
                skip_val(&mut r, ty, v1)?;
            }
        }
    }

    Some(meta)
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
