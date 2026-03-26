//! Model browser — curated catalog of popular GGUF models with download/delete support.

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::Emitter;
use tokio::io::AsyncWriteExt;

use crate::state::SharedState;

// ─── Catalog types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubQuant {
    pub quant: String,
    pub size_gb: f32,
    pub url: String,
    pub filename: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubModel {
    pub id: String,
    pub name: String,
    pub family: String,
    pub params: String,
    pub description: String,
    pub tags: Vec<String>,
    pub quants: Vec<HubQuant>,
}

fn bw(repo: &str, filename: &str, quant: &str, size_gb: f32) -> HubQuant {
    HubQuant {
        quant: quant.to_string(),
        size_gb,
        url: format!(
            "https://huggingface.co/bartowski/{}/resolve/main/{}",
            repo, filename
        ),
        filename: filename.to_string(),
    }
}

pub fn catalog() -> Vec<HubModel> {
    vec![
        // ── Qwen3.5 ──────────────────────────────────────────────────────────
        HubModel {
            id: "qwen3.5-0.6b".into(),
            name: "Qwen3.5-0.6B".into(),
            family: "Qwen3.5".into(),
            params: "0.6B".into(),
            description: "Alibaba's latest reasoning model. 262K context, thinking mode, tool use. Smallest variant.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3.5-0.6B-GGUF", "Qwen3.5-0.6B-Q4_K_M.gguf", "Q4_K_M", 0.52),
                bw("Qwen3.5-0.6B-GGUF", "Qwen3.5-0.6B-Q8_0.gguf", "Q8_0", 0.72),
            ],
        },
        HubModel {
            id: "qwen3.5-1.7b".into(),
            name: "Qwen3.5-1.7B".into(),
            family: "Qwen3.5".into(),
            params: "1.7B".into(),
            description: "Qwen3.5 1.7B — 262K context, thinking mode, tool use. Excellent quality/speed for edge devices.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3.5-1.7B-GGUF", "Qwen3.5-1.7B-Q4_K_M.gguf", "Q4_K_M", 1.1),
                bw("Qwen3.5-1.7B-GGUF", "Qwen3.5-1.7B-Q8_0.gguf", "Q8_0", 1.9),
            ],
        },
        HubModel {
            id: "qwen3.5-4b".into(),
            name: "Qwen3.5-4B".into(),
            family: "Qwen3.5".into(),
            params: "4B".into(),
            description: "Qwen3.5 4B — great quality/speed balance. 262K context, thinking mode optional.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", "Q4_K_M", 2.5),
                bw("Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q8_0.gguf", "Q8_0", 4.3),
            ],
        },
        HubModel {
            id: "qwen3.5-8b".into(),
            name: "Qwen3.5-8B".into(),
            family: "Qwen3.5".into(),
            params: "8B".into(),
            description: "Qwen3.5 8B — strong reasoning, coding, multilingual. Recommended daily driver for 8GB+ VRAM.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3.5-8B-GGUF", "Qwen3.5-8B-Q4_K_M.gguf", "Q4_K_M", 4.9),
                bw("Qwen3.5-8B-GGUF", "Qwen3.5-8B-Q8_0.gguf", "Q8_0", 8.5),
            ],
        },
        HubModel {
            id: "qwen3.5-14b".into(),
            name: "Qwen3.5-14B".into(),
            family: "Qwen3.5".into(),
            params: "14B".into(),
            description: "Qwen3.5 14B — near-frontier quality for coding, reasoning, and agentic tasks. Needs 12GB+ VRAM.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3.5-14B-GGUF", "Qwen3.5-14B-Q4_K_M.gguf", "Q4_K_M", 8.6),
                bw("Qwen3.5-14B-GGUF", "Qwen3.5-14B-Q8_0.gguf", "Q8_0", 14.7),
            ],
        },
        HubModel {
            id: "qwen3.5-32b".into(),
            name: "Qwen3.5-32B".into(),
            family: "Qwen3.5".into(),
            params: "32B".into(),
            description: "Qwen3.5 32B — flagship dense model. Exceptional coding and reasoning. Needs 24GB+ VRAM.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3.5-32B-GGUF", "Qwen3.5-32B-Q4_K_M.gguf", "Q4_K_M", 19.4),
                bw("Qwen3.5-32B-GGUF", "Qwen3.5-32B-IQ4_XS.gguf", "IQ4_XS", 17.0),
            ],
        },
        HubModel {
            id: "qwen3.5-30b-a3b".into(),
            name: "Qwen3.5-30B-A3B".into(),
            family: "Qwen3.5".into(),
            params: "30B MoE (3.5B active)".into(),
            description: "Qwen3.5 MoE — 30B total, 3.5B active. Near 30B quality at ~4B inference cost. 262K context.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into(), "moe".into()],
            quants: vec![
                bw("Qwen3.5-30B-A3B-GGUF", "Qwen3.5-30B-A3B-Q4_K_M.gguf", "Q4_K_M", 17.5),
                bw("Qwen3.5-30B-A3B-GGUF", "Qwen3.5-30B-A3B-IQ4_XS.gguf", "IQ4_XS", 15.0),
            ],
        },
        // ── Qwen3 ────────────────────────────────────────────────────────────
        HubModel {
            id: "qwen3-0.6b".into(),
            name: "Qwen3-0.6B".into(),
            family: "Qwen3".into(),
            params: "0.6B".into(),
            description: "Qwen3 0.6B — tiny reasoning model with thinking mode. Ideal for constrained environments.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", "Q4_K_M", 0.52),
                bw("Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf", "Q8_0", 0.72),
            ],
        },
        HubModel {
            id: "qwen3-4b".into(),
            name: "Qwen3-4B".into(),
            family: "Qwen3".into(),
            params: "4B".into(),
            description: "Qwen3 4B — punches above its weight for coding and reasoning. Popular daily driver.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", "Q4_K_M", 2.5),
                bw("Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", "Q8_0", 4.3),
            ],
        },
        HubModel {
            id: "qwen3-8b".into(),
            name: "Qwen3-8B".into(),
            family: "Qwen3".into(),
            params: "8B".into(),
            description: "Qwen3 8B — best-in-class 8B reasoning model. Highly recommended for 8GB VRAM.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", "Q4_K_M", 5.2),
                bw("Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf", "Q8_0", 8.5),
            ],
        },
        HubModel {
            id: "qwen3-14b".into(),
            name: "Qwen3-14B".into(),
            family: "Qwen3".into(),
            params: "14B".into(),
            description: "Qwen3 14B — excellent instruction following, coding, and multilingual capabilities.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf", "Q4_K_M", 8.6),
                bw("Qwen3-14B-GGUF", "Qwen3-14B-Q8_0.gguf", "Q8_0", 14.7),
            ],
        },
        HubModel {
            id: "qwen3-32b".into(),
            name: "Qwen3-32B".into(),
            family: "Qwen3".into(),
            params: "32B".into(),
            description: "Qwen3 32B — top open-source reasoning model. Matches frontier models on coding benchmarks.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into()],
            quants: vec![
                bw("Qwen3-32B-GGUF", "Qwen3-32B-Q4_K_M.gguf", "Q4_K_M", 19.4),
                bw("Qwen3-32B-GGUF", "Qwen3-32B-IQ4_XS.gguf", "IQ4_XS", 17.0),
            ],
        },
        HubModel {
            id: "qwen3-30b-a3b".into(),
            name: "Qwen3-30B-A3B".into(),
            family: "Qwen3".into(),
            params: "30B MoE (3B active)".into(),
            description: "Qwen3 MoE — 30B parameters, 3B active. Best efficiency for reasoning at low inference cost.".into(),
            tags: vec!["reasoning".into(), "tools".into(), "thinking".into(), "moe".into()],
            quants: vec![
                bw("Qwen3-30B-A3B-GGUF", "Qwen3-30B-A3B-Q4_K_M.gguf", "Q4_K_M", 17.5),
            ],
        },
        // ── Llama 3.x ────────────────────────────────────────────────────────
        HubModel {
            id: "llama-3.2-1b".into(),
            name: "Llama 3.2 1B".into(),
            family: "Llama3".into(),
            params: "1B".into(),
            description: "Meta's smallest Llama 3.2. 128K context. Fast and lightweight for simple tasks.".into(),
            tags: vec!["chat".into()],
            quants: vec![
                bw("Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Q4_K_M", 0.77),
                bw("Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf", "Q8_0", 1.32),
            ],
        },
        HubModel {
            id: "llama-3.2-3b".into(),
            name: "Llama 3.2 3B".into(),
            family: "Llama3".into(),
            params: "3B".into(),
            description: "Meta's Llama 3.2 3B. 128K context. Strong instruction following and tool use for its size.".into(),
            tags: vec!["chat".into(), "tools".into()],
            quants: vec![
                bw("Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "Q4_K_M", 1.9),
                bw("Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q8_0.gguf", "Q8_0", 3.2),
            ],
        },
        HubModel {
            id: "llama-3.1-8b".into(),
            name: "Llama 3.1 8B".into(),
            family: "Llama3".into(),
            params: "8B".into(),
            description: "Meta's Llama 3.1 8B Instruct. 128K context. Excellent for chat, coding, and function calling.".into(),
            tags: vec!["chat".into(), "tools".into()],
            quants: vec![
                bw("Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Q4_K_M", 4.9),
                bw("Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", "Q8_0", 8.5),
            ],
        },
        HubModel {
            id: "llama-3.3-70b".into(),
            name: "Llama 3.3 70B".into(),
            family: "Llama3".into(),
            params: "70B".into(),
            description: "Meta's best open Llama. 128K context. Competitive with GPT-4 class models. Needs 40GB+ VRAM.".into(),
            tags: vec!["chat".into(), "tools".into()],
            quants: vec![
                bw("Llama-3.3-70B-Instruct-GGUF", "Llama-3.3-70B-Instruct-Q4_K_M.gguf", "Q4_K_M", 42.5),
                bw("Llama-3.3-70B-Instruct-GGUF", "Llama-3.3-70B-Instruct-IQ4_XS.gguf", "IQ4_XS", 37.0),
            ],
        },
        // ── DeepSeek R1 ──────────────────────────────────────────────────────
        HubModel {
            id: "deepseek-r1-distill-7b".into(),
            name: "DeepSeek-R1 Distill 7B".into(),
            family: "DeepSeekR1".into(),
            params: "7B".into(),
            description: "DeepSeek R1 distilled into Qwen 7B. Always reasons before answering. Exceptional for math, coding, and logic.".into(),
            tags: vec!["reasoning".into(), "thinking".into(), "math".into()],
            quants: vec![
                bw("DeepSeek-R1-Distill-Qwen-7B-GGUF", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", "Q4_K_M", 4.7),
                bw("DeepSeek-R1-Distill-Qwen-7B-GGUF", "DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf", "Q8_0", 7.9),
            ],
        },
        HubModel {
            id: "deepseek-r1-distill-14b".into(),
            name: "DeepSeek-R1 Distill 14B".into(),
            family: "DeepSeekR1".into(),
            params: "14B".into(),
            description: "DeepSeek R1 distilled into Qwen 14B. Best-in-class 14B reasoning. Outstanding for math and proofs.".into(),
            tags: vec!["reasoning".into(), "thinking".into(), "math".into()],
            quants: vec![
                bw("DeepSeek-R1-Distill-Qwen-14B-GGUF", "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf", "Q4_K_M", 8.9),
                bw("DeepSeek-R1-Distill-Qwen-14B-GGUF", "DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf", "Q8_0", 15.1),
            ],
        },
        HubModel {
            id: "deepseek-r1-distill-32b".into(),
            name: "DeepSeek-R1 Distill 32B".into(),
            family: "DeepSeekR1".into(),
            params: "32B".into(),
            description: "DeepSeek R1 distilled into Qwen 32B. Near full R1 quality. Needs 24GB VRAM.".into(),
            tags: vec!["reasoning".into(), "thinking".into(), "math".into()],
            quants: vec![
                bw("DeepSeek-R1-Distill-Qwen-32B-GGUF", "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", "Q4_K_M", 19.4),
                bw("DeepSeek-R1-Distill-Qwen-32B-GGUF", "DeepSeek-R1-Distill-Qwen-32B-IQ4_XS.gguf", "IQ4_XS", 17.0),
            ],
        },
        // ── Phi ──────────────────────────────────────────────────────────────
        HubModel {
            id: "phi-4".into(),
            name: "Phi-4".into(),
            family: "Phi".into(),
            params: "14B".into(),
            description: "Microsoft's Phi-4. 128K context. Exceptional reasoning per parameter. Best Phi yet.".into(),
            tags: vec!["reasoning".into(), "chat".into(), "tools".into()],
            quants: vec![
                bw("phi-4-GGUF", "phi-4-Q4_K_M.gguf", "Q4_K_M", 8.5),
                bw("phi-4-GGUF", "phi-4-Q8_0.gguf", "Q8_0", 14.7),
            ],
        },
        HubModel {
            id: "phi-4-mini".into(),
            name: "Phi-4 Mini".into(),
            family: "Phi".into(),
            params: "3.8B".into(),
            description: "Microsoft's compact Phi-4 Mini. 128K context. Impressive reasoning for a 4B scale model.".into(),
            tags: vec!["reasoning".into(), "chat".into()],
            quants: vec![
                bw("Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct-Q4_K_M.gguf", "Q4_K_M", 2.3),
                bw("Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct-Q8_0.gguf", "Q8_0", 4.0),
            ],
        },
        // ── Mistral ──────────────────────────────────────────────────────────
        HubModel {
            id: "mistral-7b-v0.3".into(),
            name: "Mistral 7B v0.3".into(),
            family: "Mistral".into(),
            params: "7B".into(),
            description: "Mistral's classic 7B instruct. 32K context. Fast, reliable, great for tool use.".into(),
            tags: vec!["chat".into(), "tools".into()],
            quants: vec![
                bw("Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "Q4_K_M", 4.4),
                bw("Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q8_0.gguf", "Q8_0", 7.7),
            ],
        },
        HubModel {
            id: "mistral-nemo".into(),
            name: "Mistral Nemo 12B".into(),
            family: "Mistral".into(),
            params: "12B".into(),
            description: "Mistral Nemo 12B. 128K context. Best Mistral instruct — strong coding and chat.".into(),
            tags: vec!["chat".into(), "tools".into()],
            quants: vec![
                bw("Mistral-Nemo-Instruct-2407-GGUF", "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf", "Q4_K_M", 7.1),
                bw("Mistral-Nemo-Instruct-2407-GGUF", "Mistral-Nemo-Instruct-2407-Q8_0.gguf", "Q8_0", 12.3),
            ],
        },
    ]
}

// ─── HuggingFace live search ──────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
struct HfSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct HfApiModel {
    model_id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

fn extract_quant(filename: &str) -> String {
    const KNOWN: &[&str] = &[
        "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_XS", "IQ2_XXS", "IQ2_XS", "Q8_0", "Q6_K", "Q5_K_M",
        "Q5_K_S", "Q5_1", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "Q3_K_M", "Q3_K_L", "Q3_K_S",
        "Q2_K", "F16", "BF16",
    ];
    let upper = filename.to_uppercase();
    for &k in KNOWN {
        if upper.contains(k) {
            return k.to_string();
        }
    }
    filename
        .trim_end_matches(".gguf")
        .rsplit('-')
        .next()
        .unwrap_or("GGUF")
        .to_uppercase()
}

fn hf_api_to_hub(m: HfApiModel) -> Option<HubModel> {
    let gguf_files: Vec<&HfSibling> = m
        .siblings
        .iter()
        .filter(|s| s.rfilename.to_lowercase().ends_with(".gguf"))
        .collect();
    if gguf_files.is_empty() {
        return None;
    }

    let quants: Vec<HubQuant> = gguf_files
        .iter()
        .map(|s| HubQuant {
            quant: extract_quant(&s.rfilename),
            size_gb: s.size.map(|sz| sz as f32 / 1_073_741_824.0).unwrap_or(0.0),
            url: format!(
                "https://huggingface.co/{}/resolve/main/{}",
                m.model_id, s.rfilename
            ),
            filename: s.rfilename.clone(),
        })
        .collect();

    let name = m
        .model_id
        .split('/')
        .last()
        .unwrap_or(&m.model_id)
        .replace('-', " ")
        .replace('_', " ");

    let tags: Vec<String> = m
        .tags
        .into_iter()
        .filter(|t| !t.contains(':') && !t.starts_with("base_model") && t.len() < 24)
        .take(5)
        .collect();

    Some(HubModel {
        id: m.model_id.clone(),
        name,
        family: "HuggingFace".into(),
        params: String::new(),
        description: format!("{:>10} downloads · {}", m.downloads, m.model_id),
        tags,
        quants,
    })
}

/// Search HuggingFace for GGUF models. Returns up to 20 results sorted by downloads.
/// `offset` is the number of results to skip (for pagination).
#[tauri::command]
pub async fn search_hub_models(query: String, offset: u32) -> Result<Vec<HubModel>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("InferenceBridge/1.0")
        .build()
        .map_err(|e| e.to_string())?;

    let offset_str = offset.to_string();
    let resp = client
        .get("https://huggingface.co/api/models")
        .query(&[
            ("filter", "gguf"),
            ("search", query.as_str()),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", "20"),
            ("offset", offset_str.as_str()),
            ("full", "true"),
        ])
        .send()
        .await
        .map_err(|e| format!("HuggingFace request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HuggingFace returned HTTP {}", resp.status()));
    }

    let models: Vec<HfApiModel> = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse HuggingFace response: {e}"))?;

    Ok(models.into_iter().filter_map(hf_api_to_hub).collect())
}

// ─── Download progress event ──────────────────────────────────────────────────

#[derive(Clone, Serialize)]
pub struct DownloadProgress {
    pub filename: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f32,
    pub done: bool,
    pub error: Option<String>,
}

// ─── Tauri commands ───────────────────────────────────────────────────────────

/// Open the containing folder for a path in the native file manager.
/// On Windows, selects the file itself in Explorer.
#[tauri::command]
pub async fn show_in_folder(path: String) -> Result<(), String> {
    let p = std::path::Path::new(&path);

    #[cfg(target_os = "windows")]
    {
        // /select highlights the specific file inside Explorer
        std::process::Command::new("explorer")
            .arg("/select,")
            .arg(p)
            .spawn()
            .map_err(|e| format!("Failed to open Explorer: {e}"))?;
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg("-R")
            .arg(p)
            .spawn()
            .map_err(|e| format!("Failed to open Finder: {e}"))?;
    }
    #[cfg(target_os = "linux")]
    {
        let dir = p.parent().unwrap_or(p);
        std::process::Command::new("xdg-open")
            .arg(dir)
            .spawn()
            .map_err(|e| format!("Failed to open file manager: {e}"))?;
    }
    Ok(())
}

/// Return the curated model catalog (synchronous, no I/O).
#[tauri::command]
pub async fn list_hub_models() -> Vec<HubModel> {
    catalog()
}

/// Stream-download a GGUF file into the first configured scan directory.
/// Emits `model-download-progress` events with live byte counts (~4/s).
#[tauri::command]
pub async fn download_hub_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
    url: String,
    filename: String,
) -> Result<String, String> {
    let dest_dir: std::path::PathBuf = {
        let s = state.read().await;
        match s.config.models.scan_dirs.first() {
            Some(d) => std::path::PathBuf::from(d),
            None => {
                return Err(
                    "No model directory configured. Add one in Settings → Model Directories."
                        .to_string(),
                )
            }
        }
    };

    tokio::fs::create_dir_all(&dest_dir)
        .await
        .map_err(|e| format!("Cannot create {}: {e}", dest_dir.display()))?;

    let dest_path = dest_dir.join(&filename);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(7200)) // 2-hour ceiling for large models
        .build()
        .map_err(|e| e.to_string())?;

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Server returned HTTP {} for {}",
            resp.status(),
            url
        ));
    }

    let total_bytes = resp.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(&dest_path)
        .await
        .map_err(|e| format!("Cannot create {}: {e}", dest_path.display()))?;

    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_emit = std::time::Instant::now();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download error: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {e}"))?;
        downloaded += chunk.len() as u64;

        if last_emit.elapsed().as_millis() >= 250 {
            let percent = if total_bytes > 0 {
                downloaded as f32 / total_bytes as f32
            } else {
                0.0
            };
            let _ = app.emit(
                "model-download-progress",
                DownloadProgress {
                    filename: filename.clone(),
                    downloaded_bytes: downloaded,
                    total_bytes,
                    percent,
                    done: false,
                    error: None,
                },
            );
            last_emit = std::time::Instant::now();
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("Flush error: {e}"))?;
    drop(file);

    // Rescan so the new model appears immediately in the registry and UI
    {
        let s = state.read().await;
        let dirs = s.config.models.scan_dirs.clone();
        drop(s);
        let scanned = tokio::task::spawn_blocking(move || crate::models::scanner::scan_all(&dirs))
            .await
            .unwrap_or_default();
        state.write().await.model_registry.update(scanned);
    }

    let _ = app.emit(
        "model-download-progress",
        DownloadProgress {
            filename: filename.clone(),
            downloaded_bytes: downloaded,
            total_bytes,
            percent: 1.0,
            done: true,
            error: None,
        },
    );

    Ok(dest_path.to_string_lossy().to_string())
}

/// Delete a local .gguf file and refresh the model registry.
#[tauri::command]
pub async fn delete_model_file(
    state: tauri::State<'_, SharedState>,
    path: String,
) -> Result<(), String> {
    let p = std::path::Path::new(&path);

    match p.extension().and_then(|e| e.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("gguf") => {}
        _ => return Err("Only .gguf files can be deleted via this command.".to_string()),
    }

    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }

    tokio::fs::remove_file(p)
        .await
        .map_err(|e| format!("Delete failed for {}: {e}", p.display()))?;

    // Rescan so deleted model vanishes from the UI
    {
        let s = state.read().await;
        let dirs = s.config.models.scan_dirs.clone();
        drop(s);
        let scanned = tokio::task::spawn_blocking(move || crate::models::scanner::scan_all(&dirs))
            .await
            .unwrap_or_default();
        state.write().await.model_registry.update(scanned);
    }

    Ok(())
}
