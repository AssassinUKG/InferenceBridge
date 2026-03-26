//! HTTP client for llama-server's /completion endpoint.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Request to llama-server's /completion endpoint.
#[derive(Debug, Serialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Enable parsing of special tokens (e.g. <|im_start|>) in the prompt text.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub special: bool,
}

/// Response from llama-server's /completion endpoint (non-streaming).
#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub stop: bool,
    pub tokens_predicted: Option<u32>,
    pub tokens_evaluated: Option<u32>,
    pub timings: Option<Timings>,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub struct Timings {
    pub predicted_per_second: Option<f64>,
    pub prompt_per_second: Option<f64>,
}

/// Slot status from llama-server's /slots endpoint.
#[derive(Debug, Deserialize)]
pub struct SlotInfo {
    pub id: u32,
    pub n_ctx: u32,
    /// May be absent in newer llama-server versions.
    #[serde(default)]
    pub n_past: u32,
    /// May be absent in newer llama-server versions.
    #[serde(default)]
    pub state: u32,
    /// Nested token info (newer builds).
    #[serde(default)]
    pub next_token: Option<NextTokenInfo>,
    #[serde(default)]
    pub is_processing: bool,
}

#[derive(Debug, Deserialize)]
pub struct NextTokenInfo {
    #[serde(default)]
    pub n_decoded: u32,
    #[serde(default)]
    pub n_remain: i64,
}

/// Server properties from /props endpoint (always available).
#[derive(Debug, Deserialize)]
pub struct ServerProps {
    pub default_generation_settings: Option<GenerationSettings>,
    #[serde(default)]
    pub build_info: Option<String>,
    #[serde(default)]
    pub total_slots: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct GenerationSettings {
    pub n_ctx: Option<u32>,
}

/// Client for communicating with a local llama-server instance.
pub struct LlamaClient {
    client: reqwest::Client,
    base_url: String,
}

impl LlamaClient {
    pub fn new(port: u16) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_default();
        Self {
            client,
            base_url: format!("http://127.0.0.1:{}", port),
        }
    }

    pub fn set_port(&mut self, port: u16) {
        self.base_url = format!("http://127.0.0.1:{}", port);
    }

    /// Send a non-streaming completion request.
    pub async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let url = format!("{}/completion", self.base_url);
        let resp = self.client.post(&url).json(request).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("llama-server returned {status}: {body}");
        }
        Ok(resp.json().await?)
    }

    /// Send a streaming completion request, returning an SSE stream.
    pub async fn complete_stream(&self, request: &CompletionRequest) -> Result<reqwest::Response> {
        let url = format!("{}/completion", self.base_url);
        let resp = self.client.post(&url).json(request).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("llama-server returned {status}: {body}");
        }
        Ok(resp)
    }

    /// Query slot information for context/KV monitoring.
    pub async fn get_slots(&self) -> Result<Vec<SlotInfo>> {
        let url = format!("{}/slots", self.base_url);
        let resp = self.client.get(&url).send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("Failed to get slots: {}", resp.status());
        }
        Ok(resp.json().await?)
    }

    /// Query server properties (always available, provides n_ctx).
    pub async fn get_props(&self) -> Result<ServerProps> {
        let url = format!("{}/props", self.base_url);
        let resp = self.client.get(&url).send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("Failed to get props: {}", resp.status());
        }
        Ok(resp.json().await?)
    }

    /// Check health status.
    pub async fn health(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}
