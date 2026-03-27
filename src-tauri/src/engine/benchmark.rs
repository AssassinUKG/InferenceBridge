//! Model benchmarking and test utilities

use crate::engine::client::{CompletionRequest, LlamaClient, Timings};
use crate::engine::process::LaunchConfig;
use crate::state::SharedState;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelTestStats {
    pub model: String,
    pub context_size: u32,
    pub prompt: String,
    pub response: String,
    pub timings: Option<Timings>,
    pub elapsed_ms: u128,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
    pub prompt_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub end_to_end_tokens_per_second: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
}

/// Test a model with the given prompt and settings, returning stats.
pub async fn test_model(
    shared_state: SharedState,
    model_name: &str,
    context_size: u32,
    prompt: &str,
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    seed: Option<i64>,
) -> anyhow::Result<ModelTestStats> {
    // 1. Find model in registry
    let (model, profile, model_path): (
        crate::models::scanner::ScannedModel,
        crate::models::profiles::ModelProfile,
        std::path::PathBuf,
    ) = {
        let state = shared_state.read().await;
        let model = state
            .model_registry
            .find_by_name(model_name)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_name))?;
        (model.clone(), model.profile.clone(), model.path.clone())
    };

    // 2. Launch model (if not already loaded)
    {
        let mut state = shared_state.write().await;
        let config = LaunchConfig {
            model_path: model_path.clone(),
            context_size: Some(context_size),
            gpu_layers: state.config.process.gpu_layers,
            threads: state.config.process.threads,
            threads_batch: state.config.process.threads_batch,
            port: state.process.port(),
            backend_preference: state.config.process.backend_preference.clone(),
            batch_size: state.config.process.batch_size,
            ubatch_size: state.config.process.ubatch_size,
            flash_attn: state.config.process.flash_attn,
            use_mmap: state.config.process.use_mmap,
            use_mlock: state.config.process.use_mlock,
            cont_batching: state.config.process.cont_batching,
            parallel_slots: state.config.process.parallel_slots,
            main_gpu: state.config.process.main_gpu,
            defrag_thold: state.config.process.defrag_thold,
            rope_freq_scale: state.config.process.rope_freq_scale,
        };
        state.process.launch(config).await?;
        state.loaded_model = Some(model.filename.clone());
    }

    // 3. Prepare prompt
    let rendered = crate::templates::engine::render_prompt(
        &[crate::templates::engine::ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        &profile,
    );
    let request = CompletionRequest {
        prompt: rendered.clone(),
        n_predict: Some(max_tokens as i32),
        temperature: temperature.or(profile.default_temperature),
        top_p: top_p.or(profile.default_top_p),
        top_k: top_k.or(profile.default_top_k),
        min_p: profile.default_min_p,
        presence_penalty: profile.default_presence_penalty,
        frequency_penalty: None,
        repeat_penalty: None,
        seed,
        stream: false,
        stop: profile.stop_markers.clone(),
        special: true,
        image_data: vec![],
    };
    let port = {
        let state = shared_state.read().await;
        state.process.port()
    };
    let client = LlamaClient::new(port);
    let start = std::time::Instant::now();
    let resp = client.complete(&request).await?;
    let elapsed = start.elapsed().as_millis();
    let response_text = crate::normalize::think_strip::strip_think_tags_with_style(
        &resp.content,
        profile.think_tag_style,
    );
    let prompt_tokens = resp.tokens_evaluated;
    let completion_tokens = resp.tokens_predicted;
    let total_tokens = match (prompt_tokens, completion_tokens) {
        (Some(prompt_tokens), Some(completion_tokens)) => Some(prompt_tokens + completion_tokens),
        _ => None,
    };
    let prompt_tokens_per_second = resp.timings.as_ref().and_then(|timings| timings.prompt_per_second);
    let decode_tokens_per_second =
        resp.timings.as_ref().and_then(|timings| timings.predicted_per_second);
    let prefill_ms = match (prompt_tokens, prompt_tokens_per_second) {
        (Some(tokens), Some(tokens_per_second)) if tokens_per_second > 0.0 => {
            Some((tokens as f64 / tokens_per_second) * 1000.0)
        }
        _ => None,
    };
    let decode_ms = match (completion_tokens, decode_tokens_per_second) {
        (Some(tokens), Some(tokens_per_second)) if tokens_per_second > 0.0 => {
            Some((tokens as f64 / tokens_per_second) * 1000.0)
        }
        _ => None,
    };
    let elapsed_secs = elapsed as f64 / 1000.0;
    let end_to_end_tokens_per_second = match (completion_tokens, elapsed_secs) {
        (Some(tokens), elapsed_secs) if elapsed_secs > 0.0 => Some(tokens as f64 / elapsed_secs),
        _ => None,
    };
    Ok(ModelTestStats {
        model: model_name.to_string(),
        context_size,
        prompt: prompt.to_string(),
        response: response_text,
        timings: resp.timings,
        elapsed_ms: elapsed,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        prompt_tokens_per_second,
        decode_tokens_per_second,
        end_to_end_tokens_per_second,
        prefill_ms,
        decode_ms,
    })
}
