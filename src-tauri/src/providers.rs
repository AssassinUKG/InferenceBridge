use crate::state::SharedState;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeDoctorReport {
    pub checked_at: String,
    pub app_api: AppApiDoctor,
    pub active_runtime: ActiveRuntimeDoctor,
    pub providers: Vec<ProviderProbe>,
    pub summary: RuntimeDoctorSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeDoctorSummary {
    pub reachable_providers: usize,
    pub total_providers: usize,
    pub loaded_model: Option<String>,
    pub preferred_next_step: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AppApiDoctor {
    pub state: String,
    pub url: String,
    pub reachable: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActiveRuntimeDoctor {
    pub managed: bool,
    pub state: String,
    pub model: Option<String>,
    pub port: Option<u16>,
    pub backend: Option<String>,
    pub launch_context_size: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderProbe {
    pub id: String,
    pub provider_type: ProviderType,
    pub name: String,
    pub base_url: String,
    pub managed: bool,
    pub reachable: bool,
    pub status: String,
    pub models: Vec<ProviderModelInfo>,
    pub model_count: usize,
    pub context_limit: Option<u32>,
    pub output_limit: Option<u32>,
    pub endpoints: ProviderEndpointSupport,
    pub build_info: Option<String>,
    pub error: Option<String>,
    pub hints: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    ManagedLlamaCpp,
    ExternalLlamaCpp,
    LmStudio,
    Ollama,
    OpenAiCompatible,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderEndpointSupport {
    pub health: bool,
    pub props: bool,
    pub slots: bool,
    pub openai_models: bool,
    pub ollama_tags: bool,
}

impl ProviderEndpointSupport {
    fn empty() -> Self {
        Self {
            health: false,
            props: false,
            slots: false,
            openai_models: false,
            ollama_tags: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderModelInfo {
    pub id: String,
    pub name: Option<String>,
    pub context_limit: Option<u32>,
    pub output_limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModelsResponse {
    #[serde(default)]
    data: Vec<OpenAiModelObject>,
}

#[derive(Debug, Deserialize)]
struct OpenAiModelObject {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    max_context_length: Option<u32>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    context_window: Option<u32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    #[serde(default)]
    models: Vec<OllamaModelObject>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelObject {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

pub async fn collect_runtime_doctor(state: SharedState) -> RuntimeDoctorReport {
    let snapshot = {
        let s = state.read().await;
        let api_url =
            crate::api::server::reachable_api_url(&s.config.server.host, s.config.server.port);
        let detected_backend = s.process.detected_backend();
        (
            format!("{:?}", s.api_server_state),
            api_url,
            s.api_server_error.clone(),
            s.api_server_state.clone(),
            s.loaded_model.clone(),
            s.process.port(),
            s.process.state(),
            s.last_launch_preview.clone(),
            s.process.find_server_binary().is_some(),
            detected_backend,
        )
    };

    let backend = snapshot.9.lock().await.clone();
    let app_api_reachable = probe_ok(&format!("{}/health", snapshot.1.trim_end_matches('/'))).await;
    let app_api = AppApiDoctor {
        state: snapshot.0,
        url: snapshot.1,
        reachable: app_api_reachable,
        error: snapshot.2,
    };

    let active_runtime = ActiveRuntimeDoctor {
        managed: snapshot.8,
        state: format!("{:?}", snapshot.6),
        model: snapshot.4.clone(),
        port: (snapshot.5 > 0).then_some(snapshot.5),
        backend,
        launch_context_size: snapshot.7.as_ref().and_then(|preview| preview.context_size),
    };

    let mut providers = Vec::new();
    providers.push(
        probe_managed_provider(
            active_runtime.port,
            snapshot.4.clone(),
            snapshot.7.as_ref().and_then(|preview| preview.context_size),
            snapshot.8,
        )
        .await,
    );

    let skip_port = active_runtime.port;
    providers.push(
        probe_llamacpp_endpoint(
            "external-llamacpp-8080",
            "Standalone llama.cpp",
            8080,
            skip_port,
        )
        .await,
    );
    providers.push(probe_ollama_endpoint().await);
    let configured_lm_studio = {
        let s = state.read().await;
        (
            s.config.providers.lm_studio.enabled,
            s.config.providers.lm_studio.base_url.clone(),
        )
    };

    if configured_lm_studio.0 {
        providers.push(
            probe_openai_compatible_endpoint(
                "lm-studio-configured",
                ProviderType::LmStudio,
                "LM Studio configured",
                &configured_lm_studio.1,
            )
            .await,
        );
    }

    providers.push(
        probe_openai_compatible_endpoint(
            "lm-studio-1234",
            ProviderType::LmStudio,
            "LM Studio",
            "http://127.0.0.1:1234",
        )
        .await,
    );
    providers.push(
        probe_openai_compatible_endpoint(
            "lm-studio-1235",
            ProviderType::LmStudio,
            "LM Studio alternate",
            "http://127.0.0.1:1235",
        )
        .await,
    );

    let reachable_providers = providers
        .iter()
        .filter(|provider| provider.reachable)
        .count();
    let preferred_next_step = if reachable_providers == 0 {
        "Start InferenceBridge managed llama.cpp, LM Studio, Ollama, or a standalone llama-server."
            .to_string()
    } else if active_runtime.model.is_none() {
        "Select or load a model, then re-run doctor to confirm context and endpoint support."
            .to_string()
    } else {
        "Runtime is discoverable. Use /v1/runtime/status for live state and /v1/context/status for KV pressure.".to_string()
    };

    RuntimeDoctorReport {
        checked_at: chrono::Utc::now().to_rfc3339(),
        app_api,
        active_runtime,
        summary: RuntimeDoctorSummary {
            reachable_providers,
            total_providers: providers.len(),
            loaded_model: snapshot.4,
            preferred_next_step,
        },
        providers,
    }
}

async fn probe_managed_provider(
    port: Option<u16>,
    loaded_model: Option<String>,
    context_limit: Option<u32>,
    has_binary: bool,
) -> ProviderProbe {
    let Some(port) = port else {
        let models = loaded_model
            .map(|id| {
                vec![ProviderModelInfo {
                    id,
                    name: None,
                    context_limit,
                    output_limit: None,
                }]
            })
            .unwrap_or_default();
        let model_count = models.len();
        return ProviderProbe {
            id: "managed-llamacpp".to_string(),
            provider_type: ProviderType::ManagedLlamaCpp,
            name: "Managed llama.cpp".to_string(),
            base_url: "internal".to_string(),
            managed: true,
            reachable: false,
            status: if has_binary {
                "idle".to_string()
            } else {
                "missing-binary".to_string()
            },
            models,
            model_count,
            context_limit,
            output_limit: None,
            endpoints: ProviderEndpointSupport::empty(),
            build_info: None,
            error: None,
            hints: vec![if has_binary {
                "Managed llama-server is installed but no backend is currently loaded.".to_string()
            } else {
                "Download a managed llama-server build from Settings.".to_string()
            }],
        };
    };

    let base_url = format!("http://127.0.0.1:{port}");
    let mut probe = probe_llamacpp_base(
        "managed-llamacpp",
        ProviderType::ManagedLlamaCpp,
        "Managed llama.cpp",
        &base_url,
    )
    .await;
    probe.managed = true;
    if probe.models.is_empty() {
        if let Some(id) = loaded_model {
            probe.models.push(ProviderModelInfo {
                id,
                name: None,
                context_limit,
                output_limit: None,
            });
            probe.model_count = probe.models.len();
        }
    }
    if probe.context_limit.is_none() {
        probe.context_limit = context_limit;
    }
    probe
}

async fn probe_llamacpp_endpoint(
    id: &str,
    name: &str,
    port: u16,
    skip_port: Option<u16>,
) -> ProviderProbe {
    if skip_port == Some(port) {
        return ProviderProbe {
            id: id.to_string(),
            provider_type: ProviderType::ExternalLlamaCpp,
            name: name.to_string(),
            base_url: format!("http://127.0.0.1:{port}"),
            managed: false,
            reachable: false,
            status: "skipped-managed-port".to_string(),
            models: Vec::new(),
            model_count: 0,
            context_limit: None,
            output_limit: None,
            endpoints: ProviderEndpointSupport::empty(),
            build_info: None,
            error: None,
            hints: vec!["This port is currently used by the managed backend.".to_string()],
        };
    }

    probe_llamacpp_base(
        id,
        ProviderType::ExternalLlamaCpp,
        name,
        &format!("http://127.0.0.1:{port}"),
    )
    .await
}

async fn probe_llamacpp_base(
    id: &str,
    provider_type: ProviderType,
    name: &str,
    base_url: &str,
) -> ProviderProbe {
    let mut endpoints = ProviderEndpointSupport::empty();
    endpoints.health = probe_ok(&format!("{base_url}/health")).await;

    let props = get_json(&format!("{base_url}/props")).await;
    endpoints.props = props.is_ok();
    let props_value = props.ok();
    let context_limit = props_value
        .as_ref()
        .and_then(|value| value.pointer("/default_generation_settings/n_ctx"))
        .and_then(|value| value.as_u64())
        .map(|value| value as u32);
    let build_info = props_value
        .as_ref()
        .and_then(|value| value.get("build_info"))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());

    endpoints.slots = get_json(&format!("{base_url}/slots")).await.is_ok();
    let model_result = get_openai_models(&format!("{base_url}/v1/models")).await;
    endpoints.openai_models = model_result.is_ok();
    let models = model_result.unwrap_or_default();

    let reachable = endpoints.health || endpoints.props || endpoints.openai_models;
    ProviderProbe {
        id: id.to_string(),
        provider_type,
        name: name.to_string(),
        base_url: base_url.to_string(),
        managed: false,
        reachable,
        status: if reachable {
            "reachable".to_string()
        } else {
            "unreachable".to_string()
        },
        model_count: models.len(),
        models,
        context_limit,
        output_limit: None,
        endpoints,
        build_info,
        error: (!reachable).then(|| "No llama.cpp-compatible endpoint responded.".to_string()),
        hints: if reachable {
            vec!["llama.cpp endpoint responded to health, props, or model probes.".to_string()]
        } else {
            vec![
                "Start llama-server on this port or configure InferenceBridge to manage one."
                    .to_string(),
            ]
        },
    }
}

async fn probe_ollama_endpoint() -> ProviderProbe {
    let base_url = "http://127.0.0.1:11434";
    let mut probe =
        probe_openai_compatible_endpoint("ollama-11434", ProviderType::Ollama, "Ollama", base_url)
            .await;

    match get_json(&format!("{base_url}/api/tags")).await {
        Ok(value) => {
            probe.endpoints.ollama_tags = true;
            if probe.models.is_empty() {
                probe.models = parse_ollama_tags(value);
                probe.model_count = probe.models.len();
            }
            probe.reachable = true;
            probe.status = "reachable".to_string();
            probe.error = None;
        }
        Err(error) => {
            if !probe.reachable {
                probe.error = Some(format!(
                    "Ollama did not respond on /v1/models or /api/tags: {error}"
                ));
            }
        }
    }

    if probe.reachable {
        probe.hints = vec![
            "Ollama is reachable. Model context is usually controlled with options.num_ctx."
                .to_string(),
        ];
    }
    probe
}

async fn probe_openai_compatible_endpoint(
    id: &str,
    provider_type: ProviderType,
    name: &str,
    base_url: &str,
) -> ProviderProbe {
    let mut endpoints = ProviderEndpointSupport::empty();
    let normalized_base = normalize_openai_base_url(base_url);
    let models_result = get_openai_models(&format!("{normalized_base}/models")).await;
    endpoints.openai_models = models_result.is_ok();
    let models = models_result.unwrap_or_default();
    let reachable = endpoints.openai_models;
    let context_limit = models.iter().filter_map(|model| model.context_limit).max();
    let output_limit = models.iter().filter_map(|model| model.output_limit).max();

    ProviderProbe {
        id: id.to_string(),
        provider_type,
        name: name.to_string(),
        base_url: normalized_base,
        managed: false,
        reachable,
        status: if reachable {
            "reachable".to_string()
        } else {
            "unreachable".to_string()
        },
        model_count: models.len(),
        models,
        context_limit,
        output_limit,
        endpoints,
        build_info: None,
        error: (!reachable).then(|| "No OpenAI-compatible /v1/models response.".to_string()),
        hints: if reachable {
            vec!["OpenAI-compatible model listing is reachable.".to_string()]
        } else {
            vec![
                "Start this provider or update its base URL in the future provider settings."
                    .to_string(),
            ]
        },
    }
}

pub fn normalize_openai_base_url(base_url: &str) -> String {
    let mut value = base_url.trim().trim_end_matches('/').to_string();
    if value.ends_with("/v1") {
        return value;
    }
    value.push_str("/v1");
    value
}

async fn probe_ok(url: &str) -> bool {
    get_json(url).await.is_ok()
}

async fn get_json(url: &str) -> anyhow::Result<Value> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(1200))
        .connect_timeout(Duration::from_millis(500))
        .build()?;
    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        anyhow::bail!("HTTP {}", response.status());
    }
    Ok(response.json::<Value>().await?)
}

async fn get_openai_models(url: &str) -> anyhow::Result<Vec<ProviderModelInfo>> {
    let value = get_json(url).await?;
    parse_openai_models(value)
}

fn parse_openai_models(value: Value) -> anyhow::Result<Vec<ProviderModelInfo>> {
    let parsed: OpenAiModelsResponse = serde_json::from_value(value)?;
    Ok(parsed
        .data
        .into_iter()
        .map(|model| ProviderModelInfo {
            id: model.id,
            name: model.name,
            context_limit: model
                .max_context_length
                .or(model.context_length)
                .or(model.context_window),
            output_limit: model.max_output_tokens,
        })
        .collect())
}

fn parse_ollama_tags(value: Value) -> Vec<ProviderModelInfo> {
    serde_json::from_value::<OllamaTagsResponse>(value)
        .map(|response| {
            response
                .models
                .into_iter()
                .filter_map(|model| model.name.or(model.model))
                .map(|id| ProviderModelInfo {
                    id,
                    name: None,
                    context_limit: None,
                    output_limit: None,
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{parse_ollama_tags, parse_openai_models};
    use serde_json::json;

    #[test]
    fn parses_openai_model_limits() {
        let models = parse_openai_models(json!({
            "object": "list",
            "data": [{
                "id": "qwen3",
                "max_context_length": 32768,
                "max_output_tokens": 4096
            }]
        }))
        .expect("models should parse");

        assert_eq!(models[0].id, "qwen3");
        assert_eq!(models[0].context_limit, Some(32768));
        assert_eq!(models[0].output_limit, Some(4096));
    }

    #[test]
    fn parses_ollama_tags() {
        let models = parse_ollama_tags(json!({
            "models": [{ "name": "llama3.2:latest" }]
        }));

        assert_eq!(models[0].id, "llama3.2:latest");
    }
}
