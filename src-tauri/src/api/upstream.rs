use axum::body::{Body, Bytes};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use futures_util::StreamExt;

use crate::api::errors::ApiErrorResponse;
use crate::config::ProvidersConfig;
use crate::state::SharedState;

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub provider_type: String,
    pub enabled: bool,
}

pub fn configured_openai_providers(config: &ProvidersConfig) -> Vec<OpenAiProvider> {
    vec![
        OpenAiProvider {
            id: "lm_studio".to_string(),
            name: "LM Studio".to_string(),
            base_url: crate::providers::normalize_openai_base_url(&config.lm_studio.base_url),
            api_key: config.lm_studio.api_key.clone(),
            provider_type: "lm_studio".to_string(),
            enabled: config.lm_studio.enabled,
        },
        OpenAiProvider {
            id: "sglang".to_string(),
            name: "SGLang".to_string(),
            base_url: crate::providers::normalize_openai_base_url(&config.sglang.base_url),
            api_key: config.sglang.api_key.clone(),
            provider_type: "sglang".to_string(),
            enabled: config.sglang.enabled,
        },
    ]
}

pub async fn active_openai_provider(state: &SharedState) -> Option<OpenAiProvider> {
    let s = state.read().await;
    let active = s.config.providers.active.clone();

    configured_openai_providers(&s.config.providers)
        .into_iter()
        .find(|provider| provider.enabled && provider.id == active)
}

pub async fn proxy_json_to_openai_provider(
    state: SharedState,
    provider: OpenAiProvider,
    endpoint: &str,
    body: serde_json::Value,
) -> Result<Response, ApiErrorResponse> {
    let model = body
        .get("model")
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("upstream")
        .to_string();
    let generation = crate::state::begin_api_generation(&state, model).await;
    let request_id = generation.request_id.clone();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|error| ApiErrorResponse::service_unavailable(error.to_string()))?;
    let url = format!(
        "{}/{}",
        provider.base_url.trim_end_matches('/'),
        endpoint.trim_start_matches('/')
    );
    let mut request = client
        .post(url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .json(&body);

    if let Some(api_key) = provider.api_key.filter(|value| !value.trim().is_empty()) {
        request = request.bearer_auth(api_key);
    }

    let upstream = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            crate::state::append_live_stream_delta_for_request(
                &state,
                &request_id,
                "error",
                &format!("{} provider request failed: {error}", provider.name),
            )
            .await;
            crate::state::finish_api_generation_for_request(&state, &request_id, "error").await;
            return Err(ApiErrorResponse::service_unavailable(format!(
                "{} provider request failed: {error}",
                provider.name
            )));
        }
    };

    response_from_upstream(state, request_id, provider.name, upstream).await
}

async fn response_from_upstream(
    state: SharedState,
    request_id: String,
    provider_name: String,
    upstream: reqwest::Response,
) -> Result<Response, ApiErrorResponse> {
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| HeaderValue::from_bytes(value.as_bytes()).ok());

    let state_for_stream = state.clone();
    let request_id_for_stream = request_id.clone();
    let provider_for_stream = provider_name.clone();
    let mut upstream_stream = upstream.bytes_stream();
    let stream = async_stream::stream! {
        while let Some(chunk) = upstream_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    crate::state::append_live_stream_delta_for_request(
                        &state_for_stream,
                        &request_id_for_stream,
                        "raw",
                        &text,
                    ).await;
                    yield Ok::<Bytes, std::io::Error>(bytes);
                }
                Err(error) => {
                    let message = format!("{provider_for_stream} provider response failed: {error}");
                    crate::state::append_live_stream_delta_for_request(
                        &state_for_stream,
                        &request_id_for_stream,
                        "error",
                        &message,
                    ).await;
                    crate::state::finish_api_generation_for_request(
                        &state_for_stream,
                        &request_id_for_stream,
                        "error",
                    ).await;
                    yield Err::<Bytes, std::io::Error>(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        message,
                    ));
                    return;
                }
            }
        }
        crate::state::finish_api_generation_for_request(
            &state_for_stream,
            &request_id_for_stream,
            if status.is_success() { "completed" } else { "error" },
        ).await;
    };

    let mut response = Response::builder().status(status);
    if let Some(content_type) = content_type {
        response = response.header(header::CONTENT_TYPE, content_type);
    }
    response
        .body(Body::from_stream(stream))
        .map_err(|error| ApiErrorResponse::service_unavailable(error.to_string()))
}
