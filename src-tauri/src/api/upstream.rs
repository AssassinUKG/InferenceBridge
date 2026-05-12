use axum::body::Body;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use futures_util::TryStreamExt;

use crate::api::errors::ApiErrorResponse;
use crate::state::SharedState;

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    pub name: String,
    pub base_url: String,
    pub api_key: Option<String>,
}

pub async fn active_openai_provider(state: &SharedState) -> Option<OpenAiProvider> {
    let s = state.read().await;
    match s.config.providers.active.as_str() {
        "lm_studio" if s.config.providers.lm_studio.enabled => Some(OpenAiProvider {
            name: "LM Studio".to_string(),
            base_url: crate::providers::normalize_openai_base_url(
                &s.config.providers.lm_studio.base_url,
            ),
            api_key: s.config.providers.lm_studio.api_key.clone(),
        }),
        _ => None,
    }
}

pub async fn proxy_json_to_openai_provider(
    provider: OpenAiProvider,
    endpoint: &str,
    body: serde_json::Value,
) -> Result<Response, ApiErrorResponse> {
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

    let upstream = request.send().await.map_err(|error| {
        ApiErrorResponse::service_unavailable(format!(
            "{} provider request failed: {error}",
            provider.name
        ))
    })?;

    response_from_upstream(provider.name, upstream).await
}

async fn response_from_upstream(
    provider_name: String,
    upstream: reqwest::Response,
) -> Result<Response, ApiErrorResponse> {
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| HeaderValue::from_bytes(value.as_bytes()).ok());

    let stream = upstream.bytes_stream().map_err(move |error| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("{provider_name} provider response failed: {error}"),
        )
    });

    let mut response = Response::builder().status(status);
    if let Some(content_type) = content_type {
        response = response.header(header::CONTENT_TYPE, content_type);
    }
    response
        .body(Body::from_stream(stream))
        .map_err(|error| ApiErrorResponse::service_unavailable(error.to_string()))
}
