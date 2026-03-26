//! OpenAI-compatible error response types.
//!
//! Spec: https://platform.openai.com/docs/guides/error-codes
//!
//! All API endpoints should return errors in this format so that standard
//! OpenAI client libraries can parse them correctly.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
    pub param: Option<String>,
}

impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<Self>) {
        let error_type = match status {
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
                "invalid_request_error"
            }
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::NOT_FOUND => "invalid_request_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            StatusCode::SERVICE_UNAVAILABLE => "server_error",
            _ => "api_error",
        };

        (
            status,
            Json(Self {
                error: ApiErrorBody {
                    message: message.into(),
                    error_type: error_type.to_string(),
                    code: None,
                    param: None,
                },
            }),
        )
    }

    pub fn no_model() -> (StatusCode, Json<Self>) {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "No model is currently loaded. Load a model via POST /v1/models/load or by specifying the `model` field in your request.",
        )
    }

    pub fn model_not_found(name: &str) -> (StatusCode, Json<Self>) {
        Self::new(
            StatusCode::NOT_FOUND,
            format!("Model '{name}' not found. Use GET /v1/models to see available models."),
        )
    }

    pub fn inference_failed(detail: &str) -> (StatusCode, Json<Self>) {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Inference failed: {detail}"),
        )
    }

    pub fn bad_request(detail: impl Into<String>) -> (StatusCode, Json<Self>) {
        Self::new(StatusCode::BAD_REQUEST, detail)
    }
}

/// A type alias for returning OpenAI-format errors from Axum handlers.
pub type ApiResult<T> = Result<T, ApiErrorResponse>;

/// Newtype wrapping `(StatusCode, Json<ApiError>)` so it implements `IntoResponse`.
pub struct ApiErrorResponse(pub StatusCode, pub Json<ApiError>);

impl IntoResponse for ApiErrorResponse {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

impl ApiErrorResponse {
    pub fn no_model() -> Self {
        let (s, j) = ApiError::no_model();
        Self(s, j)
    }

    pub fn model_not_found(name: &str) -> Self {
        let (s, j) = ApiError::model_not_found(name);
        Self(s, j)
    }

    pub fn inference_failed(detail: &str) -> Self {
        let (s, j) = ApiError::inference_failed(detail);
        Self(s, j)
    }

    pub fn bad_request(detail: impl Into<String>) -> Self {
        let (s, j) = ApiError::bad_request(detail);
        Self(s, j)
    }

    pub fn service_unavailable(detail: impl Into<String>) -> Self {
        let (s, j) = ApiError::new(StatusCode::SERVICE_UNAVAILABLE, detail);
        Self(s, j)
    }
}
