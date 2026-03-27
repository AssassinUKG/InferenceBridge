//! Axum HTTP server that runs alongside Tauri and serves the OpenAI-compatible API.

use axum::Router;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::oneshot;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::ApiServerState;
use crate::state::SharedState;

static API_SERVER_ACTIVE: AtomicBool = AtomicBool::new(false);
static API_SERVER_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
const PUBLIC_API_BIND_RETRIES: u32 = 40;

pub(crate) fn reachable_probe_host(host: &str) -> String {
    match host.trim() {
        "0.0.0.0" => "127.0.0.1".to_string(),
        "::" | "[::]" => "::1".to_string(),
        other => other.to_string(),
    }
}

pub(crate) fn reachable_api_url(host: &str, port: u16) -> String {
    let probe_host = reachable_probe_host(host);
    if probe_host.contains(':') && !probe_host.starts_with('[') {
        format!("http://[{probe_host}]:{port}/v1")
    } else {
        format!("http://{probe_host}:{port}/v1")
    }
}

/// Start the API server on the configured host and port.
pub async fn start_api_server(
    state: SharedState,
    host: &str,
    port: u16,
    source: &'static str,
) -> anyhow::Result<()> {
    serve_api_server(state, host, port, source, None).await
}

pub async fn start_api_server_with_shutdown(
    state: SharedState,
    host: &str,
    port: u16,
    source: &'static str,
    shutdown: oneshot::Receiver<()>,
) -> anyhow::Result<()> {
    serve_api_server(state, host, port, source, Some(shutdown)).await
}

async fn serve_api_server(
    state: SharedState,
    host: &str,
    port: u16,
    source: &'static str,
    shutdown: Option<oneshot::Receiver<()>>,
) -> anyhow::Result<()> {
    let attempt = API_SERVER_ATTEMPTS.fetch_add(1, Ordering::SeqCst) + 1;
    let pid = std::process::id();
    tracing::info!(
        pid,
        attempt,
        source,
        host,
        port,
        "API server start requested"
    );

    let Some(_guard) = ApiServerGuard::acquire() else {
        tracing::warn!(
            pid,
            attempt,
            source,
            host,
            port,
            backtrace = %std::backtrace::Backtrace::force_capture(),
            "Ignoring duplicate API server start request in the same process"
        );
        return Ok(());
    };

    update_api_status(&state, ApiServerState::Starting, None).await;

    let app = Router::new()
        .nest("/v1", api_routes())
        .nest("/api/v1", native_api_routes())
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    let addr = format!("{host}:{port}");
    tracing::info!(pid, attempt, source, %addr, "Binding InferenceBridge API server");

    let listener = match bind_public_api_listener(host, port, attempt, source).await {
        Ok(listener) => listener,
        Err((error, message)) => {
            tracing::error!(
                pid,
                attempt,
                source,
                %addr,
                error = %message,
                "API server bind failed"
            );
            update_api_status(&state, ApiServerState::Error, Some(message.clone())).await;
            return Err(error.into());
        }
    };

    update_api_status(&state, ApiServerState::Running, None).await;
    spawn_startup_probe(state.clone(), host.to_string(), port, attempt, source);
    tracing::info!(pid, attempt, source, %addr, "API server bound successfully");

    let server = axum::serve(listener, app);
    let serve_result = if let Some(shutdown) = shutdown {
        server
            .with_graceful_shutdown(async move {
                let _ = shutdown.await;
            })
            .await
    } else {
        server.await
    };

    if let Err(error) = serve_result {
        let message = format!("API server stopped unexpectedly: {error}");
        tracing::error!(
            pid,
            attempt,
            source,
            %addr,
            error = %message,
            "API server exited"
        );
        update_api_status(&state, ApiServerState::Error, Some(message.clone())).await;
        return Err(error.into());
    }

    tracing::info!(pid, attempt, source, %addr, "API server exited cleanly");
    update_api_status(&state, ApiServerState::Idle, None).await;
    Ok(())
}

/// Axum middleware: enforce Bearer token auth when `config.server.api_key` is set.
/// Health checks and CORS preflight are always allowed through.
async fn require_api_key(
    axum::extract::State(state): axum::extract::State<SharedState>,
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    // Always pass health probes and CORS preflight
    if req.uri().path().ends_with("/health") || req.method() == axum::http::Method::OPTIONS {
        return next.run(req).await;
    }

    let api_key = {
        let s = state.read().await;
        s.config.server.api_key.clone().unwrap_or_default()
    };

    // No key configured → open access
    if api_key.is_empty() {
        return next.run(req).await;
    }

    let auth = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if auth.strip_prefix("Bearer ").unwrap_or("") == api_key {
        return next.run(req).await;
    }

    axum::http::Response::builder()
        .status(401)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .header("WWW-Authenticate", "Bearer")
        .body(axum::body::Body::from(
            r#"{"error":{"message":"Invalid API key. Provide your key as: Authorization: Bearer <key>","type":"invalid_request_error","code":"invalid_api_key"}}"#,
        ))
        .unwrap_or_default()
}

fn api_routes() -> Router<SharedState> {
    Router::new()
        .route(
            "/responses",
            axum::routing::post(super::responses::responses),
        )
        .route(
            "/chat/completions",
            axum::routing::post(super::completions::chat_completions),
        )
        .route(
            "/completions",
            axum::routing::post(super::completions::text_completions),
        )
        .route("/models", axum::routing::get(super::models::list_models))
        .route(
            "/models/load",
            axum::routing::post(super::models::load_model),
        )
        .route(
            "/models/unload",
            axum::routing::post(super::models::unload_model),
        )
        .route(
            "/models/stats",
            axum::routing::get(super::models::current_model_stats).post(super::models::model_stats),
        )
        .route(
            "/models/:model",
            axum::routing::get(super::models::get_model),
        )
        .route(
            "/context/status",
            axum::routing::get(super::extensions::context_status),
        )
        .route(
            "/runtime/status",
            axum::routing::get(super::extensions::runtime_status),
        )
        .route(
            "/debug/profile",
            axum::routing::get(super::extensions::debug_profile),
        )
        .route(
            "/sessions",
            axum::routing::get(super::extensions::list_sessions)
                .post(super::extensions::create_session),
        )
        .route(
            "/sessions/:id",
            axum::routing::delete(super::extensions::delete_session),
        )
        .route(
            "/sessions/:id/messages",
            axum::routing::get(super::extensions::get_session_messages),
        )
        .route("/health", axum::routing::get(super::health::health_check))
}

fn native_api_routes() -> Router<SharedState> {
    Router::new()
        .route("/models", axum::routing::get(super::models::list_models))
        // LM Studio-compatible load/unload endpoints at /api/v1/models/load and /api/v1/models/unload.
        // HelixClaw and other LM Studio clients POST model-load requests (with context_length) here.
        .route(
            "/models/load",
            axum::routing::post(super::models::load_model),
        )
        .route(
            "/models/unload",
            axum::routing::post(super::models::unload_model),
        )
        .route("/health", axum::routing::get(super::health::health_check))
}

async fn update_api_status(
    state: &SharedState,
    api_server_state: ApiServerState,
    api_server_error: Option<String>,
) {
    use tauri::Emitter;

    // Update state and grab the app handle in a single write lock.
    let app_handle = {
        let mut app_state = state.write().await;
        app_state.api_server_state = api_server_state.clone();
        app_state.api_server_error = api_server_error.clone();
        app_state.app_handle.clone()
    };

    // Notify the GUI immediately so it doesn't have to wait for the 3 s poll.
    if let Some(handle) = app_handle {
        let _ = handle.emit(
            "api-server-state-changed",
            serde_json::json!({
                "state": format!("{:?}", api_server_state),
                "error": api_server_error,
            }),
        );
    }
}

fn format_bind_error(host: &str, port: u16, error: &std::io::Error) -> String {
    let api_url = reachable_api_url(host, port);
    if error.kind() == std::io::ErrorKind::AddrInUse {
        if let Some(diagnostic) = diagnose_port_conflict(port) {
            return format!(
                "API server could not start on {api_url} because port {port} is already in use. {diagnostic}"
            );
        }
        return format!(
            "API server could not start on {api_url} because port {port} is already in use. \
Close the other InferenceBridge or headless server process using that port, or change the API Surface port and restart the app."
        );
    }

    format!("API server could not start on {api_url}: {error}")
}

async fn bind_public_api_listener(
    host: &str,
    port: u16,
    attempt: u64,
    source: &'static str,
) -> Result<tokio::net::TcpListener, (std::io::Error, String)> {
    let addr = format!("{host}:{port}");
    let mut last_error: Option<std::io::Error> = None;
    let mut last_message: Option<String> = None;

    for bind_attempt in 1..=PUBLIC_API_BIND_RETRIES {
        match tokio::net::TcpListener::bind(&addr).await {
            Ok(listener) => return Ok(listener),
            Err(error) => {
                let message = format_bind_error(host, port, &error);
                let should_retry = error.kind() == std::io::ErrorKind::AddrInUse
                    && bind_attempt < PUBLIC_API_BIND_RETRIES;

                if should_retry {
                    tracing::warn!(
                        pid = std::process::id(),
                        attempt,
                        source,
                        bind_attempt,
                        %addr,
                        error = %message,
                        "Public API bind failed, retrying"
                    );
                    last_error = Some(error);
                    last_message = Some(message);
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    continue;
                }

                return Err((error, message));
            }
        }
    }

    Err((
        last_error.unwrap_or_else(|| std::io::Error::other("unknown bind failure")),
        last_message
            .unwrap_or_else(|| format!("API server could not start on http://{host}:{port}/v1")),
    ))
}

fn spawn_startup_probe(
    state: SharedState,
    host: String,
    port: u16,
    attempt: u64,
    source: &'static str,
) {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap_or_default();
        let url = format!("{}/health", reachable_api_url(&host, port));
        let pid = std::process::id();

        for probe in 1..=8 {
            match client.get(&url).send().await {
                Ok(response) => {
                    tracing::info!(
                        pid,
                        attempt,
                        source,
                        probe,
                        status = %response.status(),
                        %url,
                        "API startup self-probe succeeded"
                    );
                    return;
                }
                Err(error) => {
                    if probe == 8 {
                        tracing::error!(
                            pid,
                            attempt,
                            source,
                            probe,
                            %url,
                            error = %error,
                            "API startup self-probe failed after retries"
                        );

                        let mut app_state = state.write().await;
                        if matches!(app_state.api_server_state, ApiServerState::Running)
                            && app_state.api_server_error.is_none()
                        {
                            app_state.api_server_error = Some(format!(
                                "The API server bound on {host}:{port}, but an internal startup probe could not reach {url}. Check firewall or startup logs."
                            ));
                        }
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(150)).await;
                }
            }
        }
    });
}

struct ApiServerGuard;

impl ApiServerGuard {
    fn acquire() -> Option<Self> {
        API_SERVER_ACTIVE
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .ok()
            .map(|_| Self)
    }
}

impl Drop for ApiServerGuard {
    fn drop(&mut self) {
        API_SERVER_ACTIVE.store(false, Ordering::SeqCst);
    }
}

fn diagnose_port_conflict(port: u16) -> Option<String> {
    #[cfg(windows)]
    {
        return diagnose_port_conflict_windows(port);
    }

    #[allow(unreachable_code)]
    None
}

#[cfg(windows)]
fn diagnose_port_conflict_windows(port: u16) -> Option<String> {
    use std::os::windows::process::CommandExt;
    use std::process::Command;

    let mut command = Command::new("netstat");
    command.creation_flags(0x08000000);
    let output = command.args(["-ano", "-p", "tcp"]).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let port_suffix = format!(":{port}");
    let current_pid = std::process::id();

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let columns: Vec<&str> = line.split_whitespace().collect();
        if columns.len() < 5 {
            continue;
        }

        let proto = columns[0];
        let local_addr = columns[1];
        let state = columns[3];
        let Ok(pid) = columns[4].parse::<u32>() else {
            continue;
        };

        if !proto.eq_ignore_ascii_case("TCP")
            || !local_addr.ends_with(&port_suffix)
            || state != "LISTENING"
        {
            continue;
        }

        let process_name = process_name_for_pid_windows(pid);

        if pid == current_pid {
            return Some(format!(
                "Diagnostics: port {port} is already owned by this same InferenceBridge process (PID {pid}), which means the app tried to start the embedded API listener twice in one launch."
            ));
        }

        if let Some(name) = &process_name {
            let lower = name.to_lowercase();
            if lower.contains("inference-bridge") {
                return Some(format!(
                    "Diagnostics: port {port} is already owned by another InferenceBridge instance (PID {pid}). Close the other app window or stale process and try again."
                ));
            }

            return Some(format!(
                "Diagnostics: port {port} is currently owned by PID {pid} ({name})."
            ));
        }

        return Some(format!(
            "Diagnostics: Windows briefly reported port {port} as busy, but the owner could not be resolved. The conflict may already have cleared; retry API if the port now looks free."
        ));
    }

    Some(format!(
        "Diagnostics: Windows briefly reported port {port} as busy, but no current LISTENING owner was found in netstat output. The conflict may already have cleared; retry API if the port now looks free."
    ))
}

#[cfg(windows)]
fn process_name_for_pid_windows(pid: u32) -> Option<String> {
    use std::os::windows::process::CommandExt;
    use std::process::Command;

    let filter = format!("PID eq {pid}");
    let mut command = Command::new("tasklist");
    command.creation_flags(0x08000000);
    let output = command
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    if line.is_empty() || line.starts_with("INFO:") {
        return None;
    }

    let parts: Vec<&str> = line.split(',').collect();
    let name = parts.first()?.trim_matches('"').trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}
