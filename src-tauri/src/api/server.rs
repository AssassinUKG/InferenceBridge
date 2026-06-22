//! Axum HTTP server that runs alongside Tauri and serves the OpenAI-compatible API.

use axum::Router;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::oneshot;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::ApiServerState;
use crate::state::SharedState;

static API_SERVER_ACTIVE: AtomicBool = AtomicBool::new(false);
static API_SERVER_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static PROXY_HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn proxy_http_client() -> reqwest::Client {
    PROXY_HTTP_CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .connect_timeout(Duration::from_secs(5))
                .pool_max_idle_per_host(8)
                .build()
                .unwrap_or_default()
        })
        .clone()
}

fn is_hop_by_hop_header(name: &axum::http::HeaderName) -> bool {
    matches!(
        name.as_str().to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "host"
    )
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Core server loop
// ---------------------------------------------------------------------------

async fn serve_api_server(
    state: SharedState,
    host: &str,
    port: u16,
    source: &'static str,
    shutdown: Option<oneshot::Receiver<()>>,
) -> anyhow::Result<()> {
    let attempt = API_SERVER_ATTEMPTS.fetch_add(1, Ordering::SeqCst) + 1;
    let pid = std::process::id();
    let addr = format!("{host}:{port}");

    tracing::info!(pid, attempt, source, %addr, "API server start requested");

    let Some(_guard) = ApiServerGuard::acquire() else {
        tracing::warn!(
            pid,
            attempt,
            source,
            %addr,
            "Duplicate API server start request in same process — ignoring"
        );
        return Ok(());
    };

    update_api_status(&state, ApiServerState::Starting, None).await;

    let app = Router::new()
        .nest("/v1", api_routes())
        .nest("/api/v1", native_api_routes())
        // Transparent proxy: any route NOT matched above is forwarded to the
        // internal llama-server process on its auto-assigned ephemeral port.
        // This means ALL llama-server native endpoints (/props, /slots,
        // /tokenize, /detokenize, /embedding, etc.) work through IB
        // automatically — no explicit proxy routes needed.
        // The /v1/* and /api/v1/* routes take priority (.nest() before .fallback()).
        .fallback(backend_proxy_fallback)
        .route_layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_api_key,
        ))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    // Attempt to evict any stale process holding our port before binding.
    evict_port_blocker(host, port, pid).await;

    tracing::info!(pid, attempt, source, %addr, "Binding API listener");

    let listener = match bind_with_retry(host, port).await {
        Ok(l) => l,
        Err(e) => {
            let msg = format!("API server could not bind {addr}: {e}");
            tracing::error!(pid, attempt, source, %addr, error = %e, "API server bind failed");
            update_api_status(&state, ApiServerState::Error, Some(msg.clone())).await;
            return Err(anyhow::anyhow!(msg));
        }
    };

    update_api_endpoint(&state, host.to_string(), port).await;
    update_api_status(&state, ApiServerState::Running, None).await;
    spawn_startup_probe(state.clone(), host.to_string(), port, attempt, source);
    tracing::info!(pid, attempt, source, %addr, "API server bound and running");

    let server = axum::serve(listener, app);
    let result = if let Some(shutdown_rx) = shutdown {
        server
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    } else {
        server.await
    };

    match result {
        Ok(()) => {
            tracing::info!(pid, attempt, source, %addr, "API server exited cleanly");
            clear_api_endpoint(&state, host, port).await;
            update_api_status(&state, ApiServerState::Idle, None).await;
        }
        Err(e) => {
            let msg = format!("API server stopped unexpectedly: {e}");
            tracing::error!(pid, attempt, source, %addr, error = %e, "API server exited with error");
            clear_api_endpoint(&state, host, port).await;
            update_api_status(&state, ApiServerState::Error, Some(msg.clone())).await;
            return Err(e.into());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Port eviction (Windows) — replaces clear_stale_api_port_process,
// process_name_for_pid_windows, diagnose_port_conflict*
// ---------------------------------------------------------------------------

/// Attempt to free `port` before we try to bind it.
///
/// Steps:
///   1. Quick probe — if the port is already free, return immediately.
///   2. Find the LISTENING PID from netstat.
///   3. Resolve its process name via tasklist (best-effort).
///   4. If it's our own PID, log a bug warning and bail — we can't kill ourselves.
///   5. Attempt `taskkill /PID <n> /F` only for recognized own processes
///      (`llama-server` or `inference-bridge`). Unknown processes are never
///      killed.
///   6. Poll for up to 1.5 s for the port to actually free up.
///   7. Log the final outcome so every path is visible in logs.
async fn evict_port_blocker(host: &str, port: u16, current_pid: u32) {
    #[cfg(windows)]
    {
        use std::net::TcpListener;

        let probe_host = reachable_probe_host(host);

        // Fast path — nothing to do.
        if TcpListener::bind(format!("{probe_host}:{port}")).is_ok() {
            tracing::debug!(port, "evict_port_blocker: port is free, nothing to do");
            return;
        }

        tracing::info!(port, "evict_port_blocker: port busy, scanning for owner");

        // Find the PID that holds the LISTENING socket.
        let owner_pid = match find_listening_pid(port) {
            Some(pid) => pid,
            None => {
                // Port is busy but we can't find a LISTENING owner — may be a
                // race (process dying right now) or a transient kernel state.
                tracing::warn!(
                    port,
                    "evict_port_blocker: port busy but no LISTENING owner found in netstat \
                     — will attempt bind and let the OS error propagate"
                );
                return;
            }
        };

        // Resolve the name for logging — failure here is non-fatal.
        let owner_name = resolve_pid_name(owner_pid);

        if owner_pid == current_pid {
            tracing::warn!(
                port,
                owner_pid,
                current_pid,
                "evict_port_blocker: port is still owned by this process; waiting for previous listener to release"
            );
            if wait_until_port_bindable(&probe_host, port, Duration::from_secs(5)).await {
                tracing::info!(
                    port,
                    owner_pid,
                    "evict_port_blocker: previous self-owned listener released port"
                );
            } else {
                tracing::warn!(
                    port,
                    owner_pid,
                    "evict_port_blocker: previous self-owned listener did not release before bind attempt"
                );
            }
            return;
        }

        if !port_owner_is_killable(owner_pid, owner_name.as_deref(), current_pid) {
            tracing::warn!(
                port,
                owner_pid,
                owner_name = owner_name
                    .as_deref()
                    .unwrap_or("<unknown — tasklist returned nothing>"),
                "evict_port_blocker: port is owned by an unrecognized process; refusing to kill it"
            );
            return;
        }

        tracing::warn!(
            port,
            owner_pid,
            owner_name = owner_name
                .as_deref()
                .unwrap_or("<ghost — tasklist returned nothing>"),
            "evict_port_blocker: attempting force-kill of port owner"
        );

        let killed = force_kill_pid(owner_pid);
        tracing::info!(
            port,
            owner_pid,
            killed,
            owner_name = owner_name.as_deref().unwrap_or("<ghost>"),
            "evict_port_blocker: taskkill issued"
        );

        let freed = wait_until_port_bindable(&probe_host, port, Duration::from_secs(5)).await;

        if freed {
            tracing::info!(
                port,
                owner_pid,
                owner_name = owner_name.as_deref().unwrap_or("<ghost>"),
                "evict_port_blocker: port is now free"
            );
        } else {
            tracing::error!(
                port,
                owner_pid,
                owner_name = owner_name.as_deref().unwrap_or("<ghost>"),
                "evict_port_blocker: port STILL busy after kill attempt — \
                 bind will likely fail. PID may be protected (SYSTEM/antivirus) \
                 or the socket is held by a kernel handle that outlived the process. \
                 Reboot will clear it."
            );
        }
    }

    #[cfg(not(windows))]
    let _ = (host, port, current_pid);
}

#[cfg(windows)]
async fn wait_until_port_bindable(probe_host: &str, port: u16, timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if std::net::TcpListener::bind(format!("{probe_host}:{port}")).is_ok() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}

fn port_owner_is_killable(owner_pid: u32, owner_name: Option<&str>, current_pid: u32) -> bool {
    if owner_pid == current_pid {
        return false;
    }
    let lower = owner_name.unwrap_or_default().to_ascii_lowercase();
    lower.contains("llama-server") || lower.contains("inference-bridge")
}

/// Parse `netstat -ano -p tcp` output and return the PID of the process
/// that has a LISTENING socket on `port`.  Returns `None` if not found.
#[cfg(windows)]
fn find_listening_pid(port: u16) -> Option<u32> {
    use std::os::windows::process::CommandExt;

    let output = std::process::Command::new("netstat")
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .args(["-ano", "-p", "tcp"])
        .output()
        .map_err(|e| tracing::warn!(error = %e, "find_listening_pid: netstat failed"))
        .ok()?;

    if !output.status.success() {
        tracing::warn!(
            status = ?output.status,
            "find_listening_pid: netstat exited non-zero"
        );
        return None;
    }

    let suffix = format!(":{port}");

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        // netstat -ano line: proto  local_addr  remote_addr  state  pid
        if cols.len() < 5 {
            continue;
        }
        if !cols[0].eq_ignore_ascii_case("TCP") {
            continue;
        }
        if !cols[1].ends_with(&suffix) {
            continue;
        }
        if cols[3] != "LISTENING" {
            continue;
        }
        if let Ok(pid) = cols[4].parse::<u32>() {
            tracing::debug!(
                port,
                pid,
                local_addr = cols[1],
                "find_listening_pid: found owner"
            );
            return Some(pid);
        }
    }

    tracing::debug!(port, "find_listening_pid: no LISTENING owner found");
    None
}

/// Resolve a PID to a process name using `tasklist`.
/// Returns `None` if the process is not visible to tasklist (ghost PID,
/// protected process, or different user session).
#[cfg(windows)]
fn resolve_pid_name(pid: u32) -> Option<String> {
    use std::os::windows::process::CommandExt;

    let filter = format!("PID eq {pid}");
    let output = std::process::Command::new("tasklist")
        .creation_flags(0x08000000)
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
        .map_err(|e| {
            tracing::warn!(pid, error = %e, "resolve_pid_name: tasklist invocation failed");
        })
        .ok()?;

    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?.trim();

    // tasklist returns "INFO: No tasks are running..." when PID is not found.
    if line.is_empty() || line.starts_with("INFO:") {
        tracing::debug!(
            pid,
            "resolve_pid_name: PID not visible to tasklist (ghost/protected)"
        );
        return None;
    }

    // CSV format: "name","pid","session","num","mem"
    let name = line.split(',').next()?.trim_matches('"').trim();
    if name.is_empty() {
        return None;
    }

    tracing::debug!(pid, name, "resolve_pid_name: resolved");
    Some(name.to_string())
}

/// Issue `taskkill /PID <pid> /F`.  Returns `true` if the command exited
/// successfully.  A `false` return means the process was protected or already
/// gone — caller should still poll the port since "already gone" is fine.
#[cfg(windows)]
fn force_kill_pid(pid: u32) -> bool {
    use std::os::windows::process::CommandExt;

    match std::process::Command::new("taskkill")
        .creation_flags(0x08000000)
        .args(["/PID", &pid.to_string(), "/F"])
        .output()
    {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if out.status.success() {
                tracing::debug!(pid, stdout = %stdout.trim(), "force_kill_pid: success");
                true
            } else {
                tracing::warn!(
                    pid,
                    exit_code = ?out.status.code(),
                    stdout = %stdout.trim(),
                    stderr = %stderr.trim(),
                    "force_kill_pid: taskkill exited non-zero \
                     (process may be protected or already gone)"
                );
                false
            }
        }
        Err(e) => {
            tracing::warn!(pid, error = %e, "force_kill_pid: failed to run taskkill");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Bind with retry
// ---------------------------------------------------------------------------

/// Bind the public API listener. Retries long enough for Windows to release a
/// listener that was just stopped during settings-save/API restarts.
async fn bind_with_retry(host: &str, port: u16) -> std::io::Result<tokio::net::TcpListener> {
    let addr = format!("{host}:{port}");
    const MAX_ATTEMPTS: u32 = 20;
    const RETRY_DELAY_MS: u64 = 250;

    let mut last_err: Option<std::io::Error> = None;

    for attempt in 1..=MAX_ATTEMPTS {
        match tokio::net::TcpListener::bind(&addr).await {
            Ok(listener) => {
                if attempt > 1 {
                    tracing::info!(
                        %addr,
                        attempt,
                        "bind_with_retry: succeeded on retry"
                    );
                }
                return Ok(listener);
            }
            Err(e) => {
                tracing::warn!(
                    %addr,
                    attempt,
                    max_attempts = MAX_ATTEMPTS,
                    error = %e,
                    "bind_with_retry: bind failed"
                );
                last_err = Some(e);
                if attempt < MAX_ATTEMPTS {
                    tokio::time::sleep(Duration::from_millis(RETRY_DELAY_MS)).await;
                }
            }
        }
    }

    Err(last_err.unwrap_or_else(|| std::io::Error::other("bind failed")))
}

// ---------------------------------------------------------------------------
// Transparent backend proxy (single-port architecture)
// ---------------------------------------------------------------------------

/// Transparent proxy: forwards any unmatched request to the internal
/// llama-server process on its auto-assigned ephemeral port.
///
/// This is the core of InferenceBridge's single-port architecture: the Axum
/// server owns the only external port, and llama-server runs on an internal
/// ephemeral port invisible to clients.  Any llama-server endpoint (current
/// or future: /props, /slots, /tokenize, /detokenize, /embedding, etc.)
/// works automatically.  The `/v1/*` and `/api/v1/*` Axum routes take
/// priority; this only fires for paths that don't match those.
async fn backend_proxy_fallback(
    axum::extract::State(state): axum::extract::State<SharedState>,
    req: axum::http::Request<axum::body::Body>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    let backend_port = {
        let s = state.read().await;
        s.process.port()
    };

    let (parts, body) = req.into_parts();
    let method = parts.method.clone();
    let path = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| parts.uri.path().to_string());

    // Short-circuit: port 0 means no model has been loaded yet — skip the
    // network call entirely and return a clean error immediately.
    if backend_port == 0 {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        tracing::debug!(%path, "backend_proxy_fallback: no model loaded (port 0), returning early");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({
                "error": "No model is loaded. Load a model before sending requests.",
                "path": path,
                "hint": "Use the InferenceBridge UI (Models tab) or POST /v1/models/load to load a model."
            })),
        )
            .into_response();
    }

    let url = format!("http://127.0.0.1:{}{}", backend_port, path);

    tracing::debug!(
        %method,
        %path,
        backend_port,
        "backend_proxy_fallback: forwarding unmatched request to llama-server"
    );

    let client = proxy_http_client();

    // Read the request body (if any)
    let body_bytes = match axum::body::to_bytes(body, 128 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "backend_proxy_fallback: failed to read request body");
            return StatusCode::BAD_REQUEST.into_response();
        }
    };

    let mut backend_req = client.request(method.clone(), &url).body(body_bytes);
    for (name, value) in parts.headers.iter() {
        if is_hop_by_hop_header(name) {
            continue;
        }
        if let Ok(header_name) = reqwest::header::HeaderName::from_bytes(name.as_str().as_bytes()) {
            backend_req = backend_req.header(header_name, value.as_bytes());
        }
    }

    match backend_req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let mut builder = axum::http::Response::builder().status(status);
            for (name, value) in headers.iter() {
                if let Ok(header_name) =
                    axum::http::HeaderName::from_bytes(name.as_str().as_bytes())
                {
                    if is_hop_by_hop_header(&header_name) {
                        continue;
                    }
                    if let Ok(header_value) = axum::http::HeaderValue::from_bytes(value.as_bytes())
                    {
                        builder = builder.header(header_name, header_value);
                    }
                }
            }
            builder
                .body(axum::body::Body::from_stream(resp.bytes_stream()))
                .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
        }
        Err(e) => {
            tracing::debug!(
                %path,
                backend_port,
                error = %e,
                "backend_proxy_fallback: backend unreachable (no model loaded?)"
            );
            (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({
                    "error": format!("llama-server backend unreachable on port {backend_port}: {e}"),
                    "path": path,
                    "hint": "No model is loaded or the backend process is not running."
                })),
            )
                .into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------

async fn require_api_key(
    axum::extract::State(state): axum::extract::State<SharedState>,
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let path = req.uri().path();
    if is_api_auth_exempt(req.method(), path) {
        return next.run(req).await;
    }

    let api_key = {
        let s = state.read().await;
        s.config.server.api_key.clone().unwrap_or_default()
    };

    if api_key.is_empty() {
        return next.run(req).await;
    }

    let auth = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if constant_time_eq(auth.strip_prefix("Bearer ").unwrap_or(""), &api_key) {
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

fn is_api_auth_exempt(method: &axum::http::Method, path: &str) -> bool {
    *method == axum::http::Method::OPTIONS || matches!(path, "/v1/health" | "/api/v1/health")
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let max_len = left.len().max(right.len());
    let mut diff = left.len() ^ right.len();
    for index in 0..max_len {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        diff |= usize::from(a ^ b);
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

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
            "/runtime/doctor",
            axum::routing::get(super::extensions::runtime_doctor),
        )
        .route(
            "/debug/profile",
            axum::routing::get(super::extensions::debug_profile),
        )
        .route(
            "/debug/logs",
            axum::routing::get(super::extensions::debug_logs),
        )
        .route(
            "/reliability/agent-action/validate",
            axum::routing::post(super::extensions::validate_agent_action),
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
        .route("/metrics", axum::routing::get(super::metrics::get_metrics))
        .route(
            "/inference/cancel",
            axum::routing::post(super::metrics::cancel_inference),
        )
}

fn native_api_routes() -> Router<SharedState> {
    Router::new()
        .route("/models", axum::routing::get(super::models::list_models))
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

// ---------------------------------------------------------------------------
// State helpers
// ---------------------------------------------------------------------------

async fn update_api_endpoint(state: &SharedState, host: String, port: u16) {
    let mut s = state.write().await;
    s.api_server_host = Some(host);
    s.api_server_port = Some(port);
}

async fn clear_api_endpoint(state: &SharedState, host: &str, port: u16) {
    let mut s = state.write().await;
    let matches_endpoint =
        s.api_server_host.as_deref() == Some(host) && s.api_server_port == Some(port);
    if matches_endpoint {
        s.api_server_host = None;
        s.api_server_port = None;
    }
}

async fn update_api_status(
    state: &SharedState,
    api_server_state: ApiServerState,
    api_server_error: Option<String>,
) {
    use tauri::Emitter;

    let app_handle = {
        let mut s = state.write().await;
        s.api_server_state = api_server_state.clone();
        s.api_server_error = api_server_error.clone();
        s.app_handle.clone()
    };

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

/// After binding, probe our own `/health` endpoint to confirm the server is
/// actually accepting connections.  Logs a warning in state if it can't reach
/// itself after several attempts — useful for catching firewall blocks or
/// silent bind failures.
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

        for probe in 1..=8u32 {
            match client.get(&url).send().await {
                Ok(resp) => {
                    tracing::info!(
                        pid, attempt, source, probe,
                        status = %resp.status(),
                        %url,
                        "API startup self-probe succeeded"
                    );
                    return;
                }
                Err(e) => {
                    tracing::debug!(
                        pid, attempt, source, probe,
                        %url, error = %e,
                        "API startup self-probe attempt failed"
                    );
                    if probe == 8 {
                        tracing::error!(
                            pid, attempt, source, %url,
                            "API startup self-probe failed after all retries — \
                             server may be blocked by firewall or failed silently"
                        );
                        let mut s = state.write().await;
                        if matches!(s.api_server_state, ApiServerState::Running)
                            && s.api_server_error.is_none()
                        {
                            s.api_server_error = Some(format!(
                                "The public API is not currently reachable on {url}. \
                                 No active listener is holding port {port}. \
                                 Retry API to start it again."
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

// ---------------------------------------------------------------------------
// Duplicate-start guard
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::{constant_time_eq, is_api_auth_exempt, port_owner_is_killable};
    use axum::http::Method;

    #[test]
    fn auth_exempts_only_exact_health_and_options() {
        assert!(is_api_auth_exempt(&Method::GET, "/v1/health"));
        assert!(is_api_auth_exempt(&Method::GET, "/api/v1/health"));
        assert!(is_api_auth_exempt(&Method::OPTIONS, "/v1/chat/completions"));

        assert!(!is_api_auth_exempt(&Method::GET, "/v1/models"));
        assert!(!is_api_auth_exempt(&Method::POST, "/api/v1/models/load"));
        assert!(!is_api_auth_exempt(&Method::POST, "/completion"));
        assert!(!is_api_auth_exempt(&Method::GET, "/v1/anything/health"));
    }

    #[test]
    fn constant_time_eq_matches_string_equality() {
        assert!(constant_time_eq("secret", "secret"));
        assert!(!constant_time_eq("secret", "Secret"));
        assert!(!constant_time_eq("secret", "secret-extra"));
        assert!(!constant_time_eq("", "secret"));
    }

    #[test]
    fn port_eviction_only_kills_owned_processes() {
        assert!(port_owner_is_killable(100, Some("llama-server.exe"), 200));
        assert!(port_owner_is_killable(
            100,
            Some("inference-bridge.exe"),
            200
        ));

        assert!(!port_owner_is_killable(100, Some("node.exe"), 200));
        assert!(!port_owner_is_killable(100, None, 200));
        assert!(!port_owner_is_killable(100, Some("llama-server.exe"), 100));
    }
}
