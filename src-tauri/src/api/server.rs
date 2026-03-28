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

    update_api_status(&state, ApiServerState::Running, None).await;
    spawn_startup_probe(state.clone(), host.to_string(), port, attempt, source);
    tracing::info!(pid, attempt, source, %addr, "API server bound and running");

    let server = axum::serve(listener, app);
    let result = if let Some(shutdown_rx) = shutdown {
        server
            .with_graceful_shutdown(async move { let _ = shutdown_rx.await; })
            .await
    } else {
        server.await
    };

    match result {
        Ok(()) => {
            tracing::info!(pid, attempt, source, %addr, "API server exited cleanly");
            update_api_status(&state, ApiServerState::Idle, None).await;
        }
        Err(e) => {
            let msg = format!("API server stopped unexpectedly: {e}");
            tracing::error!(pid, attempt, source, %addr, error = %e, "API server exited with error");
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
///   5. Attempt `taskkill /PID <n> /F` regardless of whether the name was
///      resolved.  Ghost PIDs (process exited but socket leaked) also get a
///      kill attempt; taskkill is a no-op if the PID is truly gone.
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
            tracing::error!(
                port,
                owner_pid,
                current_pid,
                "evict_port_blocker: port is owned by OUR OWN PROCESS — \
                 double-start bug, ApiServerGuard should have caught this"
            );
            return;
        }

        tracing::warn!(
            port,
            owner_pid,
            owner_name = owner_name.as_deref().unwrap_or("<ghost — tasklist returned nothing>"),
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

        // Poll up to 1.5 s for the socket to actually disappear.
        let deadline = std::time::Instant::now() + Duration::from_millis(1500);
        let mut freed = false;
        while std::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(150)).await;
            if TcpListener::bind(format!("{probe_host}:{port}")).is_ok() {
                freed = true;
                break;
            }
        }

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
            tracing::debug!(port, pid, local_addr = cols[1], "find_listening_pid: found owner");
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
        tracing::debug!(pid, "resolve_pid_name: PID not visible to tasklist (ghost/protected)");
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

/// Bind the public API listener.  Retries a few times with short delays to
/// handle the window between our eviction attempt and the OS releasing the port.
async fn bind_with_retry(host: &str, port: u16) -> std::io::Result<tokio::net::TcpListener> {
    let addr = format!("{host}:{port}");
    const MAX_ATTEMPTS: u32 = 4;
    const RETRY_DELAY_MS: u64 = 300;

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
// Auth middleware
// ---------------------------------------------------------------------------

async fn require_api_key(
    axum::extract::State(state): axum::extract::State<SharedState>,
    req: axum::http::Request<axum::body::Body>,
    next: axum::middleware::Next,
) -> axum::response::Response {
    if req.uri().path().ends_with("/health") || req.method() == axum::http::Method::OPTIONS {
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

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

fn api_routes() -> Router<SharedState> {
    Router::new()
        .route("/responses", axum::routing::post(super::responses::responses))
        .route("/chat/completions", axum::routing::post(super::completions::chat_completions))
        .route("/completions", axum::routing::post(super::completions::text_completions))
        .route("/models", axum::routing::get(super::models::list_models))
        .route("/models/load", axum::routing::post(super::models::load_model))
        .route("/models/unload", axum::routing::post(super::models::unload_model))
        .route(
            "/models/stats",
            axum::routing::get(super::models::current_model_stats)
                .post(super::models::model_stats),
        )
        .route("/models/:model", axum::routing::get(super::models::get_model))
        .route("/context/status", axum::routing::get(super::extensions::context_status))
        .route("/runtime/status", axum::routing::get(super::extensions::runtime_status))
        .route("/debug/profile", axum::routing::get(super::extensions::debug_profile))
        .route(
            "/sessions",
            axum::routing::get(super::extensions::list_sessions)
                .post(super::extensions::create_session),
        )
        .route("/sessions/:id", axum::routing::delete(super::extensions::delete_session))
        .route("/sessions/:id/messages", axum::routing::get(super::extensions::get_session_messages))
        .route("/health", axum::routing::get(super::health::health_check))
        .route("/metrics", axum::routing::get(super::metrics::get_metrics))
        .route("/inference/cancel", axum::routing::post(super::metrics::cancel_inference))
}

fn native_api_routes() -> Router<SharedState> {
    Router::new()
        .route("/models", axum::routing::get(super::models::list_models))
        .route("/models/load", axum::routing::post(super::models::load_model))
        .route("/models/unload", axum::routing::post(super::models::unload_model))
        .route("/health", axum::routing::get(super::health::health_check))
}

// ---------------------------------------------------------------------------
// State helpers
// ---------------------------------------------------------------------------

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
