use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use tauri::async_runtime::JoinHandle;
use tokio::sync::oneshot;

use crate::state::{ApiServerState, SharedState};

struct ManagedApiServer {
    stop_tx: Option<oneshot::Sender<()>>,
    handle: JoinHandle<()>,
}

static MANAGED_API_SERVER: OnceLock<Mutex<Option<ManagedApiServer>>> = OnceLock::new();

fn registry() -> &'static Mutex<Option<ManagedApiServer>> {
    MANAGED_API_SERVER.get_or_init(|| Mutex::new(None))
}

pub fn start_managed(state: SharedState, host: String, port: u16, source: &'static str) -> bool {
    let registry = registry();
    let mut guard = match registry.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    if let Some(existing) = guard.as_ref() {
        if !existing.handle.inner().is_finished() {
            tracing::info!(host = %host, port, source, "Managed API server already running");
            return false;
        }
    }

    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let handle = tauri::async_runtime::spawn(async move {
        let (effective_host, effective_port) = if should_try_startup_port_fallback(source, &host, port)
        {
            match reserve_startup_api_port(state.clone(), &host, port).await {
                Ok((fallback_host, fallback_port)) => (fallback_host, fallback_port),
                Err(error) => {
                    tracing::warn!(
                        source,
                        host = %host,
                        port,
                        error = %error,
                        "Could not preflight startup API port; continuing with configured endpoint"
                    );
                    (host.clone(), port)
                }
            }
        } else {
            (host.clone(), port)
        };

        if source == "gui" {
            // Brief pause only if the port is still held from a previous instance.
            // Skip entirely when the port is already free (the common case).
            let addr = format!("{effective_host}:{effective_port}");
            if tokio::net::TcpListener::bind(&addr).await.is_err() {
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
        }
        if let Err(error) =
            crate::api::server::start_api_server_with_shutdown(
                state,
                &effective_host,
                effective_port,
                source,
                stop_rx,
            )
                .await
        {
            tracing::error!(error = %error, source, "Managed API server task exited with error");
        }
    });

    *guard = Some(ManagedApiServer {
        stop_tx: Some(stop_tx),
        handle,
    });
    true
}

pub async fn stop_managed(state: SharedState) -> bool {
    let managed = {
        let registry = registry();
        let mut guard = match registry.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.take()
    };

    let Some(mut managed) = managed else {
        let mut app_state = state.write().await;
        app_state.api_server_state = ApiServerState::Idle;
        app_state.api_server_error = None;
        return false;
    };

    if let Some(stop_tx) = managed.stop_tx.take() {
        let _ = stop_tx.send(());
    }

    tokio::time::timeout(Duration::from_secs(2), &mut managed.handle)
        .await
        .ok();

    let mut app_state = state.write().await;
    app_state.api_server_state = ApiServerState::Idle;
    app_state.api_server_error = None;
    true
}

fn should_try_startup_port_fallback(source: &'static str, host: &str, port: u16) -> bool {
    source == "gui" && host == "127.0.0.1" && port == 8800
}

async fn reserve_startup_api_port(
    state: SharedState,
    host: &str,
    port: u16,
) -> Result<(String, u16), String> {
    if tokio::net::TcpListener::bind(format!("{host}:{port}")).await.is_ok() {
        return Ok((host.to_string(), port));
    }

    for candidate in 8802..=8810 {
        if tokio::net::TcpListener::bind(format!("{host}:{candidate}"))
            .await
            .is_ok()
        {
            {
                let mut app_state = state.write().await;
                app_state.config.server.port = candidate;
                app_state
                    .config
                    .save()
                    .map_err(|e| format!("Failed to persist fallback API port {candidate}: {e}"))?;
            }

            tracing::warn!(
                requested_port = port,
                fallback_port = candidate,
                host = %host,
                "Default public API port was unavailable on startup; switched to fallback port"
            );

            return Ok((host.to_string(), candidate));
        }
    }

    Err(format!(
        "No fallback public API port was free in the startup range 8802-8810 for {host}:{port}"
    ))
}
