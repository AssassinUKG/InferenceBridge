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

    if let Some(existing) = guard.as_mut() {
        if !existing.handle.inner().is_finished() {
            let stale = state
                .try_read()
                .map(|app_state| {
                    matches!(
                        app_state.api_server_state,
                        ApiServerState::Idle | ApiServerState::Starting | ApiServerState::Error
                    )
                })
                .unwrap_or(false);

            if stale {
                tracing::warn!(
                    host = %host,
                    port,
                    source,
                    "Aborting stale managed API server task before restart"
                );
                if let Some(stop_tx) = existing.stop_tx.take() {
                    let _ = stop_tx.send(());
                }
                existing.handle.abort();
            } else {
                tracing::info!(host = %host, port, source, "Managed API server already running");
                return false;
            }
        }
    }

    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let handle = tauri::async_runtime::spawn(async move {
        if source == "gui" {
            // Brief pause only if the port is still held from a previous instance.
            // Skip entirely when the port is already free (the common case).
            let addr = format!("{host}:{port}");
            if tokio::net::TcpListener::bind(&addr).await.is_err() {
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
        }
        if let Err(error) =
            crate::api::server::start_api_server_with_shutdown(state, &host, port, source, stop_rx)
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
    let endpoint = {
        let app_state = state.read().await;
        (
            app_state
                .api_server_host
                .clone()
                .unwrap_or_else(|| app_state.config.server.host.clone()),
            app_state
                .api_server_port
                .unwrap_or(app_state.config.server.port),
        )
    };
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

    match tokio::time::timeout(Duration::from_secs(5), &mut managed.handle).await {
        Ok(join_result) => {
            if let Err(error) = join_result {
                tracing::warn!(error = %error, "Managed API server task ended with join error");
            }
        }
        Err(_) => {
            tracing::warn!("Managed API server did not stop after shutdown signal; aborting task");
            managed.handle.abort();
            let _ = managed.handle.await;
        }
    }

    wait_for_api_port_release(&endpoint.0, endpoint.1).await;

    let mut app_state = state.write().await;
    app_state.api_server_state = ApiServerState::Idle;
    app_state.api_server_error = None;
    true
}

async fn wait_for_api_port_release(host: &str, port: u16) {
    let probe_host = crate::api::server::reachable_probe_host(host);
    let addr = format!("{probe_host}:{port}");
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if std::net::TcpListener::bind(&addr).is_ok() {
            tracing::debug!(%addr, "Managed API port released after stop");
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    tracing::warn!(
        %addr,
        "Managed API port was still busy after waiting for shutdown; restart may need bind retry"
    );
}
