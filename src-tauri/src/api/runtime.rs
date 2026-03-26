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
        if source == "gui" {
            // Give a previous instance a brief moment to finish releasing the socket
            // during app restarts before we try to bind the public API again.
            tokio::time::sleep(Duration::from_millis(900)).await;
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
