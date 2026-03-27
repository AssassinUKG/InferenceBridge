use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Clone, serde::Serialize)]
pub struct SchedulerSnapshot {
    pub limit: u32,
    pub active: usize,
    pub queued: usize,
}

pub struct RequestScheduler {
    limit: u32,
    semaphore: Arc<Semaphore>,
    active: AtomicUsize,
    queued: AtomicUsize,
}

pub struct RequestPermit {
    scheduler: Arc<RequestScheduler>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        self.scheduler.active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl RequestScheduler {
    pub fn new(limit: u32) -> Self {
        let limit = limit.max(1);
        Self {
            limit,
            semaphore: Arc::new(Semaphore::new(limit as usize)),
            active: AtomicUsize::new(0),
            queued: AtomicUsize::new(0),
        }
    }

    pub async fn acquire(self: &Arc<Self>) -> RequestPermit {
        self.queued.fetch_add(1, Ordering::SeqCst);
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("request scheduler semaphore closed unexpectedly");
        self.queued.fetch_sub(1, Ordering::SeqCst);
        self.active.fetch_add(1, Ordering::SeqCst);

        RequestPermit {
            scheduler: self.clone(),
            _permit: permit,
        }
    }

    pub fn snapshot(&self) -> SchedulerSnapshot {
        SchedulerSnapshot {
            limit: self.limit,
            active: self.active.load(Ordering::SeqCst),
            queued: self.queued.load(Ordering::SeqCst),
        }
    }
}
