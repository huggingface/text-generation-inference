use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use text_generation_client::Health;

#[derive(Clone)]
pub(crate) struct HealthCheck {
    client: Arc<dyn Health + Send + Sync>,
    generation_health: Arc<AtomicBool>,
}

impl HealthCheck {
    pub(crate) fn new(
        client: Arc<dyn Health + Send + Sync>,
        generation_health: Arc<AtomicBool>,
    ) -> Self {
        Self {
            client,
            generation_health,
        }
    }

    pub(crate) async fn check(&mut self) -> bool {
        let value = if self.generation_health.load(Ordering::SeqCst) {
            // Generation is healthy, we only check that the shards can allocate on device
            self.client.device_health().await
        } else {
            self.client.model_health().await
        }
        .is_ok();
        // Update generation health
        self.generation_health.store(value, Ordering::SeqCst);
        value
    }
}
