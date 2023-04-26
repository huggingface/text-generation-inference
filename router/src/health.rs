use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use text_generation_client::{
    Batch, NextTokenChooserParameters, Request, ShardedClient, StoppingCriteriaParameters,
};

#[derive(Clone, Debug)]
pub(crate) struct Health {
    client: ShardedClient,
    generation_health: Arc<AtomicBool>,
}

impl Health {
    pub(crate) fn new(client: ShardedClient, generation_health: Arc<AtomicBool>) -> Self {
        Self {
            client,
            generation_health,
        }
    }

    pub(crate) async fn check(&mut self) -> bool {
        if self.generation_health.load(Ordering::SeqCst) {
            // Generation is healthy, we only check that the shards are answering gRPC calls
            self.client.health().await.is_ok()
        } else {
            // Generation is unhealthy or have not sent any generation request yet

            // Dummy batch of 1 token and 1 generated token
            let liveness_request = Request {
                id: u64::MAX,
                inputs: "liveness".to_string(),
                truncate: 10,
                parameters: Some(NextTokenChooserParameters {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 1.0,
                    typical_p: 1.0,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 1.0,
                    watermark: false,
                }),
                stopping_parameters: Some(StoppingCriteriaParameters {
                    max_new_tokens: 1,
                    stop_sequences: vec![],
                    ignore_eos_token: false,
                }),
            };
            let batch = Batch {
                id: u64::MAX,
                requests: vec![liveness_request],
                size: 1,
                max_tokens: 2,
            };
            // Skips the queue
            let value = self.client.prefill(batch).await.is_ok();
            // Update generation health
            self.generation_health.store(value, Ordering::SeqCst);
            value
        }
    }
}
