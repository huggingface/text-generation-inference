mod v3;

use crate::infer::InferStreamResponse;
use crate::validation::ValidGenerateRequest;
use async_trait::async_trait;
use std::sync::Arc;
use text_generation_client::ShardInfo;
use thiserror::Error;
use tokio_stream::wrappers::UnboundedReceiverStream;

#[async_trait]
pub(crate) trait Scheduler {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, SchedulerError>>, SchedulerError>;

    async fn health(&self, current_health: bool) -> bool;
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn connect_backend(
    master_shard_uds_path: String,
    max_input_tokens: usize,
    max_total_tokens: usize,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: Option<u32>,
    max_waiting_tokens: usize,
    max_batch_size: Option<usize>,
) -> Result<(Arc<dyn Scheduler + Send + Sync>, ShardInfo, u32), SchedulerError> {
    v3::connect_backend(
        master_shard_uds_path,
        max_input_tokens,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        max_waiting_tokens,
        max_batch_size,
    )
    .await
    .map_err(|err| SchedulerError::Startup(Box::new(err)))
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("Startup error: {0}")]
    Startup(Box<dyn std::error::Error + Send + Sync>),
    #[error("Request failed during generation: {0}")]
    Generation(Box<dyn std::error::Error + Send + Sync>),
    #[error("Backend error: {0}")]
    Backend(Box<dyn std::error::Error + Send + Sync>),
}
