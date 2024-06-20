mod v3;

use crate::infer::InferStreamResponse;
use crate::validation::ValidGenerateRequest;
use async_trait::async_trait;
use serde::Serialize;
use std::fmt::Debug;
use thiserror::Error;
use tokio_stream::wrappers::UnboundedReceiverStream;
use utoipa::ToSchema;

#[async_trait]
pub(crate) trait Backend {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, BackendError>>, BackendError>;

    async fn health(&self, current_health: bool) -> bool;
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub(crate) struct BackendInfo {
    /// Mandatory
    #[schema(example = "cuda")]
    pub model_device_type: String,
    #[schema(example = "torch.float16")]
    pub model_dtype: String,
    #[schema(example = "1")]
    pub speculate: usize,

    /// Backend parameters
    #[schema(example = "1.2")]
    pub waiting_served_ratio: f32,
    #[schema(example = "32000")]
    pub max_batch_total_tokens: u32,
    #[schema(example = "20")]
    pub max_waiting_tokens: usize,
    #[schema(nullable = true, example = "null")]
    pub max_batch_size: Option<usize>,
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
) -> Result<(impl Backend, BackendInfo), BackendError> {
    let (backend, info) = v3::connect_backend(
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
    .map_err(|err| BackendError::Startup(Box::new(err)))?;

    Ok((backend, info))
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("Startup error: {0}")]
    Startup(Box<dyn std::error::Error + Send + Sync>),
    #[error("Request failed during generation: {0}")]
    Generation(Box<dyn std::error::Error + Send + Sync>),
}
