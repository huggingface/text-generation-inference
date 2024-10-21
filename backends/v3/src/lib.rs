mod backend;
pub mod block_allocator;
mod client;
mod queue;
pub mod radix;

use crate::client::{ClientError, ShardedClient};
pub(crate) use backend::BackendV3;
use serde::Serialize;
use thiserror::Error;
use utoipa::ToSchema;

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct BackendInfo {
    /// Mandatory
    #[schema(example = "cuda")]
    pub model_device_type: String,
    #[schema(example = "torch.float16")]
    pub model_dtype: String,

    /// Backend parameters
    #[schema(example = "1")]
    pub speculate: usize,
    #[schema(example = "1.2")]
    pub waiting_served_ratio: f32,
    #[schema(example = "32000")]
    pub max_batch_total_tokens: u32,
    #[schema(example = "20")]
    pub max_waiting_tokens: usize,
    #[schema(nullable = true, example = "null")]
    pub max_batch_size: Option<usize>,
    #[schema(example = "false")]
    pub support_chunking: bool,
    #[schema(example = "false")]
    pub prefix_caching: bool,
    #[schema(example = "flashinfer")]
    pub attention_impl: String,
    #[schema(example = "1")]
    pub block_size: u32,

    #[schema(example = "30000")]
    pub max_input_tokens: usize,
    #[schema(example = "32000")]
    pub max_total_tokens: usize,
}

#[allow(clippy::too_many_arguments)]
pub async fn connect_backend(
    max_input_tokens: Option<usize>,
    max_total_tokens: Option<usize>,
    master_shard_uds_path: String,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: Option<u32>,
    max_waiting_tokens: usize,
    max_batch_size: Option<usize>,
) -> Result<(BackendV3, BackendInfo), V3Error> {
    // Helper function
    let check_max_batch_total_tokens = |(
        max_supported_batch_total_tokens,
        shard_max_input_tokens,
        shard_max_total_tokens,
    ): (Option<u32>, u32, u32)|
     -> Result<(u32, usize, usize), V3Error> {
        if let Some(max_input_tokens) = max_input_tokens {
            assert_eq!(max_input_tokens as u32, shard_max_input_tokens);
        }
        if let Some(max_total_tokens) = max_total_tokens {
            assert_eq!(max_total_tokens as u32, shard_max_total_tokens);
        }
        match max_supported_batch_total_tokens {
            // Older models do not support automatic max-batch-total-tokens
            None => {
                let max_batch_total_tokens = max_batch_total_tokens.unwrap_or(
                    16000
                        .max(shard_max_total_tokens)
                        .max(max_batch_prefill_tokens),
                );
                tracing::warn!("Model does not support automatic max batch total tokens");
                Ok((
                    max_batch_total_tokens,
                    shard_max_input_tokens as usize,
                    shard_max_total_tokens as usize,
                ))
            }
            // Flash attention models return their max supported total tokens
            Some(max_supported_batch_total_tokens) => {
                // Warn if user added his own max-batch-total-tokens as we will ignore it
                if max_batch_total_tokens.is_some() {
                    tracing::warn!(
                        "`--max-batch-total-tokens` is deprecated for Flash \
                        Attention models."
                    );
                    tracing::warn!(
                        "Inferred max batch total tokens: {max_supported_batch_total_tokens}"
                    );
                }
                if shard_max_total_tokens > max_supported_batch_total_tokens {
                    return Err(V3Error::NotEnoughMemory(shard_max_total_tokens as usize));
                }

                Ok((
                    max_supported_batch_total_tokens,
                    shard_max_input_tokens as usize,
                    shard_max_total_tokens as usize,
                ))
            }
        }
    };

    let mut sharded_client = ShardedClient::connect_uds(master_shard_uds_path)
        .await
        .map_err(V3Error::Connection)?;

    // server is running on v3
    // Clear the cache; useful if the webserver rebooted
    sharded_client
        .clear_cache(None)
        .await
        .map_err(V3Error::Cache)?;
    // Get info from the shard
    let shard_info = sharded_client.info().await.map_err(V3Error::Info)?;

    // Warmup model
    tracing::info!("Warming up model");
    let answer = sharded_client
        .warmup(
            max_input_tokens.map(|p| p as u32),
            max_batch_prefill_tokens,
            max_total_tokens.map(|p| p as u32),
            max_batch_size,
        )
        .await
        .map_err(V3Error::Warmup)?;
    let (max_batch_total_tokens, max_input_tokens, max_total_tokens) =
        check_max_batch_total_tokens(answer)?;
    tracing::info!("Setting max batch total tokens to {max_batch_total_tokens}");
    metrics::gauge!("tgi_batch_max_total_tokens").set(max_batch_total_tokens);

    let backend_info = BackendInfo {
        waiting_served_ratio,
        max_batch_total_tokens,
        max_input_tokens,
        max_total_tokens,
        max_waiting_tokens,
        max_batch_size,
        model_device_type: shard_info.device_type.clone(),
        model_dtype: shard_info.dtype.clone(),
        speculate: shard_info.speculate as usize,
        support_chunking: shard_info.support_chunking,
        prefix_caching: shard_info.use_prefix_caching,
        attention_impl: shard_info.attention_impl.clone(),
        block_size: shard_info.block_size,
    };

    let backend = BackendV3::new(
        sharded_client,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        max_waiting_tokens,
        max_batch_size,
        shard_info,
    );

    tracing::info!("Using backend V3");

    Ok((backend, backend_info))
}

#[derive(Debug, Error)]
pub enum V3Error {
    #[error("Unable to clear the Python model shards cache: {0}")]
    Cache(ClientError),
    #[error("Unable to connect to the Python model shards: {0}")]
    Connection(ClientError),
    #[error("Unable to get the Python model shards info: {0}")]
    Info(ClientError),
    #[error("Unable to warmup the Python model shards: {0}")]
    Warmup(ClientError),
    #[error("Not enough memory to handle `max_total_tokens={0}`")]
    NotEnoughMemory(usize),
}
