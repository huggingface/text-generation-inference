mod block_allocator;
mod queue;
mod scheduler;

use crate::infer::schedulers::v3::scheduler::SchedulerV3;
use crate::infer::schedulers::Scheduler;
use std::sync::Arc;
use text_generation_client::v3::ShardedClient;
use text_generation_client::{ClientError, ShardInfo};
use thiserror::Error;

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
) -> Result<(Arc<dyn Scheduler + Send + Sync>, ShardInfo, u32), V3Error> {
    // Helper function
    let check_max_batch_total_tokens = |max_supported_batch_total_tokens: Option<u32>| {
        match max_supported_batch_total_tokens {
            // Older models do not support automatic max-batch-total-tokens
            None => {
                let max_batch_total_tokens = max_batch_total_tokens
                    .unwrap_or(16000.max((max_total_tokens as u32).max(max_batch_prefill_tokens)));
                tracing::warn!("Model does not support automatic max batch total tokens");
                Ok(max_batch_total_tokens)
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
                if max_total_tokens as u32 > max_supported_batch_total_tokens {
                    return Err(V3Error::NotEnoughMemory(max_total_tokens));
                }

                Ok(max_supported_batch_total_tokens)
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
    let max_batch_total_tokens = check_max_batch_total_tokens(
        sharded_client
            .warmup(
                max_input_tokens as u32,
                max_batch_prefill_tokens,
                max_total_tokens as u32,
                max_batch_size,
            )
            .await
            .map_err(V3Error::Warmup)?,
    )?;

    let scheduler = Arc::new(SchedulerV3::new(
        sharded_client,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        max_waiting_tokens,
        max_batch_size,
        shard_info.requires_padding,
        shard_info.window_size,
        shard_info.speculate,
    ));
    tracing::info!("Using scheduler V3");

    Ok((scheduler, shard_info, max_batch_total_tokens))
}

#[derive(Debug, Error)]
pub(crate) enum V3Error {
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
