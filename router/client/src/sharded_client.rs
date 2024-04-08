/// Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

/// Multi shard Client
use crate::{Batch, CachedBatch, Client, Generation, HealthResponse, ShardInfo};
use crate::{ClientError, Result};
use futures::future::join_all;
use tonic::transport::Uri;
use tracing::instrument;

#[derive(Debug, Clone)]
/// Text Generation Inference gRPC multi client
pub struct ShardedClient {
    clients: Vec<Client>,
}

impl ShardedClient {
    fn new(clients: Vec<Client>) -> Self {
        Self { clients }
    }

    /// Create a new ShardedClient from a master client. The master client will communicate with
    /// the other shards and returns all uris/unix sockets with the `service_discovery` gRPC method.
    async fn from_master_client(mut master_client: Client) -> Result<Self> {
        // Get all uris/unix sockets from the master client
        let uris = master_client.service_discovery().await?;
        let futures = uris.into_iter().map(Client::connect_uds);
        let clients: Result<Vec<Client>> = join_all(futures).await.into_iter().collect();
        Ok(Self::new(clients?))
    }

    /// Returns a client connected to the given uri
    pub async fn connect(uri: Uri) -> Result<Self> {
        let master_client = Client::connect(uri).await?;
        Self::from_master_client(master_client).await
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let master_client = Client::connect_uds(path).await?;
        Self::from_master_client(master_client).await
    }

    /// Get the model info
    #[instrument(skip(self))]
    pub async fn info(&mut self) -> Result<ShardInfo> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.info())
            .collect();
        join_all(futures).await.pop().unwrap()
    }

    /// GRPC health check
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.health())
            .collect();
        join_all(futures).await.pop().unwrap()
    }

    /// Clear the past generations cache
    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self, batch_id: Option<u64>) -> Result<()> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.clear_cache(batch_id))
            .collect();
        join_all(futures).await.into_iter().collect()
    }

    /// Filter a cached batch
    #[instrument(skip(self))]
    pub async fn filter_batch(
        &mut self,
        batch_id: u64,
        request_ids: Vec<u64>,
    ) -> Result<Option<CachedBatch>> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.filter_batch(batch_id, request_ids.clone())))
            .collect();
        // all shards return the same message
        join_all(futures).await.pop().unwrap()
    }

    /// Warmup on a max size batch
    ///
    /// Returns the maximum amount of tokens supported by the hardware
    #[instrument(skip(self))]
    pub async fn warmup(
        &mut self,
        max_input_length: u32,
        max_prefill_tokens: u32,
        max_total_tokens: u32,
        max_batch_total_tokens: Option<u32>,
    ) -> Result<Option<u32>> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| {
                Box::pin(client.warmup(max_input_length, max_prefill_tokens, max_total_tokens, max_batch_total_tokens))
            })
            .collect();
        // Take the minimum value
        let results = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<Option<u32>>>>()?;
        Ok(results.into_iter().flatten().min())
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns Generation for each request in batch
    /// and the next cached batch
    #[instrument(skip_all, fields(id = &batch.id, size = &batch.size))]
    pub async fn prefill(
        &mut self,
        batch: Batch,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>)> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.prefill(batch.clone())))
            .collect();
        let results: Result<Vec<(Vec<Generation>, Option<CachedBatch>)>> =
            join_all(futures).await.into_iter().collect();
        merge_generations(results?)
    }

    /// Generate one token for each request in the given cached batches
    ///
    /// Returns Generation for each request in batches
    /// and the next cached batch
    #[instrument(skip_all, fields(size = batches.iter().map(|batch|{batch.size}).sum::<u32>()))]
    pub async fn decode(
        &mut self,
        batches: Vec<CachedBatch>,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>)> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.decode(batches.clone())))
            .collect();
        let results: Result<Vec<(Vec<Generation>, Option<CachedBatch>)>> =
            join_all(futures).await.into_iter().collect();
        merge_generations(results?)
    }
}

/// Merge generations from the different model shards
fn merge_generations(
    mut results: Vec<(Vec<Generation>, Option<CachedBatch>)>,
) -> Result<(Vec<Generation>, Option<CachedBatch>)> {
    let (mut generations, next_batch) = results.pop().ok_or(ClientError::EmptyResults)?;

    for (mut shard_generations, _) in results.into_iter() {
        generations.append(&mut shard_generations);
    }
    Ok((generations, next_batch))
}
