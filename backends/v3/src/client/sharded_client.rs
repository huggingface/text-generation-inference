use crate::client::Health;
/// Multi shard Client
use crate::client::{ClientError, Result};

use crate::client::grpc_client::{DecodeTimings, PrefillTimings};
use crate::client::{
    Batch, CachedBatch, Client, Generation, GrammarType, HealthResponse,
    NextTokenChooserParameters, Request, StoppingCriteriaParameters,
};
use crate::client::{Chunk, InfoResponse, Input};
use async_trait::async_trait;
use futures::future::join_all;
use tonic::transport::Uri;
use tracing::instrument;

#[derive(Debug, Clone, Copy)]
pub enum Attn {
    Flashdecoding,
    Flashinfer,
    Paged,
}

impl TryFrom<&str> for Attn {
    type Error = ClientError;
    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value {
            "flashdecoding" => Ok(Attn::Flashdecoding),
            "flashinfer" => Ok(Attn::Flashinfer),
            "paged" => Ok(Attn::Paged),
            string => Err(ClientError::InvalidAttention(string.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
/// Text Generation Inference gRPC multi client
pub struct ShardedClient {
    clients: Vec<Client>,
    attention_impl: Option<Attn>,
}

impl ShardedClient {
    fn new(clients: Vec<Client>) -> Self {
        Self {
            clients,
            attention_impl: None,
        }
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
    #[allow(dead_code)]
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
    pub async fn info(&mut self) -> Result<InfoResponse> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.info())
            .collect();
        let info = join_all(futures).await.pop().unwrap()?;
        self.attention_impl = Some((&*info.attention_impl).try_into()?);
        Ok(info)
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
        max_input_length: Option<u32>,
        max_prefill_tokens: u32,
        max_total_tokens: Option<u32>,
        max_batch_size: Option<usize>,
    ) -> Result<(Option<u32>, u32, u32)> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| {
                Box::pin(client.warmup(
                    max_input_length,
                    max_prefill_tokens,
                    max_total_tokens,
                    max_batch_size,
                ))
            })
            .collect();
        let results = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<(Option<u32>, u32, u32)>>>()?;

        // Take the minimum value
        // Different shards hold different parts of vocab, might yield
        // different available block size.
        let min = results
            .iter()
            .min()
            .expect("Expect at least 1 warmup result");
        Ok(*min)
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns Generation for each request in batch
    /// and the next cached batch
    #[instrument(skip_all, fields(id = & batch.id, size = & batch.size))]
    pub async fn prefill(
        &mut self,
        batch: Batch,
        cached_batch: Option<CachedBatch>,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>, PrefillTimings)> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.prefill(batch.clone(), cached_batch.clone())))
            .collect();
        #[allow(clippy::type_complexity)]
        let results: Result<Vec<(Vec<Generation>, Option<CachedBatch>, PrefillTimings)>> =
            join_all(futures).await.into_iter().collect();
        let mut results = results?;

        let (mut generations, next_batch, mut timings) =
            results.pop().ok_or(ClientError::EmptyResults)?;

        // Merge generations from different model shards
        for (mut shard_generations, _, shard_timings) in results.into_iter() {
            generations.append(&mut shard_generations);
            // Return the timings of the slowest shard
            if shard_timings.total > timings.total {
                timings = shard_timings;
            }
        }
        Ok((generations, next_batch, timings))
    }

    /// Generate one token for each request in the given cached batches
    ///
    /// Returns Generation for each request in batches
    /// and the next cached batch
    #[instrument(skip_all, fields(size = batches.iter().map(| batch | {batch.size}).sum::< u32 > ()))]
    pub async fn decode(
        &mut self,
        batches: Vec<CachedBatch>,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>, DecodeTimings)> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.decode(batches.clone())))
            .collect();
        #[allow(clippy::type_complexity)]
        let results: Result<Vec<(Vec<Generation>, Option<CachedBatch>, DecodeTimings)>> =
            join_all(futures).await.into_iter().collect();
        let mut results = results?;

        let (mut generations, next_batch, mut timings) =
            results.pop().ok_or(ClientError::EmptyResults)?;

        // Merge generations from different model shards
        for (mut shard_generations, _, shard_timings) in results.into_iter() {
            generations.append(&mut shard_generations);
            // Return the timings of the slowest shard
            if shard_timings.total > timings.total {
                timings = shard_timings;
            }
        }
        Ok((generations, next_batch, timings))
    }
}

#[async_trait]
impl Health for ShardedClient {
    async fn device_health(&self) -> Result<()> {
        self.clone().health().await?;
        Ok(())
    }

    async fn model_health(&self) -> Result<()> {
        // Dummy batch of 1 token and 1 generated token
        //
        let (blocks, slots) = match self.attention_impl.expect("Attention to be set") {
            Attn::Paged => (vec![0], (0..2).collect()),
            Attn::Flashinfer => (vec![0, 1], (0..2).collect()),
            Attn::Flashdecoding => (vec![0], (0..2).collect()),
        };
        let liveness_request = Request {
            id: u64::MAX,
            inputs: "liveness".to_string(),
            input_chunks: Some(Input {
                chunks: vec![Chunk::Text("liveness".into()).into()],
            }),
            truncate: 10,
            add_special_tokens: true,
            prefill_logprobs: false,
            parameters: Some(NextTokenChooserParameters {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                typical_p: 1.0,
                do_sample: false,
                seed: 0,
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                watermark: false,
                grammar: String::new(),
                grammar_type: GrammarType::None as i32,
            }),
            stopping_parameters: Some(StoppingCriteriaParameters {
                max_new_tokens: 1,
                stop_sequences: vec![],
                ignore_eos_token: false,
            }),
            top_n_tokens: 0,
            blocks,
            slots,
            cache_len: 0,
            adapter_id: None,
            chunk_len: None,
        };
        let batch = Batch {
            id: u64::MAX,
            requests: vec![liveness_request],
            size: 1,
            max_tokens: 2,
            max_blocks: 1,
        };
        self.clone().prefill(batch, None).await?;
        Ok(())
    }
}
