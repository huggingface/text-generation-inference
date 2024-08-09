use crate::v3::{pb, Chunk};
use crate::{ClientError, Result, WARMUP_IMAGE_BASE64};
/// Single shard Client
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use grpc_metadata::InjectTelemetryContext;
use pb::generate::v3::text_generation_service_client::TextGenerationServiceClient;
use pb::generate::v3::*;
use std::cmp::min;
use std::time::Duration;
use tonic::transport::{Channel, Uri};
use tracing::instrument;

/// Text Generation Inference gRPC client
#[derive(Debug, Clone)]
pub struct Client {
    stub: TextGenerationServiceClient<Channel>,
}

impl Client {
    /// Returns a client connected to the given url
    pub async fn connect(uri: Uri) -> Result<Self> {
        let channel = Channel::builder(uri).connect().await?;

        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let channel = Channel::from_shared("http://[::]:50051".to_string())
            .unwrap()
            .connect_with_connector(tower::service_fn(move |_: Uri| {
                tokio::net::UnixStream::connect(path.clone())
            }))
            .await?;

        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    /// Returns a list of uris or unix sockets of all shards
    #[instrument(skip(self))]
    pub async fn service_discovery(&mut self) -> Result<Vec<String>> {
        let request = tonic::Request::new(ServiceDiscoveryRequest {}).inject_context();
        let response = self.stub.service_discovery(request).await.map_err(|_| {
            ClientError::Connection("Server does not support v3 interface".to_string())
        })?;
        let urls = response
            .into_inner()
            .urls
            .into_iter()
            // Remove unix socket prefix
            .map(|url| match url.strip_prefix("unix://") {
                None => url,
                Some(stripped_url) => stripped_url.to_string(),
            })
            .collect();
        Ok(urls)
    }

    /// Get model info
    #[instrument(skip(self))]
    pub async fn info(&mut self) -> Result<InfoResponse> {
        let request = tonic::Request::new(InfoRequest {}).inject_context();
        let response = self.stub.info(request).await?.into_inner();
        Ok(response)
    }

    /// Get model health
    #[instrument(skip(self))]
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let request = tonic::Request::new(HealthRequest {}).inject_context();
        let response = self.stub.health(request).await?.into_inner();
        Ok(response)
    }

    /// Clear the past generations cache
    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self, batch_id: Option<u64>) -> Result<()> {
        let request = tonic::Request::new(ClearCacheRequest { id: batch_id }).inject_context();
        self.stub.clear_cache(request).await?;
        Ok(())
    }

    /// Filter a cached batch
    #[instrument(skip(self))]
    pub async fn filter_batch(
        &mut self,
        batch_id: u64,
        request_ids: Vec<u64>,
    ) -> Result<Option<CachedBatch>> {
        let request = tonic::Request::new(FilterBatchRequest {
            batch_id,
            request_ids,
        })
        .inject_context();
        let filtered_batch = self.stub.filter_batch(request).await?.into_inner();
        Ok(filtered_batch.batch)
    }

    /// Warmup on a max size batch
    ///
    /// Returns the maximum amount of tokens supported by the hardware
    #[instrument(skip_all)]
    pub async fn warmup(
        &mut self,
        max_input_length: u32,
        max_prefill_tokens: u32,
        max_total_tokens: u32,
        max_batch_size: Option<usize>,
    ) -> Result<Option<u32>> {
        let mut n_tokens = 0;
        let mut requests = Vec::new();
        // Create requests
        while n_tokens < max_prefill_tokens {
            let truncate = min(max_input_length, max_prefill_tokens - n_tokens);

            let mut input_chunks = Vec::new();
            input_chunks
                .push(Chunk::Text("_test ".to_string().repeat(max_input_length as usize)).into());
            if n_tokens == 0 {
                input_chunks.push(
                    Chunk::Image(Image {
                        // Safe unwrap, because we control the data.
                        data: STANDARD.decode(WARMUP_IMAGE_BASE64).unwrap(),
                        mimetype: "image/jpeg;base64".to_string(),
                    })
                    .into(),
                );
            }

            // Send stringly-typed inputs for compatibility for backends that haven't
            // been updated to support chunks.

            let mut inputs = String::new();
            inputs.push_str(&"_test ".to_string().repeat(max_input_length as usize));
            if n_tokens == 0 {
                // 1 request is enough to test vision heads.
                // Sending images on other queries messes up easily with truncation.
                inputs.push_str(&format!(
                    "![](data:image/jpeg;base64,{WARMUP_IMAGE_BASE64})",
                ));
            }

            requests.push(Request {
                id: 0,
                inputs,
                input_chunks: Some(Input {
                    chunks: input_chunks,
                }),
                // We truncate the input on the server side to be sure that it has the correct size
                truncate,
                // Blocks and slots will be set on the server side if we use paged attention
                blocks: vec![],
                slots: vec![],
                prefix_len: 0,
                // Set sampling parameters to also take these ops into account in the max memory
                parameters: Some(NextTokenChooserParameters {
                    temperature: 0.9,
                    top_k: 10,
                    top_p: 0.9,
                    typical_p: 0.9,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 1.2,
                    frequency_penalty: 0.1,
                    watermark: true,
                    grammar: String::new(),
                    grammar_type: GrammarType::None as i32,
                }),
                stopping_parameters: Some(StoppingCriteriaParameters {
                    max_new_tokens: max_total_tokens - truncate,
                    stop_sequences: vec![],
                    ignore_eos_token: true,
                }),
                prefill_logprobs: true,
                top_n_tokens: 20,
                adapter_id: None,
            });
            n_tokens += max_input_length;

            // Check max_batch_size
            if Some(requests.len()) == max_batch_size {
                break;
            }
        }

        let batch = Batch {
            id: 0,
            size: requests.len() as u32,
            requests,
            max_tokens: max_input_length,
            max_blocks: 0,
        };

        let request = tonic::Request::new(WarmupRequest {
            batch: Some(batch),
            max_input_length,
            max_prefill_tokens,
            max_total_tokens,
        })
        .inject_context();
        let response = self.stub.warmup(request).await?.into_inner();
        Ok(response.max_supported_total_tokens)
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns Generation for each request in batch
    /// and the next cached batch
    #[instrument(skip_all, fields(id = &batch.id, size = &batch.size))]
    pub async fn prefill(
        &mut self,
        batch: Batch,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>, PrefillTimings)> {
        let request = tonic::Request::new(PrefillRequest { batch: Some(batch) }).inject_context();
        let response = self.stub.prefill(request).await?.into_inner();
        Ok((
            response.generations,
            response.batch,
            PrefillTimings::new(response.forward_ns, response.decode_ns, response.total_ns),
        ))
    }

    /// Generate one token for each request in the given cached batches
    ///
    /// Returns Generation for each request in batches
    /// and the next cached batch
    #[instrument(skip_all, fields(size = batches.iter().map(|batch|{batch.size}).sum::<u32>()))]
    pub async fn decode(
        &mut self,
        batches: Vec<CachedBatch>,
    ) -> Result<(Vec<Generation>, Option<CachedBatch>, DecodeTimings)> {
        let request = tonic::Request::new(DecodeRequest { batches }).inject_context();
        let response = self.stub.decode(request).await?.into_inner();
        Ok((
            response.generations,
            response.batch,
            DecodeTimings::new(
                response.concat_ns,
                response.forward_ns,
                response.decode_ns,
                response.total_ns,
            ),
        ))
    }
}

pub struct PrefillTimings {
    pub forward: Duration,
    pub decode: Duration,
    pub total: Duration,
}

impl PrefillTimings {
    fn new(forward_ns: u64, decode_ns: u64, total_ns: u64) -> Self {
        Self {
            forward: Duration::from_nanos(forward_ns),
            decode: Duration::from_nanos(decode_ns),
            total: Duration::from_nanos(total_ns),
        }
    }
}

pub struct DecodeTimings {
    pub concat: Option<Duration>,
    pub forward: Duration,
    pub decode: Duration,
    pub total: Duration,
}

impl DecodeTimings {
    fn new(concat_ns: Option<u64>, forward_ns: u64, decode_ns: u64, total_ns: u64) -> Self {
        Self {
            concat: concat_ns.map(Duration::from_nanos),
            forward: Duration::from_nanos(forward_ns),
            decode: Duration::from_nanos(decode_ns),
            total: Duration::from_nanos(total_ns),
        }
    }
}
