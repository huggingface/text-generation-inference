/// Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

/// Single shard Client
use crate::pb::generate::v2::text_generation_service_client::TextGenerationServiceClient;
use crate::pb::generate::v2::*;
use crate::Result;
use std::env;
use rand::{distributions::Uniform, Rng};
use grpc_metadata::InjectTelemetryContext;
use std::cmp;
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
        let response = self.stub.service_discovery(request).await?;
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
        let warmup_enabled: bool = env::var("WARMUP_ENABLED").ok().map_or(true, |value| value.to_lowercase() == "true");
        if !warmup_enabled {
            return Ok(None);
        }

        let read_env_var = |key: &str, default: u32| -> u32 {
            env::var(key).ok().map_or(default, |value| value.parse::<u32>().unwrap())
        };

        // get all possible batch sizes
        let decode_bucket_size: u32 = read_env_var("BATCH_BUCKET_SIZE", 8);
        let max_decode_batch_size: u32 = match max_batch_size {
            Some(max_batch_size) => max_batch_size as u32,
            None => decode_bucket_size
        };
        let decode_batch_sizes: Vec<u32> = (decode_bucket_size..max_decode_batch_size+1).step_by(decode_bucket_size as usize).collect();

        let prefill_bucket_size: u32 = read_env_var("PREFILL_BATCH_BUCKET_SIZE", 4);
        let mut max_prefill_batch_size: u32 = max_prefill_tokens / max_input_length;
        max_prefill_batch_size = cmp::min(max_prefill_batch_size, max_decode_batch_size);
        let prefill_batch_sizes: Vec<u32> = (prefill_bucket_size..max_prefill_batch_size+1).step_by(prefill_bucket_size as usize).collect();

        // get all possible sequence lengths for prefill
        let seq_bucket_size: u32 = read_env_var("PAD_SEQUENCE_TO_MULTIPLE_OF", 128);
        let mut seq_lengths: Vec<u32> = (seq_bucket_size..max_input_length+1).step_by(seq_bucket_size as usize).collect();
        if let Some(&last) = seq_lengths.last() {
            if last < max_input_length {
                seq_lengths.push(max_input_length);
            }
        }

        // execute batch for each combination of batch size and sequence length
        let mut shapes: Vec<(u32, u32)> = Vec::with_capacity(prefill_batch_sizes.len() * seq_lengths.len());
        for batch_size in &prefill_batch_sizes {
            for seq_length in &seq_lengths {
                shapes.push((*batch_size, *seq_length));
            }
        }

        let mut id_counter: u64 = 0;
        for shape in shapes.iter() {
            let (batch_size, seq_length) = shape;
            let mut batches: Vec<Batch> = vec![
                self.create_warmup_batch(
                    *shape,
                    &mut id_counter,
                    max_input_length,
                    max_total_tokens,
                    seq_bucket_size,
                    false,
                    None,
                )
            ];
            // if possible, create second batch in order to trigger concatenate operation
            if *batch_size < max_decode_batch_size {
                batches.push(
                    self.create_warmup_batch(
                        (1, *seq_length),
                        &mut id_counter,
                        max_input_length,
                        max_total_tokens,
                        seq_bucket_size,
                        false,
                        None,
                    )
                );
            }

            let request = tonic::Request::new(WarmupRequest {
                batches,
                max_input_length,
                max_prefill_tokens,
                max_total_tokens,
            }).inject_context();
            let _response = self.stub.warmup(request).await?.into_inner();
        }

        // send batches to warmup all possible decode shapes
        if decode_batch_sizes.len() > 1 {
            let steps_per_bucket: u32 = if decode_bucket_size <= max_prefill_batch_size {
                decode_bucket_size
            } else {
                decode_bucket_size.div_ceil(max_prefill_batch_size)
            };
            let max_new_tokens: u32 = 2 * decode_batch_sizes.len() as u32 * steps_per_bucket;

            let mut requests_send: u32 = cmp::min(max_prefill_batch_size, decode_bucket_size);
            let mut batches: Vec<Batch> = vec![
                self.create_warmup_batch(
                    (requests_send, seq_bucket_size),
                    &mut id_counter,
                    max_input_length,
                    max_total_tokens,
                    seq_bucket_size,
                    false,
                    Some(max_new_tokens),
                )
            ];

            let get_current_decode_batch_size = |num: u32| -> u32 {
                decode_batch_sizes.iter()
                    .filter(|&&x| x >= num)
                    .min()
                    .copied()
                    .unwrap()
            };

            let mut current_decode_batch_size: u32 = get_current_decode_batch_size(requests_send);
            while current_decode_batch_size < max_decode_batch_size {
                let distance_to_next_bucket = current_decode_batch_size + decode_bucket_size - requests_send;
                let num_requests: u32 = cmp::min(distance_to_next_bucket, max_prefill_batch_size);
                batches.push(
                    self.create_warmup_batch(
                        (num_requests, seq_bucket_size),
                        &mut id_counter,
                        max_input_length,
                        max_total_tokens,
                        seq_bucket_size,
                        false,
                        Some(max_new_tokens),
                    )
                );

                requests_send += num_requests;
                current_decode_batch_size = get_current_decode_batch_size(requests_send);
            }

            let request = tonic::Request::new(WarmupRequest {
                batches,
                max_input_length,
                max_prefill_tokens,
                max_total_tokens,
            }).inject_context();
            let _response = self.stub.warmup(request).await?.into_inner();
        }

        // send batches with default params to warm up Greedy search
        let mut greedy_shapes: Vec<(u32, u32)> = Vec::with_capacity(prefill_batch_sizes.len());
        for batch_size in &prefill_batch_sizes {
            greedy_shapes.push((*batch_size, seq_bucket_size.clone()));
        }
        for greedy_shape in greedy_shapes.iter() {
            let batches: Vec<Batch> = vec![
                self.create_warmup_batch(
                    *greedy_shape,
                    &mut id_counter,
                    max_input_length,
                    max_total_tokens,
                    seq_bucket_size,
                    true,
                    None,
                )
            ];
            let request = tonic::Request::new(WarmupRequest {
                batches,
                max_input_length,
                max_prefill_tokens,
                max_total_tokens,
            }).inject_context();
            let _response = self.stub.warmup(request).await?.into_inner();
        }
        Ok(None) // No support for maximum total tokens
    }

    #[instrument(skip_all)]
    fn create_warmup_batch(
        &mut self,
        shape: (u32, u32),
        id_counter: &mut u64,
        max_input_length: u32,
        max_total_tokens: u32,
        seq_bucket_size: u32,
        default_params: bool,
        max_new_tokens: Option<u32>,
    ) -> Batch {
        *id_counter += 1;
        let (batch_size, input_length) = shape;
        let mut requests = Vec::new();
        for request_id in 0..batch_size {
            let req_params = if default_params {
                Some(NextTokenChooserParameters {
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
                })
            } else {
                Some(NextTokenChooserParameters {
                    temperature: 0.9,
                    top_k: 10,
                    top_p: 0.9,
                    typical_p: 0.9,
                    do_sample: true,
                    seed: 0,
                    repetition_penalty: 1.2,
                    frequency_penalty: 0.1,
                    watermark: false,
                    grammar: String::new(),
                    grammar_type: GrammarType::None as i32,
                })
            };
            requests.push(Request {
                id: *id_counter + request_id as u64,
                inputs: self.get_random_input(input_length, seq_bucket_size),
                truncate: max_input_length,
                parameters: req_params,
                stopping_parameters: Some(StoppingCriteriaParameters {
                    max_new_tokens: cmp::min(max_new_tokens.unwrap_or(10), max_total_tokens - max_input_length),
                    stop_sequences: vec![],
                    ignore_eos_token: true,
                }),
                prefill_logprobs: false,
                top_n_tokens: 0,
            });
        }

        Batch {
            id: *id_counter,
            size: requests.len() as u32,
            requests,
            max_tokens: max_total_tokens,
        }
    }

    #[instrument(skip_all)]
    fn get_random_input(
        &mut self,
        input_length: u32,
        seq_bucket_size: u32,
    ) -> String {
        let skip_tokenizer_in_tgi: bool = env::var("SKIP_TOKENIZER_IN_TGI")
            .ok()
            .map_or(false, |value| value.to_lowercase() == "true");
        if skip_tokenizer_in_tgi {
            // generate random tokens
            let mut rng = rand::thread_rng();
            let range = Uniform::new(2, 8192);
            let tokens = if input_length % seq_bucket_size == 0 {
                input_length - seq_bucket_size / 2
            } else {
                input_length - (input_length % seq_bucket_size) / 2
            };
            (0..tokens)
                .map(|_| rng.sample(&range).to_string())
                .collect::<Vec<String>>()
                .join(", ")
        } else {
            // repeat test string to get expected input shape
            let mut bucket_id = input_length / seq_bucket_size;
            if input_length % seq_bucket_size != 0 {
                bucket_id += 1
            }
            let repeats = cmp::max(1, (bucket_id - 1) * seq_bucket_size / 2);
            "_test ".to_string().repeat(repeats as usize)
        }
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
