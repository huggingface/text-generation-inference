/// Batching and inference logic
use crate::validation::{Validation, ValidationError};
use crate::{Entry, Queue, Token};
use crate::{GenerateRequest, PrefillToken};
use futures::future::try_join_all;
use nohash_hasher::IntMap;
use std::sync::Arc;
use text_generation_client::{
    Batch, ClientError, GeneratedText, Generation, PrefillTokens, ShardedClient,
};
use thiserror::Error;
use tokio::sync::{mpsc, Notify, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tracing::{info_span, instrument, Instrument, Span};

/// Inference struct
#[derive(Clone)]
pub struct Infer {
    /// Validation
    validation: Validation,
    /// Request queue
    queue: Queue,
    /// Shared state
    shared: Arc<Shared>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
}

/// Infer shared state
struct Shared {
    /// Batching background Tokio task notifier
    batching_task: Notify,
}

impl Infer {
    pub(crate) fn new(
        client: ShardedClient,
        validation: Validation,
        max_batch_size: usize,
        max_waiting_tokens: usize,
        max_concurrent_requests: usize,
    ) -> Self {
        // Infer shared state
        let queue = Queue::new();
        let shared = Arc::new(Shared {
            batching_task: Notify::new(),
        });

        // Spawn batching background task that contains all the inference logic
        tokio::spawn(batching_task(
            client,
            max_batch_size,
            max_waiting_tokens,
            queue.clone(),
            shared.clone(),
        ));

        // Inference limit with a semaphore
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

        Self {
            validation,
            queue,
            shared,
            limit_concurrent_requests: semaphore,
        }
    }

    /// Add a new request to the queue and return a stream of InferStreamResponse
    #[instrument(skip(self))]
    pub(crate) async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        // This permit will live as long as Entry
        let permit = self
            .clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::increment_counter!("tgi_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                err
            })?;

        // Validate request
        let valid_request = self.validation.validate(request).await?;

        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        // Append the request to the queue
        self.queue.append(Entry {
            request: valid_request,
            response_tx,
            span: Span::current(),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
            _permit: permit,
        });

        // Notify the background task that we have a new entry in the queue that needs
        // to be batched
        self.shared.batching_task.notify_one();

        // Return stream
        Ok(UnboundedReceiverStream::new(response_rx))
    }

    /// Add a new request to the queue and return a InferResponse
    #[instrument(skip(self))]
    pub(crate) async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<InferResponse, InferError> {
        // Create stream
        let mut stream = self.generate_stream(request).await?;

        // Return values
        let mut result_prefill = Vec::new();
        let mut result_tokens = Vec::new();
        let mut result_generated_text = None;
        let mut result_start = None;
        let mut result_queued = None;

        // Iterate on stream
        while let Some(response) = stream.next().await {
            match response? {
                // Add prefill tokens
                InferStreamResponse::Prefill(tokens) => {
                    // Create Token objects
                    // We do that here instead of in the Python code as Rust for loops are faster
                    result_prefill = tokens
                        .ids
                        .into_iter()
                        .zip(tokens.logprobs.into_iter())
                        .zip(tokens.texts.into_iter())
                        .map(|((id, logprob), text)| PrefillToken { id, text, logprob })
                        .collect();
                }
                // Push last token
                InferStreamResponse::Token(token) => result_tokens.push(token),
                // Final message
                // Set return values
                InferStreamResponse::End {
                    token,
                    generated_text,
                    start,
                    queued,
                } => {
                    result_tokens.push(token);
                    result_generated_text = Some(generated_text);
                    result_start = Some(start);
                    result_queued = Some(queued)
                }
            }
        }

        // Check that we received a `InferStreamResponse::End` message
        if let (Some(generated_text), Some(queued), Some(start)) =
            (result_generated_text, result_queued, result_start)
        {
            Ok(InferResponse {
                prefill: result_prefill,
                tokens: result_tokens,
                generated_text,
                queued,
                start,
            })
        } else {
            let err = InferError::IncompleteGeneration;
            metrics::increment_counter!("tgi_request_failure", "err" => "incomplete");
            tracing::error!("{err}");
            Err(err)
        }
    }
    /// Add best_of new requests to the queue and return a InferResponse of the sequence with
    /// the highest log probability per token
    #[instrument(skip(self))]
    pub(crate) async fn generate_best_of(
        &self,
        request: GenerateRequest,
        best_of: usize,
    ) -> Result<(InferResponse, Vec<InferResponse>), InferError> {
        // validate  best_of parameter separately
        let best_of = self.validation.validate_best_of(best_of)?;

        // create multiple generate requests
        let mut infer_responses: Vec<InferResponse> =
            try_join_all((0..best_of).map(|_| self.generate(request.clone()))).await?;

        // get the sequence with the highest log probability per token
        let mut max_index = 0;
        let mut max_logprob: f32 = f32::MIN;

        for (i, response) in infer_responses.iter().enumerate() {
            // mean logprobs of the generated tokens
            let sequence_logprob = response
                .tokens
                .iter()
                .map(|token| token.logprob)
                .sum::<f32>()
                / response.tokens.len() as f32;

            // set best sequence
            if sequence_logprob > max_logprob {
                max_index = i;
                max_logprob = sequence_logprob;
            }
        }
        let best_response = infer_responses.remove(max_index);
        Ok((best_response, infer_responses))
    }
}

/// Batching logic
/// Will be launched in a background Tokio task
///
/// Batches requests and sends them to the inference server
async fn batching_task(
    mut client: ShardedClient,
    max_batch_size: usize,
    max_waiting_tokens: usize,
    queue: Queue,
    shared: Arc<Shared>,
) {
    // Minimum batch size after which we try to add more requests
    let limit_min_batch_size = if max_batch_size > 1 {
        (max_batch_size / 2) as u32
    } else {
        0
    };

    // Infinite loop
    loop {
        // Wait for a notification from the Infer struct
        shared.batching_task.notified().await;

        // Get the next batch from the queue
        // This batch might be smaller than the maximum batch size if there are not enough requests
        // waiting in the queue
        while let Some((mut entries, batch, span)) = queue.next_batch(None, max_batch_size).await {
            let mut cached_batch = prefill(&mut client, batch, &mut entries)
                .instrument(span)
                .await;
            let mut waiting_tokens = 1;

            // We loop until we do not receive any cached batch from the inference server (== until
            // all requests have met their stopping criteria)
            while let Some(batch) = cached_batch {
                // Get current batch info
                let batch_size = batch.size;
                let mut batches = vec![batch];
                metrics::gauge!("tgi_batch_current_size", batch_size as f64);

                // If the current batch is too small, we try to add more requests to it
                if batch_size <= limit_min_batch_size {
                    let min_size = match waiting_tokens {
                        // If we didn't onboard any new requests since >= max_waiting_tokens, we try
                        // to add a new batch even though its size might be small
                        _ if waiting_tokens >= max_waiting_tokens => None,
                        // Minimum size criteria
                        _ => Some(limit_min_batch_size as usize),
                    };

                    // Try to get a new batch
                    if let Some((mut new_entries, new_batch, span)) = queue
                        .next_batch(min_size, max_batch_size - batch_size as usize)
                        .await
                    {
                        let new_batch_size = new_batch.size;
                        entries.iter_mut().for_each(|(_, entry)| {
                            // Create a new span to add the info that this entry is waiting
                            // because a new batch is being computed
                            let entry_waiting_span =
                                info_span!(parent: &entry.span, "waiting", batch_size = new_batch_size);
                            // Add relationship
                            entry_waiting_span.follows_from(&span);
                            // Update entry
                            entry.temp_span = Some(entry_waiting_span);
                        });

                        // Generate one token for this new batch to have the attention past in cache
                        let new_cached_batch = prefill(&mut client, new_batch, &mut new_entries)
                            .instrument(span)
                            .await;
                        // Reset waiting counter
                        waiting_tokens = 1;
                        // Extend current batch with the new batch
                        if let Some(new_cached_batch) = new_cached_batch {
                            entries.extend(new_entries);
                            batches.push(new_cached_batch);
                        }
                    }
                }
                // Create span for this batch to add context to inference calls
                let next_batch_size = entries.len();
                let next_batch_span =
                    info_span!(parent: None, "batch", batch_size = next_batch_size);
                entries.iter_mut().for_each(|(_, entry)| {
                    // Create a new span to link the batch back to this entry
                    let entry_batch_span =
                        info_span!(parent: &entry.span, "infer", batch_size = next_batch_size);
                    // Add relationship
                    entry_batch_span.follows_from(&next_batch_span);
                    // Update entry
                    entry.temp_span = Some(entry_batch_span);
                });

                cached_batch = decode(&mut client, batches, &mut entries)
                    .instrument(next_batch_span)
                    .await;
                waiting_tokens += 1;
            }
            metrics::gauge!("tgi_batch_current_size", 0.0);
        }
    }
}

#[instrument(skip_all)]
async fn prefill(
    client: &mut ShardedClient,
    batch: Batch,
    entries: &mut IntMap<u64, Entry>,
) -> Option<Batch> {
    let start_time = Instant::now();

    match client.prefill(batch).await {
        Ok((generations, next_batch)) => {
            send_generations(generations, entries);
            metrics::histogram!("tgi_batch_inference_duration", start_time.elapsed(), "method" => "prefill");
            metrics::increment_counter!("tgi_batch_inference_success", "method" => "prefill");
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            send_errors(err, entries);
            metrics::increment_counter!("tgi_batch_inference_failure", "method" => "prefill");
            None
        }
    }
}

#[instrument(skip_all)]
async fn decode(
    client: &mut ShardedClient,
    batches: Vec<Batch>,
    entries: &mut IntMap<u64, Entry>,
) -> Option<Batch> {
    let start_time = Instant::now();

    match client.decode(batches).await {
        Ok((generations, next_batch)) => {
            send_generations(generations, entries);
            metrics::histogram!("tgi_batch_inference_duration", start_time.elapsed(), "method" => "decode");
            metrics::increment_counter!("tgi_batch_inference_success", "method" => "decode");
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            send_errors(err, entries);
            metrics::increment_counter!("tgi_batch_inference_failure", "method" => "decode");
            None
        }
    }
}

/// Send errors to Infer for all `entries`
#[instrument(skip_all)]
fn send_errors(error: ClientError, entries: &mut IntMap<u64, Entry>) {
    entries.drain().for_each(|(_, entry)| {
        // Create and enter a span to link this function back to the entry
        let _send_error_span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_error").entered();
        let err = InferError::GenerationError(error.to_string());
        metrics::increment_counter!("tgi_request_failure", "err" => "generation");
        tracing::error!("{err}");

        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry
            .response_tx
            .send(Err(err))
            .unwrap_or(());
    });
}

/// Send one or multiple `InferStreamResponse` to Infer for all `entries`
#[instrument(skip_all)]
fn send_generations(generations: Vec<Generation>, entries: &mut IntMap<u64, Entry>) {
    generations.into_iter().for_each(|generation| {
        // Get entry
        // We can `expect` here as the request id should always be in the entries
        let entry = entries
            .get(&generation.request_id)
            .expect("ID not found in entries. This is a bug.");

        // Create and enter a span to link this function back to the entry
        let _generation_span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_generation", generation = ?generation).entered();

        if let Some(prefill_tokens) = generation.prefill_tokens {
            // Send message
            // unwrap_or is valid here as we don't care if the receiver is gone.
            entry
                .response_tx
                .send(Ok(InferStreamResponse::Prefill(prefill_tokens)))
                .unwrap_or(());
        }

        // Create last Token
        let token = Token {
            id: generation.token_id,
            text: generation.token_text,
            logprob: generation.token_logprob,
            special: generation.token_is_special,
        };

        if let Some(generated_text) = generation.generated_text {
            // Remove entry as this is the last message
            // We can `expect` here as the request id should always be in the entries
            let entry = entries
                .remove(&generation.request_id)
                .expect("ID not found in entries. This is a bug.");

            // Send message
            // unwrap_or is valid here as we don't care if the receiver is gone.
            entry
                .response_tx
                .send(Ok(InferStreamResponse::End {
                    token,
                    generated_text,
                    queued: entry.queue_time,
                    start: entry.batch_time.unwrap(),
                }))
                .unwrap_or(());
        } else {
            // Send message
            // unwrap_or is valid here as we don't care if the receiver is gone.
            entry
                .response_tx
                .send(Ok(InferStreamResponse::Token(token)))
                .unwrap_or(());
        }
    });
}

#[derive(Debug)]
pub(crate) enum InferStreamResponse {
    // Optional first message
    Prefill(PrefillTokens),
    // Intermediate messages
    Token(Token),
    // Last message
    End {
        token: Token,
        generated_text: GeneratedText,
        start: Instant,
        queued: Instant,
    },
}

#[derive(Debug)]
pub(crate) struct InferResponse {
    pub(crate) prefill: Vec<PrefillToken>,
    pub(crate) tokens: Vec<Token>,
    pub(crate) generated_text: GeneratedText,
    pub(crate) queued: Instant,
    pub(crate) start: Instant,
}

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
    #[error("Model is overloaded")]
    Overloaded(#[from] TryAcquireError),
    #[error("Input validation error: {0}")]
    ValidationError(#[from] ValidationError),
    #[error("Incomplete generation")]
    IncompleteGeneration,
}

impl InferError {
    pub(crate) fn error_type(&self) -> &str {
        match self {
            InferError::GenerationError(_) => "generation",
            InferError::Overloaded(_) => "overloaded",
            InferError::ValidationError(_) => "validation",
            InferError::IncompleteGeneration => "incomplete_generation",
        }
    }
}
