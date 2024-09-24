use crate::client::{Batch, CachedBatch, ClientError, Generation, Health, ShardedClient};
/// Batching and inference logic
use crate::queue::{Entry, Queue};
use async_trait::async_trait;
use nohash_hasher::IntMap;
use std::sync::Arc;
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;
use text_generation_router::{Attention, FinishReason, PrefillToken, Token};
use tokio::sync::mpsc::error::SendError;
use tokio::sync::{mpsc, Notify};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{info_span, instrument, Instrument, Span};

pub struct BackendV2 {
    /// Request queue
    queue: Queue,
    /// Notify batcher on queue appends
    batching_task_notifier: Arc<Notify>,
    /// Client clone, used for health checks to skip the queue
    client: ShardedClient,
}

impl BackendV2 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        client: ShardedClient,
        waiting_served_ratio: f32,
        max_batch_prefill_tokens: u32,
        max_batch_total_tokens: u32,
        max_waiting_tokens: usize,
        max_batch_size: Option<usize>,
        requires_padding: bool,
        window_size: Option<u32>,
        speculate: u32,
    ) -> Self {
        // Infer shared state
        let attention = if let Ok(attention) = std::env::var("ATTENTION") {
            attention
                .parse()
                .unwrap_or_else(|_| panic!("Invalid attention was specified :`{attention}`"))
        } else {
            Attention::Paged
        };
        let block_size = if attention == Attention::FlashDecoding {
            256
        } else {
            16
        };
        let queue = Queue::new(requires_padding, block_size, window_size, speculate);
        let batching_task_notifier = Arc::new(Notify::new());

        // Spawn batching background task that contains all the inference logic
        tokio::spawn(batching_task(
            client.clone(),
            waiting_served_ratio,
            max_batch_prefill_tokens,
            max_batch_total_tokens,
            max_waiting_tokens,
            max_batch_size,
            queue.clone(),
            batching_task_notifier.clone(),
        ));

        Self {
            queue,
            batching_task_notifier,
            client,
        }
    }
}

#[async_trait]
impl Backend for BackendV2 {
    #[instrument(skip_all)]
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        // Append the request to the queue
        self.queue.append(Entry {
            request,
            response_tx,
            span: Span::current(),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
        });

        // Notify the background task that we have a new entry in the queue that needs
        // to be batched
        self.batching_task_notifier.notify_one();

        // Return stream
        Ok(UnboundedReceiverStream::new(response_rx))
    }

    async fn health(&self, current_health: bool) -> bool {
        if current_health {
            // Generation is healthy, we only check that the shards can allocate on device
            self.client.device_health().await
        } else {
            self.client.model_health().await
        }
        .is_ok()
    }
}

/// Batching logic
/// Will be launched in a background Tokio task
///
/// Batches requests and sends them to the inference server
#[allow(clippy::too_many_arguments)]
pub(crate) async fn batching_task(
    mut client: ShardedClient,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    max_batch_size: Option<usize>,
    queue: Queue,
    notifier: Arc<Notify>,
) {
    // Infinite loop
    loop {
        // Wait for a notification from the Infer struct
        notifier.notified().await;

        // Get the next batch from the queue
        // This batch might be smaller than the maximum batch size if there are not enough requests
        // waiting in the queue
        while let Some((mut entries, batch, span)) = queue
            .next_batch(
                None,
                max_batch_size,
                max_batch_prefill_tokens,
                max_batch_total_tokens,
            )
            .await
        {
            let mut cached_batch = prefill(&mut client, batch, &mut entries)
                .instrument(span)
                .await;
            let mut waiting_tokens = 1;

            // We loop until we do not receive any cached batch from the inference server (== until
            // all requests have met their stopping criteria)
            while let Some(batch) = cached_batch {
                // Get current batch info
                let batch_size = batch.size;
                let batch_max_tokens = batch.max_tokens;
                let mut batches = vec![batch];
                metrics::gauge!("tgi_batch_current_size").set(batch_size as f64);
                metrics::gauge!("tgi_batch_current_max_tokens").set(batch_max_tokens as f64);

                let min_size = if waiting_tokens >= max_waiting_tokens {
                    // If we didn't onboard any new requests since >= max_waiting_tokens, we try
                    // to add a new batch even though its size might be small
                    None
                } else {
                    // Minimum batch size
                    Some((batch_size as f32 * waiting_served_ratio).floor() as usize)
                };

                let token_budget = max_batch_total_tokens.saturating_sub(batch_max_tokens);
                let max_size =
                    max_batch_size.map(|max_size| max_size.saturating_sub(batch_size as usize));
                // Try to get a new batch
                if let Some((mut new_entries, new_batch, span)) = queue
                    .next_batch(min_size, max_size, max_batch_prefill_tokens, token_budget)
                    .await
                {
                    // Tracking metrics
                    if min_size.is_some() {
                        metrics::counter!("tgi_batch_concat", "reason" => "backpressure")
                            .increment(1);
                    } else {
                        metrics::counter!("tgi_batch_concat", "reason" => "wait_exceeded")
                            .increment(1);
                    }

                    entries.iter_mut().for_each(|(_, entry)| {
                        // Create a new span to add the info that this entry is waiting
                        // because a new batch is being computed
                        let entry_waiting_span = info_span!(parent: &entry.span, "waiting");
                        // Add relationships
                        span.follows_from(&entry_waiting_span);
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

                // Create span for this batch to add context to inference calls
                let next_batch_size = entries.len();
                let next_batch_span =
                    info_span!(parent: None, "batch", batch_size = next_batch_size);
                entries.iter_mut().for_each(|(_, entry)| {
                    // Create a new span to link the batch back to this entry
                    let entry_batch_span = info_span!(parent: &entry.span, "infer");
                    // Add relationships
                    next_batch_span.follows_from(&entry_batch_span);
                    entry_batch_span.follows_from(&next_batch_span);
                    // Update entry
                    entry.temp_span = Some(entry_batch_span);
                });

                cached_batch = decode(&mut client, batches, &mut entries)
                    .instrument(next_batch_span)
                    .await;
                waiting_tokens += 1;
            }
            metrics::gauge!("tgi_batch_current_size").set(0.0);
            metrics::gauge!("tgi_batch_current_max_tokens").set(0.0);
        }
    }
}

#[instrument(skip_all)]
async fn prefill(
    client: &mut ShardedClient,
    batch: Batch,
    entries: &mut IntMap<u64, Entry>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_id = batch.id;
    metrics::counter!("tgi_batch_inference_count", "method" => "prefill").increment(1);

    match client.prefill(batch).await {
        Ok((generations, next_batch, timings)) => {
            let start_filtering_time = Instant::now();
            // Send generated tokens and filter stopped entries
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

            metrics::histogram!("tgi_batch_forward_duration","method" => "prefill")
                .record(timings.forward.as_secs_f64());
            metrics::histogram!("tgi_batch_decode_duration", "method" => "prefill")
                .record(timings.decode.as_secs_f64());
            metrics::histogram!("tgi_batch_filter_duration", "method" => "prefill")
                .record(start_filtering_time.elapsed().as_secs_f64());
            metrics::histogram!("tgi_batch_inference_duration","method" => "prefill")
                .record(start_time.elapsed().as_secs_f64());
            metrics::counter!("tgi_batch_inference_success", "method" => "prefill").increment(1);
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            let _ = client.clear_cache(Some(batch_id)).await;
            send_errors(err, entries);
            metrics::counter!("tgi_batch_inference_failure", "method" => "prefill").increment(1);
            None
        }
    }
}

#[instrument(skip_all)]
async fn decode(
    client: &mut ShardedClient,
    batches: Vec<CachedBatch>,
    entries: &mut IntMap<u64, Entry>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_ids: Vec<u64> = batches.iter().map(|b| b.id).collect();
    metrics::counter!("tgi_batch_inference_count", "method" => "decode").increment(1);

    match client.decode(batches).await {
        Ok((generations, next_batch, timings)) => {
            let start_filtering_time = Instant::now();
            // Send generated tokens and filter stopped entries
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

            if let Some(concat_duration) = timings.concat {
                metrics::histogram!("tgi_batch_concat_duration", "method" => "decode")
                    .record(concat_duration.as_secs_f64());
            }
            metrics::histogram!("tgi_batch_forward_duration", "method" => "decode")
                .record(timings.forward.as_secs_f64());
            metrics::histogram!("tgi_batch_decode_duration", "method" => "decode")
                .record(timings.decode.as_secs_f64());
            metrics::histogram!("tgi_batch_filter_duration", "method" => "decode")
                .record(start_filtering_time.elapsed().as_secs_f64());
            metrics::histogram!("tgi_batch_inference_duration", "method" => "decode")
                .record(start_time.elapsed().as_secs_f64());
            metrics::counter!("tgi_batch_inference_success", "method" => "decode").increment(1);
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            for id in batch_ids {
                let _ = client.clear_cache(Some(id)).await;
            }
            send_errors(err, entries);
            metrics::counter!("tgi_batch_inference_failure", "method" => "decode").increment(1);
            None
        }
    }
}

/// Filter a `batch` and remove all requests not present in `entries`
#[instrument(skip_all)]
async fn filter_batch(
    client: &mut ShardedClient,
    next_batch: Option<CachedBatch>,
    entries: &IntMap<u64, Entry>,
) -> Option<CachedBatch> {
    let mut batch = next_batch?;

    // No need to filter
    if batch.size as usize == entries.len() {
        return Some(batch);
    }

    let id = batch.id;

    // Retain only requests that are still in entries
    batch.request_ids.retain(|id| entries.contains_key(id));

    if batch.request_ids.is_empty() {
        // All requests have been filtered out
        // Next batch is now empty
        // Clear it from the Python shards cache
        // We unwrap here as we need to panic since we cannot recover if this method fails
        client.clear_cache(Some(id)).await.unwrap();
        None
    } else {
        // Filter Python shard cache
        // We unwrap here as we need to panic since we cannot recover if this method fails
        client.filter_batch(id, batch.request_ids).await.unwrap()
    }
}

/// Send one or multiple `InferStreamResponse` to Infer for all `entries`
/// and filter entries
#[instrument(skip_all)]
fn filter_send_generations(generations: Vec<Generation>, entries: &mut IntMap<u64, Entry>) {
    generations.into_iter().for_each(|generation| {
        let id = generation.request_id;
        // Get entry
        // We can `expect` here as the request id should always be in the entries
        let entry = entries
            .get(&id)
            .expect("ID not found in entries. This is a bug.");

        // Create and enter a span to link this function back to the entry
        let _span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_generation", generation = ?generation).entered();
        // Send generation responses back to the infer task
        // If the receive an error from the Flume channel, it means that the client dropped the
        // request and we need to stop generating hence why we unwrap_or(true)
        let stopped = send_responses(generation, entry).inspect_err(|_err| {
            tracing::error!("Entry response channel error.");
            metrics::counter!("tgi_request_failure", "err" => "dropped").increment(1);
        }).unwrap_or(true);
        if stopped {
            entries.remove(&id).expect("ID not found in entries. This is a bug.");
        }
    });
}

/// Send responses through the `entry` response channel
fn send_responses(
    generation: Generation,
    entry: &Entry,
) -> Result<bool, Box<SendError<Result<InferStreamResponse, InferError>>>> {
    // Return directly if the channel is disconnected
    if entry.response_tx.is_closed() {
        metrics::counter!("tgi_request_failure", "err" => "dropped").increment(1);
        return Ok(true);
    }

    let mut stopped = false;

    if let Some(prefill_tokens) = generation.prefill_tokens {
        // Create Token objects
        // We do that here instead of in the Python code as Rust for loops are faster
        let prefill_tokens = prefill_tokens
            .ids
            .into_iter()
            .zip(prefill_tokens.logprobs)
            .zip(prefill_tokens.texts)
            .map(|((id, logprob), text)| PrefillToken { id, text, logprob })
            .collect();

        // Send message
        entry
            .response_tx
            .send(Ok(InferStreamResponse::Prefill(prefill_tokens)))?;
    }

    // Create last Token
    let tokens_ = generation.tokens.expect("Non empty tokens in generation");
    let n = tokens_.ids.len();
    metrics::histogram!("tgi_request_skipped_tokens").record((n - 1) as f64);
    let mut iterator = tokens_
        .ids
        .into_iter()
        .zip(tokens_.logprobs)
        .zip(tokens_.texts)
        .zip(tokens_.is_special)
        .enumerate()
        .peekable();
    while let Some((i, (((id, logprob), text), special))) = iterator.next() {
        let token = Token {
            id,
            text,
            logprob,
            special,
        };
        let top_tokens = if let Some(top_tokens_) = generation.top_tokens.get(i) {
            top_tokens_
                .ids
                .iter()
                .zip(top_tokens_.logprobs.iter())
                .zip(top_tokens_.texts.iter())
                .zip(top_tokens_.is_special.iter())
                .map(|(((&id, &logprob), text), &special)| Token {
                    id,
                    text: text.to_string(),
                    logprob,
                    special,
                })
                .collect()
        } else {
            vec![]
        };
        match (&generation.generated_text, iterator.peek()) {
            (Some(generated_text), None) => {
                // Generation has ended
                stopped = true;
                // Send message
                entry.response_tx.send(Ok(InferStreamResponse::End {
                    token,
                    top_tokens,
                    generated_text: GeneratedText::from(generated_text.clone()),
                    queued: entry.queue_time,
                    start: entry.batch_time.unwrap(),
                }))?;
            }
            _ => {
                // Send message
                entry
                    .response_tx
                    .send(Ok(InferStreamResponse::Intermediate { token, top_tokens }))?;
            }
        }
    }

    Ok(stopped)
}

/// Send errors to Infer for all `entries`
#[instrument(skip_all)]
fn send_errors(error: ClientError, entries: &mut IntMap<u64, Entry>) {
    entries.drain().for_each(|(_, entry)| {
        // Create and enter a span to link this function back to the entry
        let _send_error_span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_error").entered();
        let err = InferError::GenerationError(error.to_string());
        metrics::counter!("tgi_request_failure", "err" => "generation").increment(1);
        tracing::error!("{err}");

        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry
            .response_tx
            .send(Err(err))
            .unwrap_or(());
    });
}

impl From<crate::client::GeneratedText> for GeneratedText {
    fn from(value: crate::client::GeneratedText) -> Self {
        let v2_finish_reason = crate::client::FinishReason::try_from(value.finish_reason).unwrap();
        let finish_reason = match v2_finish_reason {
            crate::client::FinishReason::Length => FinishReason::Length,
            crate::client::FinishReason::EosToken => FinishReason::EndOfSequenceToken,
            crate::client::FinishReason::StopSequence => FinishReason::StopSequence,
        };

        Self {
            text: value.text,
            generated_tokens: value.generated_tokens,
            finish_reason,
            seed: value.seed,
        }
    }
}
