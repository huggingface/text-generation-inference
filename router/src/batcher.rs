/// Batching and inference logic
use crate::Entry;
use crate::{ErrorResponse, GenerateRequest};
use axum::http::StatusCode;
use axum::Json;
use std::future::Future;
use nohash_hasher::IntMap;
use text_generation_client::{Batch, ClientError, GeneratedText, ShardedClient};
use thiserror::Error;
use tokio::sync::oneshot;
use tokio::sync::mpsc::{channel, Permit, Sender};
use tokio::sync::mpsc::error::TrySendError;
use tokio::time::Instant;
use tracing::instrument;
use crate::queue::Queue;

/// Batcher
#[derive(Clone)]
pub struct Batcher {
    /// Request queue
    sender: Sender<Entry>,
}


impl Batcher {
    pub(crate) fn new(
        client: ShardedClient,
        max_batch_size: usize,
        max_waiting_tokens: usize,
        queue_size: usize,
    ) -> Self {
        // Set up queue
        let (sender, receiver) = channel(queue_size);

        // Spawn batching background task that contains all the inference logic
        tokio::spawn(batching_task(
            client,
            max_batch_size,
            max_waiting_tokens,
            Queue::new(receiver),
        ));

        Self { sender }
    }

    /// Reserve a slot in the queue for sending a request
    pub(crate) fn reserve_slot(&self) -> Result<RequestSender<'_>, TrySendError<()>> {
        self.sender.try_reserve().map(|permit| RequestSender { permit })
    }
}

pub(crate) struct RequestSender<'a> {
    permit: Permit<'a, Entry>
}

impl <'a> RequestSender<'a> {
    /// Add a new request to the queue and return a future that will generate the text
    pub(crate) async fn infer(
        self,
        input_length: usize,
        request: GenerateRequest,
    ) -> Result<InferResponse, InferError> {
        // One shot channel to communicate with the background batching task
        let (response_tx, response_rx) = oneshot::channel();

        // Try to enqueue the request
        self.permit.send(Entry {
            request,
            response_tx,
            input_length,
            time: Instant::now(),
            batch_time: None,
        });

        // Await on the response from the background task
        // We can safely unwrap as the background task will never drop the sender
        response_rx
            .await
            .unwrap()
            .map_err(|err| InferError::GenerationError(err.to_string()))
    }
}

/// Batching logic
/// Will be launched in a background Tokio task
///
/// Batches requests and sends them to the inference server
#[instrument(skip(client, queue))]
async fn batching_task(
    mut client: ShardedClient,
    max_batch_size: usize,
    max_waiting_tokens: usize,
    mut queue: Queue,
) {
    // Minimum batch size after which we try to add more requests
    let limit_min_batch_size = (max_batch_size / 2) as u32;

    // Entries corresponding to all of the in-progress requests
    let mut entries = IntMap::default();

    // Get the next batch from the queue
    // This batch might be smaller than the maximum batch size if there are not enough requests
    // waiting in the queue
    while let Some(batch) = queue.next_batch(max_batch_size, &mut entries).await {
        let mut cached_batch = wrap_future(
            client.generate(batch), None, &mut entries
        ).await;
        let mut waiting_tokens = 1;

        // We loop until we do not receive any cached batch from the inference server (== until
        // all requests have met their stopping criteria)
        while let Some(batch) = cached_batch {
            // Get current batch info
            let batch_size = batch.size;
            let mut batches = vec![batch];

            // If the current batch is too small, we try to add more requests to it
            if batch_size <= limit_min_batch_size {
                let min_size = match waiting_tokens {
                    // If we didn't onboard any new requests since >= max_waiting_tokens, we try
                    // to add a new batch even though its size might be small
                    _ if waiting_tokens >= max_waiting_tokens => 1,
                    // Minimum size criteria
                    _ => limit_min_batch_size as usize,
                };

                // Try to get a new batch
                if let Some(new_batch) = queue.try_next_batch(
                    min_size, max_batch_size - batch_size as usize, &mut entries
                ) {
                    let first_new_id = new_batch.requests.first()
                        .expect("batch can't be empty here").id;
                    // Generate one token for this new batch to have the attention past in cache
                    let new_cached_batch = wrap_future(
                        client.generate(new_batch), Some(first_new_id), &mut entries
                    ).await;

                    // Reset waiting counter
                    waiting_tokens = 1;
                    // Extend current batch with the new batch
                    if let Some(new_cached_batch) = new_cached_batch {
                        batches.push(new_cached_batch);
                    }
                }
            }

            cached_batch = wrap_future(
                client.generate_with_cache(batches), None, &mut entries
            ).await;
            waiting_tokens += 1;
        }
    }
}

/// Wrap a future inside a match statement to handle errors and send the response to the Batcher
async fn wrap_future(
    future: impl Future<Output = Result<(Vec<GeneratedText>, Option<Batch>), ClientError>>,
    // First request id in this batch if it doesn't comprise all current entries
    start_id: Option<u64>,
    entries: &mut IntMap<u64, Entry>,
) -> Option<Batch> {
    match future.await {
        Ok((generated_texts, next_batch)) => {
            send_generated(generated_texts, entries);
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            send_error(err, start_id, entries);
            None
        }
    }
}

/// Send errors to the Batcher for all failed entries
fn send_error(error: ClientError, start_id: Option<u64>, entries: &mut IntMap<u64, Entry>) {
    let to_keep = entries.drain().filter_map(|(id, entry)| match start_id {
        // Keep entries that weren't in the failed request batch
        Some(sid) if id < sid => Some((id, entry)),
        _ => {
            // unwrap_or is valid here as we don't care if the receiver is gone.
            entry.response_tx.send(Err(error.clone())).unwrap_or(());
            None
        }
    }).collect::<IntMap<u64, Entry>>();
    // Workaround since drain_filter() is not yet stable. This will be empty when start_id == None.
    entries.extend(to_keep);
}

/// Send `generated_text` to the Batcher for all `finished`
fn send_generated(finished: Vec<GeneratedText>, entries: &mut IntMap<u64, Entry>) {
    finished.into_iter().for_each(|output| {
        // We can `expect` here as the request id should always be in the map
        let entry = entries
            .remove(&output.request.unwrap().id)
            .expect("ID not found. This is a bug.");

        let response = InferResponse {
            output_text: output.output_text,
            generated_tokens: output.generated_tokens,
            token_ids: output.token_ids,
            tokens: output.tokens,
            logprobs: output.logprobs,
            finish_reason: output.finish_reason,
            queued: entry.time,
            start: entry.batch_time.unwrap(), // unwrap is always valid
            end: Instant::now(),
        };
        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry.response_tx.send(Ok(response)).unwrap_or(());
    });
}

#[derive(Debug)]
pub(crate) struct InferResponse {
    pub(crate) output_text: String,
    pub(crate) generated_tokens: u32,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) tokens: Vec<String>,
    pub(crate) logprobs: Vec<f32>,
    pub(crate) finish_reason: String,
    pub(crate) queued: Instant,
    pub(crate) start: Instant,
    pub(crate) end: Instant,
}

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
}

/// Convert to Axum supported format
impl From<InferError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: InferError) -> Self {
        match err {
            InferError::GenerationError(_) => (
                StatusCode::FAILED_DEPENDENCY,
                Json(ErrorResponse {
                    error: err.to_string(),
                }),
            ),
        }
    }
}
