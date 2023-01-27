/// Batching and inference logic
use crate::{Db, Entry, Token};
use crate::{ErrorResponse, GenerateRequest};
use axum::http::StatusCode;
use axum::Json;
use nohash_hasher::IntMap;
use std::future::Future;
use std::sync::Arc;
use text_generation_client::{Batch, ClientError, GeneratedText, Generation, ShardedClient};
use thiserror::Error;
use tokio::sync::{mpsc, Notify};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tracing::instrument;

/// Batcher
#[derive(Clone)]
pub struct Batcher {
    /// Request database
    db: Db,
    /// Shared state
    shared: Arc<Shared>,
}

/// Batcher shared state
struct Shared {
    /// Batching background Tokio task notifier
    batching_task: Notify,
}

impl Batcher {
    pub(crate) fn new(
        client: ShardedClient,
        max_batch_size: usize,
        max_waiting_tokens: usize,
    ) -> Self {
        // Batcher shared state
        let db = Db::new();
        let shared = Arc::new(Shared {
            batching_task: Notify::new(),
        });

        // Spawn batching background task that contains all the inference logic
        tokio::spawn(batching_task(
            client,
            max_batch_size,
            max_waiting_tokens,
            db.clone(),
            shared.clone(),
        ));

        Self { db, shared }
    }

    /// Add a new request to the database and return a stream of tokens
    pub(crate) fn infer_stream(
        &self,
        input_length: usize,
        request: GenerateRequest,
    ) -> UnboundedReceiverStream<Result<InferStreamResponse, InferError>> {
        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();

        // Try to append the request to the database
        self.db.append(Entry {
            request,
            response_tx,
            input_length,
            time: Instant::now(),
            batch_time: None,
        });

        // Notify the background task that we have a new entry in the database that needs
        // to be batched
        self.shared.batching_task.notify_one();

        // Return stream
        UnboundedReceiverStream::new(response_rx)
    }

    pub(crate) async fn infer(
        &self,
        input_length: usize,
        request: GenerateRequest,
    ) -> Result<InferResponse, InferError> {
        let mut stream = self.infer_stream(input_length, request);

        let mut result_tokens = Vec::new();
        let mut result_generated_text = None;
        let mut result_start = None;
        let mut result_queued = None;

        while let Some(response) = stream.next().await {
            match response? {
                InferStreamResponse::Prefill(prefill_tokens) => {
                    result_tokens.extend(prefill_tokens)
                }
                InferStreamResponse::Token(token) => result_tokens.push(token),
                InferStreamResponse::End {
                    generated_text,
                    start,
                    queued,
                } => {
                    result_generated_text = Some(generated_text);
                    result_start = Some(start);
                    result_queued = Some(queued)
                }
            }
        }
        Ok(InferResponse {
            tokens: result_tokens,
            generated_text: result_generated_text.unwrap(),
            queued: result_queued.unwrap(),
            start: result_start.unwrap(),
        })
    }
}

/// Batching logic
/// Will be launched in a background Tokio task
///
/// Batches requests and sends them to the inference server
#[instrument(skip(client, db, shared))]
async fn batching_task(
    mut client: ShardedClient,
    max_batch_size: usize,
    max_waiting_tokens: usize,
    db: Db,
    shared: Arc<Shared>,
) {
    // Minimum batch size after which we try to add more requests
    let limit_min_batch_size = (max_batch_size / 2) as u32;

    // Infinite loop
    loop {
        // Wait for a notification from the Batcher struct
        shared.batching_task.notified().await;

        // Get the next batch from the DB
        // This batch might be smaller than the maximum batch size if there are not enough requests
        // waiting in the DB
        while let Some((mut entries, batch)) = db.next_batch(None, max_batch_size) {
            let mut cached_batch = wrap_future(client.prefill(batch), &mut entries).await;
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
                        _ if waiting_tokens >= max_waiting_tokens => None,
                        // Minimum size criteria
                        _ => Some(limit_min_batch_size as usize),
                    };

                    // Try to get a new batch
                    if let Some((mut new_entries, new_batch)) =
                        db.next_batch(min_size, max_batch_size - batch_size as usize)
                    {
                        // Generate one token for this new batch to have the attention past in cache
                        let new_cached_batch =
                            wrap_future(client.prefill(new_batch), &mut new_entries).await;
                        // Reset waiting counter
                        waiting_tokens = 1;
                        // Extend current batch with the new batch
                        if let Some(new_cached_batch) = new_cached_batch {
                            entries.extend(new_entries);
                            batches.push(new_cached_batch);
                        }
                    }
                }

                cached_batch = wrap_future(client.decode(batches), &mut entries).await;
                waiting_tokens += 1;
            }
        }
    }
}

/// Wrap a future inside a match statement to handle errors and send the response to the Batcher
async fn wrap_future(
    future: impl Future<Output = Result<(Vec<Generation>, Option<Batch>), ClientError>>,
    entries: &mut IntMap<u64, Entry>,
) -> Option<Batch> {
    match future.await {
        Ok((generations, next_batch)) => {
            send_generated(generations, entries);
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            send_error(err, entries);
            None
        }
    }
}

/// Send errors to the Batcher for all `entries`
fn send_error(error: ClientError, entries: &mut IntMap<u64, Entry>) {
    entries.drain().for_each(|(_, entry)| {
        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry
            .response_tx
            .send(Err(InferError::GenerationError(error.to_string())))
            .unwrap_or(());
    });
}

/// Send `generated_text` to the Batcher for all `finished`
fn send_generated(generations: Vec<Generation>, entries: &mut IntMap<u64, Entry>) {
    generations.into_iter().for_each(|generation| {
        let entry = entries
            .get(&generation.request_id)
            .expect("ID not found in entries. This is a bug.");

        if let Some(prefill_tokens) = generation.prefill_tokens {
            let tokens = prefill_tokens
                .ids
                .into_iter()
                .zip(prefill_tokens.logprobs.into_iter())
                .zip(prefill_tokens.texts.into_iter())
                .map(|((id, logprob), text)| Token(id, text, logprob))
                .collect();
            entry
                .response_tx
                .send(Ok(InferStreamResponse::Prefill(tokens)))
                .unwrap_or(());
        }

        let token = Token(
            generation.token_id,
            generation.token_text,
            generation.token_logprob,
        );
        entry
            .response_tx
            .send(Ok(InferStreamResponse::Token(token)))
            .unwrap_or(());

        if let Some(generated_text) = generation.generated_text {
            let entry = entries
                .remove(&generation.request_id)
                .expect("ID not found in entries. This is a bug.");

            entry
                .response_tx
                .send(Ok(InferStreamResponse::End {
                    generated_text,
                    queued: entry.time,
                    start: entry.batch_time.unwrap(),
                }))
                .unwrap_or(());
        }
    });
}

#[derive(Debug)]
pub(crate) enum InferStreamResponse {
    Prefill(Vec<Token>),
    Token(Token),
    End {
        generated_text: GeneratedText,
        start: Instant,
        queued: Instant,
    },
}

#[derive(Debug)]
pub(crate) struct InferResponse {
    pub(crate) tokens: Vec<Token>,
    pub(crate) generated_text: GeneratedText,
    pub(crate) seed: Option<u64>
    pub(crate) queued: Instant,
    pub(crate) start: Instant,
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
