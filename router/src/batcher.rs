/// Batching and inference logic
use crate::{Db, Entry};
use crate::{ErrorResponse, GenerateRequest};
use axum::http::StatusCode;
use axum::Json;
use std::future::Future;
use std::sync::Arc;
use text_generation_client::{Batch, ClientError, GeneratedText, ShardedClient};
use thiserror::Error;
use tokio::sync::{oneshot, Notify};
use tokio::time::Instant;
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

    /// Add a new request to the database and return a future that will generate the text
    pub(crate) async fn infer(
        &self,
        input_length: usize,
        request: GenerateRequest,
    ) -> Result<InferResponse, InferError> {
        // One shot channel to communicate with the background batching task
        let (response_tx, response_rx) = oneshot::channel();

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
        self.shared.batching_task.notify_waiters();

        // Await on the response from the background task
        // We can safely unwrap as the background task will never drop the sender
        match response_rx.await.unwrap() {
            Ok(output) => Ok(output),
            Err(err) => Err(InferError::GenerationError(err.to_string())),
        }
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
        let mut waiting_tokens = 0;
        if let Some((request_ids, batch)) = db.next_batch(None, max_batch_size) {
            let mut cached_batch = wrap_future(client.generate(batch), request_ids, &db).await;
            waiting_tokens += 1;

            // We loop until we do not receive any cached batch from the inference server (== until
            // all requests have met their stopping criteria)
            while let Some(batch) = cached_batch {
                // Get current batch info
                let batch_size = batch.size;
                let mut request_ids: Vec<u64> = batch.requests.iter().map(|req| req.id).collect();
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
                    if let Some((new_request_ids, new_batch)) =
                        db.next_batch(min_size, max_batch_size)
                    {
                        // Reset waiting counter
                        waiting_tokens = 0;
                        // Generate one token for this new batch to have the attention past in cache
                        let new_cached_batch =
                            wrap_future(client.generate(new_batch), new_request_ids, &db).await;
                        // Extend current batch with the new batch
                        if let Some(new_cached_batch) = new_cached_batch {
                            request_ids.extend(new_cached_batch.requests.iter().map(|req| req.id));
                            batches.push(new_cached_batch);
                        }
                    }
                }

                cached_batch =
                    wrap_future(client.generate_with_cache(batches), request_ids, &db).await;
                waiting_tokens += 1;
            }
        }
    }
}

/// Wrap a future inside a match statement to handle errors and send the response to the Batcher
async fn wrap_future(
    future: impl Future<Output = Result<(Vec<GeneratedText>, Option<Batch>), ClientError>>,
    request_ids: Vec<u64>,
    db: &Db,
) -> Option<Batch> {
    match future.await {
        Ok((generated_texts, next_batch)) => {
            send_generated(generated_texts, db);
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            send_error(err, request_ids, db);
            None
        }
    }
}

/// Send errors to the Batcher for all `request_ids`
fn send_error(error: ClientError, request_ids: Vec<u64>, db: &Db) {
    request_ids.into_iter().for_each(|id| {
        // We can `expect` here as the request id should always be in the DB
        let entry = db.remove(&id).expect("ID not found in db. This is a bug.");
        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry.response_tx.send(Err(error.clone())).unwrap_or(());
    });
}

/// Send `generated_text` to the Batcher for all `finished`
fn send_generated(finished: Vec<GeneratedText>, db: &Db) {
    finished.into_iter().for_each(|output| {
        // We can `expect` here as the request id should always be in the DB
        let entry = db
            .remove(&output.request.unwrap().id)
            .expect("ID not found in db. This is a bug.");
        let response = InferResponse {
            output: output.output,
            tokens: output.tokens,
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
    pub(crate) output: String,
    pub(crate) tokens: u32,
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
