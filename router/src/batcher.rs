use crate::server::GenerateRequest;
use crate::{Db, Entry};
use axum::http::StatusCode;
use bloom_inference_client::{Batch, ClientError, GeneratedText, ShardedClient};
use std::future::Future;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{oneshot, Notify};

const MAX_LENGTH: usize = 128;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
    #[error("Model is overloaded")]
    Overloaded,
}

impl From<InferError> for (StatusCode, String) {
    fn from(err: InferError) -> Self {
        match err {
            InferError::GenerationError(_) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
            InferError::Overloaded => (StatusCode::TOO_MANY_REQUESTS, err.to_string()),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Batcher {
    db: Db,
    shared: Arc<Shared>,
}

struct Shared {
    batching_task: Notify,
}

impl Batcher {
    pub(crate) fn new(client: ShardedClient) -> Self {
        let db = Db::new();
        let shared = Arc::new(Shared {
            batching_task: Notify::new(),
        });

        tokio::spawn(batching_task(client, db.clone(), shared.clone()));

        Self { db, shared }
    }

    pub(crate) async fn infer(
        &self,
        input_length: usize,
        request: GenerateRequest,
    ) -> Result<String, InferError> {
        if self.db.len() > MAX_LENGTH {
            return Err(InferError::Overloaded);
        }
        let (request_tx, request_rx) = oneshot::channel();
        self.db.append(Entry {
            request,
            response_tx: request_tx,
            input_length,
        });
        self.shared.batching_task.notify_waiters();
        match request_rx.await.unwrap() {
            Ok(output) => Ok(output),
            Err(err) => Err(InferError::GenerationError(err.to_string())),
        }
    }
}

async fn batching_task(client: ShardedClient, db: Db, shared: Arc<Shared>) {
    loop {
        shared.batching_task.notified().await;

        if let Some(batch) = db.next_batch(32) {
            let request_ids = batch.requests.iter().map(|req| req.id).collect();
            let mut cached_batch = match batch.size {
                size if size > 16 => {
                    wrap_future(client.generate_until_finished(batch), request_ids, &db).await
                }
                _ => wrap_future(client.generate(batch), request_ids, &db).await,
            };

            while let Some(batch) = cached_batch {
                let batch_size = batch.size;
                let mut request_ids: Vec<u64> = batch.requests.iter().map(|req| req.id).collect();
                let mut batches = vec![batch];

                if batch_size <= 16 {
                    if let Some(new_batch) = db.next_batch_minimum_size(16, 48) {
                        let new_batch_request_ids =
                            new_batch.requests.iter().map(|req| req.id).collect();
                        let new_cached_batch =
                            wrap_future(client.generate(new_batch), new_batch_request_ids, &db)
                                .await;
                        if let Some(new_cached_batch) = new_cached_batch {
                            request_ids.extend(new_cached_batch.requests.iter().map(|req| req.id));
                            batches.push(new_cached_batch);
                        }
                    }
                }

                cached_batch = match batch_size {
                    size if size > 16 => {
                        wrap_future(
                            client.generate_until_finished_with_cache(batches),
                            request_ids,
                            &db,
                        )
                        .await
                    }
                    _ => wrap_future(client.generate_with_cache(batches), request_ids, &db).await,
                };
            }
        }
    }
}

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
        Err(err) => {
            send_error(err, request_ids, db);
            None
        }
    }
}

fn send_error(error: ClientError, request_ids: Vec<u64>, db: &Db) {
    request_ids.into_iter().for_each(|id| {
        let entry = db.remove(&id).expect("ID not found in db. This is a bug.");
        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry.response_tx.send(Err(error.clone())).unwrap_or(());
    });
}

fn send_generated(finished: Vec<GeneratedText>, db: &Db) {
    finished.into_iter().for_each(|output| {
        let entry = db
            .remove(&output.request.unwrap().id)
            .expect("ID not found in db. This is a bug.");
        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry.response_tx.send(Ok(output.output)).unwrap_or(());
    });
}
