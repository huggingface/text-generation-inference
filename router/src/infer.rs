use crate::{Db, GenerateRequest};
use bloom_inference_client::{Batch, BatchCached, CacheEntry, ClientError, FinishedGeneration, ShardedClient};
use std::sync::Arc;
use tokio::sync::{oneshot, Notify};

const MAX_LENGTH: usize = 128;

pub struct InferError {}

#[derive(Clone)]
pub(crate) struct Infer {
    db: Db,
    shared: Arc<Shared>,
}

struct Shared {
    batching_task: Notify,
}

impl Infer {
    pub(crate) fn new(client: ShardedClient) -> Self {
        let db = Db::new();
        let shared = Arc::new(Shared {
            batching_task: Notify::new(),
        });

        tokio::spawn(batching_task(client, db.clone(), shared.clone()));

        Self { db, shared }
    }

    pub(crate) async fn infer(&self, request: GenerateRequest) -> Result<String, InferError> {
        if self.db.len() > MAX_LENGTH {
            return Err(InferError {});
        }
        let (request_tx, request_rx) = oneshot::channel();
        self.db.append(request, request_tx);
        self.shared.batching_task.notify_waiters();
        match request_rx.await.unwrap() {
            Ok(output) => Ok(output),
            Err(_) => Err(InferError {})
        }
    }
}

async fn batching_task(client: ShardedClient, db: Db, shared: Arc<Shared>) {
    loop {
        shared.batching_task.notified().await;

        if let Some(batch) = db.next_batch(32) {
            let mut cache_entry = infer_batch(batch, &client, &db).await;

            loop {
                if let Some(entry) = cache_entry {
                    let mut batch_cached_ids = vec![entry.id];
                    let mut total_batch_size = entry.request_ids.len();
                    let mut max_sequence_length = entry.sequence_length;
                    let mut request_ids = entry.request_ids;

                    if total_batch_size <= 16 {
                        if let Some(batch) = db.next_batch_minimum_size(16, 48) {
                            let other_cache_entry = infer_batch(batch, &client, &db).await;

                            if let Some(entry) = other_cache_entry {
                                batch_cached_ids.push(entry.id);
                                total_batch_size += entry.request_ids.len();
                                max_sequence_length =
                                    max_sequence_length.max(entry.sequence_length);
                                request_ids.extend(entry.request_ids.into_iter());
                            }
                        }
                    }

                    let batch_cached = BatchCached {
                        id: entry.id,
                        batch_cached_ids,
                        total_batch_size: total_batch_size as u32,
                        max_sequence_length,
                        request_ids,
                    };
                    cache_entry = infer_batch_cached(batch_cached, &client, &db).await;
                } else {
                    break;
                }
            }
        }
    }
}

async fn infer_batch_cached(batch: BatchCached, client: &ShardedClient, db: &Db) -> Option<CacheEntry> {
    match client.generate_with_cache(batch.clone()).await {
        Ok((finished, cache_entry)) => {
            send_finished(finished, db);
            cache_entry
        }
        Err(err) => {
            println!("{:?}", err);
            send_error(err, batch.request_ids, &db);
            None
        }
    }
}

async fn infer_batch(batch: Batch, client: &ShardedClient, db: &Db) -> Option<CacheEntry> {
    match client.generate(batch.clone()).await {
        Ok((finished, cache_entry)) => {
            send_finished(finished, db);
            cache_entry
        }
        Err(err) => {
            println!("{:?}", err);
            send_error(err, batch.requests.into_iter().map(|req| req.id).collect(), &db);
            None
        }
    }
}

fn send_error(error: ClientError, request_ids: Vec<u64>, db: &Db) {
    request_ids.into_iter().for_each(|id| {
        let (_, response_tx) = db.remove(&id).unwrap();
        response_tx.send(Err(error.clone())).unwrap_or(());
    });
}

fn send_finished(finished: Vec<FinishedGeneration>, db: &Db) {
    finished.into_iter().for_each(|output| {
        let (_, response_tx) = db.remove(&output.id).unwrap();
        response_tx.send(Ok(output.output)).unwrap_or(());
    });
}
