/// This code is massively inspired by Tokio mini-redis
use crate::infer::InferError;
use crate::infer::InferStreamResponse;
use crate::validation::ValidGenerateRequest;
use nohash_hasher::{BuildNoHashHasher, IntMap};
use std::cmp::min;
use text_generation_client::{Batch, Request};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit};
use tokio::time::Instant;

/// Database entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: ValidGenerateRequest,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: UnboundedSender<Result<InferStreamResponse, InferError>>,
    /// Instant when this entry was created
    pub time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
    /// Permit
    pub _permit: OwnedSemaphorePermit,
}

/// Request Database
#[derive(Debug, Clone)]
pub(crate) struct Db {
    /// Channel to communicate with the background database task
    sender: UnboundedSender<DatabaseCommand>,
}

impl Db {
    pub(crate) fn new() -> Self {
        // Create channel
        let (db_sender, db_receiver) = mpsc::unbounded_channel();

        // Launch background database task
        tokio::spawn(database_task(db_receiver));

        Self { sender: db_sender }
    }

    /// Append an entry to the database
    pub(crate) fn append(&self, entry: Entry) {
        // Send append command to the background task managing the state
        // Unwrap is safe here
        self.sender.send(DatabaseCommand::Append(entry)).unwrap();
    }

    // Get the next batch
    pub(crate) async fn next_batch(
        &self,
        min_size: Option<usize>,
        max_size: usize,
    ) -> Option<NextBatch> {
        // Create response channel
        let (sender, receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.sender
            .send(DatabaseCommand::NextBatch {
                min_size,
                max_size,
                response_rx: sender,
            })
            .unwrap();
        // Await on response channel
        // Unwrap is safe here
        receiver.await.unwrap()
    }
}

// Background task responsible of the database state
async fn database_task(mut receiver: UnboundedReceiver<DatabaseCommand>) {
    let mut state = State::new();

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            DatabaseCommand::Append(entry) => state.append(entry),
            DatabaseCommand::NextBatch {
                min_size,
                max_size,
                response_rx,
            } => {
                let next_batch = state.next_batch(min_size, max_size);
                response_rx.send(next_batch).unwrap_or(());
            }
        }
    }
}

/// Database State
#[derive(Debug)]
struct State {
    /// Database entries organized in a Vec
    entries: Vec<(u64, Entry)>,

    /// Id of the next entry
    next_id: u64,

    /// Id of the next batch
    next_batch_id: u64,
}

impl State {
    fn new() -> Self {
        Self {
            entries: Vec::with_capacity(128),
            next_id: 0,
            next_batch_id: 0,
        }
    }

    /// Append an entry to the database
    fn append(&mut self, entry: Entry) {
        self.entries.push((self.next_id, entry));
        self.next_id += 1;
    }

    // Get the next batch
    fn next_batch(&mut self, min_size: Option<usize>, max_size: usize) -> Option<NextBatch> {
        // Check if we have enough entries in DB by comparing next batch id and current id
        if let Some(min_size) = min_size {
            if self.entries.len() < min_size {
                return None;
            }
        }

        // If both ids are equal, the DB is empty
        if self.entries.is_empty() {
            return None;
        }

        let next_batch_size = min(self.entries.len(), max_size);

        // Iterates for max_size over the BTreemap starting from next_batch_start_id
        let mut batch_requests = Vec::new();
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(next_batch_size, BuildNoHashHasher::default());

        self.entries
            .drain(..next_batch_size)
            .for_each(|(id, mut entry)| {
                batch_requests.push(Request {
                    id,
                    inputs: entry.request.inputs.clone(),
                    input_length: entry.request.input_length,
                    parameters: Some(entry.request.parameters.clone()),
                    stopping_parameters: Some(entry.request.stopping_parameters.clone()),
                });
                // Set batch_time
                entry.batch_time = Some(Instant::now());
                // Insert in entries IntMap
                batch_entries.insert(id, entry);
            });

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size: next_batch_size as u32,
        };
        // Increment batch id
        self.next_batch_id += 1;

        Some((batch_entries, batch))
    }
}

type NextBatch = (IntMap<u64, Entry>, Batch);

#[derive(Debug)]
enum DatabaseCommand {
    Append(Entry),
    NextBatch {
        min_size: Option<usize>,
        max_size: usize,
        response_rx: oneshot::Sender<Option<NextBatch>>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use text_generation_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
    use tokio::sync::{mpsc, Semaphore};
    use std::sync::Arc;

    fn default_entry() -> Entry {
        let semaphore = Arc::new(Semaphore::new(1));
        let (response_tx, _) = mpsc::unbounded_channel();
        let permit = semaphore.try_acquire_owned().unwrap();

        Entry {
            request: ValidGenerateRequest {
                inputs: "".to_string(),
                input_length: 0,
                parameters: NextTokenChooserParameters {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 0.0,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 0.0,
                },
                stopping_parameters: StoppingCriteriaParameters {
                    max_new_tokens: 0,
                    stop_sequences: vec![],
                },
            },
            response_tx,
            time: Instant::now(),
            batch_time: None,
            _permit: permit,
        }
    }

    #[test]
    fn test_append() {
        let mut state = State::new();
        let entry = default_entry();

        assert_eq!(state.next_id, 0);
        assert_eq!(state.entries.len(), 0);

        state.append(entry);

        assert_eq!(state.next_id, 1);
        assert_eq!(state.entries.len(), 1);
        let (id, _) = state.entries.remove(0);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_next_batch_empty() {
        let mut state = State::new();

        assert!(state.next_batch(None, 1).is_none());
        assert!(state.next_batch(Some(1), 1).is_none());
    }

    #[test]
    fn test_next_batch_min_size() {
        let mut state = State::new();
        state.append(default_entry());
        state.append(default_entry());

        let (entries, batch) = state.next_batch(None, 2).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        assert_eq!(state.next_id, 2);
        assert_eq!(state.entries.len(), 0);
        assert_eq!(state.next_batch_id, 1);

        state.append(default_entry());

        assert!(state.next_batch(Some(2), 2).is_none());

        assert_eq!(state.next_id, 3);
        assert_eq!(state.entries.len(), 1);
        let (id, _) = state.entries.remove(0);
        assert_eq!(id, 2);
    }

    #[test]
    fn test_next_batch_max_size() {
        let mut state = State::new();
        state.append(default_entry());
        state.append(default_entry());

        let (entries, batch) = state.next_batch(None, 1).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        assert_eq!(state.next_id, 2);
        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.next_batch_id, 1);

        state.append(default_entry());

        let (entries, batch) = state.next_batch(None, 3).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);

        assert_eq!(state.next_id, 3);
        assert_eq!(state.entries.len(), 0);
        assert_eq!(state.next_batch_id, 2);
    }

    #[tokio::test]
    async fn test_db_append() {
        let db = Db::new();
        db.append(default_entry());
    }

    #[tokio::test]
    async fn test_db_next_batch_empty() {
        let db = Db::new();

        assert!(db.next_batch(None, 1).await.is_none());
        assert!(db.next_batch(Some(1), 1).await.is_none());
    }

    #[tokio::test]
    async fn test_db_next_batch_min_size() {
        let db = Db::new();
        db.append(default_entry());
        db.append(default_entry());

        let (entries, batch) = db.next_batch(None, 2).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        db.append(default_entry());

        assert!(db.next_batch(Some(2), 2).await.is_none());
    }

    #[tokio::test]
    async fn test_db_next_batch_max_size() {
        let db = Db::new();
        db.append(default_entry());
        db.append(default_entry());

        let (entries, batch) = db.next_batch(None, 1).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        db.append(default_entry());

        let (entries, batch) = db.next_batch(None, 3).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);
    }
}
