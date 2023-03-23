use crate::infer::InferError;
use crate::infer::InferStreamResponse;
use crate::validation::ValidGenerateRequest;
use nohash_hasher::{BuildNoHashHasher, IntMap};
use std::cmp::min;
use text_generation_client::{Batch, Request};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit};
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

/// Queue entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: ValidGenerateRequest,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: UnboundedSender<Result<InferStreamResponse, InferError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
    /// Permit
    pub _permit: OwnedSemaphorePermit,
}

/// Request Queue
#[derive(Debug, Clone)]
pub(crate) struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: UnboundedSender<QueueCommand>,
}

impl Queue {
    pub(crate) fn new() -> Self {
        // Create channel
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::spawn(queue_task(queue_receiver));

        Self { queue_sender }
    }

    /// Append an entry to the queue
    #[instrument(skip_all)]
    pub(crate) fn append(&self, entry: Entry) {
        // Send append command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::Append(entry, Span::current()))
            .unwrap();
    }

    // Get the next batch
    #[instrument(skip(self))]
    pub(crate) async fn next_batch(
        &self,
        min_size: Option<usize>,
        max_size: usize,
    ) -> Option<NextBatch> {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::NextBatch {
                min_size,
                max_size,
                response_sender,
                span: Span::current(),
            })
            .unwrap();
        // Await on response channel
        // Unwrap is safe here
        response_receiver.await.unwrap()
    }
}

// Background task responsible of the queue state
async fn queue_task(mut receiver: UnboundedReceiver<QueueCommand>) {
    let mut state = State::new();

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            QueueCommand::Append(entry, span) => span.in_scope(|| state.append(entry)),
            QueueCommand::NextBatch {
                min_size,
                max_size,
                response_sender,
                span,
            } => span.in_scope(|| {
                let next_batch = state.next_batch(min_size, max_size);
                response_sender.send(next_batch).unwrap_or(());
            }),
        }
    }
}

/// Queue State
#[derive(Debug)]
struct State {
    /// Queue entries organized in a Vec
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

    /// Append an entry to the queue
    fn append(&mut self, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        // Push entry in the queue
        self.entries.push((self.next_id, entry));
        self.next_id += 1;
        metrics::increment_gauge!("tgi_queue_size", 1.0);
    }

    // Get the next batch
    fn next_batch(&mut self, min_size: Option<usize>, max_size: usize) -> Option<NextBatch> {
        if self.entries.is_empty() {
            return None;
        }

        // Check if we have enough entries
        if let Some(min_size) = min_size {
            if self.entries.len() < min_size {
                return None;
            }
        }

        let next_batch_size = min(self.entries.len(), max_size);

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = next_batch_size);
        next_batch_span.follows_from(&Span::current());

        let mut batch_requests = Vec::with_capacity(next_batch_size);
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(next_batch_size, BuildNoHashHasher::default());

        // Drain next_batch_size entries
        self.entries
            .drain(..next_batch_size)
            .for_each(|(id, mut entry)| {
                // Create a new span to link the batch back to this entry
                let entry_batch_span =
                    info_span!(parent: &entry.span, "infer", batch_size = next_batch_size);
                // Add relationships
                next_batch_span.follows_from(&entry_batch_span);
                entry_batch_span.follows_from(&next_batch_span);
                // Update entry
                entry.temp_span = Some(entry_batch_span);

                batch_requests.push(Request {
                    id,
                    inputs: entry.request.inputs.clone(),
                    parameters: Some(entry.request.parameters.clone()),
                    stopping_parameters: Some(entry.request.stopping_parameters.clone()),
                });
                // Set batch_time
                entry.batch_time = Some(Instant::now());
                // Insert in batch_entries IntMap
                batch_entries.insert(id, entry);
            });

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size: next_batch_size as u32,
        };
        // Increment batch id
        self.next_batch_id += 1;

        metrics::gauge!("tgi_queue_size", self.entries.len() as f64);
        metrics::histogram!("tgi_batch_next_size", batch.size as f64);
        Some((batch_entries, batch, next_batch_span))
    }
}

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

#[derive(Debug)]
enum QueueCommand {
    Append(Entry, Span),
    NextBatch {
        min_size: Option<usize>,
        max_size: usize,
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use text_generation_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
    use tokio::sync::{mpsc, Semaphore};
    use tracing::info_span;

    fn default_entry() -> Entry {
        let semaphore = Arc::new(Semaphore::new(1));
        let (response_tx, _) = mpsc::unbounded_channel();
        let permit = semaphore.try_acquire_owned().unwrap();

        Entry {
            request: ValidGenerateRequest {
                inputs: "".to_string(),
                parameters: NextTokenChooserParameters {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 0.0,
                    typical_p: 0.0,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 0.0,
                    watermark: false,
                },
                stopping_parameters: StoppingCriteriaParameters {
                    max_new_tokens: 0,
                    stop_sequences: vec![],
                },
            },
            response_tx,
            span: info_span!("entry"),
            temp_span: None,
            queue_time: Instant::now(),
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

        let (entries, batch, _) = state.next_batch(None, 2).unwrap();
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

        let (entries, batch, _) = state.next_batch(None, 1).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        assert_eq!(state.next_id, 2);
        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.next_batch_id, 1);

        state.append(default_entry());

        let (entries, batch, _) = state.next_batch(None, 3).unwrap();
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
    async fn test_queue_append() {
        let queue = Queue::new();
        queue.append(default_entry());
    }

    #[tokio::test]
    async fn test_queue_next_batch_empty() {
        let queue = Queue::new();

        assert!(queue.next_batch(None, 1).await.is_none());
        assert!(queue.next_batch(Some(1), 1).await.is_none());
    }

    #[tokio::test]
    async fn test_queue_next_batch_min_size() {
        let queue = Queue::new();
        queue.append(default_entry());
        queue.append(default_entry());

        let (entries, batch, _) = queue.next_batch(None, 2).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        queue.append(default_entry());

        assert!(queue.next_batch(Some(2), 2).await.is_none());
    }

    #[tokio::test]
    async fn test_queue_next_batch_max_size() {
        let queue = Queue::new();
        queue.append(default_entry());
        queue.append(default_entry());

        let (entries, batch, _) = queue.next_batch(None, 1).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        queue.append(default_entry());

        let (entries, batch, _) = queue.next_batch(None, 3).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);
    }
}
