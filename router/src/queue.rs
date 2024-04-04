use crate::infer::InferError;
use crate::infer::InferStreamResponse;
use crate::validation::ValidGenerateRequest;
use nohash_hasher::{BuildNoHashHasher, IntMap};
use std::cmp::min;
use std::cmp::{Eq, Ord, PartialEq, PartialOrd};
use std::collections::BinaryHeap;
use std::env;
use std::time::Duration;
use text_generation_client::{Batch, Request};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

/// Queue entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: ValidGenerateRequest,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>,
    /// Span that will live as long as entry
    pub span: Span,
    /// Temporary span used as a guard when logging inference, wait times...
    pub temp_span: Option<Span>,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
}

/// Request Queue
#[derive(Debug, Clone)]
pub(crate) struct Queue {
    /// Channel to communicate with the background queue task
    queue_sender: mpsc::UnboundedSender<QueueCommand>,
}

impl Queue {
    pub(crate) fn new(
        requires_padding: bool,
        max_input_length: u32,
        max_total_tokens: u32,
        block_size: u32,
        window_size: Option<u32>
    ) -> Self {
        // Create channel
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::spawn(queue_task(
            requires_padding,
            max_input_length,
            max_total_tokens,
            block_size,
            window_size,
            queue_receiver,
        ));

        Self { queue_sender }
    }

    /// Append an entry to the queue
    #[instrument(skip_all)]
    pub(crate) fn append(&self, entry: Entry) {
        // Send append command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::Append(Box::new(entry), Span::current()))
            .unwrap();
    }

    // Get the next batch
    #[instrument(skip(self))]
    pub(crate) async fn next_batch(
        &self,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::NextBatch {
                min_size,
                prefill_token_budget,
                token_budget,
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
async fn queue_task(
    requires_padding: bool,
    max_input_length: u32,
    max_total_tokens: u32,
    block_size: u32,
    window_size: Option<u32>,
    mut receiver: mpsc::UnboundedReceiver<QueueCommand>,
) {
    let mut state = State::new(
        requires_padding,
        max_input_length,
        max_total_tokens,
        block_size,
        window_size
    );

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            QueueCommand::Append(entry, span) => {
                span.in_scope(|| state.append(*entry));
                metrics::increment_gauge!("tgi_queue_size", 1.0);
            }
            QueueCommand::NextBatch {
                min_size,
                prefill_token_budget,
                token_budget,
                response_sender,
                span,
            } => span.in_scope(|| {
                let next_batch = state.next_batch(min_size, prefill_token_budget, token_budget);
                response_sender.send(next_batch).unwrap();
                metrics::gauge!("tgi_queue_size", state.entries.len() as f64);
            }),
        }
    }
}

#[derive(Debug)]
struct IdentifiableEntry(u64, Entry);

impl Eq for IdentifiableEntry {}

impl PartialEq for IdentifiableEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Ord for IdentifiableEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let ordering = match self
            .1
            .request
            .input_length
            .cmp(&other.1.request.input_length)
        {
            std::cmp::Ordering::Equal => self.0.cmp(&other.0),
            any => any,
        };

        // inverse to get min heap
        return ordering.reverse();
    }
}

impl PartialOrd for IdentifiableEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct QueueImpl {
    regular_entries: BinaryHeap<IdentifiableEntry>,
    overdue_entries: BinaryHeap<IdentifiableEntry>,
    overdue_threshold: Duration,
}

impl QueueImpl {
    fn new(capacity: usize, overdue_threshold: Duration) -> Self {
        Self {
            regular_entries: BinaryHeap::with_capacity(capacity),
            overdue_entries: BinaryHeap::with_capacity(capacity),
            overdue_threshold,
        }
    }

    fn update(&mut self) {
        if self.regular_entries.is_empty() {
            return;
        }

        let mut left = BinaryHeap::with_capacity(self.regular_entries.capacity());

        for entry in self.regular_entries.drain() {
            if entry.1.queue_time.elapsed() > self.overdue_threshold {
                self.overdue_entries.push(entry);
            } else {
                left.push(entry);
            }
        }

        self.regular_entries = left;
    }

    fn push(&mut self, entry: IdentifiableEntry) {
        if entry.1.queue_time.elapsed() > self.overdue_threshold {
            self.overdue_entries.push(entry);
        } else {
            self.regular_entries.push(entry);
        }
    }

    fn pop(&mut self) -> Option<IdentifiableEntry> {
        if !self.overdue_entries.is_empty() {
            self.overdue_entries.pop()
        } else {
            self.regular_entries.pop()
        }
    }

    fn is_empty(&self) -> bool {
        self.regular_entries.is_empty() && self.overdue_entries.is_empty()
    }

    fn len(&self) -> usize {
        self.regular_entries.len() + self.overdue_entries.len()
    }
}

/// Queue State
#[derive(Debug)]
struct State {
    /// Queue entries
    entries: QueueImpl,

    /// Id of the next entry
    next_id: u64,

    /// Id of the next batch
    next_batch_id: u64,

    /// Whether the model is using padding
    requires_padding: bool,

    /// Maximum input length, required for padding scenario
    max_input_length: u32,

    /// Maximum input and output length, required for padding scenario
    max_total_tokens: u32,

    /// Paged Attention block size
    block_size: u32,

    /// Sliding window
    window_size: Option<u32>,
}

impl State {
    fn new(
        requires_padding: bool,
        max_input_length: u32,
        max_total_tokens: u32,
        block_size: u32,
        window_size: Option<u32>,
    ) -> Self {
        let default_threshold: u64 = 120;
        let threshold: u64 = match env::var("QUEUE_THRESHOLD_MS") {
            Ok(val) => val.parse().unwrap_or(default_threshold),
            Err(_) => default_threshold,
        };

        Self {
            entries: QueueImpl::new(128, Duration::from_millis(threshold)),
            next_id: 0,
            next_batch_id: 0,
            requires_padding,
            max_input_length,
            max_total_tokens,
            block_size,
            window_size,
        }
    }

    /// Append an entry to the queue
    fn append(&mut self, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        // Push entry in the queue
        self.entries.push(IdentifiableEntry(self.next_id, entry));
        self.next_id += 1;
    }

    // Get the next batch
    fn next_batch(
        &mut self,
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        if self.entries.is_empty() {
            return None;
        }

        // Check if we have enough entries
        if let Some(min_size) = min_size {
            if self.entries.len() < min_size {
                return None;
            }
        }

        self.entries.update();

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(&Span::current());

        let mut batch_requests = Vec::with_capacity(self.entries.len());
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(self.entries.len(), BuildNoHashHasher::default());

        let mut prefill_tokens: u32 = 0;
        let mut decode_tokens: u32 = 0;

        // Pop entries starting from the front of the queue
        while let Some(IdentifiableEntry(id, mut entry)) = self.entries.pop() {
            // Filter entries where the response receiver was dropped (== entries where the request
            // was dropped by the client)
            if entry.response_tx.is_closed() {
                metrics::increment_counter!("tgi_request_failure", "err" => "dropped");
                continue;
            }

            if self.requires_padding {
                // We pad to max input length in the Python shards
                // We need to take these padding tokens into the equation
                prefill_tokens = (batch_requests.len() + 1) as u32 * self.max_input_length;
            } else {
                // pad to block size
                prefill_tokens += ((entry.request.input_length + self.block_size - 1)
                    / self.block_size)
                    * self.block_size;
            }

            if self.requires_padding {
                // We pad to max total tokens in the Python shards
                // We need to take these padding tokens into the equation
                decode_tokens = (batch_requests.len() + 1) as u32 * (self.max_total_tokens - self.max_input_length);
            } else {
                let max_new_tokens = match self.window_size {
                    None => entry.request.stopping_parameters.max_new_tokens,
                    Some(window_size) => min(
                        window_size.saturating_sub(entry.request.input_length),
                        entry.request.stopping_parameters.max_new_tokens,
                    ),
                };

                // pad to block size
                decode_tokens +=
                    ((max_new_tokens + self.block_size - 1) / self.block_size) * self.block_size;
            }

            if prefill_tokens > prefill_token_budget
                || (prefill_tokens + decode_tokens) > token_budget
            {
                // Entry is over budget
                // Add it back to the front
                self.entries.push(IdentifiableEntry(id, entry));
                break;
            }

            // Create a new span to link the batch back to this entry
            let entry_batch_span = info_span!(parent: &entry.span, "infer");
            // Add relationships
            next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);

            batch_requests.push(Request {
                id,
                prefill_logprobs: entry.request.decoder_input_details,
                inputs: entry.request.inputs.clone(),
                truncate: entry.request.truncate,
                parameters: Some(entry.request.parameters.clone()),
                stopping_parameters: Some(entry.request.stopping_parameters.clone()),
                top_n_tokens: entry.request.top_n_tokens,
            });
            // Set batch_time
            entry.batch_time = Some(Instant::now());
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
        }

        // Empty batch
        if batch_requests.is_empty() {
            return None;
        }

        // Check if our batch is big enough
        if let Some(min_size) = min_size {
            // Batch is too small
            if batch_requests.len() < min_size {
                // Add back entries to the queue in the correct order
                for r in batch_requests.into_iter().rev() {
                    let id = r.id;
                    let entry = batch_entries.remove(&id).unwrap();
                    self.entries.push(IdentifiableEntry(id, entry));
                }

                return None;
            }
        }

        // Final batch size
        let size = batch_requests.len() as u32;
        next_batch_span.record("batch_size", size);

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size,
            max_tokens: (prefill_tokens + decode_tokens),
        };
        // Increment batch id
        self.next_batch_id += 1;

        metrics::histogram!("tgi_batch_next_size", batch.size as f64);

        Some((batch_entries, batch, next_batch_span))
    }
}

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>, Span),
    NextBatch {
        min_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use text_generation_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
    use tracing::info_span;

    fn default_queue() -> Queue {
        Queue::new(
            true, 1, 2, 1, None
        )
    }

    fn default_state() -> State {
        State::new(
            true, 1, 2, 1, None
        )
    }

    fn default_entry() -> (
        Entry,
        mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>,
    ) {
        let (response_tx, receiver_tx) = mpsc::unbounded_channel();

        let entry = Entry {
            request: ValidGenerateRequest {
                inputs: "".to_string(),
                input_length: 0,
                truncate: 0,
                decoder_input_details: false,
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
                    ignore_eos_token: false,
                    max_new_tokens: 1,
                    stop_sequences: vec![],
                },
                top_n_tokens: 0,
            },
            response_tx,
            span: info_span!("entry"),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
        };
        (entry, receiver_tx)
    }

    #[test]
    fn test_append() {
        let mut state = default_state();
        let (entry, _guard) = default_entry();

        assert_eq!(state.next_id, 0);
        assert_eq!(state.entries.len(), 0);

        state.append(entry);

        assert_eq!(state.next_id, 1);
        assert_eq!(state.entries.len(), 1);
        let id = state.entries.pop().unwrap().0;
        assert_eq!(id, 0);
    }

    #[test]
    fn test_next_batch_empty() {
        let mut state = default_state();

        assert!(state.next_batch(None, 1, 1).is_none());
        assert!(state.next_batch(Some(1), 1, 1).is_none());
    }

    #[test]
    fn test_next_batch_min_size() {
        let mut state = default_state();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.append(entry1);
        state.append(entry2);

        let (entries, batch, _) = state.next_batch(None, 2, 4).unwrap();
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

        let (entry3, _guard3) = default_entry();
        state.append(entry3);

        assert!(state.next_batch(Some(2), 2, 2).is_none());

        assert_eq!(state.next_id, 3);
        assert_eq!(state.entries.len(), 1);
        let IdentifiableEntry(id, _) = state.entries.pop().unwrap();
        assert_eq!(id, 2);
    }

    #[test]
    fn test_next_batch_token_budget() {
        let mut state = default_state();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.append(entry1);
        state.append(entry2);

        let (entries, batch, _) = state.next_batch(None, 1, 2).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        assert_eq!(state.next_id, 2);
        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.next_batch_id, 1);

        let (entry3, _guard3) = default_entry();
        state.append(entry3);

        let (entries, batch, _) = state.next_batch(None, 3, 6).unwrap();
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
        let queue = default_queue();
        let (entry, _guard) = default_entry();
        queue.append(entry);
    }

    #[tokio::test]
    async fn test_queue_next_batch_empty() {
        let queue = default_queue();

        assert!(queue.next_batch(None, 1, 1).await.is_none());
        assert!(queue.next_batch(Some(1), 1, 1).await.is_none());
    }

    #[tokio::test]
    async fn test_queue_next_batch_min_size() {
        let queue = default_queue();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        let (entries, batch, _) = queue.next_batch(None, 2, 4).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert!(entries.get(&1).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);

        let (entry3, _guard3) = default_entry();
        queue.append(entry3);

        // Not enough requests pending
        assert!(queue.next_batch(Some(2), 2, 2).await.is_none());
        // Not enough token budget
        assert!(queue.next_batch(Some(1), 0, 0).await.is_none());
        // Ok
        let (entries2, batch2, _) = queue.next_batch(Some(1), 1, 2).await.unwrap();
        assert_eq!(entries2.len(), 1);
        assert!(entries2.contains_key(&2));
        assert!(entries2.get(&2).unwrap().batch_time.is_some());
        assert_eq!(batch2.id, 1);
        assert_eq!(batch2.size, 1);
    }

    #[tokio::test]
    async fn test_queue_next_batch_token_budget() {
        let queue = default_queue();
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        let (entries, batch, _) = queue.next_batch(None, 1, 2).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        let (entry3, _guard3) = default_entry();
        queue.append(entry3);

        let (entries, batch, _) = queue.next_batch(None, 2, 4).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);
    }

    #[tokio::test]
    async fn test_queue_next_batch_dropped_receiver() {
        let queue = default_queue();
        let (entry, _) = default_entry();
        queue.append(entry);

        assert!(queue.next_batch(None, 1, 1).await.is_none());
    }
}
