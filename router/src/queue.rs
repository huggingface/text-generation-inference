use crate::infer::InferError;
use crate::infer::InferStreamResponse;
use crate::validation::ValidGenerateRequest;
use nohash_hasher::{BuildNoHashHasher, IntMap};
use std::collections::{BTreeSet, VecDeque};
use std::ops::Add;
use std::time::Duration;
use text_generation_client::{Batch, Request};
use tokio::sync::oneshot;
use tokio::time::Instant;
use tracing::{info_span, instrument, Span};

/// Queue entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: ValidGenerateRequest,
    /// Count of tokens generated so far
    pub generated_tokens: usize,
    /// Response sender to communicate between the Infer struct and the batching_task
    pub response_tx: flume::Sender<Result<InferStreamResponse, InferError>>,
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
    queue_sender: flume::Sender<QueueCommand>,
}

impl Queue {
    pub(crate) fn new(config: BatchingConfig) -> Self {
        // Create channel
        let (queue_sender, queue_receiver) = flume::unbounded();

        // Launch background queue task
        tokio::spawn(queue_task(queue_receiver, config));

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

    // Get the next batch - existing batch is returned unchanged
    #[instrument(skip(self))]
    pub(crate) async fn next_batch(
        &self,
        entries: Option<ExistingBatch>,
    ) -> (Option<ExistingBatch>, Option<NextBatch>) {
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::NextBatch {
                entries,
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
async fn queue_task(receiver: flume::Receiver<QueueCommand>, config: BatchingConfig) {
    let mut state = State::new(config);

    while let Ok(cmd) = receiver.recv_async().await {
        match cmd {
            QueueCommand::Append(entry, span) => span.in_scope(|| state.append(entry)),
            QueueCommand::NextBatch {
                entries,
                response_sender,
                span,
            } => span.in_scope(|| {
                let response = state.next_batch(entries);
                response_sender.send(response).unwrap_or(());
            }),
        }
    }
}

#[derive(Debug)]
pub(crate) struct BatchingConfig {
    pub(crate) size_limit: usize,
    pub(crate) weight_limit: usize,
    pub(crate) prefill_weight_limit: usize,
}

/// Queue State
#[derive(Debug)]
struct State {
    /// Batching configuration
    config: BatchingConfig,

    /// Queue entries organized in a Vec
    entries: VecDeque<(u64, Entry)>,

    /// Id of the next entry
    next_id: u64,

    /// Id of the next batch
    next_batch_id: u64,

    // Remembered size of the last batch, used to determine
    // when entries have completed between calls to the
    // next_batch function
    last_seen_batch_size: usize,

    // Index in the queue up to which entries have been
    // checked to see if they can fit into the current batch.
    // Reset to zero when any existing entries complete
    checked_request_count: usize,

    /// true if it's known that the current size of the
    /// requests in the queue is too small to prefill an add-on batch
    buffer_contents_insufficient: bool,

    /// Just a constant empty map to reuse
    empty_map: ExistingBatch,
}

// Could also make these configurable

/// Longest that requests can be waiting before we ignore the minimum
/// size requirement when adding to a new batch
const MAX_WAITING_DURATION: Duration = Duration::from_secs(1);

/// Maximum difference in arrival time that smaller requests can jump
/// ahead of larger ones in the queue
const CUTOFF_DURATION: Duration = Duration::from_secs(1);

impl State {
    fn new(config: BatchingConfig) -> Self {
        Self {
            config,
            entries: VecDeque::with_capacity(128),
            next_id: 0,
            next_batch_id: 0,
            last_seen_batch_size: 0,
            checked_request_count: 0,
            buffer_contents_insufficient: false,
            empty_map: IntMap::default(),
        }
    }

    /// Append an entry to the queue
    fn append(&mut self, mut entry: Entry) {
        // Create a span that will live as long as the entry is in the queue waiting to be batched
        let queue_span = info_span!(parent: &entry.span, "queued");
        entry.temp_span = Some(queue_span);

        // Push entry in the queue
        self.entries.push_back((self.next_id, entry));
        self.next_id += 1;
        metrics::increment_gauge!("tgi_queue_size", 1.0);
    }

    // Get the next batch
    fn next_batch(
        &mut self, existing_entries_opt: Option<ExistingBatch>,
    ) -> (Option<ExistingBatch>, Option<NextBatch>) {

        // Use ref to empty map in None case to simplify subsequent logic
        let existing_entries = existing_entries_opt.as_ref().unwrap_or(&self.empty_map);

        let config = &self.config;
        let mut total_count = existing_entries.len();
        if total_count >= config.size_limit {
            // We are already at max batch size
            return (existing_entries_opt, None)
        }

        if total_count != self.last_seen_batch_size {
            // Reset the count of checked requests if any completed since last check
            self.checked_request_count = 0;
            self.last_seen_batch_size = total_count
        }

        // Filter entries where the response receiver was dropped (== entries where the request
        // was dropped by the client)
        let queue_len_before = self.entries.len();
        self.entries.retain_mut(|(_, entry)| !entry.response_tx.is_disconnected());
        if queue_len_before != self.entries.len() {
            // Reset the count of checked requests if any in the queue were cancelled since last check
            self.checked_request_count = 0;
        }

        // This will generally be zero, but if no requests have been completed
        // since last time, we don't need to reconsider those already checked
        let mut checked_up_to_index = self.checked_request_count;

        if !existing_entries.is_empty() {
            // If we don't have any new requests in the buffer to check
            if self.entries.len() <= checked_up_to_index ||
                // Or the current buffer isn't large enough to satisfy the min prefill requirement
                self.buffer_contents_insufficient && !self.next_entry_waiting_too_long() {
                return (existing_entries_opt, None)
            }
        }

        // Indices into buffer of entries chosen to add to next batch or remove due to expiry
        let mut chosen_indices = vec![];
        // Indices to drop due to client cancellation
        let mut indices_to_drop = vec![];
        let mut btree = None;
        let mut time_cutoff = None;
        let mut hit_prefill_weight_limit = false;

        let mut total_token_count = existing_entries.iter().map(
            |(_, e)| e.request.stopping_parameters.max_new_tokens + e.request.truncate
        ).sum::<u32>() as usize;

        let mut prefill_size = 0;
        // We first do a read-only pass over the queue to allow skipping over large entries
        // that don't fit in the current batch to reach smaller entries that do
        let mut queue_index = checked_up_to_index;
        'queue_loop: for (entry_id, entry) in self.entries.range(queue_index..) {
            if matches!(time_cutoff, Some(t) if entry.queue_time > t) {
                break
            }
            queue_index += 1;
            if entry.response_tx.is_disconnected() {
                // Eject cancelled entry from queue
                indices_to_drop.push(queue_index);
                continue
            }
            // This is the index into the queue after cancelled entries
            // have been pruned
            checked_up_to_index += 1;

            let input_len = entry.request.truncate as usize;
            let output_len = entry.request.stopping_parameters.max_new_tokens as usize;
            let next_total_token_count = total_token_count + input_len + output_len;

            // Avoid more granular analysis if possible
            if next_total_token_count > config.weight_limit {
                // We aren't sure whether this next request will fit, so populate
                // a btree with the current batch of requests, the set of
                // requests already evaluated, and this one, and perform more
                // granular analysis to verify that the batch shape won't exceed
                // the limit at any point

                // Allocate btree the first time it's required
                let tree = btree.get_or_insert_with(|| {
                    let mut t = Box::new(BTreeSet::new());
                    // Populate with records corresponding to all existing and pending entries
                    let pending = chosen_indices.iter()
                        .map(|i| self.entries.get(*i).unwrap())
                        .map(|(eid, e)| (eid, e));
                    for (eid, e) in existing_entries.iter().chain(pending) {
                        let generated_count = e.generated_tokens;
                        t.insert((
                            e.request.stopping_parameters.max_new_tokens as usize - generated_count,
                            e.request.truncate as usize + e.generated_tokens,
                            eid,
                        ));
                    }
                    t
                });
                // Add the current entry
                tree.insert((output_len, input_len, entry_id));

                // Perform analysis
                let mut in_sum = 0;
                // Work backwards from longest projected entry
                for (bs, (ol, il, _)) in tree.iter().rev().enumerate() {
                    let this_ol = *ol;
                    in_sum += *il;
                    if this_ol <= output_len {
                        // Check if we breach max space for this segment
                        let token_count = in_sum + (bs + 1) * this_ol;
                        if token_count > config.weight_limit {
                            // Remove our tuple from the set
                            tree.remove(&(output_len, input_len, entry_id));
                            time_cutoff.get_or_insert_with(|| entry.queue_time.add(CUTOFF_DURATION));
                            continue 'queue_loop
                        }
                    }
                }
            } else if let Some(tree) = btree.as_mut() {
                // If we initialized the btree for a prior request, keep it updated
                tree.insert((output_len, input_len, entry_id));
            }
            // Here, we can add this request to the batch without breaching memory limit

            if config.prefill_weight_limit > 0 {
                // Also check whether adding this request will make the batch of new requests
                // too expensive latency-wise to perform in a single forward-pass.
                if prefill_size + input_len > config.prefill_weight_limit {
                    if let Some(tree) = btree.as_mut() {
                        // Remove our tuple from the set
                        tree.remove(&(output_len, input_len, entry_id));
                        hit_prefill_weight_limit = true;
                    }
                    time_cutoff.get_or_insert_with(|| entry.queue_time.add(CUTOFF_DURATION));
                    continue
                }
            }

            total_token_count = next_total_token_count;
            prefill_size += input_len;

            chosen_indices.push(queue_index - 1);
            total_count += 1;
            if total_count >= config.size_limit {
                break
            }
        }

        // Drop any cancelled requests
        if !indices_to_drop.is_empty() {
            indices_to_drop.iter().for_each(|i| {
                self.entries.remove(*i);
            });
            metrics::gauge!("tgi_queue_size", self.entries.len() as f64);
        }

        let next_batch_size = chosen_indices.len();
        if next_batch_size == 0 {
            // This gets reset to zero when any requests in the existing batch are removed
            self.checked_request_count = checked_up_to_index;
            return (existing_entries_opt, None)
        }
        self.checked_request_count = 0;

        if !hit_prefill_weight_limit && !existing_entries.is_empty() {
            // If this is to be added to an existing batch, ensure it meets urgency or size
            // requirements to avoid too frequent prefills
            if !self.next_entry_waiting_too_long() {
                if total_token_count < config.weight_limit / 2 {
                    // Don't add this new batch yet because it's not large enough
                    self.checked_request_count = checked_up_to_index;
                    self.buffer_contents_insufficient = true;
                    return (existing_entries_opt, None)
                }
            }
        }

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = next_batch_size);
        next_batch_span.follows_from(&Span::current());

        let mut batch_entries =
            IntMap::with_capacity_and_hasher(next_batch_size, BuildNoHashHasher::default());

        let some_now = Some(Instant::now());
        let batch_requests = chosen_indices.iter().enumerate().map(|(i, index)| {
            let (id, mut entry) = self.entries.remove(index - i).expect("bug");
            // Create a new span to link the batch back to this entry
            let entry_batch_span =
                info_span!(parent: &entry.span, "infer", batch_size = next_batch_size);
            // Add relationships
            next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);

            let request = Request {
                id,
                inputs: entry.request.inputs.clone(),
                truncate: entry.request.truncate,
                parameters: Some(entry.request.parameters.clone()),
                stopping_parameters: Some(entry.request.stopping_parameters.clone()),
            };
            // Set batch_time
            entry.batch_time = some_now;
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
            request
        }).collect::<Vec<Request>>();

        metrics::gauge!("tgi_queue_size", self.entries.len() as f64);

        // Final batch size once we dropped entries
        let size = batch_requests.len() as u32;
        next_batch_span.record("batch_size", size);

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size,
        };
        // Increment batch id
        self.next_batch_id += 1;
        self.buffer_contents_insufficient = false;

        metrics::histogram!("tgi_batch_next_size", batch.size as f64);
        (existing_entries_opt, Some((batch_entries, batch, next_batch_span)))
    }

    /// Returns true if the entry at the front of the queue has been waiting for longer
    /// than MAX_WAITING_DURATION
    fn next_entry_waiting_too_long(&self) -> bool {
        matches!(
            self.entries.front(), Some((_, e)) if e.queue_time.elapsed() > MAX_WAITING_DURATION
        )
    }
}

type ExistingBatch = IntMap<u64, Entry>;
type NextBatch = (IntMap<u64, Entry>, Batch, Span);

#[derive(Debug)]
enum QueueCommand {
    Append(Entry, Span),
    NextBatch {
        entries: Option<ExistingBatch>,
        response_sender: oneshot::Sender<(Option<ExistingBatch>, Option<NextBatch>)>,
        span: Span,
    },
}
