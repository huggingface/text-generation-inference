use crate::block_allocator::{BlockAllocation, BlockAllocator};
use crate::client;
use crate::client::{
    Batch, GrammarType, NextTokenChooserParameters, Request, StoppingCriteriaParameters,
};
use nohash_hasher::{BuildNoHashHasher, IntMap};
use std::cmp::max;
use std::collections::VecDeque;
use text_generation_router::infer::InferError;
use text_generation_router::infer::InferStreamResponse;
use text_generation_router::usage_stats::Env;
use text_generation_router::validation::{
    Chunk, ChunksToString, ValidGenerateRequest, ValidGrammar, ValidParameters,
    ValidStoppingParameters,
};
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use tracing::{info_span, instrument, Instrument, Span};

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
    /// Block Allocation
    pub block_allocation: Option<BlockAllocation>,
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
        block_size: u32,
        prefix_caching: bool,
        window_size: Option<u32>,
        speculate: u32,
        max_batch_total_tokens: u32,
        support_chunking: bool,
    ) -> Self {
        // Create channel
        let (queue_sender, queue_receiver) = mpsc::unbounded_channel();

        // Launch background queue task
        tokio::spawn(queue_task(
            requires_padding,
            block_size,
            prefix_caching,
            window_size,
            speculate,
            max_batch_total_tokens,
            support_chunking,
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
        max_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        if prefill_token_budget == 0 || token_budget == 0 {
            return None;
        };

        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send next batch command to the background task managing the state
        // Unwrap is safe here
        self.queue_sender
            .send(QueueCommand::NextBatch {
                min_size,
                max_size,
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
#[allow(clippy::too_many_arguments)]
async fn queue_task(
    requires_padding: bool,
    block_size: u32,
    prefix_caching: bool,
    window_size: Option<u32>,
    speculate: u32,
    max_batch_total_tokens: u32,
    support_chunking: bool,
    mut receiver: mpsc::UnboundedReceiver<QueueCommand>,
) {
    let mut state = State::new(
        requires_padding,
        block_size,
        prefix_caching,
        window_size,
        speculate,
        max_batch_total_tokens,
        support_chunking,
    );

    while let Some(cmd) = receiver.recv().await {
        match cmd {
            QueueCommand::Append(entry, span) => {
                span.in_scope(|| state.append(*entry));
                metrics::gauge!("tgi_queue_size").increment(1.0);
            }
            QueueCommand::NextBatch {
                min_size,
                max_size,
                prefill_token_budget,
                token_budget,
                response_sender,
                span,
            } => {
                let next_batch = state
                    .next_batch(min_size, max_size, prefill_token_budget, token_budget)
                    .instrument(span)
                    .await;
                response_sender.send(next_batch).unwrap();
                metrics::gauge!("tgi_queue_size").set(state.entries.len() as f64);
            }
        }
    }
}

/// Queue State
#[derive(Debug)]
struct State {
    /// Queue entries organized in a Vec
    entries: VecDeque<(u64, Entry)>,

    /// Id of the next entry
    next_id: u64,

    /// Id of the next batch
    next_batch_id: u64,

    /// Paged Attention block size
    block_size: u32,

    /// Speculation amount
    speculate: u32,

    /// Whether the model allow the prefill chunking
    /// If it does, the last request in the batch will be split to exactly match the prefill
    /// token budget
    support_chunking: bool,

    /// Paged Attention Block Allocation
    block_allocator: Option<BlockAllocator>,

    /// indicate if it's hpu device, the hpu device needs padding to generate first token.
    is_hpu_device: bool,
}

impl State {
    fn new(
        requires_padding: bool,
        block_size: u32,
        prefix_caching: bool,
        window_size: Option<u32>,
        speculate: u32,
        max_batch_total_tokens: u32,
        support_chunking: bool,
    ) -> Self {
        let block_allocator = (!requires_padding).then(|| {
            BlockAllocator::new(
                max_batch_total_tokens,
                block_size,
                prefix_caching,
                window_size,
            )
        });

        Self {
            entries: VecDeque::with_capacity(128),
            next_id: 0,
            next_batch_id: 0,
            block_size,
            speculate,
            support_chunking,
            block_allocator,
            is_hpu_device: Env::new().is_hpu_device(),
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
    }

    // Get the next batch
    async fn next_batch(
        &mut self,
        min_size: Option<usize>,
        max_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
    ) -> Option<NextBatch> {
        if self.entries.is_empty() {
            tracing::debug!("No queue");
            return None;
        }

        // Check if we have enough entries
        if let Some(min_size) = min_size {
            if self.entries.len() < min_size {
                tracing::debug!("Not enough entries");
                return None;
            }
        }

        if let Some(max_size) = max_size {
            if max_size == 0 {
                tracing::debug!("No capacity");
                return None;
            }
        }

        // Pad prefill_token_budget to be a multiple of block size
        let prefill_token_budget = prefill_token_budget.div_ceil(self.block_size) * self.block_size;

        // Create span for this batch to add context to inference calls
        let next_batch_span = info_span!(parent: None, "batch", batch_size = tracing::field::Empty);
        next_batch_span.follows_from(Span::current());

        let mut batch = Vec::with_capacity(self.entries.len());
        let mut max_input_length = 0;
        let mut prefill_tokens: u32 = 0;
        let mut decode_tokens: u32 = 0;
        let mut max_blocks = 0;

        // Pop entries starting from the front of the queue
        'entry_loop: while let Some((id, entry)) = self.entries.pop_front() {
            // Filter entries where the response receiver was dropped (== entries where the request
            // was dropped by the client)
            if entry.response_tx.is_closed() {
                metrics::counter!("tgi_request_failure", "err" => "dropped").increment(1);
                tracing::debug!("Dropping entry");
                continue;
            }

            let block_allocation = match &self.block_allocator {
                None => {
                    // We pad to max input length in the Python shards
                    // We need to take these padding tokens into the equation
                    max_input_length = max_input_length.max(entry.request.input_length);
                    prefill_tokens = (batch.len() + 1) as u32 * max_input_length;

                    decode_tokens += entry.request.stopping_parameters.max_new_tokens;
                    let total_tokens = prefill_tokens + decode_tokens + self.speculate;

                    if prefill_tokens > prefill_token_budget || total_tokens > token_budget {
                        // Entry is over budget
                        // Add it back to the front
                        tracing::debug!("Over budget: prefill_tokens={prefill_tokens} > {prefill_token_budget} || {prefill_tokens} + {decode_tokens} + {} > {token_budget}", self.speculate);
                        self.entries.push_front((id, entry));
                        break 'entry_loop;
                    }
                    None
                }
                Some(block_allocator) => {
                    // If users wants the prefill logprobs, we cannot reuse the cache.
                    // So no input_ids for the radix tree.
                    let input_ids = if entry.request.decoder_input_details {
                        None
                    } else {
                        entry.request.input_ids.clone()
                    };

                    let tokens = entry.request.input_length
                        + entry.request.stopping_parameters.max_new_tokens
                        + self.speculate
                        - 1;
                    // tracing::debug!("Allocating {tokens} with {input_ids:?}");

                    let block_allocation = match block_allocator.allocate(tokens, input_ids).await {
                        None => {
                            // Entry is over budget
                            // Add it back to the front
                            tracing::debug!("Over budget: not enough free blocks");
                            self.entries.push_front((id, entry));
                            break 'entry_loop;
                        }
                        Some(mut block_allocation) => {
                            // tracing::debug!("Allocation: {block_allocation:?}");
                            max_blocks = max(max_blocks, block_allocation.blocks.len() as u32);

                            if block_allocation.prefix_len == entry.request.input_length {
                                // The whole request was found in the radix trie
                                // However, for the transformer forward to work, we need to
                                // have at least one token of postfix.
                                block_allocation.prefix_len -= 1;
                            }

                            block_allocation
                        }
                    };

                    let postfix_len = entry.request.input_length - block_allocation.prefix_len;

                    if prefill_tokens + postfix_len > prefill_token_budget {
                        // Entry is over budget
                        if self.support_chunking {
                            // We support chunking, just set postfix_len to exactly match prefill_token_budget
                            let chunk_len = prefill_token_budget.saturating_sub(prefill_tokens);
                            if chunk_len > 0 {
                                // Push this entry inside the batch
                                batch.push((id, entry, Some(block_allocation), Some(chunk_len)));
                            } else {
                                // We cannot prefill even one token for this entry
                                // Add it back to the queue
                                self.entries.push_front((id, entry));
                            }
                            tracing::debug!(
                                "Matched budget: prefill_tokens={} == {prefill_token_budget}",
                                prefill_tokens + postfix_len
                            );
                            break 'entry_loop;
                        } else {
                            // We don't support chunking, this entry needs to go back to the buffer
                            // Add it back to the front
                            tracing::debug!(
                                "Over budget: prefill_tokens={} > {prefill_token_budget}",
                                prefill_tokens + postfix_len
                            );
                            self.entries.push_front((id, entry));
                            break 'entry_loop;
                        }
                    }

                    if self.is_hpu_device {
                        //HPU needs to pad for the prefill
                        max_input_length = max_input_length.max(entry.request.input_length);
                        let actual_prefill_tokens_for_hpu =
                            (batch.len() + 1) as u32 * max_input_length;

                        if actual_prefill_tokens_for_hpu > prefill_token_budget {
                            // Entry is over budget
                            // Add it back to the front
                            tracing::debug!("Over budget: prefill_tokens={actual_prefill_tokens_for_hpu} > {prefill_token_budget}");
                            self.entries.push_front((id, entry));
                            break 'entry_loop;
                        }
                    }

                    prefill_tokens += postfix_len;

                    Some(block_allocation)
                }
            };
            batch.push((id, entry, block_allocation, None));
            if Some(batch.len()) == max_size {
                break;
            }
        }

        // Empty batch
        if batch.is_empty() {
            tracing::debug!("Filterered out all entries");
            return None;
        }

        // XXX We haven't allocated yet, so we're allowed to ditch the results.
        // Check if our batch is big enough
        if let Some(min_size) = min_size {
            // Batch is too small
            if batch.len() < min_size {
                // Add back entries to the queue in the correct order
                for (id, entry, _, _) in batch.into_iter().rev() {
                    self.entries.push_front((id, entry));
                }
                return None;
            }
        }

        let mut batch_requests = Vec::with_capacity(self.entries.len());
        let mut batch_entries =
            IntMap::with_capacity_and_hasher(self.entries.len(), BuildNoHashHasher::default());

        for (id, mut entry, block_allocation, chunk_len) in batch {
            // Create a new span to link the batch back to this entry
            let entry_batch_span = info_span!(parent: &entry.span, "infer");
            // Add relationships
            next_batch_span.follows_from(&entry_batch_span);
            entry_batch_span.follows_from(&next_batch_span);
            // Update entry
            entry.temp_span = Some(entry_batch_span);

            let (blocks, slots, prefix_len) = match &block_allocation {
                None => (Vec::new(), Vec::new(), 0),
                Some(block_allocation) => (
                    block_allocation.blocks.clone(),
                    block_allocation.slots.clone(),
                    block_allocation.prefix_len,
                ),
            };

            entry.block_allocation = block_allocation;

            batch_requests.push(Request {
                id,
                prefill_logprobs: entry.request.decoder_input_details,
                input_chunks: Some(client::Input {
                    chunks: entry
                        .request
                        .inputs
                        .clone()
                        .into_iter()
                        .map(|c| client::InputChunk {
                            chunk: Some(match c {
                                Chunk::Text(text) => client::Chunk::Text(text),
                                Chunk::Image(image) => client::Chunk::Image(client::Image {
                                    data: image.data,
                                    mimetype: image.mimetype,
                                }),
                            }),
                        })
                        .collect(),
                }),
                inputs: entry.request.inputs.chunks_to_string(),
                truncate: entry.request.truncate,
                add_special_tokens: entry.request.add_special_tokens,
                parameters: Some(NextTokenChooserParameters::from(
                    entry.request.parameters.clone(),
                )),
                stopping_parameters: Some(StoppingCriteriaParameters::from(
                    entry.request.stopping_parameters.clone(),
                )),
                top_n_tokens: entry.request.top_n_tokens,
                blocks,
                slots,
                cache_len: prefix_len,
                adapter_id: entry.request.adapter_id.clone(),
                chunk_len,
            });
            // Set batch_time
            entry.batch_time = Some(Instant::now());
            // Insert in batch_entries IntMap
            batch_entries.insert(id, entry);
        }

        // Final batch size
        let size = batch_requests.len() as u32;
        next_batch_span.record("batch_size", size);

        let batch = Batch {
            id: self.next_batch_id,
            requests: batch_requests,
            size,
            max_tokens: (prefill_tokens + decode_tokens),
            max_blocks,
        };
        // Increment batch id
        self.next_batch_id += 1;

        metrics::histogram!("tgi_batch_next_size").record(batch.size as f64);

        Some((batch_entries, batch, next_batch_span))
    }
}

type NextBatch = (IntMap<u64, Entry>, Batch, Span);

#[derive(Debug)]
enum QueueCommand {
    Append(Box<Entry>, Span),
    NextBatch {
        min_size: Option<usize>,
        max_size: Option<usize>,
        prefill_token_budget: u32,
        token_budget: u32,
        response_sender: oneshot::Sender<Option<NextBatch>>,
        span: Span,
    },
}

impl From<ValidParameters> for NextTokenChooserParameters {
    fn from(value: ValidParameters) -> Self {
        let (grammar, grammar_type) = match value.grammar {
            None => (String::new(), GrammarType::None),

            Some(grammar) => match grammar {
                ValidGrammar::Json(grammar_string) => (grammar_string, GrammarType::Json),
                ValidGrammar::Regex(grammar_string) => (grammar_string, GrammarType::Regex),
            },
        };

        Self {
            temperature: value.temperature,
            top_k: value.top_k,
            top_p: value.top_p,
            typical_p: value.typical_p,
            do_sample: value.do_sample,
            seed: value.seed,
            repetition_penalty: value.repetition_penalty,
            frequency_penalty: value.frequency_penalty,
            watermark: value.watermark,
            grammar,
            grammar_type: grammar_type.into(),
        }
    }
}

impl From<ValidStoppingParameters> for StoppingCriteriaParameters {
    fn from(value: ValidStoppingParameters) -> Self {
        Self {
            max_new_tokens: value.max_new_tokens,
            stop_sequences: value.stop_sequences,
            ignore_eos_token: value.ignore_eos_token,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use tracing::info_span;

    fn default_entry() -> (
        Entry,
        mpsc::UnboundedReceiver<Result<InferStreamResponse, InferError>>,
    ) {
        let (response_tx, receiver_tx) = mpsc::unbounded_channel();

        let entry = Entry {
            request: ValidGenerateRequest {
                inputs: vec![],
                input_ids: Some(Arc::new(vec![])),
                input_length: 1,
                add_special_tokens: true,
                truncate: 0,
                decoder_input_details: false,
                parameters: ValidParameters {
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 0.0,
                    typical_p: 0.0,
                    do_sample: false,
                    seed: 0,
                    repetition_penalty: 0.0,
                    frequency_penalty: 0.0,
                    watermark: false,
                    grammar: None,
                },
                stopping_parameters: ValidStoppingParameters {
                    ignore_eos_token: false,
                    max_new_tokens: 1,
                    max_total_new_tokens: 1024,
                    stop_sequences: vec![],
                },
                top_n_tokens: 0,
                adapter_id: None,
            },
            response_tx,
            span: info_span!("entry"),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
            block_allocation: None,
        };
        (entry, receiver_tx)
    }

    #[tokio::test]
    async fn test_append() {
        let mut state = State::new(false, 1, false, None, 0, 16, false);
        let (entry, _guard) = default_entry();

        assert_eq!(state.next_id, 0);
        assert_eq!(state.entries.len(), 0);

        state.append(entry);

        assert_eq!(state.next_id, 1);
        assert_eq!(state.entries.len(), 1);
        let (id, _) = state.entries.remove(0).unwrap();
        assert_eq!(id, 0);
    }

    #[tokio::test]
    async fn test_next_batch_empty() {
        let mut state = State::new(false, 1, false, None, 0, 16, false);

        assert!(state.next_batch(None, None, 1, 1).await.is_none());
        assert!(state.next_batch(Some(1), None, 1, 1).await.is_none());
    }

    #[tokio::test]
    async fn test_next_batch_min_size() {
        let mut state = State::new(false, 1, false, None, 0, 16, false);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.append(entry1);
        state.append(entry2);

        let (entries, batch, _) = state.next_batch(None, None, 2, 2).await.unwrap();
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

        assert!(state.next_batch(Some(2), None, 2, 2).await.is_none());

        assert_eq!(state.next_id, 3);
        assert_eq!(state.entries.len(), 1);
        let (id, _) = state.entries.remove(0).unwrap();
        assert_eq!(id, 2);
    }

    #[tokio::test]
    async fn test_next_batch_max_size() {
        let mut state = State::new(false, 1, false, None, 0, 16, false);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.append(entry1);
        state.append(entry2);

        let (entries, batch, _) = state.next_batch(None, Some(1), 2, 2).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        assert_eq!(state.next_id, 2);
        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.next_batch_id, 1);
    }

    #[tokio::test]
    async fn test_next_batch_token_budget() {
        let mut state = State::new(false, 1, false, None, 0, 16, false);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        state.append(entry1);
        state.append(entry2);

        let (entries, batch, _) = state.next_batch(None, None, 1, 1).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        assert_eq!(state.next_id, 2);
        assert_eq!(state.entries.len(), 1);
        assert_eq!(state.next_batch_id, 1);

        let (entry3, _guard3) = default_entry();
        state.append(entry3);

        let (entries, batch, _) = state.next_batch(None, None, 3, 3).await.unwrap();
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
        let queue = Queue::new(false, 1, false, None, 0, 16, false);
        let (entry, _guard) = default_entry();
        queue.append(entry);
    }

    #[tokio::test]
    async fn test_queue_next_batch_empty() {
        let queue = Queue::new(false, 1, false, None, 0, 16, false);

        assert!(queue.next_batch(None, None, 1, 1).await.is_none());
        assert!(queue.next_batch(Some(1), None, 1, 1).await.is_none());
    }

    #[tokio::test]
    async fn test_queue_next_batch_min_size() {
        let queue = Queue::new(false, 1, false, None, 0, 16, false);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        let (entries, batch, _) = queue.next_batch(None, None, 2, 2).await.unwrap();
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
        assert!(queue.next_batch(Some(2), None, 2, 2).await.is_none());
        // Not enough token budget
        assert!(queue.next_batch(Some(1), None, 0, 0).await.is_none());
        // Ok
        let (entries2, batch2, _) = queue.next_batch(Some(1), None, 2, 2).await.unwrap();
        assert_eq!(entries2.len(), 1);
        assert!(entries2.contains_key(&2));
        assert!(entries2.get(&2).unwrap().batch_time.is_some());
        assert_eq!(batch2.id, 1);
        assert_eq!(batch2.size, 1);
    }

    #[tokio::test]
    async fn test_queue_next_batch_max_size() {
        let queue = Queue::new(false, 1, false, None, 0, 16, false);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        let (entries, batch, _) = queue.next_batch(None, Some(1), 2, 2).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert!(entries.get(&0).unwrap().batch_time.is_some());
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);
    }

    #[tokio::test]
    async fn test_queue_next_batch_token_budget() {
        let queue = Queue::new(false, 1, false, None, 0, 16, false);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        let (entries, batch, _) = queue.next_batch(None, None, 1, 1).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&0));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 1);

        let (entry3, _guard3) = default_entry();
        queue.append(entry3);

        let (entries, batch, _) = queue.next_batch(None, None, 3, 3).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&1));
        assert!(entries.contains_key(&2));
        assert_eq!(batch.id, 1);
        assert_eq!(batch.size, 2);
    }

    #[tokio::test]
    async fn test_queue_next_batch_token_speculate() {
        let queue = Queue::new(true, 1, false, None, 2, 16, false);
        let (entry1, _guard1) = default_entry();
        let (entry2, _guard2) = default_entry();
        queue.append(entry1);
        queue.append(entry2);

        // Budget of 1 is not enough
        assert!(queue.next_batch(None, None, 1, 1).await.is_none());

        let (entries, batch, _) = queue.next_batch(None, None, 6, 6).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries.contains_key(&0));
        assert!(entries.contains_key(&1));
        assert_eq!(batch.id, 0);
        assert_eq!(batch.size, 2);
    }

    #[tokio::test]
    async fn test_queue_next_batch_dropped_receiver() {
        let queue = Queue::new(false, 1, false, None, 0, 16, false);
        let (entry, _) = default_entry();
        queue.append(entry);

        assert!(queue.next_batch(None, None, 1, 1).await.is_none());
    }
}
