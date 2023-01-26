/// This code is massively inspired by Tokio mini-redis
use crate::InferResponse;
use crate::{GenerateParameters, GenerateRequest};
use nohash_hasher::{BuildNoHashHasher, IntMap};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::Arc;
use text_generation_client::{
    Batch, ClientError, NextTokenChooserParameters, Request, StoppingCriteriaParameters,
};
use tokio::sync::oneshot::Sender;
use tokio::time::Instant;

/// Database entry
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: GenerateRequest,
    /// Response sender to communicate between the Batcher and the batching_task
    pub response_tx: Sender<Result<InferResponse, ClientError>>,
    /// Number of tokens in the input
    pub input_length: usize,
    /// Instant when this entry was created
    pub time: Instant,
    /// Instant when this entry was added to a batch
    pub batch_time: Option<Instant>,
}

/// Request Database
#[derive(Debug, Clone)]
pub(crate) struct Db {
    pub shared: Arc<Shared>,
}

/// Shared state
#[derive(Debug)]
pub struct Shared {
    state: Mutex<State>,
}

/// Database State
#[derive(Debug)]
struct State {
    /// Database entries organized in a BTreeMap to be able to iterate over them in order
    entries: BTreeMap<u64, Entry>,

    /// Id of the next entry
    next_id: u64,

    /// Id of the next batch
    next_batch_id: u64,

    /// Start ID of the next batch. Used to iterate inside the entries BTreeMap
    next_batch_start_id: u64,
}

impl State {
    /// Get the next requests
    fn next_requests(&self, max_size: usize) -> Option<(Vec<u64>, Vec<Request>)> {
        // Iterates for max_size over the BTreemap starting from next_batch_start_id
        let mut requests = Vec::new();
        let mut ids = Vec::new();

        for (id, entry) in self
            .entries
            // Start from next_batch_start_id
            .range(self.next_batch_start_id..)
            // Take max_size
            .take(max_size)
        {
            requests.push(Request {
                id: *id,
                inputs: entry.request.inputs.clone(),
                input_length: entry.input_length as u32,
                parameters: Some((&entry.request.parameters).into()),
                stopping_parameters: Some(entry.request.parameters.clone().into()),
            });

            ids.push(*id);
        }

        if requests.is_empty() {
            None
        } else {
            Some((ids, requests))
        }
    }
}

impl Db {
    pub(crate) fn new() -> Self {
        // Shared state
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                entries: BTreeMap::new(),
                next_id: 0,
                next_batch_id: 0,
                next_batch_start_id: 0,
            }),
        });

        Self { shared }
    }

    /// Append an entry to the database
    pub(crate) fn append(&self, entry: Entry) {
        // Acquire lock
        let mut state = self.shared.state.lock();

        // Insert entry
        let id = state.next_id;
        state.next_id += 1;
        state.entries.insert(id, entry);
    }

    // Get the next batch
    pub(crate) fn next_batch(
        &self,
        min_size: Option<usize>,
        max_size: usize,
    ) -> Option<(IntMap<u64, Entry>, Batch)> {
        // Acquire lock
        let mut state = self.shared.state.lock();

        // Get requests from the database
        if let Some((ids, requests)) = state.next_requests(max_size) {
            if let Some(min_size) = min_size {
                // If min_size is set, only return a batch if there are enough requests
                if requests.len() < min_size {
                    return None;
                }
            }
            // Batch size
            let size = requests.len();

            let mut entries = IntMap::with_capacity_and_hasher(size, BuildNoHashHasher::default());
            ids.iter().for_each(|id| {
                // Remove entry from db
                let mut entry = state.entries.remove(id).unwrap();
                // Set batch_time
                entry.batch_time = Some(Instant::now());
                // Insert in entries IntMap
                entries.insert(*id, entry);
            });

            let batch = Batch {
                id: state.next_batch_id,
                requests,
                size: size as u32,
            };
            // Update next_batch_start_id to the last id in the batch + 1
            state.next_batch_start_id = ids.last().unwrap() + 1;
            // Increment batch id
            state.next_batch_id += 1;

            return Some((entries, batch));
        }
        None
    }
}

impl From<&GenerateParameters> for NextTokenChooserParameters {
    fn from(parameters: &GenerateParameters) -> Self {
        Self {
            temperature: parameters.temperature,
            top_k: parameters.top_k as u32,
            top_p: parameters.top_p,
            do_sample: parameters.do_sample,
        }
    }
}

impl From<GenerateParameters> for StoppingCriteriaParameters {
    fn from(parameters: GenerateParameters) -> Self {
        Self {
            stop_sequences: parameters.stop,
            max_new_tokens: parameters.max_new_tokens,
        }
    }
}
