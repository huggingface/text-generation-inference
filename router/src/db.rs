/// This code is massively inspired by Tokio mini-redis
use crate::server::GenerateRequest;
use bloom_inference_client::{Batch, ClientError, LogitsWarperParameters, Request};
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::oneshot::Sender;

#[derive(Debug, Clone)]
pub(crate) struct Db {
    pub shared: Arc<Shared>,
}

#[derive(Debug)]
pub struct Shared {
    state: RwLock<State>,
}

#[derive(Debug)]
struct State {
    entries: BTreeMap<u64, (Request, Sender<Result<String, ClientError>>)>,

    /// Identifier to use for the next expiration. Each expiration is associated
    /// with a unique identifier. See above for why.
    next_id: u64,

    next_batch_id: u64,

    /// Current batch id
    next_batch_start_id: u64,
}

impl Db {
    pub(crate) fn new() -> Self {
        let shared = Arc::new(Shared {
            state: RwLock::new(State {
                entries: BTreeMap::new(),
                next_id: 0,
                next_batch_id: 0,
                next_batch_start_id: 0,
            }),
        });

        Self { shared }
    }

    pub(crate) fn append(
        &self,
        input_length: usize,
        request: GenerateRequest,
        sender: Sender<Result<String, ClientError>>,
    ) {
        let mut state = self.shared.state.write();

        let id = state.next_id;
        state.next_id += 1;

        let parameters = Some(LogitsWarperParameters {
            temperature: request.parameters.temperature,
            top_k: request.parameters.top_k,
            top_p: request.parameters.top_p,
            do_sample: request.parameters.do_sample,
        });
        let request = Request {
            id,
            inputs: request.inputs,
            input_length: input_length as u32,
            parameters,
            max_new_tokens: request.parameters.max_new_tokens,
        };
        state.entries.insert(id, (request, sender));
    }

    pub(crate) fn remove(
        &self,
        id: &u64,
    ) -> Option<(Request, Sender<Result<String, ClientError>>)> {
        let mut state = self.shared.state.write();
        state.entries.remove(id)
    }

    pub(crate) fn len(&self) -> usize {
        let state = self.shared.state.read();
        state.entries.len()
    }

    fn next_requests(&self, max_size: usize) -> Option<(u64, Vec<Request>)> {
        let state = self.shared.state.read();

        let requests: Vec<Request> = state
            .entries
            .range(state.next_batch_start_id..)
            .take(max_size)
            .map(|(_, (request, _))| request.clone())
            .collect();

        if requests.is_empty() {
            None
        } else {
            let last_id = requests.last().unwrap().id;
            Some((last_id, requests))
        }
    }

    pub(crate) fn next_batch(&self, max_size: usize) -> Option<Batch> {
        if let Some((last_id, requests)) = self.next_requests(max_size) {
            let mut state = self.shared.state.write();
            let size = requests.len();
            let max_sequence_length = requests.iter().map(|r| r.input_length).max().unwrap();
            let batch = Batch {
                id: state.next_batch_id,
                requests,
                size: size as u32,
                max_sequence_length,
            };
            state.next_batch_start_id = last_id + 1;
            state.next_batch_id += 1;
            return Some(batch);
        }
        None
    }

    pub(crate) fn next_batch_minimum_size(
        &self,
        min_size: usize,
        max_size: usize,
    ) -> Option<Batch> {
        if let Some((last_id, requests)) = self.next_requests(max_size) {
            if requests.len() >= min_size {
                let mut state = self.shared.state.write();
                let size = requests.len();
                let max_sequence_length = requests.iter().map(|r| r.input_length).max().unwrap();
                let batch = Batch {
                    id: state.next_batch_id,
                    requests,
                    size: size as u32,
                    max_sequence_length,
                };
                state.next_batch_start_id = last_id + 1;
                state.next_batch_id += 1;
                return Some(batch);
            }
        }
        None
    }
}
