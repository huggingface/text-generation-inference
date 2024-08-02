use std::hint;
use std::ops::Deref;
use std::path::Path;
use std::sync::OnceLock;

use async_trait::async_trait;
use cxx::UniquePtr;
use hashbrown::HashMap;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, info, Level, span};

use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::infer::InferError::GenerationError;
use text_generation_router::validation::ValidGenerateRequest;

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{create_tensorrt_llm_backend, TensorRtLlmBackendImpl};

// Value used to poll the state of the generation stream
static POLLING_INTERVAL_US: OnceLock<u64> = OnceLock::new();

// It's safe to send the backend between threads
unsafe impl Send for TensorRtLlmBackendImpl {}

type InferResult<T> = Result<T, InferError>;

fn executor_status_poller(
    mut backend: UniquePtr<TensorRtLlmBackendImpl>,
    mut waiting_requests: UnboundedReceiver<GenerationContext>,
) {
    // Track the tuple (request_id, stream) for each request
    let mut in_flights = HashMap::<u64, GenerationContext>::with_capacity(128);

    // TODO: Does it need a spin-loop?
    loop {
        span!(Level::DEBUG, "in-flight submit").in_scope(|| {
            // Is there any request pending to be scheduled?
            let awaiting_requests = waiting_requests.len();
            if awaiting_requests > 0 {
                // Retrieve all the requests
                let mut requests = Vec::with_capacity(awaiting_requests);
                let _ = waiting_requests.recv_many(&mut requests, awaiting_requests);

                // Submit all the request to the executor and move the context to the in-flight tracker
                for ctx in requests {
                    let request = &ctx.request;
                    let generation_params = &request.parameters;
                    let stopping_params = &request.stopping_parameters;

                    // Submit to the TensorRT-LLM executor for scheduling
                    match backend.pin_mut().submit(
                        &vec![],
                        stopping_params.max_new_tokens,
                        generation_params.top_k as i32,
                        generation_params.top_p,
                        generation_params.temperature,
                        generation_params.repetition_penalty,
                        generation_params.frequency_penalty,
                        generation_params.seed,
                    ) {
                        Ok(request_id) => {
                            // Insert the context linked to the generated request id in the tracker
                            in_flights.insert(request_id, ctx);
                        }
                        Err(e) => {
                            // Return to the caller
                            let what = Err(InferError::SchedulingError(e.to_string()));
                            if let Err(e) = ctx.streamer.send(what) {
                                error!("Failed to send back through the channel: {}", e);
                            }
                        }
                    };
                }
            }
        });

        span!(Level::DEBUG, "in-flight poll").in_scope(|| {
            if backend.num_responses_ready() > 0 {
                match backend.pin_mut().pull_tokens() {
                    Ok(responses) => {
                        for step in responses.deref() {
                            let request_id = step.request_id;
                            match in_flights.get(&request_id) {
                                Some(ctx) => {
                                    info!("New token for {} -> {}", request_id, step.token_id);

                                    if step.is_final {
                                        let _ = in_flights.remove(&step.request_id);
                                    }
                                }
                                None => {
                                    error!("Got step for untracked request {}", request_id);
                                }
                            }
                        }
                    }
                    Err(err) => {
                        error!("Failed to retrieve tokens from the executor: {}", err);
                    }
                }
            }
        });

        // Hint the CPU we are spin-locking
        hint::spin_loop();
    }
}

struct GenerationContext {
    request: ValidGenerateRequest,
    streamer: UnboundedSender<InferResult<InferStreamResponse>>,
}

pub struct TensorRtLlmBackendV2 {
    tokenizer: Tokenizer,
    looper: JoinHandle<()>,
    queue: UnboundedSender<GenerationContext>,
}

impl TensorRtLlmBackendV2 {
    pub fn new<P: AsRef<Path> + Send, PP: AsRef<Path> + Send>(
        tokenizer: Tokenizer,
        engine_folder: P,
        executor_worker_path: PP,
    ) -> Result<Self, TensorRtLlmBackendError> {
        // Retrieve paths as &str for the backend creation
        let engine_folder = engine_folder.as_ref();
        let executor_worker_path = executor_worker_path.as_ref();

        let engine_folder = String::from(
            engine_folder
                .to_str()
                .expect("Failed to convert engine_folder to valid UTF-8"),
        );

        let executor_worker_path = String::from(
            executor_worker_path
                .to_str()
                .expect("Failed to convert executor_worker_path to valid UTF-8"),
        );

        // Allocate the IPC layer to communicate with the backend
        let (requests_sender, requests_receiver) = unbounded_channel::<GenerationContext>();

        // Create the FFI backend
        let backend = create_tensorrt_llm_backend(&engine_folder, &executor_worker_path)
            .map_err(|e| TensorRtLlmBackendError::Runtime(e.what().to_string()))?;

        // Looper is responsible for scheduling and pulling requests state at regular interval
        let looper =
            tokio::task::spawn_blocking(move || executor_status_poller(backend, requests_receiver));

        Ok(TensorRtLlmBackendV2 {
            tokenizer,
            looper,
            queue: requests_sender,
        })
    }
}

#[async_trait]
impl Backend for TensorRtLlmBackendV2 {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        let (streamer, receiver) = unbounded_channel::<InferResult<InferStreamResponse>>();
        match self.queue.send(GenerationContext { request, streamer }) {
            Ok(_) => Ok(UnboundedReceiverStream::new(receiver)),
            Err(_) => Err(GenerationError(
                "Failed to submit request to the backend".into(),
            )),
        }
    }

    async fn health(&self, current_health: bool) -> bool {
        current_health & !self.looper.is_finished()
    }
}
