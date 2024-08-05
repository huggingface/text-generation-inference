use std::hint;
use std::ops::Deref;
use std::path::Path;
use std::sync::OnceLock;

use async_trait::async_trait;
use cxx::UniquePtr;
use hashbrown::HashMap;
use log::warn;
use tokenizers::{Encoding, Tokenizer};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::mpsc::error::SendError;
use tokio::task::{JoinHandle, spawn_blocking};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, info, Level, span};

use text_generation_router::{FinishReason, Token};
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::infer::InferError::GenerationError;
use text_generation_router::validation::{Chunk, ValidationError, ValidGenerateRequest};
use text_generation_router::validation::ValidationError::UnsupportedModality;

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{create_tensorrt_llm_backend, GenerationStep, TensorRtLlmBackendImpl};
use crate::utils::first_line;

// Value used to poll the state of the generation stream
static POLLING_INTERVAL_US: OnceLock<u64> = OnceLock::new();

// It's safe to send the backend between threads
unsafe impl Send for TensorRtLlmBackendImpl {}

type InferResult<T> = Result<T, InferError>;

struct ValidGenerateRequestWithTokens {
    encoding: Encoding,
    inner: ValidGenerateRequest,
}

struct DecodedTokenContext {
    tokens: Vec<GenerationStep>,
    ctx: UnboundedSender<InferResult<InferStreamResponse>>,
}

fn executor_status_poller(
    mut backend: UniquePtr<TensorRtLlmBackendImpl>,
    mut waiting_requests: UnboundedReceiver<GenerationContext>,
    mut post_processor_sender: UnboundedSender<DecodedTokenContext>,
) {
    // Track the tuple (request_id, stream) for each request
    let mut in_flights = HashMap::<u64, GenerationContext>::with_capacity(128);

    // TODO: Does it need a spin-loop?
    'executor: loop {
        span!(Level::DEBUG, "[in-flight][submit]").in_scope(|| {
            // Is there any request pending to be scheduled?
            let awaiting_requests = waiting_requests.len();
            if awaiting_requests > 0 {
                // Retrieve all the requests
                let mut requests = Vec::with_capacity(awaiting_requests);
                let _ = waiting_requests.recv_many(&mut requests, awaiting_requests);

                // Submit all the request to the executor and move the context to the in-flight tracker
                for ctx in requests {
                    let request = &ctx.request;
                    let generation_params = &request.inner.parameters;
                    let stopping_params = &request.inner.stopping_parameters;

                    // Submit to the TensorRT-LLM executor for scheduling
                    match backend.pin_mut().submit(
                        request.encoding.get_ids(),
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

        if let Err(e) = span!(Level::DEBUG, "[in-flight][poll]").in_scope(|| {
            if backend.num_responses_ready() > 0 {
                match backend.pin_mut().pull_tokens() {
                    Ok(responses) => {
                        // worse case scenario is one token for each response: with_capacity(responses.len())
                        // grouper will group decoded tokens per request to decode multiple tokens
                        let mut grouper: HashMap<u64, DecodedTokenContext> =
                            HashMap::with_capacity(responses.len());

                        // Iterate through all the decoded token
                        for step in responses.deref() {
                            let request_id = step.request_id;

                            match in_flights.get(&request_id) {
                                Some(ctx) => {
                                    info!("New token for {} -> {}", request_id, step.token_id);

                                    if !step.has_error {
                                        let req_group = grouper.entry(request_id).or_insert(
                                            DecodedTokenContext {
                                                tokens: vec![],
                                                ctx: ctx.streamer.clone(), // Arc::clone() = cheap
                                            },
                                        );
                                        req_group.tokens.push(step.clone()); // Should be ultra cheap

                                        if step.is_final {
                                            let _ = in_flights.remove(&step.request_id);
                                        }
                                    } else {
                                        warn!(
                                            "Error for request: {} -> {}",
                                            request_id, &step.error_msg
                                        );
                                    }
                                }
                                None => {
                                    error!("Got step for untracked request {}", request_id);
                                }
                            }
                        }

                        grouper
                            .into_values()
                            .map(|ctx| post_processor_sender.send(ctx))
                            .collect::<Result<(), SendError<DecodedTokenContext>>>()?;
                    }
                    Err(err) => {
                        error!("Failed to retrieve tokens from the executor: {}", err);
                    }
                }
            }

            Ok::<(), SendError<DecodedTokenContext>>(())
        }) {
            error!(
                "Caught an fatal error in the executor's loop, about to exit. {}",
                e
            );
            break 'executor;
        }

        // Hint the CPU we are spin-locking
        hint::spin_loop();
    }
}

fn post_processor_looper(
    tokenizer: Tokenizer,
    mut decoded_tokens: UnboundedReceiver<DecodedTokenContext>,
) {
    'post_processor: loop {
        if decoded_tokens.is_closed() {
            warn!("Post processor IPC is closed, loop will exit now.");
            break 'post_processor;
        }

        if let Some(ctx) = decoded_tokens.blocking_recv() {
            ctx.tokens.iter().for_each(|step| {
                let out = match tokenizer.decode(&[step.token_id], true) {
                    Ok(text) => {
                        let is_special = tokenizer.get_added_vocabulary().is_special_token(&text);
                        let token = Token {
                            id: step.token_id,
                            text,
                            logprob: step.log_prob,
                            special: is_special,
                        };

                        let response = if !step.is_final {
                            InferStreamResponse::Intermediate {
                                token,
                                top_tokens: vec![],
                            }
                        } else {
                            InferStreamResponse::End {
                                token,
                                top_tokens: vec![],
                                generated_text: GeneratedText {
                                    text: String::from(""),
                                    generated_tokens: 0,
                                    finish_reason: FinishReason::Length,
                                    seed: None,
                                },
                                start: Instant::now(),  // Handle start time
                                queued: Instant::now(), // Handle queued time
                            }
                        };

                        Ok(response)
                    }
                    Err(e) => Err(GenerationError(e.to_string())),
                };

                if let Err(e) = ctx.ctx.send(out) {
                    warn!("Failed to send back the decoded tokens: {}", e);
                };
            });
        }
    }
}

struct GenerationContext {
    request: ValidGenerateRequestWithTokens,
    streamer: UnboundedSender<InferResult<InferStreamResponse>>,
}

pub struct TensorRtLlmBackendV2 {
    tokenizer: Tokenizer,
    executor_looper: JoinHandle<()>,
    post_processor_looper: JoinHandle<()>,
    executor: UnboundedSender<GenerationContext>,
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
        let (executor_sender, executor_receiver) = unbounded_channel();
        let (post_processor_sender, post_processor_receiver) = unbounded_channel();

        // Create the FFI backend
        let backend = create_tensorrt_llm_backend(&engine_folder, &executor_worker_path)
            .map_err(|e| TensorRtLlmBackendError::Runtime(first_line(e.what(), "Unknown error")))?;

        // Executor looper is responsible for scheduling and pulling requests state at regular interval
        let executor_looper = spawn_blocking(move || {
            executor_status_poller(backend, executor_receiver, post_processor_sender)
        });

        // Post processor looper is responsible from receiving a bunch of tokens, decoding them and sending them back to the user
        let tokenizer_ = tokenizer.clone();
        let post_processor_looper =
            spawn_blocking(move || post_processor_looper(tokenizer_, post_processor_receiver));

        Ok(TensorRtLlmBackendV2 {
            tokenizer,
            executor_looper,
            post_processor_looper,
            executor: executor_sender,
        })
    }

    fn validate(request: &ValidGenerateRequest) -> InferResult<&String> {
        if request.top_n_tokens > 1 {
            return Err(InferError::ValidationError(
                ValidationError::TopNTokensDisabled,
            ));
        }

        // TODO: Is it really needed? How can it be validated before?
        if request.parameters.grammar.is_some() {
            return Err(InferError::ValidationError(ValidationError::Grammar));
        }

        match request.inputs.len() {
            0 => Err(InferError::ValidationError(ValidationError::EmptyInput)),
            2.. => Err(InferError::GenerationError(
                "TensorRT-LLM backend don't support multi-chunk".into(),
            )),
            1 => match request.inputs.first().expect("Single item-chunk") {
                Chunk::Text(text) => Ok(text),
                Chunk::Image(_) => Err(InferError::ValidationError(UnsupportedModality("image"))),
            },
        }
    }
}

#[async_trait]
impl Backend for TensorRtLlmBackendV2 {
    fn schedule(
        &self,
        inner: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        let prompt = Self::validate(&inner)?;

        // We encode the prompt in every request context/thread
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| GenerationError(format!("Tokenization failed {}", e.to_string())))?;

        let request = ValidGenerateRequestWithTokens { encoding, inner };

        // Open-up the stream to send tokens
        let (streamer, receiver) = unbounded_channel::<InferResult<InferStreamResponse>>();

        // Send the context to the executor for scheduling
        match self.executor.send(GenerationContext { request, streamer }) {
            Ok(_) => Ok(UnboundedReceiverStream::new(receiver)),
            Err(_) => Err(GenerationError(
                "Failed to submit request to the backend".into(),
            )),
        }
    }

    async fn health(&self, current_health: bool) -> bool {
        current_health
            & !self.executor_looper.is_finished()
            & !self.post_processor_looper.is_finished()
    }
}
