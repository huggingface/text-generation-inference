use std::hint;
use std::ops::Deref;
use std::path::Path;

use async_trait::async_trait;
use cxx::UniquePtr;
use hashbrown::HashMap;
use log::warn;
use tokenizers::{Encoding, Tokenizer};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::task::{JoinHandle, spawn_blocking};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error};

use text_generation_router::{FinishReason, Token};
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::infer::InferError::{GenerationError, ValidationError};
use text_generation_router::validation::{Chunk, ValidGenerateRequest};
use text_generation_router::validation::ValidationError::{
    EmptyInput, Grammar, TopNTokensDisabled, UnsupportedModality,
};

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{create_tensorrt_llm_backend, GenerationStep, TensorRtLlmBackendImpl};
use crate::utils::first_line;

type InferResult<T> = Result<T, InferError>;

struct IdentifiableRequest<T> {
    request_id: u64,
    inner: T,
}

/// Wrap the TGI server forwarded ValidGenerateRequest with the tokenized view of the prompt
struct ValidGenerateRequestWithTokens {
    encoding: Encoding,
    inner: ValidGenerateRequest,
}

/// Wrap the requests along with the channel used to stream back to the client the decoded tokens
struct GenerationContext {
    request: ValidGenerateRequestWithTokens,
    start: Option<Instant>,
    queued: Instant,
    streamer: UnboundedSender<InferResult<InferStreamResponse>>,
}

#[derive(Debug, Copy, Clone)]
struct DecodedToken {
    id: u32,
    log_prob: f32,
    is_final: bool,
}

impl<'step> TryFrom<&'step GenerationStep> for DecodedToken {
    type Error = InferError;

    fn try_from(step: &'step GenerationStep) -> Result<Self, Self::Error> {
        if !step.has_error {
            Ok(Self {
                id: step.token_id,
                log_prob: step.log_prob,
                is_final: step.is_final,
            })
        } else {
            Err(GenerationError(step.error_msg.clone()))
        }
    }
}

/// Wraps the decoded token with the channel used to stream back to the client the decoded tokens
struct DecodedTokenContext {
    token: DecodedToken,
    start: Option<Instant>,
    queued: Instant,
    channel: UnboundedSender<InferResult<InferStreamResponse>>,
}

fn executor_status_looper(
    mut backend: UniquePtr<TensorRtLlmBackendImpl>,
    mut waiting_requests: UnboundedReceiver<GenerationContext>,
    post_processor_sender: UnboundedSender<(u64, InferResult<DecodedTokenContext>)>,
) {
    // Track the tuple (request_id, stream) for each request
    let mut in_flights = HashMap::<u64, GenerationContext>::with_capacity(128);

    // TODO: Does it need a spin-loop?
    'scheduler: loop {
        // Is there any request pending to be scheduled?
        let awaiting_requests = waiting_requests.len();
        for _ in 0..awaiting_requests {
            // Retrieve all the requests
            if let Some(mut ctx) = waiting_requests.blocking_recv() {
                // Submit all the request to the executor and move the context to the in-flight tracker
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
                        debug!("[in-flight] Added {}", request_id);
                        ctx.start = Some(Instant::now());
                        in_flights.insert(request_id, ctx);
                    }
                    Err(e) => {
                        // Return to the caller
                        let what = e.to_string();
                        error!(error = what.as_str(), "Failed to schedule request");

                        let err = Err(InferError::SchedulingError(what));
                        if let Err(_) = ctx.streamer.send(err) {
                            error!("Failed to send back error to the client");
                        }
                    }
                };
            }
        }

        if backend.num_responses_ready() > 0 {
            match backend.pin_mut().pull_tokens() {
                Ok(responses) => {
                    // Iterate through all the decoded token
                    for step in responses.deref() {
                        if let Some(ctx) = in_flights.get(&step.request_id) {
                            // Remove from tracked requests
                            let parcel =
                                DecodedToken::try_from(step).map(|dt| DecodedTokenContext {
                                    token: dt,
                                    start: ctx.start,
                                    queued: ctx.queued,
                                    channel: ctx.streamer.clone(),
                                });

                            // Submit the work to p:the post_processor
                            let posted = post_processor_sender.send((step.request_id, parcel));

                            if posted.is_err() || step.is_final {
                                debug!("Removing {}", step.request_id);
                                let _ = in_flights.remove(&step.request_id);
                            }
                        } else {
                            warn!("Untracked request {}", step.request_id,);
                        }
                    }
                }
                Err(ref err) => {
                    error!("Failed to get responses from the executor: {}.", err.what());
                    break 'scheduler;
                }
            }
        }

        // Hint the CPU we are spin-locking
        hint::spin_loop();
    }
}

fn post_processor_looper(
    tokenizer: Tokenizer,
    mut decoded_tokens: UnboundedReceiver<(u64, InferResult<DecodedTokenContext>)>,
) {
    'post_processor: loop {
        if decoded_tokens.is_closed() {
            warn!("Post processor IPC is closed, loop will exit now.");
            break 'post_processor;
        }

        let mut states: HashMap<u64, Vec<u32>> = HashMap::with_capacity(128);

        if let Some((request_id, decoded)) = decoded_tokens.blocking_recv() {
            let state = states.entry(request_id).or_insert(vec![]);

            match decoded {
                Ok(ctx) => {
                    state.push(ctx.token.id);
                    let out = match tokenizer.decode(&[ctx.token.id], false) {
                        Ok(text) => {
                            let is_special =
                                tokenizer.get_added_vocabulary().is_special_token(&text);
                            let token = Token {
                                id: ctx.token.id,
                                text,
                                logprob: ctx.token.log_prob,
                                special: is_special,
                            };

                            let out = if !ctx.token.is_final {
                                InferStreamResponse::Intermediate {
                                    token,
                                    top_tokens: vec![],
                                }
                            } else {
                                let text = tokenizer.decode(&state, true);
                                let generated_text = GeneratedText {
                                    text: text.unwrap(),
                                    generated_tokens: state.len() as u32,
                                    finish_reason: FinishReason::EndOfSequenceToken,
                                    seed: None,
                                };

                                InferStreamResponse::End {
                                    token,
                                    top_tokens: vec![],
                                    generated_text,
                                    start: ctx.start.unwrap(),
                                    queued: ctx.queued,
                                }
                            };

                            Ok(out)
                        }
                        Err(err) => Err(GenerationError(err.to_string())),
                    };

                    if let Err(_) = ctx.channel.send(out) {
                        warn!("Failed to send decoded token back to the user")
                    }
                }
                Err(err) => {}
            }
        }
    }
}

unsafe impl Send for TensorRtLlmBackendImpl {}

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
            executor_status_looper(backend, executor_receiver, post_processor_sender)
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
            return Err(ValidationError(TopNTokensDisabled));
        }

        // TODO: Is it really needed? How can it be validated before?
        if request.parameters.grammar.is_some() {
            return Err(ValidationError(Grammar));
        }

        match request.inputs.len() {
            0 => Err(ValidationError(EmptyInput)),
            2.. => Err(GenerationError(
                "TensorRT-LLM backend don't support multi-chunk".into(),
            )),
            1 => match request.inputs.first().expect("Single item-chunk") {
                Chunk::Text(text) => Ok(text),
                Chunk::Image(_) => Err(ValidationError(UnsupportedModality("image"))),
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
        let queued = Instant::now();
        match self.executor.send(GenerationContext {
            request,
            start: None,
            queued,
            streamer,
        }) {
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
