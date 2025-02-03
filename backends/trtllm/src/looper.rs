use async_trait::async_trait;
use cxx::UniquePtr;
use hashbrown::HashMap;
use std::hint;
use std::ops::Deref;
use std::path::Path;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::TryAcquireError;
use tokio::task::spawn_blocking;
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};

use text_generation_router::infer::InferError::{GenerationError, ValidationError};
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::ValidationError::{
    EmptyInput, Grammar, TopNTokensDisabled, UnsupportedModality,
};
use text_generation_router::validation::{Chunk, ValidGenerateRequest};
use text_generation_router::Token;

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{
    create_backend_from_engine_folder, FinishReason, GenerationStep, TensorRtLlmBackendImpl,
};
use crate::utils::first_line;

type InferResult<T> = Result<T, InferError>;

/// Wrap the requests along with the channel used to stream back to the client the decoded tokens
struct GenerationContext {
    request: ValidGenerateRequest,
    streamer: UnboundedSender<InferResult<InferStreamResponse>>,
    tokens: Vec<u32>,
    start: Option<Instant>,
    queued: Instant,
}

#[derive(Debug, Copy, Clone)]
struct DecodedToken {
    id: u32,
    log_prob: f32,
    is_final: bool,
    finish_reason: FinishReason,
}

impl<'step> TryFrom<&'step GenerationStep> for DecodedToken {
    type Error = InferError;

    fn try_from(step: &'step GenerationStep) -> Result<Self, Self::Error> {
        if !step.has_error {
            Ok(Self {
                id: step.token_id,
                log_prob: step.log_prob,
                is_final: step.is_final,
                finish_reason: step.finish_reason,
            })
        } else {
            Err(GenerationError(step.error_msg.clone()))
        }
    }
}

fn executor_status_looper(
    max_inflight_requests: usize,
    tokenizer: Tokenizer,
    mut backend: UniquePtr<TensorRtLlmBackendImpl>,
    mut backlog: UnboundedReceiver<GenerationContext>,
) {
    // Track the tuple (request_id, stream) for each request
    let mut in_flights =
        HashMap::<u64, GenerationContext>::with_capacity(max_inflight_requests * 2);

    'scheduler: loop {
        // Is there any request pending to be scheduled?
        let awaiting_requests = backlog.len();
        for _ in 0..awaiting_requests {
            // Retrieve all the requests
            if let Some(ctx) = backlog.blocking_recv() {
                // Submit all the request to the executor and move the context to the in-flight tracker
                let request = &ctx.request;
                let generation_params = &request.parameters;
                let stopping_params = &request.stopping_parameters;
                let input_ids = request.input_ids.as_deref();

                // Submit to the TensorRT-LLM executor for scheduling
                match backend.pin_mut().submit(
                    &input_ids.unwrap(), // This is checked beforehand in validate()
                    stopping_params.max_new_tokens,
                    generation_params.top_k,
                    generation_params.top_p,
                    generation_params.temperature,
                    generation_params.repetition_penalty,
                    generation_params.frequency_penalty,
                    generation_params.seed,
                ) {
                    Ok(request_id) => {
                        // Insert the context linked to the generated request id in the tracker
                        debug!("[in-flight] Added {}", request_id);
                        in_flights.insert(request_id, ctx);
                    }
                    Err(e) => {
                        // Return to the caller
                        let what = e.to_string();
                        error!(error = what.as_str(), "Failed to schedule request");

                        let err = Err(InferError::Overloaded(TryAcquireError::NoPermits));
                        if let Err(_) = ctx.streamer.send(err) {
                            error!("Failed to send back error to the client");
                        }
                    }
                };
            } else {
                break 'scheduler;
            }
        }

        if backend.num_tokens_ready() > 0 {
            let mut backend = backend.pin_mut();
            match backend.as_mut().pull_tokens() {
                Ok(responses) => {
                    // Iterate through all the decoded token
                    for step in responses.deref() {
                        if let Some(ctx) = in_flights.get_mut(&step.request_id) {
                            // Update the starting timestamp if not set
                            // This value might not be the actual real starting time of the request
                            // on the executor side - Need to expose more info from the executor to
                            // retrieve this value
                            // TODO : Expose actual real starting time for a request on FFI layer
                            if ctx.start.is_none() {
                                ctx.start = Some(Instant::now());
                            }

                            // Try to map the generation step to a DecodedToken
                            let response = match DecodedToken::try_from(step) {
                                Ok(decoded_token) => {
                                    post_process_decoded_token(&tokenizer, ctx, decoded_token)
                                }
                                Err(err) => Err(err),
                            };

                            // Attempt to send back the response to the client
                            if let Err(_) = ctx.streamer.send(response) {
                                // Client has dropped, remove from tracked requests
                                debug!(
                                    "Client dropped - removing request {} from tracked requests",
                                    step.request_id
                                );
                                backend.as_mut().cancel(step.request_id);
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

fn post_process_decoded_token(
    tokenizer: &Tokenizer,
    ctx: &mut GenerationContext,
    decoded_token: DecodedToken,
) -> InferResult<InferStreamResponse> {
    match tokenizer.decode(&[decoded_token.id], false) {
        Ok(text) => {
            let is_special = tokenizer.get_added_vocabulary().is_special_token(&text);
            let token = Token {
                id: decoded_token.id,
                text,
                logprob: decoded_token.log_prob,
                special: is_special,
            };

            // Append the token to the tracked generated tokens
            ctx.tokens.push(token.id);

            // Map the correct response depending on the step is final or not
            let out = if !decoded_token.is_final {
                InferStreamResponse::Intermediate {
                    token,
                    top_tokens: vec![],
                }
            } else {
                let text = tokenizer.decode(&ctx.tokens, true);
                let generated_text = GeneratedText {
                    text: text.unwrap(),
                    generated_tokens: ctx.tokens.len() as u32,
                    finish_reason: decoded_token.finish_reason.into(),
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
    }
}

fn ensure_paths_exist<P: AsRef<Path>, PP: AsRef<Path>>(
    engine_folder: P,
    executor_worker_path: PP,
) -> Result<(String, String), TensorRtLlmBackendError> {
    // Retrieve paths as &str for the backend creation
    let engine_folder = engine_folder.as_ref();
    let executor_worker_path = executor_worker_path.as_ref();

    // Ensure the engine folder exists
    if !engine_folder.exists() {
        let err = TensorRtLlmBackendError::EngineFolderDoesntExists(engine_folder.to_path_buf());

        error!("Path validation failed: {}", err,);
        return Err(err);
    }

    // Ensure executor worker binary exists
    if !executor_worker_path.exists() {
        let err = TensorRtLlmBackendError::ExecutorWorkerNotFound(engine_folder.to_path_buf());

        error!("Path validation failed: {}", err,);
        return Err(err);
    }

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

    Ok((engine_folder, executor_worker_path))
}

unsafe impl Send for TensorRtLlmBackendImpl {}

pub struct TensorRtLlmBackendV2(UnboundedSender<GenerationContext>);

impl TensorRtLlmBackendV2 {
    pub fn new<P: AsRef<Path> + Send, PP: AsRef<Path> + Send>(
        tokenizer: Tokenizer,
        engine_folder: P,
        executor_worker_path: PP,
        max_inflight_requests: usize,
    ) -> Result<Self, TensorRtLlmBackendError> {
        let (engine_folder, executor_worker_path) =
            ensure_paths_exist(engine_folder, executor_worker_path)?;

        // Allocate the IPC layer to communicate with the backend
        let (executor_sender, executor_receiver) = unbounded_channel();

        // Create the FFI backend
        let backend = create_backend_from_engine_folder(&engine_folder, &executor_worker_path)
            .map_err(|e| TensorRtLlmBackendError::Runtime(first_line(e.what(), "Unknown error")))?;

        // Executor looper is responsible for scheduling and pulling requests state at regular interval
        spawn_blocking(move || {
            executor_status_looper(max_inflight_requests, tokenizer, backend, executor_receiver)
        });

        Ok(TensorRtLlmBackendV2(executor_sender))
    }

    fn validate(request: &ValidGenerateRequest) -> InferResult<()> {
        if request.input_ids.is_none() {
            return Err(ValidationError(UnsupportedModality("No token provided")));
        }

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
                Chunk::Text(_) => Ok(()),
                Chunk::Image(_) => Err(ValidationError(UnsupportedModality("image"))),
            },
        }
    }
}

#[async_trait]
impl Backend for TensorRtLlmBackendV2 {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        Self::validate(&request)?;

        // Open-up the stream to send tokens
        let (streamer, receiver) = unbounded_channel::<InferResult<InferStreamResponse>>();

        // Send the context to the executor for scheduling
        let queued = Instant::now();
        match self.0.send(GenerationContext {
            request,
            streamer,
            tokens: Vec::with_capacity(256),
            start: None,
            queued,
        }) {
            Ok(_) => Ok(UnboundedReceiverStream::new(receiver)),
            Err(_) => Err(GenerationError(
                "Failed to submit request to the backend".into(),
            )),
        }
    }

    async fn health(&self, _: bool) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "TensorRT-LLM"
    }
}
