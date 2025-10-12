use async_trait::async_trait;
use cxx::UniquePtr;
use hashbrown::HashMap;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::TryAcquireError;
use tokio::task::spawn_blocking;
use tokio::time::{Duration, Instant};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};

use text_generation_router::infer::InferError::{GenerationError, ValidationError};
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::ValidationError::{
    EmptyInput, Grammar, TopNTokensDisabled, UnsupportedModality,
};
use text_generation_router::validation::{Chunk, ValidGenerateRequest, ValidGrammar};
use text_generation_router::Token;

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{
    create_backend_from_engine_folder, FinishReason, GenerationStep, GrammarType,
    TensorRtLlmBackendImpl,
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

    /// output_buffer stores the output for detecting stop sequences
    output_buffer: Option<String>,
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
        if step.has_error {
            return Err(GenerationError(step.error_msg.clone()));
        }

        if !step.token_id_valid {
            return Err(GenerationError(
                "GenerationStep contains no token_id".to_string(),
            ));
        }

        if !step.log_prob_valid {
            return Err(GenerationError(
                "GenerationStep contains no log_prob".to_string(),
            ));
        }

        Ok(Self {
            id: step.token_id,
            log_prob: step.log_prob,
            is_final: step.is_final,
            finish_reason: step.finish_reason,
        })
    }
}

struct InFlightRequest {
    request_id: u64,
    ctx: GenerationContext,
}

/// request_looper reads from the backlog, sends the request to backend,
/// and then transfer the request context to the response_looper via in_flights.
fn request_looper(
    backend: Arc<UniquePtr<TensorRtLlmBackendImpl>>,
    mut backlog: UnboundedReceiver<GenerationContext>,
    in_flights: UnboundedSender<InFlightRequest>,
) {
    loop {
        let Some(ctx) = backlog.blocking_recv() else {
            break;
        };
        // Submit all the request to the executor and move the context to the in-flight tracker
        let request = &ctx.request;
        let generation_params = &request.parameters;
        let stopping_params = &request.stopping_parameters;
        let input_ids = request.input_ids.as_deref();
        let top_k = if generation_params.do_sample {
            generation_params.top_k
        } else {
            1
        };

        let (grammar_type, grammar_value): (GrammarType, &str) =
            if let Some(grammar) = &generation_params.grammar {
                match grammar {
                    ValidGrammar::Json(v) => (GrammarType::Json, v),
                    ValidGrammar::Regex(v) => (GrammarType::Regex, v),
                }
            } else {
                (GrammarType::None, "")
            };

        // Submit to the TensorRT-LLM executor for scheduling
        match backend.submit(
            &input_ids.unwrap(), // This is checked beforehand in validate()
            stopping_params.max_new_tokens,
            top_k,
            generation_params.top_p,
            generation_params.temperature,
            generation_params.repetition_penalty,
            generation_params.frequency_penalty,
            generation_params.seed,
            grammar_type,
            grammar_value,
        ) {
            Ok(request_id) => {
                // Insert the context linked to the generated request id in the tracker
                debug!("[in-flight] Added {}", request_id);
                if let Err(err) = in_flights.send(InFlightRequest { request_id, ctx }) {
                    error!("[in-flight] Send failed {}", err);
                    return;
                }
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
    }
}

/// response_looper awaits requests from in_flights if there are no active ones
/// or awaits for tokens from backend. The tokens are processed and sent back.
fn response_looper(
    max_inflight_requests: usize,
    tokenizer: Tokenizer,
    created_time: Instant,
    backend: Arc<UniquePtr<TensorRtLlmBackendImpl>>,
    mut in_flight_recv: UnboundedReceiver<InFlightRequest>,
) {
    // // Track the tuple (request_id, stream) for each request
    let mut in_flights =
        HashMap::<u64, GenerationContext>::with_capacity(max_inflight_requests * 2);
    loop {
        if in_flights.is_empty() {
            // If there are no active requests, block on Rust channel instead of C++ side.
            let Some(req) = in_flight_recv.blocking_recv() else {
                return;
            };
            in_flights.insert(req.request_id, req.ctx);
        }
        match backend.pull_tokens() {
            Ok(responses) => {
                // Fetch all pending requests, in case we are receiving tokens from them.
                loop {
                    match in_flight_recv.try_recv() {
                        Ok(req) => in_flights.insert(req.request_id, req.ctx),
                        Err(err) => match err {
                            TryRecvError::Empty => break,
                            TryRecvError::Disconnected => return,
                        },
                    };
                }

                // Iterate through all the decoded token
                for step in responses.deref() {
                    if let Some(ctx) = in_flights.get_mut(&step.request_id) {
                        // Update the starting timestamp if not set
                        if ctx.start.is_none() {
                            if step.first_scheduled_time_ns_valid {
                                if step.first_scheduled_time_ns >= 0 {
                                    ctx.start = created_time.checked_add(Duration::from_nanos(
                                        step.first_scheduled_time_ns as u64,
                                    ));
                                } else {
                                    ctx.start = created_time.checked_sub(Duration::from_nanos(
                                        -step.first_scheduled_time_ns as u64,
                                    ));
                                }
                            }

                            if ctx.start.is_none() {
                                ctx.start = Some(Instant::now());
                            }
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
                            backend.cancel(step.request_id);
                            let _ = in_flights.remove(&step.request_id);
                        }
                    } else {
                        match step.finish_reason {
                            FinishReason::Cancelled => {
                                // The client has canceled the request, so this should not generate a
                                // warning.
                                debug!("Cancelled request {}", step.request_id);
                            }
                            _ => {
                                warn!("Untracked request {}", step.request_id);
                            }
                        }
                    }
                }
            }
            Err(ref err) => {
                error!("Failed to get responses from the executor: {}.", err.what());
                break;
            }
        }
    }
}

fn post_process_decoded_token(
    tokenizer: &Tokenizer,
    ctx: &mut GenerationContext,
    mut decoded_token: DecodedToken,
) -> InferResult<InferStreamResponse> {
    match tokenizer.decode(&[decoded_token.id], false) {
        Ok(text) => {
            let is_special = tokenizer.get_added_vocabulary().is_special_token(&text);

            if let Some(buf) = ctx.output_buffer.as_mut() {
                if buf.len() + text.len() > buf.capacity() {
                    let mut start = buf.len() + text.len() - buf.capacity();
                    while start <= buf.len() && !buf.is_char_boundary(start) {
                        start += 1;
                    }
                    buf.drain(..start);
                }
                buf.push_str(&text);

                for stop_seq in &ctx.request.stopping_parameters.stop_sequences {
                    let start = if 1 + buf.len() > text.len() + stop_seq.len() {
                        let mut start = 1 + buf.len() - text.len() - stop_seq.len();
                        while start > 0 && !buf.is_char_boundary(start) {
                            start -= 1;
                        }
                        start
                    } else {
                        0
                    };
                    if buf[start..].contains(stop_seq) {
                        decoded_token.is_final = true;
                        decoded_token.finish_reason = FinishReason::StopWords;
                    }
                }
            }

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

    let mut config_path = PathBuf::from(engine_folder);
    config_path.push("config.json");

    if !config_path.exists() {
        let err = TensorRtLlmBackendError::ConfigNotFound(engine_folder.to_path_buf());

        error!("Path validation failed: {}", err,);
        return Err(err);
    }

    let mut generation_config_path = PathBuf::from(engine_folder);
    generation_config_path.push("generation_config.json");

    if !generation_config_path.exists() {
        let err = TensorRtLlmBackendError::GenerationConfigNotFound(engine_folder.to_path_buf());

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
unsafe impl Sync for TensorRtLlmBackendImpl {}

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

        let (in_flight_sender, in_flight_receiver) = unbounded_channel();

        // This is a reference point to convert time from c++ time_point
        // to rust Instant.
        let created_time = Instant::now();

        let encoded_vocab = {
            let vocab = tokenizer.get_vocab(true);
            let mut tokens: Vec<String> = vocab.keys().map(|x| x.clone()).collect();
            tokens.sort_by(|a, b| vocab.get(a).cmp(&vocab.get(b)));
            tokens
        };

        let tokenizer_str = tokenizer
            .to_string(false)
            .map_err(|e| TensorRtLlmBackendError::Tokenizer(e.to_string()))?;

        // Create the FFI backend
        let backend = create_backend_from_engine_folder(
            &engine_folder,
            &executor_worker_path,
            &tokenizer_str,
            encoded_vocab,
        )
        .map_err(|e| TensorRtLlmBackendError::Runtime(first_line(e.what(), "Unknown error")))?;

        let backend = Arc::new(backend);
        let backend_response = backend.clone();

        // Request looper is responsible for scheduling requests
        spawn_blocking(move || request_looper(backend, executor_receiver, in_flight_sender));

        // Response looper is responsible for awaiting tokens and send them back
        spawn_blocking(move || {
            response_looper(
                max_inflight_requests,
                tokenizer,
                created_time,
                backend_response,
                in_flight_receiver,
            )
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
        let output_buffer = request
            .stopping_parameters
            .stop_sequences
            .iter()
            .map(|x| x.len())
            .max()
            .map(|m| String::with_capacity(m + 32)); // TODO: is this number enough?
        match self.0.send(GenerationContext {
            request,
            streamer,
            tokens: Vec::with_capacity(256),
            start: None,
            queued,
            output_buffer,
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
