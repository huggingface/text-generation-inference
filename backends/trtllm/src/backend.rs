use std::future::Future;
use std::path::Path;
use std::pin::{pin, Pin};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use cxx::UniquePtr;
use log::{error, warn};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::RwLock;
use tokio::time::{sleep, Instant};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::{Stream, StreamExt};
use tracing::{instrument, span, Level};

use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::ValidationError::UnsupportedModality;
use text_generation_router::validation::{Chunk, ValidGenerateRequest, ValidationError};
use text_generation_router::{FinishReason, Token};

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{create_tensorrt_llm_backend, GenerationStep, TensorRtLlmBackendImpl};

// Value used to poll the state of the generation stream
static POLLING_INTERVAL_US: OnceLock<u64> = OnceLock::new();

type InferResult<T> = Result<T, InferError>;

pub(crate) struct Generation {
    executor: Arc<RwLock<UniquePtr<TensorRtLlmBackendImpl>>>,
    done: Arc<AtomicBool>,
}

/// Holds the user provided input to be executed along with a channel allowing
/// to bubble up all the generated tokens for that tokens the to end stream.
pub struct GenerationContext {
    sender: UnboundedSender<InferResult<InferStreamResponse>>,
    tokenizer: Arc<Tokenizer>,
    tokens: Vec<u32>,
    done: Arc<AtomicBool>,
    queued: Instant,
    start: Option<Instant>,
}

impl Stream for Generation {
    type Item = usize;

    fn poll_next(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let interval = POLLING_INTERVAL_US.get_or_init(|| {
            u64::from_str(option_env!("TRTLLM_BACKEND_POLLING_INTERVAL_US").unwrap_or("100"))
                .expect("Invalid value provided for envvar POLLING_INTERVAL_US")
        });

        if !self.done.load(Ordering::Relaxed) {
            let backend = pin!(self.executor.read());
            let status = match backend.poll(ctx) {
                Poll::Ready(executor_r) => {
                    let ready = executor_r.num_responses_ready();
                    if ready == 0 {
                        Poll::Pending
                    } else {
                        Poll::Ready(Some(ready))
                    }
                }
                Poll::Pending => Poll::Pending,
            };

            let waker = ctx.waker().clone();
            tokio::spawn(async {
                sleep(Duration::from_micros(*interval)).await;
                waker.wake();
            });

            status
        } else {
            Poll::Ready(None) // end of stream
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}

unsafe impl Send for TensorRtLlmBackendImpl {}
unsafe impl Sync for TensorRtLlmBackendImpl {}

/// Implements the logic to execute generation with TensorRT-LLM executor API in background
pub struct TensorRtLlmBackend {
    tokenizer: Arc<Tokenizer>,

    // Backing the backend behind a RwLock to allow concurrent read access to retrieve
    // the number of available tokens (read only) in the Generation stream
    backend: Arc<RwLock<UniquePtr<TensorRtLlmBackendImpl>>>,
}

impl TensorRtLlmBackend {
    pub fn new<P: AsRef<Path> + Send + 'static, PP: AsRef<Path> + Send + 'static>(
        tokenizer: Tokenizer,
        engine_folder: P,
        executor_worker_path: PP,
    ) -> Result<Self, TensorRtLlmBackendError> {
        Ok(TensorRtLlmBackend {
            tokenizer: Arc::new(tokenizer),
            backend: Arc::new(RwLock::new(create_tensorrt_llm_backend(
                engine_folder.as_ref().to_str().unwrap(),
                executor_worker_path.as_ref().to_str().unwrap(),
            ))),
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

    fn generate(
        &self,
        sender: UnboundedSender<InferResult<InferStreamResponse>>,
        tokens: Vec<u32>,
        top_k: u32,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        frequency_penalty: f32,
        seed: u64,
    ) {
        let tokenizer = Arc::clone(&self.tokenizer);
        let executor = Arc::clone(&self.backend);

        // Let's push this in async context
        tokio::spawn(async move {
            // Define the generation state
            let mut generation = Generation {
                executor: executor.clone(),
                done: Arc::new(AtomicBool::new(false)),
            };

            // Define the context over the generation
            // TODO(asap): Do we really need so many shared-ownership?
            let ctx = Box::new(GenerationContext {
                sender: sender.clone(),
                tokenizer,
                tokens: vec![],
                done: Arc::clone(&generation.done),
                start: None,
                queued: Instant::now(),
            });

            // We are leaking the context on-purpose to avoid the box being dropped while there are
            // still computation ongoing
            // TODO(asap): Can we achieve the same with an Arc<Box<T>> without the need to go unsafe?
            let ctx_ = Box::leak(ctx);

            // Submit the request to the batcher
            let request_id = span!(Level::DEBUG, "submit")
                .in_scope(|| async {
                    let mut handle = executor.write().await;
                    let request_id = handle.pin_mut().submit(
                        &tokens,
                        top_k as i32,
                        top_p,
                        temperature,
                        repetition_penalty,
                        frequency_penalty,
                        seed,
                    );

                    request_id
                })
                .await;

            while let Some(_) = generation.next().await {
                let mut executor_w = executor.write().await;
                let executor = executor_w.pin_mut();

                span!(Level::DEBUG, "decode")
                    .in_scope(|| async {
                        unsafe {
                            executor.stream_tokens(
                                request_id,
                                ctx_,
                                |ctx: *mut GenerationContext, step: GenerationStep| {
                                    let inner_ctx = &mut *ctx;

                                    // Update the timestamp at which the request started effectively
                                    // Can be a bit off, would need to be before the callback, let's see
                                    inner_ctx.start.get_or_insert(Instant::now());
                                    inner_ctx.done.store(step.is_final, Ordering::Relaxed);

                                    // Ensure we are not running into errors
                                    let parcel = if !step.has_error {
                                        // Insert the latest generated token to the tracker
                                        inner_ctx.tokens.push(step.token_id);

                                        // Decode the token
                                        let text = inner_ctx
                                            .tokenizer
                                            .decode(&[step.token_id], true)
                                            .expect("Failed to decode token");

                                        let special = inner_ctx
                                            .tokenizer
                                            .get_added_vocabulary()
                                            .is_special_token(&text);

                                        // Create the structure holding the token
                                        let token = Token {
                                            id: step.token_id,
                                            text,
                                            logprob: step.log_prob,
                                            special,
                                        };

                                        if step.is_final {
                                            let generated_text = inner_ctx
                                                .tokenizer
                                                .decode(&inner_ctx.tokens, true)
                                                .expect("Failed to decode generated_tokens");

                                            Ok(InferStreamResponse::End {
                                                token,
                                                top_tokens: vec![],
                                                generated_text: GeneratedText {
                                                    text: generated_text,
                                                    generated_tokens: inner_ctx.tokens.len() as u32,
                                                    finish_reason: FinishReason::EndOfSequenceToken,
                                                    seed: None,
                                                },
                                                start: inner_ctx.start.unwrap_or(Instant::now()),
                                                queued: inner_ctx.queued,
                                            })
                                        } else {
                                            Ok(InferStreamResponse::Intermediate {
                                                token,
                                                top_tokens: vec![],
                                            })
                                        }
                                    } else {
                                        error!("Error caught while decoding: {}", &step.error_msg);
                                        Err(InferError::GenerationError(step.error_msg))
                                    };

                                    // Send the parcel to the client
                                    inner_ctx
                                        .sender
                                        .send(parcel)
                                        .expect("Failed to sent msg through the channel");
                                },
                            );
                        }
                    })
                    .await;
            }

            // "Properly" free the shared context...
            // TODO: clean that piece of sh** asap
            unsafe {
                let _ = Box::from_raw(ctx_);
            }
        });
    }
}

#[async_trait]
impl Backend for TensorRtLlmBackend {
    #[instrument(skip_all)]
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> InferResult<UnboundedReceiverStream<InferResult<InferStreamResponse>>> {
        // Let's add a few more validation
        let input = TensorRtLlmBackend::validate(&request)?;

        // Channel to stream the generated token as they come from the worker thread back to the transport layer
        let (sender, receiver) = unbounded_channel();

        // Unpack parameters
        let params = &request.parameters;

        // Preprocess the inputs to send to TRTLLM backend
        let encoding = self
            .tokenizer
            .encode(input.as_str(), true)
            .map_err(|e| InferError::GenerationError(e.to_string()))?;

        // Generate the response
        self.generate(
            sender,
            Vec::from(encoding.get_ids()),
            params.top_k,
            params.top_p,
            params.temperature,
            params.repetition_penalty,
            params.frequency_penalty,
            params.seed,
        );

        Ok(UnboundedReceiverStream::new(receiver))
    }

    async fn health(&self, _current_health: bool) -> bool {
        true
    }
}
