use std::future::Future;
use std::path::Path;
use std::pin::{pin, Pin};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use cxx::UniquePtr;
use log::{info, warn};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::RwLock;
use tokio::time::{Instant, sleep};
use tokio_stream::{Stream, StreamExt};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{instrument, Level, span};

use text_generation_router::{FinishReason, Token};
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{create_tensorrt_llm_backend, TensorRtLlmBackendImpl};

// macro_rules! propagate {
//     ($ctx: expr, $res: expr) => {
//         $ctx.sender
//             .send($res)
//             .expect("Failed to propagate error back to the transport layer")
//     };
// }

type InferResult<T> = Result<T, InferError>;

/// Holds the user provided input to be executed along with a channel allowing
/// to bubble up all the generated tokens for that tokens the to end stream.
// pub struct InferenceContext {
//     /// User provided request
//     request: ValidGenerateRequest,
//
//     /// Inter-process communication handler moving token from the executor thread to the HTTP server
//     sender: UnboundedSender<InferResult<InferStreamResponse>>,
//
//     /// Pin the instant this inference context was submitted
//     when: Instant,
//
//     /// Span that will live as long as entry
//     span: Span,
// }

pub(crate) struct Generation {
    executor: Arc<RwLock<UniquePtr<TensorRtLlmBackendImpl>>>,
    done: Arc<AtomicBool>,
}

pub struct GenerationContext(
    UnboundedSender<InferResult<InferStreamResponse>>,
    Arc<AtomicBool>,
);

impl Stream for Generation {
    type Item = usize;

    fn poll_next(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done.load(Ordering::Relaxed) {
            Poll::Ready(None)
        } else {
            let pinned = pin!(self.executor.read());
            match pinned.poll(ctx) {
                Poll::Ready(executor_r) => {
                    let ready = executor_r.num_responses_ready();
                    if ready == 0 {
                        let waker = ctx.waker().clone();
                        tokio::spawn(async {
                            sleep(Duration::from_millis(10)).await;
                            waker.wake();
                        });
                        Poll::Pending
                    } else {
                        info!("Ready: {}", ready);
                        let waker = ctx.waker().clone();
                        tokio::spawn(async {
                            sleep(Duration::from_millis(100)).await;
                            waker.wake();
                        });
                        Poll::Ready(Some(ready))
                    }
                }
                Poll::Pending => {
                    let waker = ctx.waker().clone();
                    tokio::spawn(async {
                        sleep(Duration::from_millis(100)).await;
                        waker.wake();
                    });
                    Poll::Pending
                }
            }
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
    // Allowing sending user requests to the TensorRT-LLM executor thread
    // batcher: UnboundedSender<InferenceContext>,
    backend: Arc<RwLock<UniquePtr<TensorRtLlmBackendImpl>>>,
}

impl TensorRtLlmBackend {
    pub fn new<P: AsRef<Path> + Send + 'static, PP: AsRef<Path> + Send + 'static>(
        _tokenizer: Tokenizer,
        engine_folder: P,
        _executor_worker_path: Option<PP>,
    ) -> Result<Self, TensorRtLlmBackendError> {
        Ok(TensorRtLlmBackend {
            backend: Arc::new(RwLock::new(create_tensorrt_llm_backend(
                engine_folder.as_ref().to_str().unwrap(),
                "",
            ))),
        })
    }
}

#[async_trait]
impl Backend for TensorRtLlmBackend {
    #[instrument(skip_all)]
    fn schedule(
        &self,
        _request: ValidGenerateRequest,
    ) -> InferResult<UnboundedReceiverStream<InferResult<InferStreamResponse>>> {
        // Channel to stream the generated token as they come from the worker thread back to the transport layer
        let (sender, receiver) = unbounded_channel();

        let executor = self.backend.clone();
        tokio::spawn(async move {
            // Submit the request to the batcher
            let request_id = span!(Level::DEBUG, "[EXECUTOR][SUBMIT]")
                .in_scope(|| async {
                    info!("Acquiring lock for submit");
                    let mut handle = executor.write().await;
                    let request_id = handle.pin_mut().submit(
                        &vec![2, 2926, 1503, 603, 20189],
                        50,
                        1.0,
                        1.0,
                        2014,
                    );

                    info!("Releasing lock for submit");
                    return request_id;
                })
                .await;

            let mut generation = Generation {
                executor: executor.clone(),
                done: Arc::new(AtomicBool::new(false)),
            };

            while let Some(num_tokens_ready) = generation.next().await {
                span!(
                    Level::DEBUG,
                    "[EXECUTOR][GENERATE]",
                    request_id = request_id,
                    num_tokens_ready = num_tokens_ready
                )
                .in_scope(|| async {
                    let ctx = Box::new(GenerationContext(
                        sender.clone(),
                        Arc::clone(&generation.done),
                    ));
                    let mut executor_w = executor.write().await;

                    info!("Acquired write lock stream");
                    executor_w.pin_mut().stream_tokens(
                        request_id,
                        ctx,
                        |ctx: Box<GenerationContext>, token: u32, logprob: f32, is_final: bool| {
                            info!("Sending token: {} (final: {})", token, is_final);
                            let out = if is_final {
                                ctx.1.store(true, Ordering::Relaxed);
                                InferStreamResponse::End {
                                    token: Token {
                                        id: token,
                                        text: "".into(),
                                        logprob,
                                        special: false,
                                    },
                                    top_tokens: vec![],
                                    generated_text: GeneratedText {
                                        text: "".into(),
                                        generated_tokens: u32::MAX,
                                        finish_reason: FinishReason::EndOfSequenceToken,
                                        seed: None,
                                    },
                                    start: Instant::now(),
                                    queued: Instant::now(),
                                }
                            } else {
                                InferStreamResponse::Intermediate {
                                    token: Token {
                                        id: token,
                                        text: "".into(),
                                        logprob,
                                        special: false,
                                    },
                                    top_tokens: vec![],
                                }
                            };
                            ctx.0
                                .send(Ok(out))
                                .expect("Failed to send back generated token");
                        },
                    );
                    info!("Releasing write lock stream")
                })
                .await;
            }
        });

        Ok(UnboundedReceiverStream::new(receiver))
    }

    async fn health(&self, _current_health: bool) -> bool {
        true
    }
}

// async fn background_looper<P: AsRef<Path>, PP: AsRef<Path>>(
//     engine_folder: P,
//     _executor_worker: Option<PP>,
//     tokenizer: Tokenizer,
//     mut receiver: UnboundedReceiver<InferenceContext>,
// ) {
//     let mut backend = create_tensorrt_llm_backend(engine_folder.as_ref().to_str().unwrap(), "");
//
//     while !(receiver.is_closed()) {
//         // Receive the incoming request
//         if let Some(ctx) = receiver.recv().await {
//             debug!("Processing new incoming request");
//
//             // We only support single, textual chunk
//             if ctx.request.inputs.len() != 1 {
//                 propagate!(
//                     ctx,
//                     Err(InferError::GenerationError(format!(
//                         "Unsupported multi-chunk ({}) input",
//                         ctx.request.inputs.len()
//                     )))
//                 );
//             }
//
//             let input = ctx
//                 .request
//                 .inputs
//                 .first()
//                 .expect("Single chunk checked above");
//             let params = ctx.request.parameters;
//         }
//     }

// Receive the incoming request
// if let Some(ctx) = receiver.recv().await {
//     debug!("Processing new incoming request");

//     // We only support single, textual chunk
//     if ctx.request.inputs.len() != 1 {
//         propagate!(
//             ctx,
//             Err(InferError::GenerationError(format!(
//                 "Unsupported multi-chunk ({}) input",
//                 ctx.request.inputs.len()
//             )))
//         );
//     }
//
//     // Unpack parameters
//     let inputs = ctx.request.inputs;
//     let params = ctx.request.parameters;
//
//     match inputs.first().unwrap() {
//         Chunk::Text(text) => match tokenizer.encode(text.as_str(), true) {
//             Err(err) => {
//                 propagate!(ctx, Err(InferError::GenerationError(err.to_string())))
//             }
//             Ok(encoding) => {
//                 // spawn_blocking(|| {
//                 //     info!("Submitting request to TensorRT-LLM executor");
//                 //     let mut executor = backend.blocking_write();
//                 // })
//                 // .await
//                 // .expect("");
//             }
//         },
//         Chunk::Image(_) => propagate!(
//             ctx,
//             Err(InferError::GenerationError(
//                 "Image input is not supported yet.".into(),
//             ))
//         ),
//     }
// };
// }
