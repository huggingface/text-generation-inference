use crate::ffi::{
    create_single_worker_backend, GenerationParams, LlamaCppBackendImpl, SamplingParams,
};
use async_trait::async_trait;
use cxx::UniquePtr;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::{
    ValidGenerateRequest, ValidParameters, ValidStoppingParameters,
};
use text_generation_router::{FinishReason, Token};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info};

type InferResult = Result<InferStreamResponse, InferError>;

unsafe impl Send for LlamaCppBackendImpl {}

impl From<&ValidParameters> for SamplingParams {
    fn from(v: &ValidParameters) -> Self {
        Self {
            top_k: v.top_k,
            top_p: v.top_p,
            frequency_penalty: v.frequency_penalty,
            repetition_penalty: v.repetition_penalty,
            seed: v.seed,
        }
    }
}

impl From<&ValidStoppingParameters> for GenerationParams {
    fn from(v: &ValidStoppingParameters) -> Self {
        Self {
            max_new_tokens: v.max_new_tokens,
            ignore_eos_token: v.ignore_eos_token,
        }
    }
}

#[cfg_attr(debug_assertions, derive(Debug))]
pub(crate) struct GenerationContext {
    pub(crate) input_tokens: Arc<Vec<u32>>,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) generation_params: GenerationParams,
    pub(crate) sampling_params: SamplingParams,
}

pub(crate) struct InferContext {
    pub(crate) start: Instant,
    pub(crate) stream: UnboundedSender<InferResult>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) generation: GenerationContext,
}

#[derive(Debug, Error)]
pub enum LlamaCppBackendError {
    #[error("Provided GGUF model path {0} doesn't exist")]
    ModelFileDoesntExist(String),

    #[error("Failed to initialize model from GGUF file {0}: {1}")]
    ModelInitializationFailed(PathBuf, String),
}

pub struct LlamaCppBackend {
    backlog: Sender<(GenerationContext, UnboundedSender<InferResult>)>,
    _scheduler_handle: JoinHandle<()>,
}

impl LlamaCppBackend {
    pub fn new<P: AsRef<Path> + Send>(
        model_path: P,
        tokenizer: Tokenizer,
    ) -> Result<Self, LlamaCppBackendError> {
        let path = Arc::new(model_path.as_ref());
        if !path.exists() {
            return Err(LlamaCppBackendError::ModelFileDoesntExist(
                path.display().to_string(),
            ));
        }

        let backend = create_single_worker_backend(path.to_str().unwrap()).map_err(|err| {
            LlamaCppBackendError::ModelInitializationFailed(
                path.to_path_buf(),
                err.what().to_string(),
            )
        })?;

        info!(
            "Successfully initialized llama.cpp backend from {}",
            path.display()
        );

        let (submitter, receiver) = channel();
        let handle = unsafe { spawn(|| scheduler_loop(backend, tokenizer, receiver)) };
        Ok(Self {
            backlog: submitter,
            _scheduler_handle: handle,
        })
    }
}

fn llama_generate_callback(
    ctx: *mut InferContext,
    new_token_id: u32,
    new_token_logit: f32,
    is_final: bool,
    n_generated_tokens: usize,
) -> bool {
    debug!("Generated token: {new_token_id} -> logits={new_token_logit}, is_final={is_final} ({n_generated_tokens})");

    let ctx = unsafe { &mut *ctx };

    // Append the new token to the generated ones
    ctx.generation.generated_tokens.push(new_token_id);

    // Decode token
    let token = match ctx.tokenizer.decode(&[new_token_id], false) {
        Ok(text) => {
            let special = ctx.tokenizer.get_added_vocabulary().is_special_token(&text);
            Ok(Token {
                id: new_token_id,
                text,
                logprob: new_token_logit,
                special,
            })
        }
        Err(ref err) => Err(InferError::GenerationError(err.to_string())),
    };

    // Create the streamed response
    let response = match token {
        Ok(token) => {
            match is_final {
                false => Ok(InferStreamResponse::Intermediate {
                    token,
                    top_tokens: vec![],
                }),
                true => {
                    // Decode the whole text
                    match ctx
                        .tokenizer
                        .decode(&ctx.generation.generated_tokens, false)
                    {
                        Ok(text) => Ok(InferStreamResponse::End {
                            token,
                            top_tokens: vec![],
                            generated_text: GeneratedText {
                                text,
                                generated_tokens: n_generated_tokens as u32,
                                finish_reason: FinishReason::Length,
                                seed: Some(ctx.generation.sampling_params.seed),
                            },
                            start: ctx.start,
                            queued: ctx.start,
                        }),
                        Err(err) => Err(InferError::GenerationError(err.to_string())),
                    }
                }
            }
        }
        Err(err) => Err(err),
    };

    // Send back to the client
    let should_stop = if let Err(ref _err) = ctx.stream.send(response) {
        error!("Failed to send back the response to the client, cancelling request");
        true
    } else {
        true
    };

    should_stop
}

unsafe fn scheduler_loop(
    mut backend: UniquePtr<LlamaCppBackendImpl>,
    tokenizer: Tokenizer,
    backlog: Receiver<(GenerationContext, UnboundedSender<InferResult>)>,
) {
    // This loop will mostly decode single token at every step, so no need to rely on parallelism
    tokenizers::utils::parallelism::set_parallelism(false);

    loop {
        if let Ok((generation, stream)) = backlog.recv() {
            let start = Instant::now();
            let tokenizer = tokenizer.clone();
            let generation_params = generation.generation_params; // copy
            let sampling_params = generation.sampling_params; // copy
            let input_tokens = Arc::clone(&generation.input_tokens);

            // Creating the whole InferContext and pushing it to the heap
            {
                let ctx = Box::new(InferContext {
                    start,
                    stream,
                    tokenizer,
                    generation,
                });

                let boxed_ctx = Box::into_raw(ctx);

                if let Err(e) = backend.pin_mut().stream(
                    &input_tokens,
                    generation_params,
                    &sampling_params,
                    boxed_ctx,
                    llama_generate_callback,
                ) {
                    error!("Error while decoding tokens... {}", e.what());
                }

                // Make sure we re-keep track of the OpaqueStream box
                let _ = Box::from_raw(boxed_ctx);
            }
        } else {
            info!("IPC channel is closed, exiting the scheduler loop");
            break;
        }
    }
}

#[async_trait]
impl Backend for LlamaCppBackend {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<InferResult>, InferError> {
        if let Some(input_ids) = request.input_ids {
            let (sx, rx) = unbounded_channel();
            let sampling_params = SamplingParams::from(&request.parameters);
            let generation_params = GenerationParams::from(&request.stopping_parameters);

            let ctx = GenerationContext {
                input_tokens: Arc::clone(&input_ids),
                generated_tokens: Vec::with_capacity(generation_params.max_new_tokens as usize),
                generation_params,
                sampling_params,
            };

            match self.backlog.send((ctx, sx)) {
                Ok(_) => Ok(UnboundedReceiverStream::new(rx)),
                Err(_) => Err(InferError::GenerationError(
                    "Failed to sent the request".to_string(),
                )),
            }
        } else {
            Err(InferError::GenerationError(
                "Unsupported modalities".to_string(),
            ))
        }
    }

    async fn health(&self, _: bool) -> bool {
        true
    }
}
