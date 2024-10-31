use crate::ffi::{
    create_single_worker_backend, GenerationParams, LlamaCppBackendImpl, SamplingParams,
};
use async_trait::async_trait;
use cxx::UniquePtr;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, SendError, Sender};
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};
use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::{
    ValidGenerateRequest, ValidParameters, ValidStoppingParameters,
};
use text_generation_router::Token;
use thiserror::Error;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::TryAcquireError;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, info};

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
struct InferContext {
    pub(crate) stream: UnboundedSender<Result<InferStreamResponse, InferError>>,
    pub(crate) input_tokens: Arc<Vec<u32>>,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) generation_params: GenerationParams,
    pub(crate) sampling_params: SamplingParams,
}

#[derive(Debug, Error)]
pub enum LlamaCppBackendError {
    #[error("Provided GGUF model path {0} doesn't exist")]
    ModelFileDoesntExist(String),

    #[error("Failed to initialize model from GGUF file {0}: {1}")]
    ModelInitializationFailed(PathBuf, String),
}

pub struct LlamaCppBackend {
    backlog: Sender<InferContext>,
    scheduler_handle: JoinHandle<()>,
}

impl LlamaCppBackend {
    pub fn new<P: AsRef<Path> + Send>(model_path: P) -> Result<Self, LlamaCppBackendError> {
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
        let handle = spawn(|| scheduler_loop(backend, receiver));
        Ok(Self {
            backlog: submitter,
            scheduler_handle: handle,
        })
    }
}

fn scheduler_loop(
    mut backend: UniquePtr<LlamaCppBackendImpl>,
    mut backlog: Receiver<InferContext>,
) {
    loop {
        println!("Looping");
        if let Ok(mut ctx) = backlog.recv() {
            println!("{ctx:?}, {}", &ctx.generated_tokens.capacity());
            match backend.pin_mut().generate(
                &ctx.input_tokens,
                &mut ctx.generated_tokens,
                &ctx.generation_params,
                &ctx.sampling_params,
                |new_token_id: u32, new_token_logit: f32, is_eos: bool| {
                    let response = InferStreamResponse::Intermediate {
                        token: Token {
                            id: new_token_id,
                            text: "".to_string(),
                            logprob: new_token_logit,
                            special: false,
                        },
                        top_tokens: vec![],
                    };
                    println!("Generated token: {response:?}");
                    // let _ = tokio::spawn(async {
                    //     match ctx.stream.send(Ok(response)) {
                    //         Ok(_) => {}
                    //         Err(ref err) => {
                    //             error!(
                    //                 "Failed to send back token to the client: {}",
                    //                 err.to_string()
                    //             );
                    //         }
                    //     }
                    // });
                },
            ) {
                Ok(n_tokens) => {
                    unsafe {
                        ctx.generated_tokens.set_len(n_tokens);
                    }
                    println!(
                        "Generated {} tokens -> {:?}",
                        n_tokens, &ctx.generated_tokens
                    );
                }
                Err(err) => println!("Error: {}", err),
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
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        if let Some(input_ids) = request.input_ids {
            let (sx, rx) = unbounded_channel();
            let sampling_params = SamplingParams::from(&request.parameters);
            let generation_params = GenerationParams::from(&request.stopping_parameters);

            let ctx = InferContext {
                stream: sx,
                input_tokens: Arc::clone(&input_ids),
                generated_tokens: Vec::with_capacity(generation_params.max_new_tokens as usize),
                generation_params,
                sampling_params,
            };

            match self.backlog.send(ctx) {
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
