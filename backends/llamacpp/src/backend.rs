use crate::ffi::{
    create_worker_frontend, set_numactl_core_affinity, GenerationParams, LlamaCppWorkerFrontend,
    SamplingParams,
};
use async_trait::async_trait;
use cxx::UniquePtr;
use log::warn;
use std::cell::RefCell;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::spawn;
use text_generation_router::infer::InferError::GenerationError;
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::{
    ValidGenerateRequest, ValidParameters, ValidStoppingParameters,
};
use text_generation_router::{FinishReason, Token};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::{Semaphore, SemaphorePermit, TryAcquireError};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info};

macro_rules! send_or_warn {
    ($send: expr, $err: expr) => {
        if let Err(se) = $send.send(err) {
            warn!(
                "Failed to send message back to the user: {}. Originating error: {}",
                se, e
            );
        }
    };
}

fn get_num_cores() -> usize {
    match option_env!("TGI_USE_PHYSICAL_CORES")
        .unwrap_or("OFF")
        .to_uppercase()
        .as_str()
    {
        "ON" => {
            info!("Using only physical cores on the machine");
            num_cpus::get_physical()
        }
        _ => {
            info!("Using physical and logical cores on the machine");
            num_cpus::get()
        }
    }
}

type InferResult = Result<InferStreamResponse, InferError>;

unsafe impl Send for LlamaCppWorkerFrontend {}

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

pub(crate) struct InferContext<'a> {
    pub(crate) start: Instant,
    pub(crate) stream: UnboundedSender<InferResult>,
    pub(crate) tokenizer: &'a Tokenizer,
    pub(crate) generation: GenerationContext,
}

#[derive(Debug, Error)]
pub enum LlamaCppBackendError {
    #[error("Provided GGUF model path {0} doesn't exist")]
    ModelFileDoesntExist(String),

    #[error("Failed to initialize model from GGUF file {0}: {1}")]
    ModelInitializationFailed(PathBuf, String),
}

struct LlamaCppWorker {
    sender: Sender<(GenerationContext, UnboundedSender<InferResult>)>,
}

impl LlamaCppWorker {
    fn submit(&self, ctx: GenerationContext, sx: UnboundedSender<InferResult>) {
        if let Err(err) = self.sender.send((ctx, sx)) {
            // TODO: What do we do?
        }
    }
}

pub struct LlamaCppBackend {
    scheduler_sender: UnboundedSender<(GenerationContext, UnboundedSender<InferResult>)>,
    scheduler_handle: JoinHandle<()>,
}

impl LlamaCppBackend {
    fn allocate_worker(
        path: &Path,
    ) -> Result<UniquePtr<LlamaCppWorkerFrontend>, LlamaCppBackendError> {
        create_worker_frontend(&path.display().to_string()).map_err(|ref err| {
            LlamaCppBackendError::ModelInitializationFailed(path.to_path_buf(), err.to_string())
        })
    }

    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer: Arc<Tokenizer>,
        num_cores_per_instance: u16,
    ) -> Result<Self, LlamaCppBackendError> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(LlamaCppBackendError::ModelFileDoesntExist(
                path.display().to_string(),
            ));
        }

        let cores_allocation = get_cores_allocation(num_cores_per_instance as usize);

        // Allocate all the workers
        let streams = cores_allocation
            .iter()
            .map(|affinity| match Self::allocate_worker(path) {
                Ok(worker) => {
                    let tokenizer = Arc::clone(&tokenizer);
                    let (sender, receiver) = channel();
                    let affinity = affinity.clone().collect::<Vec<_>>();
                    spawn(move || worker_loop(worker, affinity, tokenizer, receiver));

                    Ok(LlamaCppWorker { sender })
                }
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Start the scheduler loop
        let (scheduler_sender, scheduler_receiver) = unbounded_channel();
        let scheduler_handle = tokio::spawn(scheduler_loop(scheduler_receiver, streams));
        Ok(Self {
            scheduler_sender,
            scheduler_handle,
        })
    }
}

fn get_cores_allocation(num_cores_per_instance: usize) -> Vec<Range<usize>> {
    // Get the total number of cores on the CPU
    let cores_count = get_num_cores();

    // Make sure each instance has some cores available
    let mut effective_num_cores_per_instance = match num_cores_per_instance {
        0 => cores_count,
        _ => num_cores_per_instance,
    };

    // If we have spare cores, let's see if we can give everyone one more core
    let mut num_instances = cores_count / effective_num_cores_per_instance;
    if cores_count - (num_instances * effective_num_cores_per_instance) >= num_instances {
        effective_num_cores_per_instance = effective_num_cores_per_instance + 1;
        warn!("Overriding cores allocation to {effective_num_cores_per_instance} per instance");
    }

    (0..num_instances)
        .map(|ordinal| {
            let start = ordinal * effective_num_cores_per_instance;
            let end = (ordinal + 1) * effective_num_cores_per_instance - 1;
            (start..end)
        })
        .collect()
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

    // Generate response
    let response = match ctx.tokenizer.decode(&[new_token_id], false) {
        Ok(text) => {
            let special = ctx.tokenizer.get_added_vocabulary().is_special_token(&text);
            let token = Token {
                id: new_token_id,
                text,
                logprob: new_token_logit,
                special,
            };

            // Should we generate an ending or intermediate response?
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
                        Err(err) => Err(GenerationError(err.to_string())),
                    }
                }
            }
        }
        Err(ref err) => Err(GenerationError(err.to_string())),
    };

    // Send back to the client
    let status = ctx.stream.send(response).inspect_err(|err| {
        error!("Failed to send back the response: {}", err);
    });
    status.is_err()
}

async fn scheduler_loop(
    mut queue: UnboundedReceiver<(GenerationContext, UnboundedSender<InferResult>)>,
    mut workers: Vec<LlamaCppWorker>,
) {
    // Semaphore allows us to wait for a worker to become available
    let permits = Semaphore::new(workers.len());

    // Let's receive incoming requests
    loop {
        match queue.recv().await {
            None => break,
            Some((ctx, sender)) => {
                let permit = permits.try_acquire();
                if let Err(err) = permit {
                    let _ = sender.send(Err(InferError::Overloaded(err)));
                }

                // We can unwrap because we wouldn't have a semaphore available otherwise
                let worker = workers.pop().unwrap();
                worker.submit(ctx, sender);
            }
        }
    }
}

fn worker_loop(
    mut backend: UniquePtr<LlamaCppWorkerFrontend>,
    affinity: Vec<usize>,
    tokenizer: Arc<Tokenizer>,
    backlog: Receiver<(GenerationContext, UnboundedSender<InferResult>)>,
) {
    // This loop will mostly decode single token at every step, so no need to rely on parallelism
    tokenizers::utils::parallelism::set_parallelism(false);

    // Bind cores for the current thread
    set_numactl_core_affinity(&affinity);

    loop {
        if let Ok((generation, stream)) = backlog.recv() {
            let start = Instant::now();
            let generation_params = generation.generation_params; // copy
            let sampling_params = generation.sampling_params; // copy
            let input_tokens = Arc::clone(&generation.input_tokens);

            // Creating the whole InferContext and pushing it to the heap
            let ctx = Box::new(InferContext {
                start,
                stream,
                tokenizer: &tokenizer,
                generation,
            });

            // We leak the box to avoid it being freed after the first callback call
            // when going out of scope
            unsafe {
                let boxed_ctx = Box::into_raw(ctx);
                if let Err(e) = backend.pin_mut().stream(
                    &input_tokens,
                    generation_params,
                    &sampling_params,
                    boxed_ctx,
                    llama_generate_callback,
                ) {
                    error!("Error while decoding tokens... {}", e.what());
                    // TODO: What error to give back to the user?
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

            // We send the workload to the scheduler
            if let Err(e) = self.scheduler_sender.send((ctx, sx)) {
                Err(InferError::IncompleteGenerationStream)
            } else {
                // We are returning the associated channel as early as we can, potentially closing it up
                Ok(UnboundedReceiverStream::new(rx))
            }
        } else {
            Err(GenerationError("Unsupported modalities".to_string()))
        }
    }

    async fn health(&self, _: bool) -> bool {
        true
    }
}
