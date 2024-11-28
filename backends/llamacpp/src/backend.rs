use crate::ffi::{
    create_worker_frontend, set_numa_core_affinity, update_numa_affinity, GenerationParams,
    LlamaCppWorkerFrontend, SamplingParams,
};
use async_channel::{unbounded as mpmc_unbounded, Receiver as MpmcReceiver, Sender as MpmcSender};
use async_trait::async_trait;
use cxx::UniquePtr;
use log::warn;
use std::ops::Range;
use std::path::{Path, PathBuf};
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
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info};

/// Detect the number of CPU cores on the machine
///
/// returns: usize Integer greater than 0 representing the number of CPU cores on the machine
///
#[cfg(not(test))]
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

#[cfg(test)]
fn get_num_cores() -> usize {
    match option_env!("TGI_USE_PHYSICAL_CORES")
        .unwrap_or("OFF")
        .to_uppercase()
        .as_str()
    {
        "ON" => 16,
        _ => 32,
    }
}

/// Subdivide the set of CPU cores available on the system to equal, non-overlapping, subsets of CPU cores
///
/// # Arguments
///
/// * `num_cores_per_instance`: Minimum number of cores for each instance
///
/// returns: Vec<Range<usize>, Global>
///
/// # Examples
///
/// ```
///
/// ```
fn get_cores_allocation(num_cores_per_instance: usize) -> Vec<Range<usize>> {
    // Get the total number of cores on the CPU
    let cores_count = get_num_cores();

    // Make sure each instance has some cores available
    let mut effective_num_cores_per_instance = match num_cores_per_instance {
        0 => cores_count,
        _ => num_cores_per_instance,
    };

    // If we have spare cores, let's see if we can give everyone one more core
    let num_instances = cores_count / effective_num_cores_per_instance;

    (0..num_instances)
        .map(|ordinal| {
            let start = ordinal * effective_num_cores_per_instance;
            let end = (ordinal + 1) * effective_num_cores_per_instance - 1;
            start..end
        })
        .collect()
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

pub struct LlamaCppBackend {
    scheduler_sender: UnboundedSender<(GenerationContext, UnboundedSender<InferResult>)>,
    scheduler_handle: JoinHandle<()>,
}

impl LlamaCppBackend {
    /// Attempt to create a new llama.cpp worker from the provided model path
    ///
    /// # Arguments
    ///
    /// * `path`: Path to the GGUF model file to load
    /// * `num_threads`: Number of cores the model is allowed to spawn for its computations
    ///
    /// returns: Result<UniquePtr<LlamaCppWorkerFrontend>, LlamaCppBackendError>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    fn allocate_worker(
        path: &Path,
        num_threads: u32,
    ) -> Result<UniquePtr<LlamaCppWorkerFrontend>, LlamaCppBackendError> {
        create_worker_frontend(&path.display().to_string(), num_threads).map_err(|ref err| {
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

        // Allocate the multi-consumer queue to orchestrate all the workers
        let (backlog_submitter, backlog_receiver) = mpmc_unbounded();

        // Allocate all the workers
        let cores_allocation = get_cores_allocation(num_cores_per_instance as usize);
        cores_allocation.iter().for_each(|affinity| {
            match Self::allocate_worker(path, affinity.len() as u32) {
                Ok(worker) => {
                    let tokenizer = Arc::clone(&tokenizer);
                    let affinity = affinity.clone().collect::<Vec<_>>();
                    let backlog_receiver = backlog_receiver.clone();
                    spawn(move || worker_loop(worker, affinity, tokenizer, backlog_receiver));
                }
                Err(e) => {}
            }
        });

        // Start the scheduler loop
        let (scheduler_sender, scheduler_receiver) = unbounded_channel();
        let scheduler_handle = tokio::spawn(scheduler_loop(scheduler_receiver, backlog_submitter));
        Ok(Self {
            scheduler_sender,
            scheduler_handle,
        })
    }
}

/// llama.cpp worker actual streaming callback, called everytime a new token is being generated
///
/// # Arguments
///
/// * `ctx`: InferContext holding the channel to stream back generated token to the client.
/// *UNSAFE* This parameter is unsafe and represented as a mutable pointer to avoid automatic drop of its
/// referenced resources after the first iteration step.
/// It's the responsibility of the caller to ensure a `Box::from_raw` is taking back full ownership of the pointer
/// for correct deletion.
/// * `new_token_id`: The sampled token identifier
/// * `new_token_logit`: the sampled token identifier log probability
/// * `is_final`: Flag indicating if the sampled token is a final one
/// * `n_generated_tokens`: Counter representing the actual number of token generated at this stage
///
/// returns: bool `true` if the worker should stop the generation at this stage, `false` to continue
///
/// # Examples
///
/// ```
///
/// ```
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

/// Main loop allowing scheduling incoming requests without blocking the main execution thread
///
/// # Arguments
///
/// * `queue`: Synchronized container to receive new request
/// * `backlog`: Synchronized container to dispatch new request towards all the workers for one to pick it up.
///
/// returns: ()
///
/// # Examples
///
/// ```
///
/// ```
async fn scheduler_loop(
    mut queue: UnboundedReceiver<(GenerationContext, UnboundedSender<InferResult>)>,
    backlog: MpmcSender<(GenerationContext, UnboundedSender<InferResult>)>,
) {
    // Let's receive incoming requests
    loop {
        match queue.recv().await {
            None => break,
            Some((ctx, sender)) => {
                if let Err(e) = backlog.send((ctx, sender)).await {
                    todo!("What do we do")
                }
            }
        }
    }
}

/// llama.cpp worker thread receiving incoming requests from the scheduler and handling all generation
/// process along with the streaming logic back to the client.
///
/// # Arguments
///
/// * `backend`: Owned llama.cpp worker with allocated execution resources
/// * `affinity`: Set of CPUs to bind the worker's thread for scheduling
/// * `tokenizer`: Tokenizer to use to decode generated token
/// * `backlog`: Multi-consumers queue holding the requests waiting to be handled by a worker
///
/// returns: ()
///
/// # Examples
///
/// ```
///
/// ```
fn worker_loop(
    mut backend: UniquePtr<LlamaCppWorkerFrontend>,
    affinity: Vec<usize>,
    tokenizer: Arc<Tokenizer>,
    backlog: MpmcReceiver<(GenerationContext, UnboundedSender<InferResult>)>,
) {
    // This loop will mostly decode single token at every step, so no need to rely on parallelism
    tokenizers::utils::parallelism::set_parallelism(false);

    // Bind cores for the current thread and make sure it's taken into account
    set_numa_core_affinity(&affinity);
    update_numa_affinity();

    loop {
        if let Ok((generation, stream)) = backlog.recv_blocking() {
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

#[cfg(test)]
mod tests {
    use crate::backend::{get_cores_allocation, get_num_cores};

    fn test_get_num_cores() {
        std::env::set_var("TGI_USE_PHYSICAL_CORES", "OFF");
        assert_eq!(get_num_cores(), 32);

        std::env::set_var("TGI_USE_PHYSICAL_CORES", "ON");
        assert_eq!(get_num_cores(), 16);
    }

    fn test_get_cores_allocation_single_instance() {
        std::env::set_var("TGI_USE_PHYSICAL_CORES", "OFF");
        let smt_allocation = get_cores_allocation(0);
        assert_eq!(smt_allocation.len(), 1);
        assert_eq!(
            smt_allocation[0].clone().collect::<Vec<_>>(),
            (0..32).collect::<Vec<_>>()
        );

        std::env::set_var("TGI_USE_PHYSICAL_CORES", "ON");
        let smt_allocation = get_cores_allocation(0);
        assert_eq!(smt_allocation.len(), 1);
        assert_eq!(
            smt_allocation[0].clone().collect::<Vec<_>>(),
            (0..16).collect::<Vec<_>>()
        );
    }

    fn test_get_cores_allocation_multi_instances() {
        for cores_per_instance in [1, 2, 4, 8, 16, 3, 7] {
            std::env::set_var("TGI_USE_PHYSICAL_CORES", "OFF");

            let num_instances = 32 / cores_per_instance;
            let smt_allocation = get_cores_allocation(cores_per_instance);

            for i in 0..num_instances {
                let start = i * cores_per_instance;
                let end = start + cores_per_instance;
                assert_eq!(
                    smt_allocation[i].clone().collect::<Vec<_>>(),
                    (start..end).collect::<Vec<_>>()
                );
            }

            std::env::set_var("TGI_USE_PHYSICAL_CORES", "ON");
            let num_instances = 16 / cores_per_instance;
            let smt_allocation = get_cores_allocation(cores_per_instance);
            assert_eq!(smt_allocation.len(), num_instances);

            for i in 0..num_instances {
                let start = i * cores_per_instance;
                let end = start + cores_per_instance;
                assert_eq!(
                    smt_allocation[i].clone().collect::<Vec<_>>(),
                    (start..end).collect::<Vec<_>>()
                );
            }
        }
    }
}
