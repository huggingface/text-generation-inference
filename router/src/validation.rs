/// Payload validation logic
use crate::{GenerateParameters, GenerateRequest};
use rand::rngs::ThreadRng;
use rand::Rng;
use text_generation_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};

const MAX_MAX_NEW_TOKENS: u32 = 512;
const MAX_STOP_SEQUENCES: usize = 4;

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Channel to communicate with the background validation task
    sender: mpsc::Sender<ValidationRequest>,
}

impl Validation {
    pub(crate) fn new(workers: usize, tokenizer: Tokenizer, max_input_length: usize) -> Self {
        // Create channel
        let (validation_sender, validation_receiver) = mpsc::channel(128);

        // Launch background validation task
        tokio::spawn(validation_task(
            workers,
            tokenizer,
            max_input_length,
            validation_receiver,
        ));

        Self {
            sender: validation_sender,
        }
    }

    /// Validate a payload and get the number of tokens in the input
    pub(crate) async fn validate(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, ValidationError> {
        // Create response channel
        let (sender, receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        self.sender.send((request, sender)).await.unwrap();
        // Await on response channel
        // Unwrap is safe here
        receiver.await.unwrap()
    }
}

/// Validation task
/// Load balance the validation requests between multiple validation workers
async fn validation_task(
    workers: usize,
    tokenizer: Tokenizer,
    max_input_length: usize,
    mut receiver: mpsc::Receiver<ValidationRequest>,
) {
    let mut workers_senders = Vec::with_capacity(workers);

    // Create workers
    for _ in 0..workers {
        let tokenizer_clone: Tokenizer = tokenizer.clone().into();
        // Create channel to communicate with worker
        let (worker_sender, worker_receiver) = mpsc::channel(workers);
        workers_senders.push(worker_sender);

        // Spawn worker
        tokio::task::spawn_blocking(move || {
            validation_worker(tokenizer_clone, max_input_length, worker_receiver)
        });
    }

    loop {
        // Load balance requests between workers
        for sender in workers_senders.iter() {
            if let Some(validation_request) = receiver.recv().await {
                sender.send(validation_request).await.unwrap();
            } else {
                return;
            }
        }
    }
}

/// Check the parameters inside the payload and get the number of tokens inside the input using
/// the tokenizer
fn validation_worker(
    tokenizer: Tokenizer,
    max_input_length: usize,
    mut receiver: mpsc::Receiver<ValidationRequest>,
) {
    // Seed rng
    let mut rng = rand::thread_rng();

    // Loop over requests
    while let Some((request, response_tx)) = receiver.blocking_recv() {
        response_tx
            .send(validate(request, &tokenizer, max_input_length, &mut rng))
            .unwrap_or(())
    }
}

fn validate(
    request: GenerateRequest,
    tokenizer: &Tokenizer,
    max_input_length: usize,
    rng: &mut ThreadRng,
) -> Result<ValidGenerateRequest, ValidationError> {
    if request.parameters.temperature <= 0.0 {
        return Err(ValidationError::Temperature);
    }
    if request.parameters.repetition_penalty <= 0.0 {
        return Err(ValidationError::RepetitionPenalty);
    }
    if request.parameters.top_p <= 0.0 || request.parameters.top_p > 1.0 {
        return Err(ValidationError::TopP);
    }
    if request.parameters.top_k < 0 {
        return Err(ValidationError::TopK);
    }
    if request.parameters.max_new_tokens > MAX_MAX_NEW_TOKENS {
        return Err(ValidationError::MaxNewTokens(MAX_MAX_NEW_TOKENS));
    }
    if request.parameters.stop.len() > MAX_STOP_SEQUENCES {
        return Err(ValidationError::StopSequence(
            MAX_STOP_SEQUENCES,
            request.parameters.stop.len(),
        ));
    }

    // If seed is None, assign a random one
    let seed = match request.parameters.seed {
        None => rng.gen(),
        Some(seed) => seed,
    };

    // Get the number of tokens in the input
    match tokenizer.encode(request.inputs.clone(), true) {
        Ok(encoding) => {
            let input_length = encoding.len();

            if input_length > max_input_length {
                Err(ValidationError::InputLength(input_length, max_input_length))
            } else {
                // Return ValidGenerateRequest
                let GenerateParameters {
                    temperature,
                    repetition_penalty,
                    top_k,
                    top_p,
                    do_sample,
                    max_new_tokens,
                    stop: stop_sequences,
                    ..
                } = request.parameters;

                let parameters = NextTokenChooserParameters {
                    temperature,
                    repetition_penalty,
                    top_k: top_k as u32,
                    top_p,
                    do_sample,
                    seed,
                };
                let stopping_parameters = StoppingCriteriaParameters {
                    max_new_tokens,
                    stop_sequences,
                };

                Ok(ValidGenerateRequest {
                    inputs: request.inputs,
                    input_length: input_length as u32,
                    parameters,
                    stopping_parameters,
                })
            }
        }
        Err(err) => Err(ValidationError::Tokenizer(err.to_string())),
    }
}

type ValidationRequest = (
    GenerateRequest,
    oneshot::Sender<Result<ValidGenerateRequest, ValidationError>>,
);

#[derive(Debug)]
pub(crate) struct ValidGenerateRequest {
    pub inputs: String,
    pub input_length: u32,
    pub parameters: NextTokenChooserParameters,
    pub stopping_parameters: StoppingCriteriaParameters,
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("temperature must be strictly positive")]
    Temperature,
    #[error("repetition_penalty must be strictly positive")]
    RepetitionPenalty,
    #[error("top_p must be > 0.0 and <= 1.0")]
    TopP,
    #[error("top_k must be strictly positive")]
    TopK,
    #[error("max_new_tokens must be <= {0}")]
    MaxNewTokens(u32),
    #[error("inputs must have less than {1} tokens. Given: {0}")]
    InputLength(usize, usize),
    #[error("stop supports up to {0} stop sequences. Given: {1}")]
    StopSequence(usize, usize),
    #[error("tokenizer error {0}")]
    Tokenizer(String),
}
