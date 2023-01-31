/// Payload validation logic
use crate::{ErrorResponse, GenerateRequest};
use axum::http::StatusCode;
use axum::Json;
use rand::rngs::ThreadRng;
use rand::Rng;
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
        // Crate channel
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
    ) -> Result<(usize, GenerateRequest), ValidationError> {
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
    mut request: GenerateRequest,
    tokenizer: &Tokenizer,
    max_input_length: usize,
    rng: &mut ThreadRng,
) -> Result<(usize, GenerateRequest), ValidationError> {
    if request.parameters.temperature <= 0.0 {
        return Err(ValidationError::Temperature);
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
    if request.parameters.seed.is_none() {
        request.parameters.seed = Some(rng.gen());
    }

    // Get the number of tokens in the input
    match tokenizer.encode(request.inputs.clone(), true) {
        Ok(inputs) => {
            let input_length = inputs.len();

            if input_length > max_input_length {
                Err(ValidationError::InputLength(input_length, max_input_length))
            } else {
                Ok((input_length, request))
            }
        }
        Err(err) => Err(ValidationError::Tokenizer(err.to_string())),
    }
}

type ValidationRequest = (
    GenerateRequest,
    oneshot::Sender<Result<(usize, GenerateRequest), ValidationError>>,
);

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("temperature must be strictly positive")]
    Temperature,
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

impl From<ValidationError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: ValidationError) -> Self {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    }
}
