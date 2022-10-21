/// Payload validation logic
use crate::GenerateRequest;
use axum::http::StatusCode;
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::{
    DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper,
    TokenizerImpl,
};
use tokio::sync::{mpsc, oneshot};

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
        let tokenizer_clone = tokenizer.clone();
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
    tokenizer: TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    max_input_length: usize,
    mut receiver: mpsc::Receiver<ValidationRequest>,
) {
    // Loop over requests
    while let Some((request, response_tx)) = receiver.blocking_recv() {
        if request.parameters.temperature < 0.0 {
            response_tx
                .send(Err(ValidationError::Temperature))
                .unwrap_or(());
            continue;
        }
        if request.parameters.top_p <= 0.0 || request.parameters.top_p > 1.0 {
            response_tx.send(Err(ValidationError::TopP)).unwrap_or(());
            continue;
        }
        if request.parameters.top_k < 0 {
            response_tx.send(Err(ValidationError::TopK)).unwrap_or(());
            continue;
        }
        if request.parameters.max_new_tokens > 512 {
            response_tx
                .send(Err(ValidationError::MaxNewTokens))
                .unwrap_or(());
            continue;
        }

        // Get the number of tokens in the input
        let inputs = tokenizer.encode(request.inputs.clone(), false).unwrap();
        let input_length = inputs.len();

        if input_length > max_input_length {
            response_tx
                .send(Err(ValidationError::InputLength(
                    input_length,
                    max_input_length,
                )))
                .unwrap_or(());
            continue;
        }

        response_tx.send(Ok((input_length, request))).unwrap_or(());
    }
}

type ValidationRequest = (
    GenerateRequest,
    oneshot::Sender<Result<(usize, GenerateRequest), ValidationError>>,
);

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Temperature must be strictly positive")]
    Temperature,
    #[error("Top p must be >= 0.0 or < 1.0")]
    TopP,
    #[error("Top k must be strictly positive")]
    TopK,
    #[error("Max New Tokens must be <= 512")]
    MaxNewTokens,
    #[error("Inputs must have less than {1} tokens. Given: {0}")]
    InputLength(usize, usize),
}

impl From<ValidationError> for (StatusCode, String) {
    fn from(err: ValidationError) -> Self {
        (StatusCode::BAD_REQUEST, err.to_string())
    }
}
