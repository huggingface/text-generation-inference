use crate::validation::ValidationError::EmptyInput;
/// Payload validation logic
use crate::{GenerateParameters, GenerateRequest};
use rand::rngs::ThreadRng;
use rand::Rng;
use text_generation_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use tracing::{instrument, Span};

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Channel to communicate with the background validation task
    sender: mpsc::UnboundedSender<ValidationRequest>,
}

impl Validation {
    pub(crate) fn new(
        workers: usize,
        tokenizer: Tokenizer,
        max_stop_sequences: usize,
        max_input_length: usize,
        max_total_tokens: usize,
    ) -> Self {
        // Create channel
        let (validation_sender, validation_receiver) = mpsc::unbounded_channel();

        // Launch background validation task
        tokio::spawn(validation_task(
            workers,
            tokenizer,
            max_stop_sequences,
            max_input_length,
            max_total_tokens,
            validation_receiver,
        ));

        Self {
            sender: validation_sender,
        }
    }

    /// Validate a payload and get the number of tokens in the input
    #[instrument(skip_all)]
    pub(crate) async fn validate(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, ValidationError> {
        // Create response channel
        let (sender, receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        self.sender
            .send((request, sender, Span::current()))
            .unwrap();
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
    max_stop_sequences: usize,
    max_input_length: usize,
    max_total_tokens: usize,
    mut receiver: mpsc::UnboundedReceiver<ValidationRequest>,
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
            validation_worker(
                tokenizer_clone,
                max_stop_sequences,
                max_input_length,
                max_total_tokens,
                worker_receiver,
            )
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
    max_stop_sequences: usize,
    max_input_length: usize,
    max_total_tokens: usize,
    mut receiver: mpsc::Receiver<ValidationRequest>,
) {
    // Seed rng
    let mut rng = rand::thread_rng();

    // Loop over requests
    while let Some((request, response_tx, parent_span)) = receiver.blocking_recv() {
        parent_span.in_scope(|| {
            response_tx
                .send(
                    validate(
                        request,
                        &tokenizer,
                        max_stop_sequences,
                        max_input_length,
                        max_total_tokens,
                        &mut rng,
                    )
                    .map_err(|err| {
                        metrics::increment_counter!("tgi_request_failure", "err" => "validation");
                        tracing::error!("{err}");
                        err
                    }),
                )
                .unwrap_or(())
        })
    }
}

fn validate(
    request: GenerateRequest,
    tokenizer: &Tokenizer,
    max_stop_sequences: usize,
    max_input_length: usize,
    max_total_tokens: usize,
    rng: &mut ThreadRng,
) -> Result<ValidGenerateRequest, ValidationError> {
    let GenerateParameters {
        temperature,
        repetition_penalty,
        top_k,
        top_p,
        do_sample,
        max_new_tokens,
        stop: stop_sequences,
        seed,
        watermark,
        ..
    } = request.parameters;

    let temperature = temperature.unwrap_or(1.0);
    if temperature <= 0.0 {
        return Err(ValidationError::Temperature);
    }

    let repetition_penalty = repetition_penalty.unwrap_or(1.0);
    if repetition_penalty <= 0.0 {
        return Err(ValidationError::RepetitionPenalty);
    }

    let top_p = top_p.unwrap_or(1.0);
    if top_p <= 0.0 || top_p > 1.0 {
        return Err(ValidationError::TopP);
    }

    // Different because the proto default value is 0 while it is not a valid value
    // for the user
    let top_k: u32 = match top_k {
        None => Ok(0),
        Some(top_k) => {
            if top_k <= 0 {
                return Err(ValidationError::TopK);
            }
            Ok(top_k as u32)
        }
    }?;

    if max_new_tokens == 0 {
        return Err(ValidationError::MaxNewTokens);
    }

    if stop_sequences.len() > max_stop_sequences {
        return Err(ValidationError::StopSequence(
            max_stop_sequences,
            stop_sequences.len(),
        ));
    }

    // If seed is None, assign a random one
    let seed = match seed {
        None => rng.gen(),
        Some(seed) => seed,
    };

    // Check if inputs is empty
    if request.inputs.is_empty() {
        return Err(EmptyInput);
    }

    // Get the number of tokens in the input
    match tokenizer.encode(request.inputs.clone(), true) {
        Ok(encoding) => {
            let input_length = encoding.len();
            let total_tokens = input_length + max_new_tokens as usize;

            if input_length > max_input_length {
                Err(ValidationError::InputLength(max_input_length, input_length))
            } else if total_tokens > max_total_tokens {
                Err(ValidationError::MaxTotalTokens(
                    max_total_tokens,
                    input_length,
                    max_new_tokens,
                ))
            } else {
                // Return ValidGenerateRequest
                let parameters = NextTokenChooserParameters {
                    temperature,
                    repetition_penalty,
                    top_k,
                    top_p,
                    do_sample,
                    seed,
                    watermark,
                };
                let stopping_parameters = StoppingCriteriaParameters {
                    max_new_tokens,
                    stop_sequences,
                };

                metrics::histogram!("tgi_request_input_length", input_length as f64);
                metrics::histogram!("tgi_request_max_new_tokens", max_new_tokens as f64);

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
    Span,
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
    #[error("max_new_tokens must be strictly positive")]
    MaxNewTokens,
    #[error("input tokens + max_new_tokens must be <= {0}. Given: {1} input tokens and {2} max_new_tokens")]
    MaxTotalTokens(usize, usize, u32),
    #[error("inputs must have less than {0} tokens. Given: {1}")]
    InputLength(usize, usize),
    #[error("inputs cannot be empty")]
    EmptyInput,
    #[error("stop supports up to {0} stop sequences. Given: {1}")]
    StopSequence(usize, usize),
    #[error("tokenizer error {0}")]
    Tokenizer(String),
}
