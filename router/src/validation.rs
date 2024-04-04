/// Payload validation logic
use crate::validation::ValidationError::{BestOfSampling, BestOfSeed, EmptyInput};
use crate::{GenerateParameters, GenerateRequest};
use rand::{thread_rng, Rng};
use std::env;
use text_generation_client::{NextTokenChooserParameters, StoppingCriteriaParameters};
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::TruncationDirection;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::{instrument, Span};

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Validation parameters
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_length: usize,
    max_total_tokens: usize,
    /// Channel to communicate with the background tokenization task
    sender: Option<mpsc::UnboundedSender<TokenizerRequest>>,
    skip_tokenizer_in_tgi: bool,
}

impl Validation {
    pub(crate) fn new(
        workers: usize,
        tokenizer: Option<Tokenizer>,
        max_best_of: usize,
        max_stop_sequences: usize,
        max_top_n_tokens: u32,
        max_input_length: usize,
        max_total_tokens: usize,
    ) -> Self {
        // If we have a fast tokenizer
        let sender = if let Some(tokenizer) = tokenizer {
            // Create round robin channel
            let (validation_sender, validation_round_robin_receiver) = mpsc::unbounded_channel();
            let mut senders = Vec::with_capacity(workers);

            // Create workers
            for _ in 0..workers {
                let tokenizer_clone = tokenizer.clone();
                let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();
                senders.push(tokenizer_sender);

                // Spawn worker
                tokio::task::spawn_blocking(move || {
                    tokenizer_worker(tokenizer_clone, tokenizer_receiver)
                });
            }

            // Create tokenization round robin task
            tokio::spawn(round_robin_task(validation_round_robin_receiver, senders));

            Some(validation_sender)
        } else {
            None
        };

        let skip_tokenizer_in_tgi = env::var("SKIP_TOKENIZER_IN_TGI")
            .ok()
            .map_or(false, |value| value.to_lowercase() == "true");

        Self {
            max_best_of,
            sender,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            skip_tokenizer_in_tgi,
        }
    }

    #[instrument(skip(self, inputs))]
    async fn validate_input(
        &self,
        inputs: String,
        truncate: Option<usize>,
        max_new_tokens: Option<u32>,
    ) -> Result<(String, usize, u32), ValidationError> {
        // If we have a fast tokenizer
        if let Some(sender) = &self.sender {
            // Create response channel
            let (response_sender, response_receiver) = oneshot::channel();
            // Send request to the background validation task
            // Unwrap is safe here
            sender
                .send(((inputs, truncate), response_sender, Span::current()))
                .unwrap();

            // Await on response channel
            // Unwrap is safe here
            let (inputs, _) = response_receiver.await.unwrap()?;

            let input_length = if self.skip_tokenizer_in_tgi {
                inputs.chars().filter(|&c| c == ',').count() + 1
            } else {
                truncate.unwrap_or(self.max_input_length)
            };

            // Get total tokens
            let max_new_tokens: u32 = if let Some(max_new_tokens) = max_new_tokens {
                max_new_tokens
            } else {
                self.max_total_tokens.saturating_sub(input_length) as u32
            };
            let total_tokens = input_length + max_new_tokens as usize;

            // Validate MaxTotalTokens
            if total_tokens > self.max_total_tokens {
                return Err(ValidationError::MaxTotalTokens(
                    self.max_total_tokens,
                    input_length,
                    max_new_tokens,
                ));
            }

            // Validate InputLength
            if input_length > self.max_input_length {
                return Err(ValidationError::InputLength(
                    self.max_input_length,
                    input_length,
                ));
            }

            metrics::histogram!("tgi_request_input_length", input_length as f64);
            Ok((inputs, input_length, max_new_tokens))
        }
        // Return inputs without validation
        else {
            // In this case, we don't know the real length in tokens of the inputs
            // However, the inputs will be truncated by the python servers
            // We make sure that truncate + max_new_tokens <= self.max_total_tokens
            let max_new_tokens: u32 = if let Some(max_new_tokens) = max_new_tokens {
                max_new_tokens
            } else if let Some(truncate) = truncate {
                self.max_total_tokens.saturating_sub(truncate) as u32
            } else {
                return Err(ValidationError::UnsetMaxNewTokens);
            };
            let input_length = if self.skip_tokenizer_in_tgi {
                inputs.chars().filter(|&c| c == ',').count() + 1
            } else {
                truncate.unwrap_or(self.max_input_length)
            };

            // Validate MaxNewTokens
            if (input_length as u32 + max_new_tokens) > self.max_total_tokens as u32 {
                return Err(ValidationError::MaxNewTokens(
                    self.max_total_tokens - self.max_input_length,
                    max_new_tokens,
                ));
            }

            Ok((inputs, input_length, max_new_tokens))
        }
    }

    /// Validate a payload and get the number of tokens in the input
    #[instrument(skip_all)]
    pub(crate) async fn validate(
        &self,
        request: GenerateRequest,
    ) -> Result<ValidGenerateRequest, ValidationError> {
        let GenerateParameters {
            best_of,
            temperature,
            repetition_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            max_new_tokens,
            stop: stop_sequences,
            truncate,
            seed,
            watermark,
            decoder_input_details,
            top_n_tokens,
            ..
        } = request.parameters;

        // sampling must be true when best_of > 1
        let best_of = best_of.unwrap_or(1);
        let sampling = do_sample
            || temperature.is_some()
            || top_k.is_some()
            || top_p.is_some()
            || typical_p.is_some();

        if best_of > 1 && !sampling {
            return Err(BestOfSampling);
        }

        let temperature = temperature.unwrap_or(1.0);
        if temperature <= 0.0 {
            return Err(ValidationError::Temperature);
        }

        let repetition_penalty = repetition_penalty.unwrap_or(1.0);
        if repetition_penalty <= 0.0 {
            return Err(ValidationError::RepetitionPenalty);
        }

        // TODO: enable watermark with fp8 quantization
        let quantization_enabled = env::var("QUANT_CONFIG")
            .ok()
            .map_or(false, |value| !value.is_empty());
        if watermark && quantization_enabled {
            return Err(ValidationError::WatermarkWithQuantization)
        }

        // Different because the proto default value is not a valid value
        // for the user
        let top_p = top_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TopP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let typical_p = typical_p
            .map(|value| {
                if value <= 0.0 || value >= 1.0 {
                    return Err(ValidationError::TypicalP);
                }
                Ok(value)
            })
            .unwrap_or(Ok(1.0))?;

        let top_k: u32 = top_k
            .map(|value| {
                if value <= 0 {
                    return Err(ValidationError::TopK);
                }
                Ok(value as u32)
            })
            .unwrap_or(Ok(0))?;

        if max_new_tokens == Some(0) {
            return Err(ValidationError::NegativeMaxNewTokens);
        }

        if stop_sequences.len() > self.max_stop_sequences {
            return Err(ValidationError::StopSequence(
                self.max_stop_sequences,
                stop_sequences.len(),
            ));
        }

        // If seed is None, assign a random one
        let seed = match seed {
            None => thread_rng().gen(),
            Some(seed) => {
                if best_of > 1 {
                    return Err(BestOfSeed);
                }
                seed
            }
        };

        let top_n_tokens = top_n_tokens
            .map(|value| {
                if value > self.max_top_n_tokens {
                    return Err(ValidationError::TopNTokens(self.max_top_n_tokens, value));
                }
                Ok(value)
            })
            .unwrap_or(Ok(0))?;

        // Check if inputs is empty
        if request.inputs.is_empty() {
            return Err(EmptyInput);
        }

        // Check if truncate is strictly positive and less than max_input_length
        let truncate = truncate
            .map(|value| {
                if value == 0 || value > self.max_input_length {
                    return Err(ValidationError::Truncate(self.max_input_length, value));
                }
                Ok(Some(value))
            })
            .unwrap_or(Ok(None))?;

        // Validate inputs
        let (inputs, input_length, max_new_tokens) = self
            .validate_input(request.inputs, truncate, max_new_tokens)
            .await?;

        let parameters = NextTokenChooserParameters {
            temperature,
            repetition_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            seed,
            watermark,
        };
        let stopping_parameters = StoppingCriteriaParameters {
            max_new_tokens,
            stop_sequences,
            ignore_eos_token: false,
        };

        metrics::histogram!("tgi_request_max_new_tokens", max_new_tokens as f64);

        Ok(ValidGenerateRequest {
            inputs,
            decoder_input_details,
            input_length: input_length as u32,
            truncate: truncate.unwrap_or(self.max_input_length) as u32,
            parameters,
            stopping_parameters,
            top_n_tokens,
        })
    }

    /// Validate the best_of parameter
    #[instrument(skip_all)]
    pub(crate) fn validate_best_of(&self, best_of: usize) -> Result<usize, ValidationError> {
        if self.max_best_of == 1 && best_of != 1 {
            return Err(ValidationError::BestOfDisabled);
        }

        if best_of > self.max_best_of {
            return Err(ValidationError::BestOf(self.max_best_of, best_of));
        }

        Ok(best_of)
    }
}

/// Round robin tokenization task
async fn round_robin_task(
    mut receiver: mpsc::UnboundedReceiver<TokenizerRequest>,
    senders: Vec<mpsc::UnboundedSender<TokenizerRequest>>,
) {
    loop {
        for sender in &senders {
            match receiver.recv().await {
                None => return,
                Some(request) => sender.send(request).unwrap(),
            };
        }
    }
}

/// Start tokenization workers
fn tokenizer_worker(tokenizer: Tokenizer, mut receiver: mpsc::UnboundedReceiver<TokenizerRequest>) {
    // Loop over requests
    while let Some(((inputs, truncate), response_tx, parent_span)) = receiver.blocking_recv() {
        parent_span.in_scope(|| {
            response_tx
                .send(prepare_input(inputs, truncate, &tokenizer))
                .unwrap_or(())
        })
    }
}

/// Get input length and optionally truncate it
fn prepare_input(
    inputs: String,
    truncate: Option<usize>,
    tokenizer: &Tokenizer,
) -> Result<(String, usize), ValidationError> {
    // Get the number of tokens in the input
    let mut encoding = tokenizer
        .encode(inputs.clone(), true)
        .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;

    // Optionally truncate
    let (inputs, input_length) = match truncate {
        // Truncate is some and < encoding length
        Some(truncate) if truncate < encoding.len() => {
            // truncate encoding and decode new inputs
            encoding.truncate(truncate, 0, TruncationDirection::Left);
            let inputs = tokenizer
                .decode(encoding.get_ids(), false)
                .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;
            (inputs, encoding.len())
        }
        // Nothing to do
        _ => (inputs, encoding.len()),
    };

    Ok((inputs, input_length))
}

type TokenizerRequest = (
    (String, Option<usize>),
    oneshot::Sender<Result<(String, usize), ValidationError>>,
    Span,
);

#[derive(Debug)]
pub(crate) struct ValidGenerateRequest {
    pub inputs: String,
    pub input_length: u32,
    pub truncate: u32,
    pub decoder_input_details: bool,
    pub parameters: NextTokenChooserParameters,
    pub stopping_parameters: StoppingCriteriaParameters,
    pub top_n_tokens: u32,
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("`best_of` must be > 0 and <= {0}. Given: {1}")]
    BestOf(usize, usize),
    #[error("`best_of` != 1 is not allowed for this endpoint")]
    BestOfDisabled,
    #[error("you must use sampling when `best_of` is > 1")]
    BestOfSampling,
    #[error("`seed` must not be set when `best_of` > 1")]
    BestOfSeed,
    #[error("`best_of` != 1 is not supported when streaming tokens")]
    BestOfStream,
    #[error("`top_n_tokens` must be >= 0 and <= {0}. Given: {1}")]
    TopNTokens(u32, u32),
    #[error("`top_n_tokens` != 0 is not allowed for this endpoint")]
    TopNTokensDisabled,
    #[error("`decoder_input_details` == true is not supported when streaming tokens")]
    PrefillDetailsStream,
    #[error("`temperature` must be strictly positive")]
    Temperature,
    #[error("`repetition_penalty` must be strictly positive")]
    RepetitionPenalty,
    #[error("`top_p` must be > 0.0 and < 1.0")]
    TopP,
    #[error("`top_k` must be strictly positive")]
    TopK,
    #[error("`truncate` must be strictly positive and less than {0}. Given: {1}")]
    Truncate(usize, usize),
    #[error("`typical_p` must be > 0.0 and < 1.0")]
    TypicalP,
    #[error("one of `max_new_tokens` or `truncate` must be set if a fast tokenizer is not in use")]
    UnsetMaxNewTokens,
    #[error("`max_new_tokens` must be strictly positive")]
    NegativeMaxNewTokens,
    #[error("`max_new_tokens` must be <= {0}. Given: {1}")]
    MaxNewTokens(usize, u32),
    #[error("`inputs` tokens + `max_new_tokens` must be <= {0}. Given: {1} `inputs` tokens and {2} `max_new_tokens`")]
    MaxTotalTokens(usize, usize, u32),
    #[error("`inputs` must have less than {0} tokens. Given: {1}")]
    InputLength(usize, usize),
    #[error("`inputs` cannot be empty")]
    EmptyInput,
    #[error("`stop` supports up to {0} stop sequences. Given: {1}")]
    StopSequence(usize, usize),
    #[error("tokenizer error {0}")]
    Tokenizer(String),
    #[error("`watermark` = true is not allowed with FP8 quantization.")]
    WatermarkWithQuantization,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_parameters;
    use crate::tests::get_tokenizer;

    #[tokio::test]
    async fn test_validation_max_new_tokens() {
        let tokenizer = None;
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
        );

        let max_new_tokens = 10;
        match validation
            .validate_input("Hello".to_string(), None, Some(max_new_tokens))
            .await
        {
            Err(ValidationError::MaxNewTokens(1, 10)) => (),
            _ => panic!("Unexpected not max new tokens"),
        }
    }

    #[tokio::test]
    async fn test_validation_input_length() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
        );

        let max_new_tokens = 10;
        match validation
            .validate_input("Hello".to_string(), None, Some(max_new_tokens))
            .await
        {
            Err(ValidationError::MaxTotalTokens(6, 5, 10)) => (),
            _ => panic!("Unexpected not max new tokens"),
        }
    }

    #[tokio::test]
    async fn test_validation_best_of_sampling() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
        );
        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    best_of: Some(2),
                    do_sample: false,
                    ..default_parameters()
                },
            })
            .await
        {
            Err(ValidationError::BestOfSampling) => (),
            _ => panic!("Unexpected not best of sampling"),
        }
    }

    #[tokio::test]
    async fn test_validation_top_p() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
        );
        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_p: Some(1.0),
                    ..default_parameters()
                },
            })
            .await
        {
            Err(ValidationError::TopP) => (),
            _ => panic!("Unexpected top_p"),
        }

        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_p: Some(0.99),
                    ..default_parameters()
                },
            })
            .await
        {
            Ok(_) => (),
            _ => panic!("Unexpected top_p error"),
        }

        let valid_request = validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_p: None,
                    ..default_parameters()
                },
            })
            .await
            .unwrap();
        // top_p == 1.0 is invalid for users to ask for but it's the default resolved value.
        assert_eq!(valid_request.parameters.top_p, 1.0);
    }

    #[tokio::test]
    async fn test_validation_top_n_tokens() {
        let tokenizer = Some(get_tokenizer().await);
        let max_best_of = 2;
        let max_stop_sequences = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let workers = 1;
        let validation = Validation::new(
            workers,
            tokenizer,
            max_best_of,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
        );
        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_n_tokens: Some(5),
                    ..default_parameters()
                },
            })
            .await
        {
            Err(ValidationError::TopNTokens(4, 5)) => (),
            _ => panic!("Unexpected top_n_tokens"),
        }

        validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_n_tokens: Some(4),
                    ..default_parameters()
                },
            })
            .await
            .unwrap();

        validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_n_tokens: Some(0),
                    ..default_parameters()
                },
            })
            .await
            .unwrap();

        let valid_request = validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_n_tokens: None,
                    ..default_parameters()
                },
            })
            .await
            .unwrap();

        assert_eq!(valid_request.top_n_tokens, 0);
    }
}
