/// Payload validation logic
use crate::config::Config;
use crate::validation::ValidationError::{BestOfSampling, BestOfSeed, EmptyInput};
use crate::{
    GenerateParameters, GenerateRequest, GrammarType, HubPreprocessorConfig, Idefics2Preprocessor,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{io::Reader as ImageReader, ImageFormat};
use jsonschema::{Draft, JSONSchema};
use rand::{thread_rng, Rng};
use serde_json::Value;
use std::io::Cursor;
use std::iter;
use text_generation_client::{Chunk, Image, InputChunk};
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::{instrument, Span};
use {once_cell::sync::Lazy, regex::Regex};

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Validation parameters
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_length: usize,
    max_total_tokens: usize,
    disable_grammar_support: bool,
    /// Channel to communicate with the background tokenization task
    sender: Option<mpsc::UnboundedSender<TokenizerRequest>>,
}

impl Validation {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        workers: usize,
        tokenizer: Option<Tokenizer>,
        config: Option<Config>,
        preprocessor_config: Option<HubPreprocessorConfig>,
        max_best_of: usize,
        max_stop_sequences: usize,
        max_top_n_tokens: u32,
        max_input_length: usize,
        max_total_tokens: usize,
        disable_grammar_support: bool,
    ) -> Self {
        // If we have a fast tokenizer
        let sender = if let Some(tokenizer) = tokenizer {
            // Create round robin channel
            let (validation_sender, validation_round_robin_receiver) = mpsc::unbounded_channel();
            let mut senders = Vec::with_capacity(workers);

            // Create workers
            for _ in 0..workers {
                let tokenizer_clone = tokenizer.clone();
                let config_clone = config.clone();
                let preprocessor_config_clone = preprocessor_config.clone();
                let (tokenizer_sender, tokenizer_receiver) = mpsc::unbounded_channel();
                senders.push(tokenizer_sender);

                // Spawn worker
                tokio::task::spawn_blocking(move || {
                    tokenizer_worker(
                        tokenizer_clone,
                        config_clone,
                        preprocessor_config_clone,
                        tokenizer_receiver,
                    )
                });
            }

            // Create tokenization round robin task
            tokio::spawn(round_robin_task(validation_round_robin_receiver, senders));

            Some(validation_sender)
        } else {
            None
        };

        Self {
            max_best_of,
            sender,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        }
    }

    #[instrument(skip(self, inputs))]
    pub async fn tokenize(
        &self,
        inputs: String,
        truncate: Option<usize>,
    ) -> Result<Option<(tokenizers::Encoding, Vec<InputChunk>)>, ValidationError> {
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
            let encoding = response_receiver.await.unwrap()?;
            Ok(Some(encoding))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self, inputs))]
    async fn validate_input(
        &self,
        inputs: String,
        truncate: Option<usize>,
        max_new_tokens: Option<u32>,
    ) -> Result<(Vec<InputChunk>, usize, u32), ValidationError> {
        // If we have a fast tokenizer
        if let Some((encoding, inputs)) = self.tokenize(inputs.clone(), truncate).await? {
            // Create response channel
            let input_length = if let Some(truncate) = truncate {
                std::cmp::min(encoding.len(), truncate)
            } else {
                encoding.len()
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

            metrics::histogram!("tgi_request_input_length").record(input_length as f64);
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
            let mut input_length = truncate.unwrap_or(self.max_input_length);

            // We don't have a tokenizer, therefore we have no idea how long is the query, let
            // them through and hope for the best.
            // Validate MaxNewTokens
            if (input_length as u32 + max_new_tokens) > self.max_total_tokens as u32 {
                input_length = input_length.saturating_sub(max_new_tokens as usize);
            }

            Ok((
                vec![Chunk::Text(inputs).into()],
                input_length,
                max_new_tokens,
            ))
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
            frequency_penalty,
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
            grammar,
            adapter_id,
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

        let frequency_penalty = frequency_penalty.unwrap_or(0.0);
        if !(-2.0..=2.0).contains(&frequency_penalty) {
            return Err(ValidationError::FrequencyPenalty);
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

        // TODO: we should build the FSM here and pass the compiled FSM instead of the grammar
        // NOTE: this is currently difficult because we need the tokenizer in Python to build
        // the FSM and we'd have to load a copy of the tokenizer into our Pyo3 instance which
        // may be slow and memory intensive. Best case is to have a Rust implementation of the FSM
        // compiler and use that to build the FSM here.

        // Validate grammar and unpack the grammar and type for the proto message
        let grammar = match grammar {
            Some(grammar) => {
                // Ensure that grammar is not set if it's not supported
                if self.disable_grammar_support {
                    return Err(ValidationError::Grammar);
                }
                let valid_grammar = match grammar {
                    GrammarType::Json(json) => {
                        let json = match json {
                            // if value is a string, we need to parse it again to make sure its
                            // a valid json
                            Value::String(s) => serde_json::from_str(&s)
                                .map_err(|e| ValidationError::InvalidGrammar(e.to_string())),
                            Value::Object(_) => Ok(json),
                            _ => Err(ValidationError::Grammar),
                        }?;

                        // Check if the json is a valid JSONSchema
                        JSONSchema::options()
                            .with_draft(Draft::Draft202012)
                            .compile(&json)
                            .map_err(|e| ValidationError::InvalidGrammar(e.to_string()))?;

                        // Serialize json to string
                        ValidGrammar::Json(
                            serde_json::to_string(&json)
                                .map_err(|e| ValidationError::InvalidGrammar(e.to_string()))?,
                        )
                    }
                    GrammarType::Regex(regex) => ValidGrammar::Regex(regex),
                };
                Some(valid_grammar)
            }
            None => None,
        };

        let parameters = ValidParameters {
            temperature,
            repetition_penalty,
            frequency_penalty,
            top_k,
            top_p,
            typical_p,
            do_sample,
            seed,
            watermark,
            grammar,
        };
        let stopping_parameters = ValidStoppingParameters {
            max_new_tokens,
            stop_sequences,
            ignore_eos_token: false,
        };

        metrics::histogram!("tgi_request_max_new_tokens").record(max_new_tokens as f64);

        Ok(ValidGenerateRequest {
            inputs,
            decoder_input_details,
            input_length: input_length as u32,
            truncate: truncate.unwrap_or(self.max_input_length) as u32,
            parameters,
            stopping_parameters,
            top_n_tokens,
            adapter_id,
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
fn tokenizer_worker(
    tokenizer: Tokenizer,
    config: Option<Config>,
    preprocessor_config: Option<HubPreprocessorConfig>,
    mut receiver: mpsc::UnboundedReceiver<TokenizerRequest>,
) {
    // Loop over requests
    while let Some(((inputs, truncate), response_tx, parent_span)) = receiver.blocking_recv() {
        parent_span.in_scope(|| {
            response_tx
                .send(prepare_input(
                    inputs,
                    truncate,
                    &tokenizer,
                    config.as_ref(),
                    preprocessor_config.as_ref(),
                ))
                .unwrap_or(())
        })
    }
}

fn format_from_mimetype(mimetype: &str) -> Option<ImageFormat> {
    match mimetype {
        "image/png" => Some(ImageFormat::Png),
        "image/jpeg" => Some(ImageFormat::Jpeg),
        "image/jpg" => Some(ImageFormat::Jpeg),
        "image/gif" => Some(ImageFormat::Gif),
        "image/webp" => Some(ImageFormat::WebP),
        "image/tiff" => Some(ImageFormat::Tiff),
        // "image/pnm"=>Some(ImageFormat::Pnm),
        // "image/tga"=>Some(ImageFormat::Tga),
        // "image/dds"=>Some(ImageFormat::Dds),
        // "image/bmp"=>Some(ImageFormat::Bmp),
        // "image/ico"=>Some(ImageFormat::Ico),
        // "image/x-exr"=>Some(ImageFormat::OpenExr),
        _ => None,
    }
}

fn format_to_mimetype(format: ImageFormat) -> String {
    match format {
        ImageFormat::Png => "image/png",
        ImageFormat::Jpeg => "image/jpeg",
        ImageFormat::Gif => "image/gif",
        ImageFormat::WebP => "image/webp",
        ImageFormat::Tiff => "image/tiff",
        _ => "application/octet-stream",
    }
    .to_string()
}

fn fetch_image(input: &str) -> Result<(Vec<u8>, String, usize, usize), ValidationError> {
    if input.starts_with("![](http://") || input.starts_with("![](https://") {
        let url = &input["![](".len()..input.len() - 1];
        let data = reqwest::blocking::get(url)?.bytes()?;

        let format = image::guess_format(&data)?;
        // TODO Remove this clone
        let img = ImageReader::with_format(Cursor::new(data.clone()), format).decode()?;
        let height: usize = img.height().try_into()?;
        let width: usize = img.width().try_into()?;
        let mimetype = format_to_mimetype(format);
        Ok((data.to_vec(), mimetype, height, width))
    } else if input.starts_with("![](data:") {
        // Remove ![](....)
        let content = &input["![](data:".len()..input.len() - 1];
        let tokens: Vec<_> = content.split(';').collect();
        if tokens.len() != 2 {
            return Err(ValidationError::InvalidImageContent(content.to_string()));
        }
        let mimetype = tokens[0];
        let content = tokens[1];

        if !content.starts_with("base64,") {
            return Err(ValidationError::InvalidImageContent(content.to_string()));
        }

        let data = STANDARD.decode(content["base64,".len()..].as_bytes())?;
        let img = if let Some(format) = format_from_mimetype(mimetype) {
            ImageReader::with_format(Cursor::new(&data), format).decode()?
        } else {
            ImageReader::new(Cursor::new(&data))
                .with_guessed_format()
                .map_err(|_io_error| ValidationError::InvalidImageContent(content.to_string()))?
                .decode()?
        };

        let height: usize = img.height().try_into()?;
        let width: usize = img.width().try_into()?;
        Ok((data, mimetype.to_string(), height, width))
    } else {
        Err(ValidationError::InvalidImageContent(input.to_string()))
    }
}

fn image_tokens(
    config: &Config,
    preprocessor_config: Option<&HubPreprocessorConfig>,
    height: usize,
    width: usize,
) -> String {
    use Config::*;
    use HubPreprocessorConfig::*;
    match config {
        Idefics => "<image>".to_string(),
        Idefics2(config) => {
            const FAKE: &str = "<fake_token_around_image>";
            const IMAGE: &str = "<image>";

            let slots = config.get_number_of_features(height, width);

            let mut image_string = String::with_capacity(2 * FAKE.len() + slots * IMAGE.len());
            image_string.push_str(FAKE);
            image_string.extend(iter::repeat(IMAGE).take(slots));
            image_string.push_str(FAKE);

            if matches!(
                preprocessor_config,
                Some(Idefics2Processor(Idefics2Preprocessor {
                    do_image_splitting: true,
                    ..
                }))
            ) {
                image_string = image_string.repeat(5);
            };

            image_string
        }
        Paligemma(config) => "<image>".repeat(config.get_number_of_features(height, width)),
        LlavaNext(config) => "<image>".repeat(config.get_number_of_features(height, width)),
        _ => unimplemented!("Images tokens are not supported for this model configuration"),
    }
}

fn image_tokens_fixup(config: &Config, text: String) -> String {
    match config {
        Config::Idefics2(_) => {
            const FAKE: &str = "<fake_token_around_image>";
            text.replace(&format!("{FAKE}{FAKE}"), FAKE)
        }
        _ => text,
    }
}

/// Get input length and optionally truncate it
fn prepare_input(
    inputs: String,
    _truncate: Option<usize>,
    tokenizer: &Tokenizer,
    config: Option<&Config>,
    preprocessor_config: Option<&HubPreprocessorConfig>,
) -> Result<(tokenizers::Encoding, Vec<InputChunk>), ValidationError> {
    use Config::*;
    static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"!\[\]\([^\)]*\)").unwrap());
    let (tokenizer_query, input_chunks) = match config {
        Some(config @ (Idefics | Idefics2(_) | Paligemma(_) | LlavaNext(_))) => {
            let mut input_chunks = Vec::new();
            let mut tokenizer_query = String::with_capacity(inputs.len());
            let mut start = 0;
            for chunk in RE.find_iter(&inputs) {
                let chunk_start = chunk.start();
                let chunk_end = chunk.end();
                if chunk_start != start {
                    input_chunks.push(Chunk::Text(inputs[start..chunk_start].to_string()).into());
                    tokenizer_query.push_str(&inputs[start..chunk_start]);
                }
                let (data, mimetype, height, width) = fetch_image(&inputs[chunk_start..chunk_end])?;
                input_chunks.push(Chunk::Image(Image { data, mimetype }).into());
                tokenizer_query.push_str(&image_tokens(config, preprocessor_config, height, width));
                start = chunk_end;
            }
            if start != inputs.len() {
                input_chunks.push(Chunk::Text(inputs[start..].to_string()).into());
                tokenizer_query.push_str(&inputs[start..]);
            }

            tokenizer_query = image_tokens_fixup(config, tokenizer_query);

            (tokenizer_query, input_chunks)
        }
        _ => (inputs.clone(), vec![Chunk::Text(inputs).into()]),
    };

    // Get the number of tokens in the input
    let encoding = tokenizer
        .encode(tokenizer_query, true)
        .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;

    Ok((encoding, input_chunks))
}

type TokenizerRequest = (
    (String, Option<usize>),
    oneshot::Sender<Result<(tokenizers::Encoding, Vec<InputChunk>), ValidationError>>,
    Span,
);

#[derive(Debug, Clone)]
pub(crate) enum ValidGrammar {
    Json(String),
    Regex(String),
}

#[derive(Debug, Clone)]
pub(crate) struct ValidParameters {
    /// / exponential scaling output probability distribution
    pub temperature: f32,
    /// / restricting to the k highest probability elements
    pub top_k: u32,
    /// / restricting to top tokens summing to prob_cut_off <= prob_cut_off
    pub top_p: f32,
    /// / restricting to top tokens summing to prob_cut_off <= prob_cut_off
    pub typical_p: f32,
    /// / apply sampling on the logits
    pub do_sample: bool,
    /// / random seed for sampling
    pub seed: u64,
    /// / repetition penalty
    pub repetition_penalty: f32,
    /// / frequency penalty
    pub frequency_penalty: f32,
    /// / token watermarking using "A Watermark for Large Language Models"
    pub watermark: bool,
    /// / grammar (applied if not empty)
    pub grammar: Option<ValidGrammar>,
}

#[derive(Debug, Clone)]
pub(crate) struct ValidStoppingParameters {
    /// / Maximum number of generated tokens
    pub max_new_tokens: u32,
    /// / Optional stopping sequences
    pub stop_sequences: Vec<String>,
    /// / Ignore end of sequence token
    /// / used for benchmarking
    pub ignore_eos_token: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ValidGenerateRequest {
    pub inputs: Vec<InputChunk>,
    pub input_length: u32,
    pub truncate: u32,
    pub decoder_input_details: bool,
    pub parameters: ValidParameters,
    pub stopping_parameters: ValidStoppingParameters,
    pub top_n_tokens: u32,
    pub adapter_id: Option<String>,
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
    #[error("`frequency_penalty` must be >= -2.0 and <= 2.0")]
    FrequencyPenalty,
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
    #[error("grammar is not supported")]
    Grammar,
    #[error("grammar is not valid: {0}")]
    InvalidGrammar(String),
    #[error("base64 encoding is invalid: {0}")]
    InvalidBase64(#[from] base64::DecodeError),
    #[error("invalid image: {0}")]
    InvalidImage(#[from] image::ImageError),
    #[error("invalid integer: {0}")]
    InvalidInt(#[from] core::num::TryFromIntError),
    #[error("invalid image content: {0}")]
    InvalidImageContent(String),
    #[error("Could not fetch image: {0}")]
    FailedFetchImage(#[from] reqwest::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Idefics2, PaliTextConfig, Paligemma};
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
        let disable_grammar_support = true;
        let config = None;
        let validation = Validation::new(
            workers,
            tokenizer,
            config,
            None,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );

        let max_new_tokens = 10;
        match validation
            .validate_input("Hello".to_string(), None, Some(max_new_tokens))
            .await
        {
            // Err(ValidationError::MaxNewTokens(1, 10)) => (),
            Ok((_s, 0, 10)) => (),
            r => panic!("Unexpected not max new tokens: {r:?}"),
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
        let disable_grammar_support = true;
        let workers = 1;
        let config = None;
        let validation = Validation::new(
            workers,
            tokenizer,
            config,
            None,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );

        let max_new_tokens = 10;
        match validation
            .validate_input("Hello".to_string(), None, Some(max_new_tokens))
            .await
        {
            Err(ValidationError::MaxTotalTokens(6, 1, 10)) => (),
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
        let disable_grammar_support = true;
        let config = None;
        let validation = Validation::new(
            workers,
            tokenizer,
            config,
            None,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
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
        let max_total_tokens = 106;
        let workers = 1;
        let disable_grammar_support = true;
        let config = None;
        let validation = Validation::new(
            workers,
            tokenizer,
            config,
            None,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );
        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_p: Some(1.0),
                    max_new_tokens: Some(5),
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
                    max_new_tokens: Some(5),
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
                    max_new_tokens: Some(5),
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
        let max_total_tokens = 106;
        let workers = 1;
        let disable_grammar_support = true;
        let config = None;
        let validation = Validation::new(
            workers,
            tokenizer,
            config,
            None,
            max_best_of,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );
        match validation
            .validate(GenerateRequest {
                inputs: "Hello".to_string(),
                parameters: GenerateParameters {
                    top_n_tokens: Some(5),
                    max_new_tokens: Some(5),
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
                    max_new_tokens: Some(5),
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
                    max_new_tokens: Some(5),
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
                    max_new_tokens: Some(5),
                    ..default_parameters()
                },
            })
            .await
            .unwrap();

        assert_eq!(valid_request.top_n_tokens, 0);
    }

    static PIXEL_GIF: &str = "R0lGODdhAQABAIEAAP///wAAAAAAAAAAACwAAAAAAQABAAAIBAABBAQAOw==";

    #[tokio::test]
    async fn test_prepare_input_chunks() {
        let pixel_data = STANDARD.decode(PIXEL_GIF).unwrap();

        let tokenizer = Some(get_tokenizer().await);

        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let disable_grammar_support = true;
        let workers = 1;
        let config = Config::Paligemma(Paligemma {
            text_config: PaliTextConfig {
                num_image_tokens: 1,
            },
        });
        let validation = Validation::new(
            workers,
            tokenizer,
            Some(config),
            None,
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );

        let chunks = match validation
            .tokenize(
                format!("test![](data:image/gif;base64,{})", PIXEL_GIF),
                None,
            )
            .await
        {
            Ok(Some((_encoding, chunks))) => chunks,
            _ => panic!("Unexpected tokenization failure"),
        };

        assert!(
            chunks
                == vec![
                    Chunk::Text("test".to_string()).into(),
                    Chunk::Image(Image {
                        data: pixel_data.clone(),
                        mimetype: "image/gif".to_string()
                    })
                    .into()
                ],
            "Failed to process images",
        );
    }

    #[tokio::test]
    async fn test_idefics2_correct_n_fake_tokens() {
        let pixel_data = STANDARD.decode(PIXEL_GIF).unwrap();

        let tokenizer = Some(get_tokenizer().await);

        let max_best_of = 2;
        let max_stop_sequence = 3;
        let max_top_n_tokens = 4;
        let max_input_length = 5;
        let max_total_tokens = 6;
        let disable_grammar_support = true;
        let workers = 1;
        let config = Config::Idefics2(Idefics2 {});
        let validation = Validation::new(
            workers,
            tokenizer,
            Some(config),
            Some(HubPreprocessorConfig::Idefics2Processor(
                Idefics2Preprocessor {
                    do_image_splitting: true,
                },
            )),
            max_best_of,
            max_stop_sequence,
            max_top_n_tokens,
            max_input_length,
            max_total_tokens,
            disable_grammar_support,
        );

        let (encoding, chunks) = match validation
            .tokenize(
                format!(
                    "test![](data:image/gif;base64,{})![](data:image/gif;base64,{})",
                    PIXEL_GIF, PIXEL_GIF
                ),
                None,
            )
            .await
        {
            Ok(Some((encoding, chunks))) => (encoding, chunks),
            _ => panic!("Unexpected tokenization failure"),
        };

        assert!(
            chunks
                == vec![
                    Chunk::Text("test".to_string()).into(),
                    Chunk::Image(Image {
                        data: pixel_data.clone(),
                        mimetype: "image/gif".to_string()
                    })
                    .into(),
                    Chunk::Image(Image {
                        data: pixel_data.clone(),
                        mimetype: "image/gif".to_string()
                    })
                    .into()
                ],
            "Failed to process images",
        );

        // Verify the number of fake tokens:
        //
        // - Two images surrounded/separated by a fake token = 3.
        // - Both are split in 5 subimages, separated by a fake token: 2 * 4
        //
        // Fake tokens get split up by the testing tokenizer, but we don't care.
        assert_eq!(
            encoding
                .get_tokens()
                .iter()
                .filter(|t| *t == "fake")
                .count(),
            11
        );
    }
}
