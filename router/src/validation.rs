use crate::config::Config;
use crate::validation::ValidationError::{BestOfSampling, BestOfSeed, EmptyInput};
use crate::{
    GenerateParameters, GenerateRequest, GrammarType, HubPreprocessorConfig, Idefics2Preprocessor,
    TokenizerTrait,
};
use crate::{PyTokenizer, Tokenizer};
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{ImageFormat, ImageReader};
use outlines_core::json_schema::to_regex as json_schema_to_regex;
use rand::{thread_rng, Rng};
use serde_json::Value;
/// Payload validation logic
use std::cmp::min;
use std::io::Cursor;
use std::iter;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::warn;
use tracing::{instrument, Span};
use {once_cell::sync::Lazy, regex::Regex};

static DEFAULT_GENERATION_LENGTH: u32 = 1024;

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
    sender: mpsc::UnboundedSender<TokenizerRequest>,
}

impl Validation {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        workers: usize,
        tokenizer: Tokenizer,
        config: Option<Config>,
        preprocessor_config: Option<HubPreprocessorConfig>,
        max_best_of: usize,
        max_stop_sequences: usize,
        max_top_n_tokens: u32,
        max_input_length: usize,
        max_total_tokens: usize,
        disable_grammar_support: bool,
    ) -> Self {
        let workers = if let Tokenizer::Python { .. } = &tokenizer {
            1
        } else {
            workers
        };
        // If we have a fast tokenizer
        let sender = {
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

            validation_sender
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
        add_special_tokens: bool,
        truncate: Option<usize>,
    ) -> Result<(tokenizers::Encoding, Vec<Chunk>), ValidationError> {
        // If we have a fast tokenizer
        // Create response channel
        let (response_sender, response_receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        let _ = &self
            .sender
            .send((
                (inputs, add_special_tokens, truncate),
                response_sender,
                Span::current(),
            ))
            .unwrap();

        // Await on response channel
        // Unwrap is safe here
        let encoding = response_receiver.await.unwrap()?;
        Ok(encoding)
    }

    #[allow(clippy::type_complexity)]
    #[instrument(skip(self, inputs))]
    async fn validate_input(
        &self,
        inputs: String,
        add_special_tokens: bool,
        truncate: Option<usize>,
        max_new_tokens: Option<u32>,
    ) -> Result<(Vec<Chunk>, Option<Vec<u32>>, usize, u32, u32), ValidationError> {
        // If we have a fast tokenizer
        let (encoding, inputs) = self
            .tokenize(inputs.clone(), add_special_tokens, truncate)
            .await?;
        // Create response channel
        let input_length = if let Some(truncate) = truncate {
            std::cmp::min(encoding.len(), truncate)
        } else {
            encoding.len()
        };

        // Get total tokens
        let (max_new_tokens, max_total_new_tokens) = if let Some(max_new_tokens) = max_new_tokens {
            // Do not accept humongous max_new_tokens queries.
            // We preallocate the default but we prevent a single user
            // from taking up all the slots in a handful of queries that consume little
            // amount of tokens. (You can have 10 token long query that creates a handful of token
            // but the requested amount to be 120k.
            let chunk_size = min(max_new_tokens, DEFAULT_GENERATION_LENGTH);
            (chunk_size, max_new_tokens)
        } else {
            // Use the maximum possible number of tokens as default
            // However, the system will re-queue the request everytime it completes
            // `DEFAULT_GENERATION_LENGTH` tokens.
            let max_new_tokens = self.max_total_tokens.saturating_sub(input_length) as u32;
            (
                min(max_new_tokens, DEFAULT_GENERATION_LENGTH),
                max_new_tokens,
            )
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

        let ids = encoding.get_ids();
        let input_ids = ids[ids.len().saturating_sub(input_length)..].to_owned();

        metrics::histogram!("tgi_request_input_length").record(input_length as f64);
        Ok((
            inputs,
            Some(input_ids),
            input_length,
            max_new_tokens,
            max_total_new_tokens,
        ))
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
        let (inputs, input_ids, input_length, max_new_tokens, max_total_new_tokens) = self
            .validate_input(
                request.inputs,
                request.add_special_tokens,
                truncate,
                max_new_tokens,
            )
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
                        jsonschema::draft202012::meta::validate(&json)
                            .map_err(|e| ValidationError::InvalidGrammar(e.to_string()))?;

                        // The schema can be valid but lack properties.
                        // We need properties for the grammar to be successfully parsed in Python.
                        // Therefore, we must check and throw an error if properties are missing.
                        json.get("properties")
                            .ok_or(ValidationError::InvalidGrammar(
                                "Grammar must have a 'properties' field".to_string(),
                            ))?;

                        // Do compilation in the router for performance. In the future, we
                        // should also move regex -> automaton compilation in the router,
                        // but this is not yet supported in pure Rust by outlines-core.
                        let grammar_regex = json_schema_to_regex(&json, None, &json)
                            .map_err(ValidationError::RegexFromSchema)?;

                        ValidGrammar::Regex(grammar_regex.to_string())
                    }
                    GrammarType::JsonSchema(schema_config) => {
                        // Extract the actual schema for validation
                        let json = &schema_config.schema;

                        // Check if the json is a valid JSONSchema
                        jsonschema::draft202012::meta::validate(json)
                            .map_err(|e| ValidationError::InvalidGrammar(e.to_string()))?;

                        // The schema can be valid but lack properties.
                        // We need properties for the grammar to be successfully parsed in Python.
                        // Therefore, we must check and throw an error if properties are missing.
                        json.get("properties")
                            .ok_or(ValidationError::InvalidGrammar(
                                "Grammar must have a 'properties' field".to_string(),
                            ))?;

                        // Do compilation in the router for performance
                        let grammar_regex = json_schema_to_regex(json, None, json)
                            .map_err(ValidationError::RegexFromSchema)?;

                        ValidGrammar::Regex(grammar_regex.to_string())
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
            max_total_new_tokens,
            stop_sequences,
            ignore_eos_token: false,
        };

        metrics::histogram!("tgi_request_max_new_tokens").record(max_new_tokens as f64);

        Ok(ValidGenerateRequest {
            inputs,
            input_ids: input_ids.map(Arc::new),
            add_special_tokens: request.add_special_tokens,
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
    match tokenizer {
        Tokenizer::Python {
            tokenizer_name,
            revision,
            trust_remote_code,
        } => {
            pyo3::Python::with_gil(|py| -> pyo3::PyResult<()> {
                let tokenizer =
                    PyTokenizer::from_py(py, tokenizer_name, revision, trust_remote_code)?;
                // Loop over requests
                while let Some(((inputs, add_special_tokens, truncate), response_tx, parent_span)) =
                    receiver.blocking_recv()
                {
                    parent_span.in_scope(|| {
                        response_tx
                            .send(prepare_input(
                                inputs,
                                truncate,
                                add_special_tokens,
                                &tokenizer,
                                config.as_ref(),
                                preprocessor_config.as_ref(),
                            ))
                            .unwrap_or(())
                    })
                }
                Ok(())
            })
            .expect("Failure in python tokenizer worker");
        }
        Tokenizer::Rust(tokenizer) => {
            while let Some(((inputs, add_special_tokens, truncate), response_tx, parent_span)) =
                receiver.blocking_recv()
            {
                parent_span.in_scope(|| {
                    response_tx
                        .send(prepare_input(
                            inputs,
                            truncate,
                            add_special_tokens,
                            &tokenizer,
                            config.as_ref(),
                            preprocessor_config.as_ref(),
                        ))
                        .unwrap_or(())
                })
            }
        }
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

        let data = STANDARD.decode(&content["base64,".len()..])?;
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
        Mllama => "<|image|>".to_string(),
        Idefics2(config) => {
            const FAKE: &str = "<fake_token_around_image>";
            const IMAGE: &str = "<image>";

            let slots = config.get_number_of_features(height, width);

            let mut image_string = String::with_capacity(2 * FAKE.len() + slots * IMAGE.len());
            image_string.push_str(FAKE);
            image_string.extend(iter::repeat_n(IMAGE, slots));
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
        Idefics3(config) => {
            const FAKE: &str = "<fake_token_around_image>";
            const IMAGE: &str = "<image>";
            const GLOBAL_IMG: &str = "<global-img>";

            let max_longest_edge_for_image_resize = config.get_max_longest_edge_for_image_resize();

            // resize image if it is larger than max_longest_edge_for_image_resize keeping aspect ratio
            let (height, width) = if height > max_longest_edge_for_image_resize
                || width > max_longest_edge_for_image_resize
            {
                let aspect_ratio = height as f32 / width as f32;
                if height > width {
                    (
                        max_longest_edge_for_image_resize,
                        (max_longest_edge_for_image_resize as f32 / aspect_ratio) as usize,
                    )
                } else {
                    (
                        (max_longest_edge_for_image_resize as f32 * aspect_ratio) as usize,
                        max_longest_edge_for_image_resize,
                    )
                }
            } else {
                (height, width)
            };

            let image_seq_len = config.get_number_of_features();
            let max_edge = config.get_max_longest_edge();

            let (image_rows, image_cols) = if height > max_edge || width > max_edge {
                (
                    (height as f32 / max_edge as f32).ceil() as usize,
                    (width as f32 / max_edge as f32).ceil() as usize,
                )
            } else {
                (0, 0)
            };

            let mut image_string = String::new();

            if image_rows == 0 && image_cols == 0 {
                // Single image case
                image_string.push_str(FAKE);
                image_string.push_str(GLOBAL_IMG);
                image_string.push_str(&IMAGE.repeat(image_seq_len));
                image_string.push_str(FAKE);
            } else {
                // Split image case
                for n_h in 0..image_rows {
                    for n_w in 0..image_cols {
                        image_string.push_str(FAKE);
                        image_string.push_str(&format!("<row_{}_col_{}>", n_h + 1, n_w + 1));
                        image_string.push_str(&IMAGE.repeat(image_seq_len));
                    }
                    image_string.push('\n');
                }

                image_string.push('\n');
                image_string.push_str(FAKE);
                image_string.push_str(GLOBAL_IMG);
                image_string.push_str(&IMAGE.repeat(image_seq_len));
                image_string.push_str(FAKE);
            }

            image_string
        }
        Paligemma(config) => "<image>".repeat(config.get_number_of_features(height, width)),
        LlavaNext(config) => "<image>".repeat(config.get_number_of_features(height, width)),
        Llama4(config) => {
            const IMAGE_START: &str = "<|image_start|>";
            const IMAGE: &str = "<|image|>";
            const IMAGE_END: &str = "<|image_end|>";
            const PATCH: &str = "<|patch|>";
            const TILE_X_SEP: &str = "<|tile_x_separator|>";
            const TILE_Y_SEP: &str = "<|tile_y_separator|>";

            let image_height = config.image_size();
            let patch_size = config.patch_size();
            let pixel_shuffle_ratio = config.pixel_shuffle_ratio();
            let max_patches = match preprocessor_config {
                Some(HubPreprocessorConfig::Llama4Processor(cfg)) => cfg.max_patches,
                _ => panic!("Expected Llama4Processor in preprocessor_config"),
            };
            let downsample_ratio =
                (1.0 / (pixel_shuffle_ratio * pixel_shuffle_ratio)).round() as usize;

            let (ratio_h, ratio_w) = config.get_aspect_ratios(height, width, max_patches);
            let image_width = image_height; // Assuming pixel shape: [H][W][C]

            let num_patches_per_chunk =
                (image_height / patch_size) * (image_width / patch_size) / downsample_ratio;

            let mut img_string = String::new();
            img_string.push_str(IMAGE_START);

            if ratio_h * ratio_w > 1 {
                for _yy in 0..ratio_h {
                    for xx in 0..ratio_w {
                        img_string.push_str(&PATCH.repeat(num_patches_per_chunk));
                        if xx < ratio_w - 1 {
                            img_string.push_str(TILE_X_SEP);
                        }
                    }
                    img_string.push_str(TILE_Y_SEP);
                }
            }

            img_string.push_str(IMAGE);
            img_string.push_str(&PATCH.repeat(num_patches_per_chunk));
            img_string.push_str(IMAGE_END);

            img_string
        }
        Qwen2Vl(config) => format!(
            "<|vision_start|>{:?}<|vision_end|>",
            "<|image_pad|>".repeat(config.get_number_of_features(height, width))
        ),
        Qwen2_5Vl(config) => format!(
            "<|vision_start|>{:?}<|vision_end|>",
            "<|image_pad|>".repeat(config.get_number_of_features(height, width))
        ),
        Gemma3(_config) => {
            // TODO: prefer using the config to determine the number of features
            let num_mm_soft_tokens_per_image = 256;
            format!(
                "\n\n<start_of_image>{}<end_of_image>\n\n",
                "<image_soft_token>".repeat(num_mm_soft_tokens_per_image)
            )
        }
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
fn prepare_input<T: TokenizerTrait>(
    inputs: String,
    _truncate: Option<usize>,
    add_special_tokens: bool,
    tokenizer: &T,
    config: Option<&Config>,
    preprocessor_config: Option<&HubPreprocessorConfig>,
) -> Result<(tokenizers::Encoding, Vec<Chunk>), ValidationError> {
    use Config::*;
    static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"!\[\]\([^\)]*\)").unwrap());
    let (tokenizer_query, input_chunks) = match config {
        Some(
            config @ (Idefics | Mllama | Idefics2(_) | Idefics3(_) | Gemma3(_) | Llama4(_)
            | Paligemma(_) | LlavaNext(_) | Qwen2Vl(_) | Qwen2_5Vl(_)),
        ) => {
            let mut input_chunks = Vec::new();
            let mut tokenizer_query = String::with_capacity(inputs.len());
            let mut start = 0;
            for chunk in RE.find_iter(&inputs) {
                let chunk_start = chunk.start();
                let chunk_end = chunk.end();
                if chunk_start != start {
                    input_chunks.push(Chunk::Text(inputs[start..chunk_start].to_string()));
                    tokenizer_query.push_str(&inputs[start..chunk_start]);
                }
                let (data, mimetype, height, width) = fetch_image(&inputs[chunk_start..chunk_end])?;
                input_chunks.push(Chunk::Image(Image { data, mimetype }));
                tokenizer_query.push_str(&image_tokens(config, preprocessor_config, height, width));
                start = chunk_end;
            }
            if start != inputs.len() {
                input_chunks.push(Chunk::Text(inputs[start..].to_string()));
                tokenizer_query.push_str(&inputs[start..]);
            }

            tokenizer_query = image_tokens_fixup(config, tokenizer_query);

            (tokenizer_query, input_chunks)
        }
        _ => (inputs.clone(), vec![Chunk::Text(inputs)]),
    };

    // Get the number of tokens in the input
    let encoding = tokenizer
        .encode_trait(tokenizer_query, add_special_tokens)
        .map_err(|err| ValidationError::Tokenizer(err.to_string()))?;

    Ok((encoding, input_chunks))
}

type TokenizerRequest = (
    (String, bool, Option<usize>),
    oneshot::Sender<Result<(tokenizers::Encoding, Vec<Chunk>), ValidationError>>,
    Span,
);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Image {
    pub data: Vec<u8>,
    pub mimetype: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Chunk {
    Text(String),
    Image(Image),
}

/// Convert input chunks to a stringly-typed input for backwards
/// compat for backends that haven't implemented chunked inputs.
pub trait ChunksToString {
    /// Convert chunks to string.
    fn chunks_to_string(&self) -> String;
}

impl ChunksToString for Vec<Chunk> {
    fn chunks_to_string(&self) -> String {
        let mut output = String::new();
        self.iter().for_each(|c| match &c {
            Chunk::Text(text) => output.push_str(text),
            Chunk::Image(Image { data, mimetype }) => {
                let encoded = STANDARD.encode(data);
                output.push_str(&format!("![](data:{};base64,{})", mimetype, encoded))
            }
        });
        output
    }
}

#[derive(Debug, Clone)]
pub enum ValidGrammar {
    Json(String),
    Regex(String),
}

#[derive(Debug, Clone)]
pub struct ValidParameters {
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
pub struct ValidStoppingParameters {
    /// / Maximum number of generated tokens
    pub max_new_tokens: u32,
    /// Maximum number of generated tokens before being re-queued by the system
    pub max_total_new_tokens: u32,
    /// / Optional stopping sequences
    pub stop_sequences: Vec<String>,
    /// / Ignore end of sequence token
    /// / used for benchmarking
    pub ignore_eos_token: bool,
}

#[derive(Debug, Clone)]
pub struct ValidGenerateRequest {
    pub inputs: Vec<Chunk>,
    pub input_ids: Option<Arc<Vec<u32>>>,
    pub input_length: u32,
    pub truncate: u32,
    pub add_special_tokens: bool,
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
    #[error("cannot compile regex from schema: {0}")]
    RegexFromSchema(anyhow::Error),
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
    #[error("{0} modality is not supported")]
    UnsupportedModality(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Idefics2, PaliTextConfig, Paligemma};
    use crate::default_parameters;
    use crate::tests::get_tokenizer;

    #[tokio::test]
    async fn test_validation_max_new_tokens() {
        let tokenizer = get_tokenizer();
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
            .validate_input("Hello".to_string(), true, None, Some(max_new_tokens))
            .await
        {
            Err(ValidationError::MaxTotalTokens(6, 1, 10)) => (),
            // Ok((_s, _, 0, 10)) => (),
            r => panic!("Unexpected not max new tokens: {r:?}"),
        }
    }

    #[tokio::test]
    async fn test_validation_input_length() {
        let tokenizer = get_tokenizer();
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
            .validate_input("Hello".to_string(), true, None, Some(max_new_tokens))
            .await
        {
            Err(ValidationError::MaxTotalTokens(6, 1, 10)) => (),
            _ => panic!("Unexpected not max new tokens"),
        }
    }

    #[tokio::test]
    async fn test_validation_best_of_sampling() {
        let tokenizer = get_tokenizer();
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
                add_special_tokens: true,
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
        let tokenizer = get_tokenizer();
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
                add_special_tokens: true,
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
                add_special_tokens: true,
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
                add_special_tokens: true,
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
        let tokenizer = get_tokenizer();
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
                add_special_tokens: true,
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
                add_special_tokens: true,
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
                add_special_tokens: true,
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
                add_special_tokens: true,
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

        let tokenizer = get_tokenizer();

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
                true,
                None,
            )
            .await
        {
            Ok((_encoding, chunks)) => chunks,
            _ => panic!("Unexpected tokenization failure"),
        };

        assert!(
            chunks
                == vec![
                    Chunk::Text("test".to_string()),
                    Chunk::Image(Image {
                        data: pixel_data.clone(),
                        mimetype: "image/gif".to_string()
                    })
                ],
            "Failed to process images",
        );
    }

    #[tokio::test]
    async fn test_idefics2_correct_n_fake_tokens() {
        let pixel_data = STANDARD.decode(PIXEL_GIF).unwrap();

        let tokenizer = get_tokenizer();

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
                true,
                None,
            )
            .await
        {
            Ok((encoding, chunks)) => (encoding, chunks),
            _ => panic!("Unexpected tokenization failure"),
        };

        assert!(
            chunks
                == vec![
                    Chunk::Text("test".to_string()),
                    Chunk::Image(Image {
                        data: pixel_data.clone(),
                        mimetype: "image/gif".to_string()
                    }),
                    Chunk::Image(Image {
                        data: pixel_data.clone(),
                        mimetype: "image/gif".to_string()
                    })
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
