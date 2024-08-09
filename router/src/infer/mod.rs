// pub(crate) mod v2;
mod chat_template;
pub mod tool_grammar;

use crate::validation::{ValidGenerateRequest, Validation, ValidationError};
use crate::GrammarType;
use crate::{
    ChatTemplateVersions, FinishReason, GenerateRequest, HubProcessorConfig, HubTokenizerConfig,
    Message, PrefillToken, Token,
};
use async_trait::async_trait;
use chat_template::ChatTemplate;
use futures::future::try_join_all;
use minijinja::ErrorKind;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tracing::instrument;

#[async_trait]
pub trait Backend {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError>;

    async fn health(&self, current_health: bool) -> bool;
}

/// Inference struct
#[derive(Clone)]
pub struct Infer {
    /// Validation
    validation: Validation,
    /// Request backend
    backend: Arc<dyn Backend + Send + Sync>,
    /// Chat template
    chat_template: Option<ChatTemplate>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
    /// Backend health
    backend_health: Arc<AtomicBool>,
}

impl Infer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        backend: impl Backend + Send + Sync + 'static,
        validation: Validation,
        max_concurrent_requests: usize,
        tokenizer_config: HubTokenizerConfig,
        processor_config: HubProcessorConfig,
    ) -> Self {
        let chat_template = tokenizer_config
            .chat_template
            .or(processor_config.chat_template)
            .and_then(|t| match t {
                ChatTemplateVersions::Single(template) => Some(template),
                ChatTemplateVersions::Multiple(templates) => templates
                    .into_iter()
                    .find(|t| t.name == "default")
                    .map(|t| t.template),
            })
            .map(|t| ChatTemplate::new(t, tokenizer_config.bos_token, tokenizer_config.eos_token));

        // Inference limit with a semaphore
        let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

        // Backend health
        let backend_health = Arc::new(AtomicBool::new(false));

        Self {
            validation,
            backend: Arc::new(backend),
            chat_template,
            limit_concurrent_requests: semaphore,
            backend_health,
        }
    }

    /// Add a new request to the queue and return a stream of InferStreamResponse
    #[instrument(skip_all)]
    pub(crate) async fn generate_stream<'a>(
        &'a self,
        request: GenerateRequest,
    ) -> Result<GenerateStreamResponse, InferError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        let permit = self
            .clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::counter!("tgi_request_failure", "err" => "overloaded").increment(1);
                tracing::error!("{err}");
                err
            })?;

        // Validate request
        let valid_request = self.validation.validate(request).await.map_err(|err| {
            metrics::counter!("tgi_request_failure", "err" => "validation").increment(1);
            tracing::error!("{err}");
            err
        })?;

        let input_length = valid_request.input_length;
        let generation_stream = self.backend.schedule(valid_request)?;

        Ok((permit, input_length, generation_stream))
    }

    /// Tokenizer the input
    #[instrument(skip_all)]
    pub(crate) async fn tokenize(
        &self,
        request: GenerateRequest,
    ) -> Result<Option<tokenizers::Encoding>, InferError> {
        // Tokenize request
        let inputs = request.inputs;
        let truncate = request.parameters.truncate;
        let encoding = self
            .validation
            .tokenize(inputs, truncate)
            .await
            .map_err(|err| {
                tracing::error!("Tokenization {err}");
                err
            })?;

        // Return Encoding
        Ok(encoding.map(|(encoding, _)| encoding))
    }

    /// Apply the chat template to the chat request
    #[instrument(skip_all)]
    pub(crate) fn apply_chat_template(
        &self,
        guideline: Option<String>,
        messages: Vec<Message>,
        grammar_with_prompt: Option<(GrammarType, String)>,
    ) -> Result<String, InferError> {
        self.chat_template
            .as_ref()
            .ok_or_else(|| InferError::TemplateError(ErrorKind::TemplateNotFound.into()))?
            .apply(guideline.as_deref(), messages, grammar_with_prompt)
            .map_err(|e| {
                metrics::counter!("tgi_request_failure", "err" => "template").increment(1);
                tracing::error!("{e}");
                e
            })
    }

    /// Add a new request to the queue and return a InferResponse
    #[instrument(skip_all)]
    pub(crate) async fn generate(
        &self,
        request: GenerateRequest,
    ) -> Result<InferResponse, InferError> {
        let use_top_tokens = request.parameters.top_n_tokens.is_some_and(|x| x > 0);

        // Create stream and keep semaphore permit as long as generate lives
        let (_permit, _input_length, stream) = self.generate_stream(request).await?;

        // Return values
        let mut result_prefill = Vec::new();
        let mut result_tokens = Vec::new();
        let mut result_top_tokens = Vec::new();
        let mut result_generated_text = None;
        let mut result_start = None;
        let mut result_queued = None;

        let mut stream = Box::pin(stream);

        // Iterate on stream
        while let Some(response) = stream.next().await {
            match response? {
                // Add prefill tokens
                InferStreamResponse::Prefill(prefill_tokens) => {
                    result_prefill = prefill_tokens;
                }
                // Push last token
                InferStreamResponse::Intermediate { token, top_tokens } => {
                    result_tokens.push(token);
                    result_top_tokens.push(top_tokens);
                }
                // Final message
                // Set return values
                InferStreamResponse::End {
                    token,
                    generated_text,
                    start,
                    queued,
                    top_tokens,
                } => {
                    result_tokens.push(token);
                    result_top_tokens.push(top_tokens);
                    result_generated_text = Some(generated_text);
                    result_start = Some(start);
                    result_queued = Some(queued)
                }
            }
        }

        // Check that we received a `InferStreamResponse::End` message
        if let (Some(generated_text), Some(queued), Some(start)) =
            (result_generated_text, result_queued, result_start)
        {
            Ok(InferResponse {
                prefill: result_prefill,
                _input_length,
                tokens: result_tokens,
                generated_text,
                queued,
                start,
                top_tokens: if use_top_tokens {
                    result_top_tokens
                } else {
                    Vec::new()
                },
            })
        } else {
            let err = InferError::IncompleteGeneration;
            metrics::counter!("tgi_request_failure", "err" => "incomplete").increment(1);
            tracing::error!("{err}");
            Err(err)
        }
    }
    /// Add best_of new requests to the queue and return a InferResponse of the sequence with
    /// the highest log probability per token
    #[instrument(skip(self, request))]
    pub(crate) async fn generate_best_of(
        &self,
        request: GenerateRequest,
        best_of: usize,
    ) -> Result<(InferResponse, Vec<InferResponse>), InferError> {
        // validate  best_of parameter separately
        let best_of = self.validation.validate_best_of(best_of)?;

        // create multiple generate requests
        let mut infer_responses: Vec<InferResponse> =
            try_join_all((0..best_of).map(|_| self.generate(request.clone()))).await?;

        // get the sequence with the highest log probability per token
        let mut max_index = 0;
        let mut max_logprob: f32 = f32::MIN;

        for (i, response) in infer_responses.iter().enumerate() {
            // mean logprobs of the generated tokens
            let sequence_logprob = response
                .tokens
                .iter()
                .map(|token| token.logprob)
                .sum::<f32>()
                / response.tokens.len() as f32;

            // set best sequence
            if sequence_logprob > max_logprob {
                max_index = i;
                max_logprob = sequence_logprob;
            }
        }
        let best_response = infer_responses.remove(max_index);
        Ok((best_response, infer_responses))
    }

    #[instrument(skip(self))]
    pub(crate) async fn health(&self) -> bool {
        let health = self
            .backend
            .health(self.backend_health.load(Ordering::SeqCst))
            .await;
        self.backend_health.store(health, Ordering::SeqCst);
        health
    }
}

/// Type alias for generation responses
pub(crate) type GenerateStreamResponse = (
    OwnedSemaphorePermit,
    u32, // input_length
    UnboundedReceiverStream<Result<InferStreamResponse, InferError>>,
);

#[derive(Debug)]
pub struct GeneratedText {
    pub text: String,
    pub generated_tokens: u32,
    pub finish_reason: FinishReason,
    pub seed: Option<u64>,
}

#[derive(Debug)]
pub enum InferStreamResponse {
    // Optional first message
    Prefill(Vec<PrefillToken>),
    // Intermediate messages
    Intermediate {
        token: Token,
        top_tokens: Vec<Token>,
    },
    // Last message
    End {
        token: Token,
        top_tokens: Vec<Token>,
        generated_text: GeneratedText,
        start: Instant,
        queued: Instant,
    },
}

#[derive(Debug)]
pub(crate) struct InferResponse {
    /// input_length is the input as perceived by the rust tokenizer in the
    /// validation pathway. It is redundant with prefill.len() but prefill
    /// has data only if the user asked for it. This will always be filled.
    pub(crate) _input_length: u32,
    pub(crate) prefill: Vec<PrefillToken>,
    pub(crate) tokens: Vec<Token>,
    pub(crate) generated_text: GeneratedText,
    pub(crate) queued: Instant,
    pub(crate) start: Instant,
    pub(crate) top_tokens: Vec<Vec<Token>>,
}

#[derive(Debug, Error)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
    #[error("Model is overloaded")]
    Overloaded(#[from] TryAcquireError),
    #[error("Input validation error: {0}")]
    ValidationError(#[from] ValidationError),
    #[error("Incomplete generation")]
    IncompleteGeneration,
    #[error("Template error: {0}")]
    TemplateError(#[from] minijinja::Error),
    #[error("Tool error: {0}")]
    ToolError(String),
}

impl InferError {
    pub(crate) fn error_type(&self) -> &str {
        match self {
            InferError::GenerationError(_) => "generation",
            InferError::Overloaded(_) => "overloaded",
            InferError::ValidationError(_) => "validation",
            InferError::IncompleteGeneration => "incomplete_generation",
            InferError::TemplateError(_) => "template_error",
            InferError::ToolError(_) => "tool_error",
        }
    }
}
