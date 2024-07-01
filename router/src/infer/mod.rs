mod health;
pub(crate) mod v2;
pub(crate) mod v3;

pub(crate) use health::HealthCheck;

use crate::validation::{ValidGenerateRequest, Validation, ValidationError};
use crate::{
    ChatTemplateInputs, ChatTemplateVersions, FinishReason, GenerateRequest, HubProcessorConfig,
    HubTokenizerConfig, Message, MessageChunk, PrefillToken, TextMessage, Token,
};
use crate::{
    FunctionRef, FunctionsMap, GrammarType, Properties, TokenizerConfigToken, Tool, ToolType, Tools,
};
use futures::future::try_join_all;
use minijinja::{Environment, ErrorKind, Template};
use minijinja_contrib::pycompat;

use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tracing::instrument;

pub(crate) trait Scheduler {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
        permit: OwnedSemaphorePermit,
    ) -> Result<GenerateStreamResponse, InferError>;
}

/// Inference struct
#[derive(Clone)]
pub struct Infer {
    /// Validation
    validation: Validation,
    /// Request scheduler
    scheduler: Arc<dyn Scheduler + Send + Sync>,
    /// Chat template
    chat_template: Option<ChatTemplate>,
    /// Inference limit
    limit_concurrent_requests: Arc<Semaphore>,
}

impl Infer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        scheduler: Arc<dyn Scheduler + Send + Sync>,
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

        Self {
            validation,
            scheduler,
            chat_template,
            limit_concurrent_requests: semaphore,
        }
    }

    /// Add a new request to the queue and return a stream of InferStreamResponse
    #[instrument(skip_all)]
    pub(crate) async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateStreamResponse, InferError> {
        // Limit concurrent requests by acquiring a permit from the semaphore
        let permit = self
            .clone()
            .limit_concurrent_requests
            .try_acquire_owned()
            .map_err(|err| {
                metrics::increment_counter!("tgi_request_failure", "err" => "overloaded");
                tracing::error!("{err}");
                err
            })?;

        // Validate request
        let valid_request = self.validation.validate(request).await.map_err(|err| {
            metrics::increment_counter!("tgi_request_failure", "err" => "validation");
            tracing::error!("{err}");
            err
        })?;

        self.scheduler.schedule(valid_request, permit)
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
        messages: Vec<Message>,
        grammar_with_prompt: Option<(GrammarType, String)>,
    ) -> Result<String, InferError> {
        self.chat_template
            .as_ref()
            .ok_or_else(|| InferError::TemplateError(ErrorKind::TemplateNotFound.into()))?
            .apply(messages, grammar_with_prompt)
            .map_err(|e| {
                metrics::increment_counter!("tgi_request_failure", "err" => "template");
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
        let (_permit, _input_length, mut stream) = self.generate_stream(request).await?;

        // Return values
        let mut result_prefill = Vec::new();
        let mut result_tokens = Vec::new();
        let mut result_top_tokens = Vec::new();
        let mut result_generated_text = None;
        let mut result_start = None;
        let mut result_queued = None;

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
            metrics::increment_counter!("tgi_request_failure", "err" => "incomplete");
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
}

/// Raise a exception (custom function) used in the chat templates
fn raise_exception(err_text: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::SyntaxError, err_text))
}

#[derive(Clone)]
struct ChatTemplate {
    template: Template<'static, 'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    use_default_tool_template: bool,
}

impl ChatTemplate {
    fn new(
        template: String,
        bos_token: Option<TokenizerConfigToken>,
        eos_token: Option<TokenizerConfigToken>,
    ) -> Self {
        let mut env = Box::new(Environment::new());
        // enable things like .strip() or .capitalize()
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        let template_str = template.into_boxed_str();
        env.add_function("raise_exception", raise_exception);

        // check if contains the tools variable within the template
        let use_default_tool_template =
            !template_str.as_ref().replace(' ', "").contains("{{tools}}");
        // leaking env and template_str as read-only, static resources for performance.
        let template = Box::leak(env)
            .template_from_str(Box::leak(template_str))
            .unwrap();

        Self {
            template,
            bos_token: bos_token.map(|token| token.as_str().to_string()),
            eos_token: eos_token.map(|token| token.as_str().to_string()),
            use_default_tool_template,
        }
    }

    fn apply(
        &self,
        mut messages: Vec<Message>,
        grammar_with_prompt: Option<(GrammarType, String)>,
    ) -> Result<String, InferError> {
        if self.use_default_tool_template {
            if let Some(last_message) = messages.last_mut() {
                if let Some((GrammarType::Json(tools), tool_prompt)) = grammar_with_prompt {
                    last_message.content.push(MessageChunk::Text {
                        text: format!("\n---\n{}\n{}", tool_prompt, tools),
                    });
                }
            }
        }

        let messages: Vec<TextMessage> = messages.into_iter().map(|c| c.into()).collect();

        self.template
            .render(ChatTemplateInputs {
                messages,
                bos_token: self.bos_token.as_deref(),
                eos_token: self.eos_token.as_deref(),
                add_generation_prompt: true,
                tools: None,
                tools_prompt: None,
            })
            .map_err(InferError::TemplateError)
    }
}

pub struct ToolGrammar {}

impl ToolGrammar {
    pub fn apply(
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolType>,
    ) -> Result<Option<Tools>, InferError> {
        if let Some((req_tools, tool_choice)) = tools.zip(tool_choice) {
            // let tool_prompt = tool_prompt.unwrap_or_default();
            let tools_to_use = match tool_choice {
                ToolType::FunctionName(name) => {
                    vec![req_tools
                        .iter()
                        .find(|tool| tool.function.name == *name)
                        .unwrap_or_else(|| panic!("Tool with name {} not found", name))
                        .clone()]
                }
                ToolType::Function { function } => {
                    let tool = req_tools
                        .iter()
                        .find(|tool| tool.function.name == function.name)
                        .unwrap_or_else(|| panic!("Tool with name {} not found", function.name))
                        .clone();
                    vec![tool]
                }
                ToolType::OneOf => req_tools.to_owned(),
            };

            // adds the error notification function for LLM feedback if required
            let mut text_response_properties = Map::new();
            text_response_properties.insert(
                "error".to_string(),
                serde_json::json!({
                    "type": "string",
                    "description": "The error or issue to notify"
                }),
            );
            text_response_properties.insert(
                "_name".to_string(),
                serde_json::json!({
                    "type": "string",
                    "const": "notify_error"
                }),
            );

            let functions: HashMap<String, serde_json::Value> = tools_to_use
                .iter()
                .map(|tool| {
                    let func = tool.function.clone();

                    // Clone the existing parameters, which are expected to be a JSON object
                    let mut params = if let Value::Object(params) = &func.arguments {
                        params.clone()
                    } else {
                        Map::new()
                    };

                    // Insert the function's description at the top level, outside of properties
                    params.insert(
                        "description".to_string(),
                        Value::String(func.description.clone().unwrap_or_default()),
                    );

                    // Ensure 'properties' exists and is an object
                    let properties = params
                        .entry("properties".to_string())
                        .or_insert_with(|| json!({}))
                        .as_object_mut()
                        .unwrap();

                    // Insert the constant for the function name inside 'properties'
                    properties.insert(
                        "_name".to_string(),
                        json!({
                            "type": "string",
                            "const": func.name.clone(),
                            // "description": "The name of the function"
                        }),
                    );

                    // Check if 'required' exists, and it is an array. If not, create an empty array.
                    let required = params
                        .entry("required".to_string())
                        .or_insert_with(|| json!([]))
                        .as_array_mut()
                        .unwrap();

                    // Add 'name' to the 'required' array if it is not already present
                    if !required.iter().any(|r| r == "_name") {
                        required.push(json!("_name"));
                    }

                    (func.name, Value::Object(params))
                })
                .chain([(
                    "notify_error".to_string(),
                    serde_json::json!({
                        "properties": text_response_properties,
                        "required": ["error", "_name"],
                        "type": "object"
                    }),
                )])
                .collect();

            let tools = Tools {
                functions_map: FunctionsMap { functions },
                properties: Properties {
                    function: tools_to_use
                        .iter()
                        .map(|tool| FunctionRef {
                            ref_path: format!("#/$functions/{}", tool.function.name.clone()),
                        })
                        .chain(std::iter::once(FunctionRef {
                            ref_path: "#/$functions/notify_error".to_string(),
                        }))
                        .collect(),
                },
            };

            return Ok(Some(tools));
        }
        // Err(InferError::ToolError("No tools provided".to_string()))
        Ok(None)
    }
}

/// Type alias for generation responses
pub(crate) type GenerateStreamResponse = (
    OwnedSemaphorePermit,
    u32, // input_length
    UnboundedReceiverStream<Result<InferStreamResponse, InferError>>,
);

#[derive(Debug)]
pub(crate) struct GeneratedText {
    pub(crate) text: String,
    pub(crate) generated_tokens: u32,
    pub(crate) finish_reason: FinishReason,
    pub(crate) seed: Option<u64>,
}

#[derive(Debug)]
pub(crate) enum InferStreamResponse {
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
