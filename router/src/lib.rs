/// Text Generation Inference Webserver
pub mod config;
pub mod infer;
pub mod server;
pub mod validation;

#[cfg(feature = "kserve")]
mod kserve;
pub mod logging;

mod sagemaker;
pub mod usage_stats;
mod vertex;

use crate::infer::{Infer, InferError};
use crate::server::prepare_chat_input;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use serde::{Deserialize, Serialize};
use tokenizers::Encoding;
use tracing::warn;
use utoipa::ToSchema;
use validation::Validation;

#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum Tokenizer {
    Python {
        tokenizer_name: String,
        revision: Option<String>,
        trust_remote_code: bool,
    },
    Rust(tokenizers::Tokenizer),
}

pub struct PyTokenizer<'a>(pyo3::Bound<'a, pyo3::PyAny>);

impl<'a> PyTokenizer<'a> {
    fn from_py(
        py: Python<'a>,
        tokenizer_name: String,
        revision: Option<String>,
        trust_remote_code: bool,
    ) -> PyResult<PyTokenizer<'a>> {
        let transformers = py.import_bound("transformers")?;
        let auto = transformers.getattr("AutoTokenizer")?;
        let from_pretrained = auto.getattr("from_pretrained")?;
        let args = (tokenizer_name,);
        let kwargs = if let Some(rev) = &revision {
            [
                ("revision", rev.to_string().into_py(py)),
                ("trust_remote_code", trust_remote_code.into_py(py)),
            ]
            .into_py_dict_bound(py)
        } else {
            [("trust_remote_code", trust_remote_code.into_py(py))].into_py_dict_bound(py)
        };
        let tokenizer = from_pretrained.call(args, Some(&kwargs))?;
        tracing::info!("Loaded a python tokenizer");
        Ok(PyTokenizer(tokenizer))
    }
}

trait TokenizerTrait {
    fn encode_trait(
        &self,
        query: String,
        add_special_tokens: bool,
    ) -> Result<tokenizers::Encoding, Box<dyn std::error::Error + Send + Sync>>;
}

impl TokenizerTrait for tokenizers::Tokenizer {
    fn encode_trait(
        &self,
        query: String,
        add_special_tokens: bool,
    ) -> Result<tokenizers::Encoding, Box<dyn std::error::Error + Send + Sync>> {
        self.encode(query, add_special_tokens)
    }
}

impl<'a> TokenizerTrait for PyTokenizer<'a> {
    fn encode_trait(
        &self,
        query: String,
        add_special_tokens: bool,
    ) -> Result<tokenizers::Encoding, Box<dyn std::error::Error + Send + Sync>> {
        let py = self.0.py();
        let kwargs = [
            ("text", query.into_py(py)),
            ("add_special_tokens", add_special_tokens.into_py(py)),
        ]
        .into_py_dict_bound(py);
        let encode = self.0.getattr("encode")?;
        let input_ids: Vec<u32> = encode.call((), Some(&kwargs))?.extract()?;
        Ok(Encoding::new(
            input_ids,
            vec![],                           // type ids
            vec![],                           // tokens (strings)
            vec![],                           // words
            vec![],                           // offsets
            vec![],                           // special_tokens_mask
            vec![],                           // attention_mask
            vec![],                           // overflowing
            std::collections::HashMap::new(), //sequence_ranges
        ))
    }
}

/// Hub type
#[derive(Clone, Debug, Deserialize)]
pub struct HubModelInfo {
    #[serde(rename(deserialize = "id"))]
    pub model_id: String,
    pub sha: Option<String>,
    pub pipeline_tag: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatTemplate {
    name: String,
    template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChatTemplateVersions {
    Single(String),
    Multiple(Vec<ChatTemplate>),
}

use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HubTokenizerConfig {
    pub chat_template: Option<ChatTemplateVersions>,
    pub completion_template: Option<String>,
    pub bos_token: Option<TokenizerConfigToken>,
    pub eos_token: Option<TokenizerConfigToken>,
    pub tokenizer_class: Option<String>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
}

impl HubTokenizerConfig {
    pub fn from_file<P: AsRef<Path>>(filename: P) -> Option<Self> {
        std::fs::read_to_string(filename)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum TokenizerConfigToken {
    String(String),
    Object { content: String },
}

impl TokenizerConfigToken {
    pub fn as_str(&self) -> &str {
        match self {
            TokenizerConfigToken::String(s) => s,
            TokenizerConfigToken::Object { content } => content,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "processor_class")]
pub enum HubPreprocessorConfig {
    Idefics2Processor(Idefics2Preprocessor),
}

impl HubPreprocessorConfig {
    pub fn from_file<P: AsRef<std::path::Path>>(filename: P) -> Option<Self> {
        let content = std::fs::read_to_string(filename).ok()?;
        serde_json::from_str(&content).ok()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Idefics2Preprocessor {
    #[serde(default)]
    do_image_splitting: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct HubProcessorConfig {
    pub chat_template: Option<ChatTemplateVersions>,
    pub image_seq_len: usize,
    pub processor_class: Option<String>,
}

impl HubProcessorConfig {
    pub fn from_file<P: AsRef<Path>>(filename: P) -> Option<Self> {
        std::fs::read_to_string(filename)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema, Serialize)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(tag = "type", content = "value")]
pub(crate) enum GrammarType {
    /// A string that represents a [JSON Schema](https://json-schema.org/).
    ///
    /// JSON Schema is a declarative language that allows to annotate JSON documents
    /// with types and descriptions.
    #[serde(rename = "json")]
    #[serde(alias = "json_object")]
    #[schema(example = json ! ({"properties": {"location":{"type": "string"}}}))]
    Json(serde_json::Value),
    #[serde(rename = "regex")]
    Regex(String),
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Info {
    /// Model info
    #[schema(example = "bigscience/blomm-560m")]
    pub model_id: String,
    #[schema(nullable = true, example = "e985a63cdc139290c5f700ff1929f0b5942cced2")]
    pub model_sha: Option<String>,
    // #[schema(example = "torch.float16")]
    // pub model_dtype: String,
    // #[schema(example = "cuda")]
    // pub model_device_type: String,
    #[schema(nullable = true, example = "text-generation")]
    pub model_pipeline_tag: Option<String>,

    /// Router Parameters
    #[schema(example = "128")]
    pub max_concurrent_requests: usize,
    #[schema(example = "2")]
    pub max_best_of: usize,
    #[schema(example = "4")]
    pub max_stop_sequences: usize,
    #[schema(example = "1024")]
    pub max_input_tokens: usize,
    #[schema(example = "2048")]
    pub max_total_tokens: usize,
    #[schema(example = "2")]
    pub validation_workers: usize,
    #[schema(example = "32")]
    pub max_client_batch_size: usize,

    /// Router Info
    #[schema(example = "text-generation-router")]
    pub router: &'static str,
    #[schema(example = "0.5.0")]
    pub version: &'static str,
    #[schema(nullable = true, example = "null")]
    pub sha: Option<&'static str>,
    #[schema(nullable = true, example = "null")]
    pub docker_label: Option<&'static str>,
}

#[derive(Clone, Debug, Deserialize, ToSchema, Default)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct GenerateParameters {
    /// Generate best_of sequences and return the one if the highest token logprobs.
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 1)]
    pub best_of: Option<usize>,

    /// The value used to module the logits distribution.
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        nullable = true,
        default = "null",
        example = 0.5
    )]
    pub temperature: Option<f32>,

    /// The parameter for repetition penalty. 1.0 means no penalty.
    /// See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        nullable = true,
        default = "null",
        example = 1.03
    )]
    pub repetition_penalty: Option<f32>,

    /// The parameter for frequency penalty. 1.0 means no penalty
    /// Penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(default)]
    #[schema(
        exclusive_minimum = -2.0,
        nullable = true,
        default = "null",
        example = 0.1
    )]
    pub frequency_penalty: Option<f32>,

    /// The number of highest probability vocabulary tokens to keep for top-k-filtering.
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 10)]
    pub top_k: Option<i32>,

    /// Top-p value for nucleus sampling.
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub top_p: Option<f32>,

    /// Typical Decoding mass
    /// See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information.
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub typical_p: Option<f32>,

    /// Activate logits sampling.
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub do_sample: bool,

    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_new_tokens")]
    #[schema(nullable = true, default = "100", example = "20")]
    pub max_new_tokens: Option<u32>,

    /// Whether to prepend the prompt to the generated text
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = false)]
    pub return_full_text: Option<bool>,

    /// Stop generating tokens if a member of `stop` is generated.
    #[serde(default)]
    #[schema(inline, max_items = 4, example = json ! (["photographer"]))]
    pub stop: Vec<String>,

    /// Truncate inputs tokens to the given size.
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub truncate: Option<usize>,

    /// Watermarking with [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226).
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub watermark: bool,

    /// Whether to return generation details.
    #[serde(default)]
    #[schema(default = "true")]
    pub details: bool,

    /// Whether to return decoder input token logprobs and ids.
    #[serde(default)]
    #[schema(default = "false")]
    pub decoder_input_details: bool,

    /// Random sampling seed.
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0,
        nullable = true,
        default = "null",
        example = "null"
    )]
    pub seed: Option<u64>,

    /// The number of highest probability vocabulary tokens to keep for top-n-filtering.
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 5)]
    pub top_n_tokens: Option<u32>,

    /// Grammar constraints for the generation.
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub grammar: Option<GrammarType>,

    /// Lora adapter id
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub adapter_id: Option<String>,
}

fn default_max_new_tokens() -> Option<u32> {
    Some(100)
}

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        best_of: None,
        temperature: None,
        repetition_penalty: None,
        frequency_penalty: None,
        top_k: None,
        top_p: None,
        typical_p: None,
        do_sample: true,
        max_new_tokens: default_max_new_tokens(),
        return_full_text: None,
        stop: Vec::new(),
        truncate: None,
        watermark: false,
        details: false,
        decoder_input_details: false,
        seed: None,
        top_n_tokens: None,
        grammar: None,
        adapter_id: None,
    }
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug)]
#[serde(try_from = "PromptDeserializer")]
pub struct Prompt(pub Vec<String>);

#[derive(Deserialize)]
#[serde(untagged)]
enum PromptDeserializer {
    Single(String),
    Multiple(Vec<String>),
}

impl TryFrom<PromptDeserializer> for Prompt {
    type Error = String;

    fn try_from(value: PromptDeserializer) -> Result<Self, Self::Error> {
        match value {
            PromptDeserializer::Single(s) => Ok(Prompt(vec![s])),
            PromptDeserializer::Multiple(v) => {
                if v.is_empty() {
                    Err(
                        "Empty array detected. Do not use an empty array for the prompt."
                            .to_string(),
                    )
                } else {
                    Ok(Prompt(v))
                }
            }
        }
    }
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug)]
pub struct CompletionRequest {
    /// UNUSED
    #[schema(example = "mistralai/Mistral-7B-Instruct-v0.2")]
    /// ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API.
    pub model: Option<String>,

    /// The prompt to generate completions for.
    #[schema(example = "What is Deep Learning?")]
    pub prompt: Prompt,

    /// The maximum number of tokens that can be generated in the chat completion.
    #[serde(default)]
    #[schema(default = "32")]
    pub max_tokens: Option<u32>,

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while
    /// lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or `top_p` but not both.
    #[serde(default)]
    #[schema(nullable = true, example = 1.0)]
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the
    /// tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    #[serde(default)]
    #[schema(nullable = true, example = 0.95)]
    pub top_p: Option<f32>,

    #[serde(default = "bool::default")]
    pub stream: bool,

    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,

    /// The text to append to the prompt. This is useful for completing sentences or generating a paragraph of text.
    /// please see the completion_template field in the model's tokenizer_config.json file for completion template.
    #[serde(default)]
    pub suffix: Option<String>,

    #[serde(default)]
    pub repetition_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(default)]
    #[schema(example = "1.0")]
    pub frequency_penalty: Option<f32>,

    /// Up to 4 sequences where the API will stop generating further tokens.
    #[serde(default)]
    #[schema(nullable = true, example = "null")]
    pub stop: Option<Vec<String>>,
}

#[derive(Clone, Serialize, ToSchema)]
#[serde(tag = "object")]
enum Completion {
    #[serde(rename = "text_completion")]
    Chunk(Chunk),
    #[serde(rename = "text_completion")]
    Final(CompletionFinal),
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Default)]
pub(crate) struct CompletionFinal {
    pub id: String,
    #[schema(example = "1706270835")]
    pub created: u64,
    #[schema(example = "mistralai/Mistral-7B-Instruct-v0.2")]
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<CompletionComplete>,
    pub usage: Usage,
}

#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct CompletionComplete {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<Vec<f32>>,
    pub finish_reason: String,
}

#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct Chunk {
    pub id: String,
    pub created: u64,
    pub choices: Vec<CompletionComplete>,
    pub model: String,
    pub system_fingerprint: String,
}

#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct ChatCompletion {
    pub id: String,
    #[schema(example = "1706270835")]
    pub created: u64,
    #[schema(example = "mistralai/Mistral-7B-Instruct-v0.2")]
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChatCompletionComplete>,
    pub usage: Usage,
}

#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct ChatCompletionComplete {
    pub index: u32,
    pub message: OutputMessage,
    pub logprobs: Option<ChatCompletionLogprobs>,
    pub finish_reason: String,
}

#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct ChatCompletionLogprobs {
    content: Vec<ChatCompletionLogprob>,
}

impl From<(Token, Vec<Token>)> for ChatCompletionLogprobs {
    fn from(value: (Token, Vec<Token>)) -> Self {
        let (token, top_tokens) = value;

        Self {
            content: vec![ChatCompletionLogprob {
                token: token.text,
                logprob: token.logprob,
                top_logprobs: top_tokens
                    .into_iter()
                    .map(|t| ChatCompletionTopLogprob {
                        token: t.text,
                        logprob: t.logprob,
                    })
                    .collect(),
            }],
        }
    }
}

impl From<(Vec<Token>, Vec<Vec<Token>>)> for ChatCompletionLogprobs {
    fn from(value: (Vec<Token>, Vec<Vec<Token>>)) -> Self {
        let (tokens, top_tokens) = value;

        // Create an iterator that produces None for top_tokens once it's exhausted
        let top_tokens_iter = top_tokens
            .into_iter()
            .map(Some)
            .chain(std::iter::repeat(None));

        let content = tokens
            .into_iter()
            .zip(top_tokens_iter)
            .map(|(t, top_t_option)| ChatCompletionLogprob {
                token: t.text,
                logprob: t.logprob,
                top_logprobs: match top_t_option {
                    Some(top_t) => top_t
                        .into_iter()
                        .map(|t| ChatCompletionTopLogprob {
                            token: t.text,
                            logprob: t.logprob,
                        })
                        .collect(),
                    None => vec![], // Handle the case where there are no top tokens
                },
            })
            .collect();

        Self { content }
    }
}

#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct ChatCompletionLogprob {
    token: String,
    logprob: f32,
    top_logprobs: Vec<ChatCompletionTopLogprob>,
}

#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct ChatCompletionTopLogprob {
    token: String,
    logprob: f32,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Default)]
pub(crate) struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Serialize, ToSchema)]
#[serde(tag = "object")]
enum CompletionType {
    #[serde(rename = "chat.completion.chunk")]
    ChatCompletionChunk(ChatCompletionChunk),
    #[serde(rename = "chat.completion")]
    ChatCompletion(ChatCompletion),
}

impl ChatCompletion {
    pub(crate) fn new(
        model: String,
        system_fingerprint: String,
        output: Option<String>,
        created: u64,
        details: Details,
        return_logprobs: bool,
        tool_calls: Option<Vec<ToolCall>>,
    ) -> Self {
        let message = match (output, tool_calls) {
            (Some(content), None) => OutputMessage::ChatMessage(TextMessage {
                role: "assistant".into(),
                content,
            }),
            (None, Some(tool_calls)) => OutputMessage::ToolCall(ToolCallMessage {
                role: "assistant".to_string(),
                tool_calls,
            }),
            (Some(output), Some(_)) => {
                warn!("Received both chat and tool call");
                OutputMessage::ChatMessage(TextMessage {
                    role: "assistant".into(),
                    content: output,
                })
            }
            (None, None) => {
                warn!("Didn't receive an answer");
                OutputMessage::ChatMessage(TextMessage {
                    role: "assistant".into(),
                    content: "".to_string(),
                })
            }
        };
        Self {
            id: String::new(),
            created,
            model,
            system_fingerprint,
            choices: vec![ChatCompletionComplete {
                index: 0,
                message,
                logprobs: return_logprobs
                    .then(|| ChatCompletionLogprobs::from((details.tokens, details.top_tokens))),
                finish_reason: details.finish_reason.format(true),
            }],
            usage: Usage {
                prompt_tokens: details.prefill.len() as u32,
                completion_tokens: details.generated_tokens,
                total_tokens: details.prefill.len() as u32 + details.generated_tokens,
            },
        }
    }
}
#[derive(Clone, Serialize, ToSchema)]
pub(crate) struct ChatCompletionChunk {
    pub id: String,
    #[schema(example = "1706270978")]
    pub created: u64,
    #[schema(example = "mistralai/Mistral-7B-Instruct-v0.2")]
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Option<Usage>,
}

#[derive(Clone, Serialize, ToSchema)]
pub(crate) struct ChatCompletionChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    pub logprobs: Option<ChatCompletionLogprobs>,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
pub struct ToolCallDelta {
    #[schema(example = "assistant")]
    role: String,
    tool_calls: DeltaToolCall,
}

#[derive(Clone, Debug, Serialize, ToSchema)]
#[serde(untagged)]
enum ChatCompletionDelta {
    Chat(TextMessage),
    Tool(ToolCallDelta),
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug, PartialEq)]
pub(crate) struct DeltaToolCall {
    pub index: u32,
    pub id: String,
    pub r#type: String,
    pub function: Function,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug, PartialEq)]
pub(crate) struct Function {
    pub name: Option<String>,
    pub arguments: String,
}

#[allow(clippy::too_many_arguments)]
impl ChatCompletionChunk {
    pub(crate) fn new(
        model: String,
        system_fingerprint: String,
        delta: Option<String>,
        tool_calls: Option<Vec<String>>,
        created: u64,
        logprobs: Option<ChatCompletionLogprobs>,
        finish_reason: Option<String>,
        usage: Option<Usage>,
    ) -> Self {
        let delta = match (delta, tool_calls) {
            (Some(delta), _) => ChatCompletionDelta::Chat(TextMessage {
                role: "assistant".to_string(),
                content: delta,
            }),
            (None, Some(tool_calls)) => ChatCompletionDelta::Tool(ToolCallDelta {
                role: "assistant".to_string(),
                tool_calls: DeltaToolCall {
                    index: 0,
                    id: String::new(),
                    r#type: "function".to_string(),
                    function: Function {
                        name: None,
                        arguments: tool_calls[0].to_string(),
                    },
                },
            }),
            (None, None) => ChatCompletionDelta::Chat(TextMessage {
                role: "assistant".to_string(),
                content: "".to_string(),
            }),
        };
        Self {
            id: String::new(),
            created,
            model,
            system_fingerprint,
            choices: vec![ChatCompletionChoice {
                index: 0,
                delta,
                logprobs,
                finish_reason,
            }],
            usage,
        }
    }
}

#[derive(Clone, Deserialize, ToSchema, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq, Default))]
pub(crate) struct ChatRequest {
    #[schema(example = "mistralai/Mistral-7B-Instruct-v0.2")]
    /// [UNUSED] ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API.
    pub model: Option<String>,

    /// A list of messages comprising the conversation so far.
    #[schema(example = "[{\"role\": \"user\", \"content\": \"What is Deep Learning?\"}]")]
    pub messages: Vec<Message>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(default)]
    #[schema(example = "1.0")]
    pub frequency_penalty: Option<f32>,

    /// UNUSED
    /// Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens
    /// (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically,
    /// the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model,
    /// but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should
    /// result in a ban or exclusive selection of the relevant token.
    #[serde(default)]
    pub logit_bias: Option<Vec<f32>>,

    /// Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each
    /// output token returned in the content of message.
    #[serde(default)]
    #[schema(example = "false")]
    pub logprobs: Option<bool>,

    /// An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with
    /// an associated log probability. logprobs must be set to true if this parameter is used.
    #[serde(default)]
    #[schema(example = "5")]
    pub top_logprobs: Option<u32>,

    /// The maximum number of tokens that can be generated in the chat completion.
    #[serde(default)]
    #[schema(example = "32")]
    pub max_tokens: Option<u32>,

    /// UNUSED
    /// How many chat completion choices to generate for each input message. Note that you will be charged based on the
    /// number of generated tokens across all of the choices. Keep n as 1 to minimize costs.
    #[serde(default)]
    #[schema(nullable = true, example = "2")]
    pub n: Option<u32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far,
    /// increasing the model's likelihood to talk about new topics
    #[serde(default)]
    #[schema(nullable = true, example = 0.1)]
    pub presence_penalty: Option<f32>,

    /// Up to 4 sequences where the API will stop generating further tokens.
    #[serde(default)]
    #[schema(nullable = true, example = "null")]
    pub stop: Option<Vec<String>>,

    #[serde(default = "bool::default")]
    pub stream: bool,

    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,

    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while
    /// lower values like 0.2 will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    #[serde(default)]
    #[schema(nullable = true, example = 1.0)]
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the
    /// tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    #[serde(default)]
    #[schema(nullable = true, example = 0.95)]
    pub top_p: Option<f32>,

    /// A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of
    /// functions the model may generate JSON inputs for.
    #[serde(default)]
    #[schema(nullable = true, example = "null")]
    pub tools: Option<Vec<Tool>>,

    /// A prompt to be appended before the tools
    #[serde(default)]
    #[schema(
        nullable = true,
        example = "Given the functions available, please respond with a JSON for a function call with its proper arguments that best answers the given prompt. Respond in the format {name: function name, parameters: dictionary of argument name and its value}.Do not use variables."
    )]
    pub tool_prompt: Option<String>,

    /// A specific tool to use. If not provided, the model will default to use any of the tools provided in the tools parameter.
    #[serde(default)]
    #[schema(nullable = true, example = "null")]
    pub tool_choice: ToolChoice,

    /// Response format constraints for the generation.
    ///
    /// NOTE: A request can use `response_format` OR `tools` but not both.
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub response_format: Option<GrammarType>,

    /// A guideline to be used in the chat_template
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub guideline: Option<String>,

    /// Options for streaming response. Only set this when you set stream: true.
    #[serde(default)]
    #[schema(nullable = true, example = "null")]
    pub stream_options: Option<StreamOptions>,
}

impl ChatRequest {
    fn try_into_generate(self, infer: &Infer) -> Result<(GenerateRequest, bool), InferError> {
        let ChatRequest {
            model,
            max_tokens,
            messages,
            seed,
            stop,
            stream,
            tools,
            tool_choice,
            tool_prompt,
            temperature,
            response_format,
            guideline,
            presence_penalty,
            frequency_penalty,
            top_p,
            top_logprobs,
            ..
        } = self;

        let repetition_penalty = presence_penalty.map(|x| x + 2.0);
        let max_new_tokens = max_tokens.or(Some(100));
        let tool_prompt = tool_prompt
            .filter(|s| !s.is_empty())
            .unwrap_or_else(default_tool_prompt);
        let stop = stop.unwrap_or_default();
        // enable greedy only when temperature is 0
        let (do_sample, temperature) = match temperature {
            Some(temperature) if temperature == 0.0 => (false, None),
            other => (true, other),
        };
        let (inputs, grammar, using_tools) = prepare_chat_input(
            infer,
            response_format,
            tools,
            tool_choice,
            &tool_prompt,
            guideline,
            messages,
        )?;

        Ok((
            GenerateRequest {
                inputs: inputs.to_string(),
                add_special_tokens: false,
                parameters: GenerateParameters {
                    best_of: None,
                    temperature,
                    repetition_penalty,
                    frequency_penalty,
                    top_k: None,
                    top_p,
                    typical_p: None,
                    do_sample,
                    max_new_tokens,
                    return_full_text: None,
                    stop,
                    truncate: None,
                    watermark: false,
                    details: true,
                    decoder_input_details: !stream,
                    seed,
                    top_n_tokens: top_logprobs,
                    grammar,
                    adapter_id: model.filter(|m| *m != "tgi").map(String::from),
                },
            },
            using_tools,
        ))
    }
}

#[derive(Clone, Deserialize, ToSchema, Serialize)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct StreamOptions {
    /// If set, an additional chunk will be streamed before the data: [DONE] message. The usage field on this chunk shows the token usage statistics for the entire request, and the choices field will always be an empty array. All other chunks will also include a usage field, but with a null value.
    #[schema(example = "true")]
    include_usage: bool,
}

pub fn default_tool_prompt() -> String {
    "\nGiven the functions available, please respond with a JSON for a function call with its proper arguments that best answers the given prompt. Respond in the format {name: function name, parameters: dictionary of argument name and its value}.Do not use variables.\n".to_string()
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ToSchema)]
#[schema(example = "auto")]
/// Controls which (if any) tool is called by the model.
pub enum ToolType {
    /// Means the model can pick between generating a message or calling one or more tools.
    #[schema(rename = "auto")]
    OneOf,
    /// Means the model will not call any tool and instead generates a message.
    #[schema(rename = "none")]
    NoTool,
    /// Forces the model to call a specific tool.
    #[schema(rename = "function")]
    Function(FunctionName),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct FunctionName {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, ToSchema)]
#[serde(from = "ToolTypeDeserializer")]
pub struct ToolChoice(pub Option<ToolType>);

#[derive(Deserialize)]
#[serde(untagged)]
enum ToolTypeDeserializer {
    Null,
    String(String),
    ToolType(ToolType),
}

impl From<ToolTypeDeserializer> for ToolChoice {
    fn from(value: ToolTypeDeserializer) -> Self {
        match value {
            ToolTypeDeserializer::Null => ToolChoice(None),
            ToolTypeDeserializer::String(s) => match s.as_str() {
                "none" => ToolChoice(Some(ToolType::NoTool)),
                "auto" => ToolChoice(Some(ToolType::OneOf)),
                _ => ToolChoice(Some(ToolType::Function(FunctionName { name: s }))),
            },
            ToolTypeDeserializer::ToolType(tool_type) => ToolChoice(Some(tool_type)),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, ToSchema, PartialEq)]
pub struct JsonSchemaTool {
    #[serde(flatten)]
    functions_map: FunctionsMap,
    properties: Properties,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct FunctionsMap {
    #[serde(rename = "$functions")]
    functions: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct FunctionRef {
    #[serde(rename = "$ref")]
    ref_path: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Properties {
    #[serde(serialize_with = "serialize_function")]
    function: Vec<FunctionRef>,
}

fn serialize_function<S>(functions: &Vec<FunctionRef>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeStruct;
    let mut state = serializer.serialize_struct("Function", 1)?;
    state.serialize_field("anyOf", functions)?;
    state.end()
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema, Default, PartialEq)]
pub(crate) struct FunctionDefinition {
    #[serde(default)]
    pub description: Option<String>,
    pub name: String,
    #[serde(alias = "parameters")]
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Tool {
    // The type of the tool. Currently, only 'function' is supported.
    #[schema(example = "function")]
    pub r#type: String,
    // Grab the tool as generic JSON for debugging purposes.
    pub function: FunctionDefinition,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct ChatTemplateInputs<'a> {
    messages: Vec<TextMessage>,
    bos_token: Option<&'a str>,
    eos_token: Option<&'a str>,
    add_generation_prompt: bool,
    tools: Option<Vec<Tool>>,
    guideline: Option<&'a str>,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Default, Debug, PartialEq)]
pub(crate) struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
pub struct Url {
    url: String,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum MessageChunk {
    Text { text: String },
    ImageUrl { image_url: Url },
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
pub struct Message {
    #[schema(example = "user")]
    role: String,
    #[schema(example = "My name is David and I")]
    pub content: MessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "\"David\"")]
    name: Option<String>,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    SingleText(String),
    MultipleChunks(Vec<MessageChunk>),
}

// Pushing a chunk to a single text message will convert it to a multiple chunks message
impl MessageContent {
    pub fn push(&mut self, chunk: MessageChunk) {
        match self {
            MessageContent::SingleText(text) => {
                *self = MessageContent::MultipleChunks(vec![
                    MessageChunk::Text { text: text.clone() },
                    chunk,
                ]);
            }
            MessageContent::MultipleChunks(chunks) => {
                chunks.push(chunk);
            }
        }
    }
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
pub struct TextMessage {
    #[schema(example = "user")]
    pub role: String,
    #[schema(example = "My name is David and I")]
    pub content: String,
}

impl From<Message> for TextMessage {
    fn from(value: Message) -> Self {
        TextMessage {
            role: value.role,
            content: match value.content {
                MessageContent::SingleText(text) => text,
                MessageContent::MultipleChunks(chunks) => chunks
                    .into_iter()
                    .map(|chunk| match chunk {
                        MessageChunk::Text { text } => text,
                        MessageChunk::ImageUrl { image_url } => format!("![]({})", image_url.url),
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            },
        }
    }
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
pub struct ToolCallMessage {
    #[schema(example = "assistant")]
    role: String,
    tool_calls: Vec<ToolCall>,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
#[serde(untagged)]
pub(crate) enum OutputMessage {
    ChatMessage(TextMessage),
    ToolCall(ToolCallMessage),
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,

    /// This is used internally because some requests
    /// already contain the templated input therefore
    /// we shouldn't add the special tokens.
    #[serde(default = "default_true", skip)]
    pub add_special_tokens: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct CompatGenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
    #[serde(default)]
    #[schema(default = "false")]
    pub stream: bool,
}

impl From<CompatGenerateRequest> for GenerateRequest {
    fn from(req: CompatGenerateRequest) -> Self {
        Self {
            inputs: req.inputs,
            add_special_tokens: true,
            parameters: req.parameters,
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PrefillToken {
    #[schema(example = 0)]
    pub id: u32,
    #[schema(example = "test")]
    pub text: String,
    #[schema(nullable = true, example = - 0.34)]
    pub logprob: f32,
}

#[derive(Debug, Serialize, ToSchema, Clone)]
pub struct Token {
    #[schema(example = 0)]
    pub id: u32,
    #[schema(example = "test")]
    pub text: String,
    #[schema(nullable = true, example = - 0.34)]
    pub logprob: f32,
    #[schema(example = "false")]
    pub special: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct SimpleToken {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(example = 0)]
    start: usize,
    #[schema(example = 2)]
    stop: usize,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all(serialize = "snake_case"))]
#[schema(example = "Length")]
pub enum FinishReason {
    #[schema(rename = "length")]
    Length,
    #[serde(rename = "eos_token")]
    #[schema(rename = "eos_token")]
    EndOfSequenceToken,
    #[schema(rename = "stop_sequence")]
    StopSequence,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::Length => write!(f, "length"),
            FinishReason::EndOfSequenceToken => write!(f, "eos_token"),
            FinishReason::StopSequence => write!(f, "stop_sequence"),
        }
    }
}

impl FinishReason {
    pub fn format(&self, use_stop: bool) -> String {
        match self {
            FinishReason::EndOfSequenceToken if use_stop => "stop".to_string(),
            _ => self.to_string(),
        }
    }
}

#[derive(Serialize, ToSchema)]
pub(crate) struct BestOfSequence {
    #[schema(example = "test")]
    pub generated_text: String,
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
    pub prefill: Vec<PrefillToken>,
    pub tokens: Vec<Token>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_tokens: Vec<Vec<Token>>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Details {
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
    pub prefill: Vec<PrefillToken>,
    pub tokens: Vec<Token>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of_sequences: Option<Vec<BestOfSequence>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_tokens: Vec<Vec<Token>>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct GenerateResponse {
    #[schema(example = "test")]
    pub generated_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Details>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ChatTokenizeResponse {
    pub(crate) tokenize_response: TokenizeResponse,
    pub(crate) templated_text: String,
}

#[derive(Serialize, ToSchema)]
#[serde(transparent)]
pub(crate) struct TokenizeResponse(Vec<SimpleToken>);

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamDetails {
    #[schema(example = "length")]
    pub finish_reason: FinishReason,
    #[schema(example = 1)]
    pub generated_tokens: u32,
    #[schema(nullable = true, example = 42)]
    pub seed: Option<u64>,
    #[schema(example = 1)]
    pub input_length: u32,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamResponse {
    pub index: u32,
    pub token: Token,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_tokens: Vec<Token>,
    #[schema(nullable = true, default = "null", example = "test")]
    pub generated_text: Option<String>,
    #[schema(nullable = true, default = "null")]
    pub details: Option<StreamDetails>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ErrorResponse {
    pub error: String,
    pub error_type: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub(crate) struct ModelInfo {
    #[schema(example = "gpt2")]
    pub id: String,
    #[schema(example = "model")]
    pub object: String,
    #[schema(example = 1686935002)]
    pub created: u64,
    #[schema(example = "openai")]
    pub owned_by: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub(crate) struct ModelsInfo {
    #[schema(example = "list")]
    pub object: String,
    pub data: Vec<ModelInfo>,
}

impl Default for ModelsInfo {
    fn default() -> Self {
        ModelsInfo {
            object: "list".to_string(),
            data: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn get_tokenizer() -> Tokenizer {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.model("gpt2".to_string());
        let filename = repo.get("tokenizer.json").unwrap();
        Tokenizer::Rust(tokenizers::Tokenizer::from_file(filename).unwrap())
    }

    #[test]
    fn test_hub_nested_tokens_tokenizer_config() {
        // this is a subset of the tokenizer.json file
        // in this case we expect the tokens to be encoded as simple strings
        let json_content = r#"{
            "chat_template": "test",
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>"
        }"#;

        let config: HubTokenizerConfig = serde_json::from_str(json_content).unwrap();

        // check that we successfully parsed the tokens
        assert_eq!(
            config.chat_template,
            Some(ChatTemplateVersions::Single("test".to_string()))
        );
        assert_eq!(
            config.bos_token,
            Some(TokenizerConfigToken::String(
                "<｜begin▁of▁sentence｜>".to_string()
            ))
        );
        assert_eq!(
            config.eos_token,
            Some(TokenizerConfigToken::String(
                "<｜end▁of▁sentence｜>".to_string()
            ))
        );

        // in this case we expect the tokens to be encoded as structured tokens
        // we want the content of the structured token
        let json_content = r#"{
            "chat_template": "test",
            "bos_token": {
              "__type": "AddedToken",
              "content": "<｜begin▁of▁sentence｜>",
              "lstrip": false,
              "normalized": true,
              "rstrip": false,
              "single_word": false
            },
            "eos_token": {
              "__type": "AddedToken",
              "content": "<｜end▁of▁sentence｜>",
              "lstrip": false,
              "normalized": true,
              "rstrip": false,
              "single_word": false
            }
        }"#;

        let config: HubTokenizerConfig = serde_json::from_str(json_content).unwrap();

        // check that we successfully parsed the tokens
        assert_eq!(
            config.chat_template,
            Some(ChatTemplateVersions::Single("test".to_string()))
        );
        assert_eq!(
            config.bos_token,
            Some(TokenizerConfigToken::Object {
                content: "<｜begin▁of▁sentence｜>".to_string()
            })
        );
        assert_eq!(
            config.eos_token,
            Some(TokenizerConfigToken::Object {
                content: "<｜end▁of▁sentence｜>".to_string()
            })
        );
    }

    #[test]
    fn test_chat_simple_string() {
        let json = json!({
            "model": "",
            "messages": [{
                "role": "user",
                "content": "What is Deep Learning?"
            }]
        });
        let request: ChatRequest = serde_json::from_str(json.to_string().as_str()).unwrap();

        assert_eq!(
            request.messages[0],
            Message {
                role: "user".to_string(),
                content: MessageContent::SingleText("What is Deep Learning?".to_string()),
                name: None
            }
        );
    }

    #[test]
    fn test_message_content_append() {
        let mut content = MessageContent::SingleText("Initial text".to_string());
        let chunk = MessageChunk::Text {
            text: "Additional text".to_string(),
        };

        content.push(chunk);

        match content {
            MessageContent::MultipleChunks(chunks) => {
                assert_eq!(chunks.len(), 2);
                assert_eq!(
                    chunks[0],
                    MessageChunk::Text {
                        text: "Initial text".to_string()
                    }
                );
                assert_eq!(
                    chunks[1],
                    MessageChunk::Text {
                        text: "Additional text".to_string()
                    }
                );
            }
            _ => panic!("Expected MultipleChunks, but got a different variant"),
        }
    }

    #[test]
    fn test_chat_request() {
        let json = json!({
            "model": "",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Whats in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"}},
                ]
            }]
        });
        let request: ChatRequest = serde_json::from_str(json.to_string().as_str()).unwrap();

        assert_eq!(
            request.messages[0],
            Message{
                role: "user".to_string(),
                content: MessageContent::MultipleChunks(vec![
                    MessageChunk::Text { text: "Whats in this image?".to_string() },
                    MessageChunk::ImageUrl { image_url: Url { url: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png".to_string() }},
                ]),
                name: None
            }
        );
    }

    #[test]
    fn text_message_convert() {
        let message = Message{
                role: "user".to_string(),
                content: MessageContent::MultipleChunks(vec![
                    MessageChunk::Text { text: "Whats in this image?".to_string() },
                    MessageChunk::ImageUrl { image_url: Url { url: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png".to_string() } }
                ]),
                name: None
            };
        let textmsg: TextMessage = message.into();
        assert_eq!(textmsg.content, "Whats in this image?![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png)");
    }

    #[test]
    fn test_chat_stream_options() {
        let json = json!({
            "model": "",
            "stream_options": {"include_usage": true},
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        });
        let request: ChatRequest = serde_json::from_str(json.to_string().as_str()).unwrap();

        assert!(matches!(
            request.stream_options,
            Some(StreamOptions {
                include_usage: true
            })
        ));
    }

    #[test]
    fn openai_output() {
        let message = OutputMessage::ChatMessage(TextMessage {
            role: "assistant".to_string(),
            content: "This is the answer".to_string(),
        });
        let serialized = serde_json::to_string(&message).unwrap();
        assert_eq!(
            serialized,
            r#"{"role":"assistant","content":"This is the answer"}"#
        );

        let message = OutputMessage::ToolCall(ToolCallMessage {
            role: "assistant".to_string(),
            tool_calls: vec![ToolCall {
                id: "0".to_string(),
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    description: None,
                    name: "myfn".to_string(),
                    arguments: json!({
                        "format": "csv"
                    }),
                },
            }],
        });
        let serialized = serde_json::to_string(&message).unwrap();
        assert_eq!(
            serialized,
            r#"{"role":"assistant","tool_calls":[{"id":"0","type":"function","function":{"description":null,"name":"myfn","arguments":{"format":"csv"}}}]}"#
        );
    }
}
