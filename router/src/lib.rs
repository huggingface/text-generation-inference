mod health;
/// Text Generation Inference Webserver
mod infer;
mod queue;
pub mod server;
mod validation;

use infer::{Infer, InferError, InferStreamResponse};
use queue::{Entry, Queue};
use serde::{Deserialize, Serialize};
use tokio::sync::OwnedSemaphorePermit;
use tokio_stream::wrappers::UnboundedReceiverStream;
use utoipa::ToSchema;
use validation::Validation;

/// Type alias for generation responses
pub(crate) type GenerateStreamResponse = (
    OwnedSemaphorePermit,
    u32, // input_length
    UnboundedReceiverStream<Result<InferStreamResponse, InferError>>,
);

#[derive(Clone, Deserialize, ToSchema)]
pub(crate) struct VertexInstance {
    #[schema(example = "What is Deep Learning?")]
    pub inputs: String,
    #[schema(nullable = true, default = "null", example = "null")]
    pub parameters: Option<GenerateParameters>,
}

#[derive(Deserialize, ToSchema)]
pub(crate) struct VertexRequest {
    #[serde(rename = "instances")]
    pub instances: Vec<VertexInstance>,
}

#[derive(Clone, Deserialize, ToSchema, Serialize)]
pub(crate) struct VertexResponse {
    pub predictions: Vec<String>,
}

/// Hub type
#[derive(Clone, Debug, Deserialize)]
pub struct HubModelInfo {
    #[serde(rename(deserialize = "id"))]
    pub model_id: String,
    pub sha: Option<String>,
    pub pipeline_tag: Option<String>,
}

#[derive(Clone, Deserialize, Default)]
pub struct HubTokenizerConfig {
    pub chat_template: Option<String>,
    pub completion_template: Option<String>,
    #[serde(deserialize_with = "token_serde::deserialize")]
    pub bos_token: Option<String>,
    #[serde(deserialize_with = "token_serde::deserialize")]
    pub eos_token: Option<String>,
}

impl HubTokenizerConfig {
    pub fn from_file(filename: &std::path::Path) -> Self {
        let content = std::fs::read_to_string(filename).unwrap();
        serde_json::from_str(&content).unwrap_or_default()
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
#[serde(tag = "type", content = "value")]
pub(crate) enum GrammarType {
    /// A string that represents a [JSON Schema](https://json-schema.org/).
    ///
    /// JSON Schema is a declarative language that allows to annotate JSON documents
    /// with types and descriptions.
    #[serde(rename = "json")]
    #[schema(example = json ! ({"properties": {"location":{"type": "string"}}}))]
    Json(serde_json::Value),
    #[serde(rename = "regex")]
    Regex(String),
}

mod token_serde {
    use super::*;
    use serde::de;
    use serde::Deserializer;
    use serde_json::Value;

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;

        match value {
            Value::String(s) => Ok(Some(s)),
            Value::Object(map) => {
                if let Some(content) = map.get("content").and_then(|v| v.as_str()) {
                    Ok(Some(content.to_string()))
                } else {
                    Err(de::Error::custom(
                        "content key not found in structured token",
                    ))
                }
            }
            _ => Err(de::Error::custom("invalid token format")),
        }
    }
}

#[derive(Clone, Debug, Serialize, ToSchema)]
pub struct Info {
    /// Model info
    #[schema(example = "bigscience/blomm-560m")]
    pub model_id: String,
    #[schema(nullable = true, example = "e985a63cdc139290c5f700ff1929f0b5942cced2")]
    pub model_sha: Option<String>,
    #[schema(example = "torch.float16")]
    pub model_dtype: String,
    #[schema(example = "cuda")]
    pub model_device_type: String,
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
    pub max_input_length: usize,
    #[schema(example = "2048")]
    pub max_total_tokens: usize,
    #[schema(example = "1.2")]
    pub waiting_served_ratio: f32,
    #[schema(example = "32000")]
    pub max_batch_total_tokens: u32,
    #[schema(example = "20")]
    pub max_waiting_tokens: usize,
    #[schema(nullable = true, example = "null")]
    pub max_batch_size: Option<usize>,
    #[schema(example = "2")]
    pub validation_workers: usize,
    /// Router Info
    #[schema(example = "0.5.0")]
    pub version: &'static str,
    #[schema(nullable = true, example = "null")]
    pub sha: Option<&'static str>,
    #[schema(nullable = true, example = "null")]
    pub docker_label: Option<&'static str>,
}

#[derive(Clone, Debug, Deserialize, ToSchema, Default)]
pub(crate) struct GenerateParameters {
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 1)]
    pub best_of: Option<usize>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        nullable = true,
        default = "null",
        example = 0.5
    )]
    pub temperature: Option<f32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        nullable = true,
        default = "null",
        example = 1.03
    )]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = -2.0,
        nullable = true,
        default = "null",
        example = 0.1
    )]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 10)]
    pub top_k: Option<i32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub top_p: Option<f32>,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0.0,
        maximum = 1.0,
        nullable = true,
        default = "null",
        example = 0.95
    )]
    pub typical_p: Option<f32>,
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub do_sample: bool,
    #[serde(default = "default_max_new_tokens")]
    #[schema(nullable = true, default = "100", example = "20")]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = false)]
    pub return_full_text: Option<bool>,
    #[serde(default)]
    #[schema(inline, max_items = 4, example = json ! (["photographer"]))]
    pub stop: Vec<String>,
    #[serde(default)]
    #[schema(nullable = true, default = "null", example = "null")]
    pub truncate: Option<usize>,
    #[serde(default)]
    #[schema(default = "false", example = true)]
    pub watermark: bool,
    #[serde(default)]
    #[schema(default = "true")]
    pub details: bool,
    #[serde(default)]
    #[schema(default = "true")]
    pub decoder_input_details: bool,
    #[serde(default)]
    #[schema(
        exclusive_minimum = 0,
        nullable = true,
        default = "null",
        example = "null"
    )]
    pub seed: Option<u64>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null", example = 5)]
    pub top_n_tokens: Option<u32>,
    #[serde(default)]
    pub grammar: Option<GrammarType>,
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
    }
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug)]
pub struct CompletionRequest {
    /// UNUSED
    #[schema(example = "mistralai/Mistral-7B-Instruct-v0.2")]
    /// ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API.
    pub model: String,

    /// The prompt to generate completions for.
    #[schema(example = "What is Deep Learning?")]
    pub prompt: String,

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
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Default)]
pub(crate) struct Completion {
    pub id: String,
    pub object: String,
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
pub(crate) struct ChatCompletion {
    pub id: String,
    pub object: String,
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
    pub message: Message,
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
        Self {
            content: tokens
                .into_iter()
                .zip(top_tokens)
                .map(|(t, top_t)| ChatCompletionLogprob {
                    token: t.text,
                    logprob: t.logprob,
                    top_logprobs: top_t
                        .into_iter()
                        .map(|t| ChatCompletionTopLogprob {
                            token: t.text,
                            logprob: t.logprob,
                        })
                        .collect(),
                })
                .collect(),
        }
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

impl ChatCompletion {
    pub(crate) fn new(
        model: String,
        system_fingerprint: String,
        output: Option<String>,
        created: u64,
        details: Details,
        return_logprobs: bool,
        tool_calls: Option<ToolCall>,
    ) -> Self {
        Self {
            id: String::new(),
            object: "text_completion".into(),
            created,
            model,
            system_fingerprint,
            choices: vec![ChatCompletionComplete {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content: output,
                    name: None,
                    tool_calls,
                },
                logprobs: return_logprobs
                    .then(|| ChatCompletionLogprobs::from((details.tokens, details.top_tokens))),
                finish_reason: details.finish_reason.to_string(),
            }],
            usage: Usage {
                prompt_tokens: details.prefill.len() as u32,
                completion_tokens: details.generated_tokens,
                total_tokens: details.prefill.len() as u32 + details.generated_tokens,
            },
        }
    }
}
#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct CompletionCompleteChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<CompletionComplete>,
    pub model: String,
    pub system_fingerprint: String,
}
#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    #[schema(example = "1706270978")]
    pub created: u64,
    #[schema(example = "mistralai/Mistral-7B-Instruct-v0.2")]
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Clone, Deserialize, Serialize, ToSchema)]
pub(crate) struct ChatCompletionChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    pub logprobs: Option<ChatCompletionLogprobs>,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub(crate) struct ChatCompletionDelta {
    #[schema(example = "user")]
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "What is Deep Learning?")]
    pub content: Option<String>,
    // default to None
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<DeltaToolCall>,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug)]
pub(crate) struct DeltaToolCall {
    pub index: u32,
    pub id: String,
    pub r#type: String,
    pub function: Function,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Debug)]
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
        index: u32,
        logprobs: Option<ChatCompletionLogprobs>,
        finish_reason: Option<String>,
    ) -> Self {
        Self {
            id: String::new(),
            object: "text_completion".to_string(),
            created,
            model,
            system_fingerprint,
            choices: vec![ChatCompletionChoice {
                index,
                delta: ChatCompletionDelta {
                    role: "assistant".to_string(),
                    content: delta,
                    tool_calls: tool_calls.map(|tc| DeltaToolCall {
                        index,
                        id: String::new(),
                        r#type: "function".to_string(),
                        function: Function {
                            name: None,
                            arguments: tc[0].to_string(),
                        },
                    }),
                },
                logprobs,
                finish_reason,
            }],
        }
    }
}

#[derive(Clone, Deserialize, ToSchema, Serialize)]
pub(crate) struct ChatRequest {
    #[schema(example = "mistralai/Mistral-7B-Instruct-v0.2")]
    /// [UNUSED] ID of the model to use. See the model endpoint compatibility table for details on which models work with the Chat API.
    pub model: String,

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
    #[serde(default = "default_tool_prompt")]
    #[schema(
        nullable = true,
        example = "\"Based on the conversation, please choose the most appropriate tool to use: \""
    )]
    pub tool_prompt: Option<String>,

    /// A specific tool to use. If not provided, the model will default to use any of the tools provided in the tools parameter.
    #[serde(default)]
    #[schema(nullable = true, example = "null")]
    #[serde(deserialize_with = "deserialize_tool_choice::deserialize")]
    pub tool_choice: Option<ToolType>,
}

fn default_tool_prompt() -> Option<String> {
    Some(
        "\nBased on the conversation, please choose the most appropriate tool to use: ".to_string(),
    )
}
#[derive(Clone, Deserialize, ToSchema, Serialize)]
enum ToolType {
    FunctionName(String),
    OneOf,
}

/// Deserialize the tool choice from the JSON input or from the function name ("none" is allowed but mapped to None)
mod deserialize_tool_choice {
    use super::*;
    use serde::de;
    use serde::Deserializer;
    use serde_json::Value;

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<ToolType>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;

        match value {
            Value::String(s) => match s.as_str() {
                "none" => Ok(None),
                "auto" => Ok(Some(ToolType::OneOf)),
                _ => Ok(Some(ToolType::FunctionName(s))),
            },
            Value::Object(map) => {
                if let Some(content) = map
                    .get("function")
                    .and_then(|v| v.get("name"))
                    .and_then(|v| v.as_str())
                {
                    Ok(Some(ToolType::FunctionName(content.to_string())))
                } else {
                    Err(de::Error::custom("function key not found in tool choice"))
                }
            }
            Value::Null => Ok(Some(ToolType::OneOf)),
            _ => Err(de::Error::custom("invalid token format")),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct Tools {
    #[serde(flatten)]
    functions_map: FunctionsMap,
    properties: Properties,
}

#[derive(Debug, Serialize, Deserialize)]
struct FunctionsMap {
    #[serde(rename = "$functions")]
    functions: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct FunctionRef {
    #[serde(rename = "$ref")]
    ref_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema, Default)]
pub(crate) struct FunctionDefinition {
    #[serde(default)]
    pub description: Option<String>,
    pub name: String,
    pub parameters: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub(crate) struct Tool {
    // The type of the tool. Currently, only 'function' is supported.
    #[schema(example = "function")]
    pub r#type: String,
    // Grab the tool as generic JSON for debugging purposes.
    pub function: FunctionDefinition,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct ChatTemplateInputs<'a> {
    messages: Vec<Message>,
    bos_token: Option<&'a str>,
    eos_token: Option<&'a str>,
    add_generation_prompt: bool,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Default, Debug)]
pub(crate) struct ToolCall {
    pub id: u32,
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Clone, Deserialize, ToSchema, Serialize)]
pub(crate) struct Message {
    #[schema(example = "user")]
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(example = "My name is David and I")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "\"David\"")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<ToolCall>,
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
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
            parameters: req.parameters,
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PrefillToken {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = - 0.34)]
    logprob: f32,
}

#[derive(Debug, Serialize, ToSchema, Clone)]
pub struct Token {
    #[schema(example = 0)]
    id: u32,
    #[schema(example = "test")]
    text: String,
    #[schema(nullable = true, example = - 0.34)]
    logprob: f32,
    #[schema(example = "false")]
    special: bool,
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

#[derive(Serialize, ToSchema)]
#[serde(rename_all(serialize = "snake_case"))]
#[schema(example = "Length")]
pub(crate) enum FinishReason {
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

#[cfg(test)]
mod tests {
    use super::*;

    use tokenizers::Tokenizer;

    pub(crate) async fn get_tokenizer() -> Tokenizer {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.model("gpt2".to_string());
        let filename = repo.get("tokenizer.json").unwrap();
        Tokenizer::from_file(filename).unwrap()
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
        assert_eq!(config.chat_template, Some("test".to_string()));
        assert_eq!(
            config.bos_token,
            Some("<｜begin▁of▁sentence｜>".to_string())
        );
        assert_eq!(config.eos_token, Some("<｜end▁of▁sentence｜>".to_string()));

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
        assert_eq!(config.chat_template, Some("test".to_string()));
        assert_eq!(
            config.bos_token,
            Some("<｜begin▁of▁sentence｜>".to_string())
        );
        assert_eq!(config.eos_token, Some("<｜end▁of▁sentence｜>".to_string()));
    }
}
