/// Text Generation Inference Webserver
mod infer;
mod queue;
pub mod server;
mod validation;

use infer::Infer;
use queue::{Entry, Queue};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validation::Validation;

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateParameters {
    #[serde(default)]
    #[schema(exclusive_minimum = 0.0, nullable = true, default = "null")]
    pub temperature: Option<f32>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0.0, nullable = true, default = "null")]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0, nullable = true, default = "null")]
    pub top_k: Option<i32>,
    #[serde(default)]
    #[schema(exclusive_minimum = 0.0, maximum = 1.0, nullable = true, default = "null")]
    pub top_p: Option<f32>,
    #[serde(default = "default_do_sample")]
    #[schema(default = "false")]
    pub do_sample: bool,
    #[serde(default = "default_max_new_tokens")]
    #[schema(exclusive_minimum = 0, exclusive_maximum = 512, default = "20")]
    pub max_new_tokens: u32,
    #[serde(default)]
    #[schema(max_items = 4, default = "null")]
    pub stop: Vec<String>,
    #[serde(default)]
    #[schema(default = "true")]
    pub details: bool,
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_do_sample() -> bool {
    false
}

fn default_max_new_tokens() -> u32 {
    20
}

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        temperature: None,
        repetition_penalty: None,
        top_k: None,
        top_p: None,
        do_sample: default_do_sample(),
        max_new_tokens: default_max_new_tokens(),
        stop: vec![],
        details: false,
        seed: None,
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema)]
pub(crate) struct GenerateRequest {
    #[schema(example = "My name is Olivier and I")]
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct Token {
    id: u32,
    text: String,
    logprob: f32,
}

#[derive(Serialize, ToSchema)]
pub(crate) enum FinishReason {
    Length,
    EndOfSequenceToken,
    StopSequence,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct Details {
    pub finish_reason: FinishReason,
    pub generated_tokens: u32,
    pub seed: Option<u64>,
    pub prefill: Option<Vec<Token>>,
    pub tokens: Option<Vec<Token>>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct GenerateResponse {
    pub generated_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Details>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamDetails {
    pub finish_reason: FinishReason,
    pub generated_tokens: u32,
    pub seed: Option<u64>,
}

#[derive(Serialize, ToSchema)]
pub(crate) struct StreamResponse {
    pub token: Token,
    pub generated_text: Option<String>,
    pub details: Option<StreamDetails>,
}

#[derive(Serialize, ToSchema)]
pub(crate) enum ErrorType {
    #[schema(example = "Request failed during generation")]
    GenerationError(String),
    #[schema(example = "Model is overloaded")]
    Overloaded(String),
    #[schema(example = "Input validation error")]
    ValidationError(String),
    #[schema(example = "Incomplete generation")]
    IncompleteGeneration(String),
}

#[derive(Serialize, ToSchema)]
pub(crate) struct ErrorResponse {
    pub error: ErrorType,
}
