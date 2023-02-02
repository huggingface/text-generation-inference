/// Text Generation Inference Webserver

mod infer;
mod queue;
pub mod server;
mod validation;

use infer::Infer;
use queue::{Entry, Queue};
use serde::{Deserialize, Serialize};
use validation::Validation;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct GenerateParameters {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_do_sample")]
    pub do_sample: bool,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: u32,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub details: bool,
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_temperature() -> f32 {
    1.0
}
fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_top_k() -> i32 {
    0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_do_sample() -> bool {
    false
}

fn default_max_new_tokens() -> u32 {
    20
}

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        temperature: default_temperature(),
        repetition_penalty: default_repetition_penalty(),
        top_k: default_top_k(),
        top_p: default_top_p(),
        do_sample: default_do_sample(),
        max_new_tokens: default_max_new_tokens(),
        stop: vec![],
        details: false,
        seed: None,
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct GenerateRequest {
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
}

#[derive(Debug, Serialize)]
pub struct Token(u32, String, f32);

#[derive(Serialize)]
pub(crate) struct Details {
    pub finish_reason: String,
    pub generated_tokens: u32,
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill: Option<Vec<Token>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<Token>>,
}

#[derive(Serialize)]
pub(crate) struct GenerateResponse {
    pub generated_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Details>,
}

#[derive(Serialize)]
pub(crate) struct StreamResponse {
    pub token: Token,
    pub generated_text: Option<String>,
    pub details: Option<Details>,
}

#[derive(Serialize)]
pub(crate) struct ErrorResponse {
    pub error: String,
}
