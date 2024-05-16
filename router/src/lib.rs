pub mod config;
mod health;
/// Text Generation Inference Webserver
mod infer;
mod queue;
pub mod server;
mod validation;

use axum::http::HeaderValue;
use config::Config;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};
use infer::{Infer, InferError, InferStreamResponse};
use opentelemetry::sdk::propagation::TraceContextPropagator;
use opentelemetry::sdk::trace;
use opentelemetry::sdk::trace::Sampler;
use opentelemetry::sdk::Resource;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use queue::{Entry, Queue};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use text_generation_client::{ClientError, ShardedClient};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::OwnedSemaphorePermit;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::AllowOrigin;
use tracing::warn;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer;
use utoipa::ToSchema;
use validation::Validation;

/// Init logging using env variables LOG_LEVEL and LOG_FORMAT:
///     - otlp_endpoint is an optional URL to an Open Telemetry collector
///     - LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
///     - LOG_FORMAT may be TEXT or JSON (default to TEXT)
///     - LOG_COLORIZE may be "false" or "true" (default to "true" or ansi supported platforms)
fn init_logging(otlp_endpoint: Option<String>, json_output: bool) {
    let mut layers = Vec::new();

    // STDOUT/STDERR layer
    let ansi = std::env::var("LOG_COLORIZE") != Ok("1".to_string());
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_ansi(ansi)
        .with_line_number(true);

    let fmt_layer = match json_output {
        true => fmt_layer.json().flatten_event(true).boxed(),
        false => fmt_layer.boxed(),
    };
    layers.push(fmt_layer);

    // OpenTelemetry tracing layer
    if let Some(otlp_endpoint) = otlp_endpoint {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint),
            )
            .with_trace_config(
                trace::config()
                    .with_resource(Resource::new(vec![KeyValue::new(
                        "service.name",
                        "text-generation-inference.router",
                    )]))
                    .with_sampler(Sampler::AlwaysOn),
            )
            .install_batch(opentelemetry::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            init_tracing_opentelemetry::init_propagator().unwrap();
        };
    }

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();
}

/// get model info from the Huggingface Hub
pub async fn get_model_info(api: &ApiRepo) -> Option<HubModelInfo> {
    let response = api.info_request().send().await.ok()?;

    if response.status().is_success() {
        let hub_model_info: HubModelInfo =
            serde_json::from_str(&response.text().await.ok()?).ok()?;
        if let Some(sha) = &hub_model_info.sha {
            tracing::info!(
                "Serving revision {sha} of model {}",
                hub_model_info.model_id
            );
        }
        Some(hub_model_info)
    } else {
        None
    }
}

/// get base tokenizer
pub async fn get_base_tokenizer(api: &Api, api_repo: &ApiRepo) -> Option<PathBuf> {
    let config_filename = api_repo.get("config.json").await.ok()?;

    // Open the file in read-only mode with buffer.
    let file = File::open(config_filename).ok()?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let config: serde_json::Value = serde_json::from_reader(reader).ok()?;

    if let Some(serde_json::Value::String(base_model_id)) = config.get("base_model_name_or_path") {
        let api_base_repo = api.repo(Repo::with_revision(
            base_model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        api_base_repo.get("tokenizer.json").await.ok()
    } else {
        None
    }
}

/// get tokenizer_config from the Huggingface Hub
pub async fn get_tokenizer_config(api_repo: &ApiRepo) -> Option<HubTokenizerConfig> {
    let tokenizer_config_filename = api_repo.get("tokenizer_config.json").await.ok()?;

    // Open the file in read-only mode with buffer.
    let file = File::open(tokenizer_config_filename).ok()?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of 'HubTokenizerConfig'.
    let tokenizer_config: HubTokenizerConfig = serde_json::from_reader(reader)
        .map_err(|e| {
            tracing::warn!("Unable to parse tokenizer config: {}", e);
            e
        })
        .ok()?;

    Some(tokenizer_config)
}

#[derive(Debug, Error)]
pub enum RouterError {
    #[error("Argument validation error: {0}")]
    ArgumentValidation(String),
    #[error("Unable to connect to the Python model shards: {0}")]
    Connection(ClientError),
    #[error("Unable to clear the Python model shards cache: {0}")]
    Cache(ClientError),
    #[error("Unable to get the Python model shards info: {0}")]
    Info(ClientError),
    #[error("Unable to warmup the Python model shards: {0}")]
    Warmup(ClientError),
    #[error("Tokio runtime failed to start: {0}")]
    Tokio(#[from] std::io::Error),
    #[error("Axum webserver failed: {0}")]
    Axum(#[from] axum::BoxError),
}

#[allow(clippy::too_many_arguments)]
pub async fn internal_main(
    max_concurrent_requests: usize,
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_tokens: usize,
    max_total_tokens: usize,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: Option<u32>,
    max_waiting_tokens: usize,
    max_batch_size: Option<usize>,
    hostname: String,
    port: u16,
    master_shard_uds_path: String,
    tokenizer_name: String,
    tokenizer_config_path: Option<String>,
    revision: Option<String>,
    validation_workers: usize,
    json_output: bool,
    otlp_endpoint: Option<String>,
    cors_allow_origin: Option<Vec<String>>,
    ngrok: bool,
    ngrok_authtoken: Option<String>,
    ngrok_edge: Option<String>,
    messages_api_enabled: bool,
    disable_grammar_support: bool,
    max_client_batch_size: usize,
) -> Result<(), RouterError> {
    // Launch Tokio runtime
    if otlp_endpoint.is_some() {
        // Initialize if OpenTelemetry is enabled
        init_logging(otlp_endpoint, json_output);
    }

    // Validate args
    if max_input_tokens >= max_total_tokens {
        return Err(RouterError::ArgumentValidation(
            "`max_input_tokens` must be < `max_total_tokens`".to_string(),
        ));
    }
    if max_input_tokens as u32 > max_batch_prefill_tokens {
        return Err(RouterError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be >= `max_input_tokens`. Given: {max_batch_prefill_tokens} and {max_input_tokens}")));
    }

    if validation_workers == 0 {
        return Err(RouterError::ArgumentValidation(
            "`validation_workers` must be > 0".to_string(),
        ));
    }

    if let Some(ref max_batch_total_tokens) = max_batch_total_tokens {
        if max_batch_prefill_tokens > *max_batch_total_tokens {
            return Err(RouterError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be <= `max_batch_total_tokens`. Given: {max_batch_prefill_tokens} and {max_batch_total_tokens}")));
        }
        if max_total_tokens as u32 > *max_batch_total_tokens {
            return Err(RouterError::ArgumentValidation(format!("`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {max_total_tokens} and {max_batch_total_tokens}")));
        }
    }

    // CORS allowed origins
    // map to go inside the option and then map to parse from String to HeaderValue
    // Finally, convert to AllowOrigin
    let cors_allow_origin: Option<AllowOrigin> = cors_allow_origin.map(|cors_allow_origin| {
        AllowOrigin::list(
            cors_allow_origin
                .iter()
                .map(|origin| origin.parse::<HeaderValue>().unwrap()),
        )
    });

    // Parse Huggingface hub token
    let authorization_token = std::env::var("HUGGING_FACE_HUB_TOKEN").ok();

    // Tokenizer instance
    // This will only be used to validate payloads
    let local_path = Path::new(&tokenizer_name);

    // Shared API builder initialization
    let api_builder = || {
        let mut builder = ApiBuilder::new()
            .with_progress(false)
            .with_token(authorization_token);

        if let Ok(cache_dir) = std::env::var("HUGGINGFACE_HUB_CACHE") {
            builder = builder.with_cache_dir(cache_dir.into());
        }

        builder
    };

    // Decide if we need to use the API based on the revision and local path
    let use_api = revision.is_some() || !local_path.exists() || !local_path.is_dir();

    // Initialize API if needed
    #[derive(Clone)]
    enum Type {
        Api(Api),
        Cache(Cache),
        None,
    }
    let api = if use_api {
        if std::env::var("HF_HUB_OFFLINE") == Ok("1".to_string()) {
            let cache = Cache::default();
            tracing::warn!("Offline mode active using cache defaults");
            Type::Cache(cache)
        } else {
            tracing::info!("Using the Hugging Face API");
            match api_builder().build() {
                Ok(api) => Type::Api(api),
                Err(_) => {
                    tracing::warn!("Unable to build the Hugging Face API");
                    Type::None
                }
            }
        }
    } else {
        Type::None
    };

    // Load tokenizer and model info
    let (tokenizer_filename, config_filename, tokenizer_config_filename, model_info) = match api {
        Type::None => (
            Some(local_path.join("tokenizer.json")),
            Some(local_path.join("config.json")),
            Some(local_path.join("tokenizer_config.json")),
            None,
        ),
        Type::Api(api) => {
            let api_repo = api.repo(Repo::with_revision(
                tokenizer_name.to_string(),
                RepoType::Model,
                revision.clone().unwrap_or_else(|| "main".to_string()),
            ));

            let tokenizer_filename = match api_repo.get("tokenizer.json").await {
                Ok(tokenizer_filename) => Some(tokenizer_filename),
                Err(_) => get_base_tokenizer(&api, &api_repo).await,
            };
            let config_filename = api_repo.get("config.json").await.ok();
            let tokenizer_config_filename = api_repo.get("tokenizer_config.json").await.ok();

            let model_info = if let Some(model_info) = get_model_info(&api_repo).await {
                Some(model_info)
            } else {
                tracing::warn!("Could not retrieve model info from the Hugging Face hub.");
                None
            };
            (
                tokenizer_filename,
                config_filename,
                tokenizer_config_filename,
                model_info,
            )
        }
        Type::Cache(cache) => {
            let repo = cache.repo(Repo::with_revision(
                tokenizer_name.to_string(),
                RepoType::Model,
                revision.clone().unwrap_or_else(|| "main".to_string()),
            ));
            (
                repo.get("tokenizer.json"),
                repo.get("config.json"),
                repo.get("tokenizer_config.json"),
                None,
            )
        }
    };
    let tokenizer: Option<Tokenizer> =
        tokenizer_filename.and_then(|filename| Tokenizer::from_file(filename).ok());
    let config: Option<Config> = config_filename.and_then(|filename| {
        std::fs::read_to_string(filename)
            .ok()
            .as_ref()
            .and_then(|c| {
                let config: Result<Config, _> = serde_json::from_str(c);
                if let Err(err) = &config {
                    tracing::warn!("Could not parse config {err:?}");
                }
                config.ok()
            })
    });
    let model_info = model_info.unwrap_or_else(|| HubModelInfo {
        model_id: tokenizer_name.to_string(),
        sha: None,
        pipeline_tag: None,
    });

    // Read the JSON contents of the file as an instance of 'HubTokenizerConfig'.
    let tokenizer_config: Option<HubTokenizerConfig> = if let Some(filename) = tokenizer_config_path
    {
        HubTokenizerConfig::from_file(filename)
    } else {
        tokenizer_config_filename.and_then(HubTokenizerConfig::from_file)
    };
    let tokenizer_config = tokenizer_config.unwrap_or_else(|| {
        tracing::warn!("Could not find tokenizer config locally and no API specified");
        HubTokenizerConfig::default()
    });

    tracing::info!("Using config {config:?}");
    if tokenizer.is_none() {
        tracing::warn!("Could not find a fast tokenizer implementation for {tokenizer_name}");
        tracing::warn!("Rust input length validation and truncation is disabled");
    }

    // if pipeline-tag == text-generation we default to return_full_text = true
    let compat_return_full_text = match &model_info.pipeline_tag {
        None => {
            tracing::warn!("no pipeline tag found for model {tokenizer_name}");
            true
        }
        Some(pipeline_tag) => pipeline_tag.as_str() == "text-generation",
    };

    // Instantiate sharded client from the master unix socket
    let mut sharded_client = ShardedClient::connect_uds(master_shard_uds_path)
        .await
        .map_err(RouterError::Connection)?;
    // Clear the cache; useful if the webserver rebooted
    sharded_client
        .clear_cache(None)
        .await
        .map_err(RouterError::Cache)?;
    // Get info from the shard
    let shard_info = sharded_client.info().await.map_err(RouterError::Info)?;

    // Warmup model
    tracing::info!("Warming up model");
    let max_supported_batch_total_tokens = match sharded_client
        .warmup(
            max_input_tokens as u32,
            max_batch_prefill_tokens,
            max_total_tokens as u32,
            max_batch_size,
        )
        .await
        .map_err(RouterError::Warmup)?
    {
        // Older models do not support automatic max-batch-total-tokens
        None => {
            let max_batch_total_tokens = max_batch_total_tokens
                .unwrap_or(16000.max((max_total_tokens as u32).max(max_batch_prefill_tokens)));
            tracing::warn!("Model does not support automatic max batch total tokens");
            max_batch_total_tokens
        }
        // Flash attention models return their max supported total tokens
        Some(max_supported_batch_total_tokens) => {
            // Warn if user added his own max-batch-total-tokens as we will ignore it
            if max_batch_total_tokens.is_some() {
                tracing::warn!(
                    "`--max-batch-total-tokens` is deprecated for Flash \
                        Attention models."
                );
                tracing::warn!(
                    "Inferred max batch total tokens: {max_supported_batch_total_tokens}"
                );
            }
            if max_total_tokens as u32 > max_supported_batch_total_tokens {
                return Err(RouterError::ArgumentValidation(format!("`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {max_total_tokens} and {max_supported_batch_total_tokens}")));
            }

            max_supported_batch_total_tokens
        }
    };
    tracing::info!("Setting max batch total tokens to {max_supported_batch_total_tokens}");
    tracing::info!("Connected");

    // Determine the server port based on the feature and environment variable.
    let port = if cfg!(feature = "google") {
        std::env::var("AIP_HTTP_PORT")
            .map(|aip_http_port| aip_http_port.parse::<u16>().unwrap_or(port))
            .unwrap_or(port)
    } else {
        port
    };

    let addr = match hostname.parse() {
        Ok(ip) => SocketAddr::new(ip, port),
        Err(_) => {
            tracing::warn!("Invalid hostname, defaulting to 0.0.0.0");
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port)
        }
    };

    // Run server
    server::run(
        model_info,
        shard_info,
        compat_return_full_text,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        max_input_tokens,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_supported_batch_total_tokens,
        max_waiting_tokens,
        max_batch_size,
        sharded_client,
        tokenizer,
        config,
        validation_workers,
        addr,
        cors_allow_origin,
        ngrok,
        ngrok_authtoken,
        ngrok_edge,
        tokenizer_config,
        messages_api_enabled,
        disable_grammar_support,
        max_client_batch_size,
    )
    .await?;
    Ok(())
}

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

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ChatTemplate {
    name: String,
    template: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChatTemplateVersions {
    Single(String),
    Multiple(Vec<ChatTemplate>),
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct HubTokenizerConfig {
    pub chat_template: Option<ChatTemplateVersions>,
    pub completion_template: Option<String>,
    #[serde(deserialize_with = "token_serde::deserialize")]
    pub bos_token: Option<String>,
    #[serde(deserialize_with = "token_serde::deserialize")]
    pub eos_token: Option<String>,
}

impl HubTokenizerConfig {
    pub fn from_file<P: AsRef<std::path::Path>>(filename: P) -> Option<Self> {
        let content = std::fs::read_to_string(filename).ok()?;
        serde_json::from_str(&content).ok()
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct HubProcessorConfig {
    pub chat_template: Option<ChatTemplateVersions>,
    pub image_seq_len: usize,
    pub processor_class: Option<String>,
}

impl HubProcessorConfig {
    pub fn from_file<P: AsRef<std::path::Path>>(filename: P) -> Option<Self> {
        let content = std::fs::read_to_string(filename).ok()?;
        serde_json::from_str(&content).ok()
    }
}

#[derive(Clone, Debug, Deserialize, ToSchema, Serialize)]
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
            Value::Null => Ok(None),
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

mod prompt_serde {
    use serde::{self, Deserialize, Deserializer};
    use serde_json::Value;

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => Ok(vec![s]),
            Value::Array(arr) if arr.is_empty() => Err(serde::de::Error::custom(
                "Empty array detected. Do not use an empty array for the prompt.",
            )),
            Value::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    Value::String(s) => Ok(s.to_owned()),
                    _ => Err(serde::de::Error::custom("Expected a string")),
                })
                .collect(),
            _ => Err(serde::de::Error::custom(
                "Expected a string or an array of strings",
            )),
        }
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
    #[serde(deserialize_with = "prompt_serde::deserialize")]
    pub prompt: Vec<String>,

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
            object: "text_completion".into(),
            created,
            model,
            system_fingerprint,
            choices: vec![ChatCompletionComplete {
                index: 0,
                message,
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

#[derive(Clone, Serialize, ToSchema)]
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
            object: "text_completion".to_string(),
            created,
            model,
            system_fingerprint,
            choices: vec![ChatCompletionChoice {
                index: 0,
                delta,
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
        example = "\"You will be presented with a JSON schema representing a set of tools.\nIf the user request lacks of sufficient information to make a precise tool selection: Do not invent any tool's properties, instead notify with an error message.\n\nJSON Schema:\n\""
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
        "\nYou will be presented with a JSON schema representing a set of tools.\nIf the user request lacks of sufficient information to make a precise tool selection: Do not invent any tool's properties, instead notify with an error message.\n\nJSON Schema:\n".to_string(),
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

#[derive(Debug, Deserialize, Serialize, ToSchema, PartialEq)]
pub struct Tools {
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
    tools: Option<&'a str>,
    tools_prompt: Option<&'a str>,
}

#[derive(Clone, Deserialize, Serialize, ToSchema, Default, Debug, PartialEq)]
pub(crate) struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
struct Url {
    url: String,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
struct ImageUrl {
    image_url: Url,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
struct Text {
    text: String,
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum MessageChunk {
    Text(Text),
    ImageUrl(ImageUrl),
}

#[derive(Clone, Deserialize, ToSchema, Serialize, Debug, PartialEq)]
pub struct Message {
    #[schema(example = "user")]
    role: String,
    #[schema(example = "My name is David and I")]
    #[serde(deserialize_with = "message_content_serde::deserialize")]
    content: Vec<MessageChunk>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(example = "\"David\"")]
    name: Option<String>,
}

mod message_content_serde {
    use super::*;
    use serde::{Deserialize, Deserializer};

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<MessageChunk>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Message {
            Text(String),
            Chunks(Vec<MessageChunk>),
        }
        let message: Message = Deserialize::deserialize(deserializer)?;
        let chunks = match message {
            Message::Text(text) => {
                vec![MessageChunk::Text(Text { text })]
            }
            Message::Chunks(s) => s,
        };
        Ok(chunks)
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
            content: value
                .content
                .into_iter()
                .map(|c| match c {
                    MessageChunk::Text(Text { text }) => text,
                    MessageChunk::ImageUrl(image) => {
                        let url = image.image_url.url;
                        format!("![]({url})")
                    }
                })
                .collect::<Vec<_>>()
                .join(""),
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
    use serde_json::json;
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
        assert_eq!(
            config.chat_template,
            Some(ChatTemplateVersions::Single("test".to_string()))
        );
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
        assert_eq!(
            config.chat_template,
            Some(ChatTemplateVersions::Single("test".to_string()))
        );
        assert_eq!(
            config.bos_token,
            Some("<｜begin▁of▁sentence｜>".to_string())
        );
        assert_eq!(config.eos_token, Some("<｜end▁of▁sentence｜>".to_string()));
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
                content: vec![MessageChunk::Text(Text {
                    text: "What is Deep Learning?".to_string()
                }),],
                name: None
            }
        );
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
                content: vec![
                    MessageChunk::Text(Text { text: "Whats in this image?".to_string() }),
                    MessageChunk::ImageUrl(ImageUrl { image_url: Url { url: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png".to_string() } })
                ],
                name: None
            }
        );
    }

    #[test]
    fn text_message_convert() {
        let message = Message{
                role: "user".to_string(),
                content: vec![
                    MessageChunk::Text(Text { text: "Whats in this image?".to_string() }),
                    MessageChunk::ImageUrl(ImageUrl { image_url: Url { url: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png".to_string() } })
                ],
                name: None
            };
        let textmsg: TextMessage = message.into();
        assert_eq!(textmsg.content, "Whats in this image?![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png)");
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
