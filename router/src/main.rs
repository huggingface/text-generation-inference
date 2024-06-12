/// Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

use axum::http::HeaderValue;
use clap::Parser;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};
use opentelemetry::sdk::propagation::TraceContextPropagator;
use opentelemetry::sdk::trace;
use opentelemetry::sdk::trace::Sampler;
use opentelemetry::sdk::Resource;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use text_generation_client::{ClientError, ShardedClient};
use text_generation_router::config::Config;
use text_generation_router::{server, HubModelInfo, HubTokenizerConfig};
use thiserror::Error;
use tokenizers::Tokenizer;
use tower_http::cors::AllowOrigin;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "2", long, env)]
    max_best_of: usize,
    #[clap(default_value = "4", long, env)]
    max_stop_sequences: usize,
    #[clap(default_value = "5", long, env)]
    max_top_n_tokens: u32,
    #[clap(default_value = "1024", long, env)]
    max_input_tokens: usize,
    #[clap(default_value = "2048", long, env)]
    max_total_tokens: usize,
    #[clap(default_value = "1.2", long, env)]
    waiting_served_ratio: f32,
    #[clap(default_value = "4096", long, env)]
    max_batch_prefill_tokens: u32,
    #[clap(long, env)]
    max_batch_total_tokens: Option<u32>,
    #[clap(default_value = "20", long, env)]
    max_waiting_tokens: usize,
    #[clap(long, env)]
    max_batch_size: Option<usize>,
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(default_value = "/tmp/text-generation-server-0", long, env)]
    master_shard_uds_path: String,
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
    #[clap(long, env)]
    tokenizer_config_path: Option<String>,
    #[clap(long, env)]
    revision: Option<String>,
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,
    #[clap(long, env)]
    json_output: bool,
    #[clap(long, env)]
    otlp_endpoint: Option<String>,
    #[clap(long, env)]
    cors_allow_origin: Option<Vec<String>>,
    #[clap(long, env)]
    ngrok: bool,
    #[clap(long, env)]
    ngrok_authtoken: Option<String>,
    #[clap(long, env)]
    ngrok_edge: Option<String>,
    #[clap(long, env, default_value_t = false)]
    messages_api_enabled: bool,
    #[clap(long, env, default_value_t = false)]
    disable_grammar_support: bool,
    #[clap(default_value = "4", long, env)]
    max_client_batch_size: usize,
}

#[tokio::main]
async fn main() -> Result<(), RouterError> {
    // Get args
    let args = Args::parse();
    // Pattern match configuration
    let Args {
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        max_input_tokens,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        max_waiting_tokens,
        max_batch_size,
        hostname,
        port,
        master_shard_uds_path,
        tokenizer_name,
        tokenizer_config_path,
        revision,
        validation_workers,
        json_output,
        otlp_endpoint,
        cors_allow_origin,
        ngrok,
        ngrok_authtoken,
        ngrok_edge,
        messages_api_enabled,
        disable_grammar_support,
        max_client_batch_size,
    } = args;

    // Launch Tokio runtime
    init_logging(otlp_endpoint, json_output);

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

    let (max_batch_size, max_batch_total_tokens) = match (max_batch_size, max_batch_total_tokens) {
        (Some(_max_batch_size), Some(_max_batch_total_tokens)) => {
            if (_max_batch_total_tokens as usize / max_total_tokens) != _max_batch_size {
                tracing::warn!("max_batch_size was set to {_max_batch_size} while max_batch_total_tokens to {_max_batch_total_tokens}");
                tracing::warn!("These values are not match, so max_batch_size will be preferred");
                (Some(_max_batch_size), Some((_max_batch_size * max_total_tokens) as u32))
            } else {
                (Some(_max_batch_size), Some(_max_batch_total_tokens))
            }
        },
        (Some(_max_batch_size), None) => (
            Some(_max_batch_size), Some((_max_batch_size * max_total_tokens) as u32)
        ),
        (None, Some(_max_batch_total_tokens)) => (
            Some(_max_batch_total_tokens as usize / max_total_tokens), Some(_max_batch_total_tokens)
        ),
        (None, None) => (None, None),
    };

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
    let skip_tokenizer_in_tgi = env::var("SKIP_TOKENIZER_IN_TGI")
        .ok()
        .map_or(false, |value| value.to_lowercase() == "true");
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
    let tokenizer: Option<Tokenizer> = if skip_tokenizer_in_tgi {
        None
    } else {
        tokenizer_filename.and_then(|filename| Tokenizer::from_file(filename).ok())
    };
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
enum RouterError {
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
