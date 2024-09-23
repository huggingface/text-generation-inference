use clap::{Parser, Subcommand};
use text_generation_router::{server, usage_stats};
use text_generation_router_v3::{connect_backend, V3Error};
use thiserror::Error;

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

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
    api_key: Option<String>,
    #[clap(long, env)]
    json_output: bool,
    #[clap(long, env)]
    otlp_endpoint: Option<String>,
    #[clap(default_value = "text-generation-inference.router", long, env)]
    otlp_service_name: String,
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
    #[clap(default_value = "on", long, env)]
    usage_stats: usage_stats::UsageStatsLevel,
}

#[derive(Debug, Subcommand)]
enum Commands {
    PrintSchema,
}

#[tokio::main]
async fn main() -> Result<(), RouterError> {
    // Get args
    let args = Args::parse();
    // Pattern match configuration
    let Args {
        command,
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
        api_key,
        json_output,
        otlp_endpoint,
        otlp_service_name,
        cors_allow_origin,
        ngrok,
        ngrok_authtoken,
        ngrok_edge,
        messages_api_enabled,
        disable_grammar_support,
        max_client_batch_size,
        usage_stats,
    } = args;

    if let Some(Commands::PrintSchema) = command {
        use utoipa::OpenApi;
        let api_doc = text_generation_router::server::ApiDoc::openapi();
        let api_doc = serde_json::to_string_pretty(&api_doc).unwrap();
        println!("{}", api_doc);
        std::process::exit(0);
    };
    text_generation_router::logging::init_logging(otlp_endpoint, otlp_service_name, json_output);

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

    if let Some(max_batch_size) = max_batch_size {
        if max_batch_size == 0 {
            return Err(RouterError::ArgumentValidation(
                "`max_batch_size` must be > 0".to_string(),
            ));
        }
    }

    let (backend, _backend_info) = connect_backend(
        max_input_tokens,
        max_total_tokens,
        master_shard_uds_path,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        max_waiting_tokens,
        max_batch_size,
    )
    .await?;

    // Run server
    server::run(
        backend,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        max_input_tokens,
        max_total_tokens,
        validation_workers,
        api_key,
        tokenizer_name,
        tokenizer_config_path,
        revision,
        hostname,
        port,
        cors_allow_origin,
        ngrok,
        ngrok_authtoken,
        ngrok_edge,
        messages_api_enabled,
        disable_grammar_support,
        max_client_batch_size,
        usage_stats,
    )
    .await?;
    Ok(())
}

#[derive(Debug, Error)]
enum RouterError {
    #[error("Argument validation error: {0}")]
    ArgumentValidation(String),
    #[error("Backend failed: {0}")]
    Backend(#[from] V3Error),
    #[error("WebServer error: {0}")]
    WebServer(#[from] server::WebServerError),
    #[error("Tokio runtime failed to start: {0}")]
    Tokio(#[from] std::io::Error),
}
