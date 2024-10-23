use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;
use text_generation_backends_trtllm::errors::TensorRtLlmBackendError;
use text_generation_backends_trtllm::TensorRtLlmBackend;
use text_generation_router::{server, usage_stats};
use tokenizers::{FromPretrainedParameters, Tokenizer};

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
    #[clap(default_value = "4096", long, env)]
    max_batch_prefill_tokens: u32,
    #[clap(long, env)]
    max_batch_total_tokens: Option<u32>,
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(long, env, required = true)]
    tokenizer_name: String,
    #[clap(long, env)]
    tokenizer_config_path: Option<String>,
    #[clap(long, env)]
    revision: Option<String>,
    #[clap(long, env)]
    model_id: String,
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,
    #[clap(long, env)]
    json_output: bool,
    #[clap(long, env)]
    otlp_endpoint: Option<String>,
    #[clap(default_value = "text-generation-inference.router", long, env)]
    otlp_service_name: String,
    #[clap(long, env)]
    cors_allow_origin: Option<Vec<String>>,
    #[clap(default_value = "4", long, env)]
    max_client_batch_size: usize,
    #[clap(long, env)]
    auth_token: Option<String>,
    #[clap(long, env, help = "Path to the TensorRT-LLM Orchestrator worker")]
    executor_worker: PathBuf,
    #[clap(default_value = "on", long, env)]
    usage_stats: usage_stats::UsageStatsLevel,
}

#[tokio::main]
async fn main() -> Result<(), TensorRtLlmBackendError> {
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
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        hostname,
        port,
        tokenizer_name,
        tokenizer_config_path,
        revision,
        model_id,
        validation_workers,
        json_output,
        otlp_endpoint,
        otlp_service_name,
        cors_allow_origin,
        max_client_batch_size,
        auth_token,
        executor_worker,
        usage_stats,
    } = args;

    // Launch Tokio runtime
    text_generation_router::logging::init_logging(otlp_endpoint, otlp_service_name, json_output);

    // Validate args
    if max_input_tokens >= max_total_tokens {
        return Err(TensorRtLlmBackendError::ArgumentValidation(
            "`max_input_tokens` must be < `max_total_tokens`".to_string(),
        ));
    }
    if max_input_tokens as u32 > max_batch_prefill_tokens {
        return Err(TensorRtLlmBackendError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be >= `max_input_tokens`. Given: {max_batch_prefill_tokens} and {max_input_tokens}")));
    }

    if validation_workers == 0 {
        return Err(TensorRtLlmBackendError::ArgumentValidation(
            "`validation_workers` must be > 0".to_string(),
        ));
    }

    if let Some(ref max_batch_total_tokens) = max_batch_total_tokens {
        if max_batch_prefill_tokens > *max_batch_total_tokens {
            return Err(TensorRtLlmBackendError::ArgumentValidation(format!("`max_batch_prefill_tokens` must be <= `max_batch_total_tokens`. Given: {max_batch_prefill_tokens} and {max_batch_total_tokens}")));
        }
        if max_total_tokens as u32 > *max_batch_total_tokens {
            return Err(TensorRtLlmBackendError::ArgumentValidation(format!("`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {max_total_tokens} and {max_batch_total_tokens}")));
        }
    }

    if !executor_worker.exists() {
        return Err(TensorRtLlmBackendError::ArgumentValidation(format!(
            "`executor_work` specified path doesn't exists: {}",
            executor_worker.display()
        )));
    }

    // Run server
    let tokenizer = Tokenizer::from_pretrained(
        tokenizer_name.clone(),
        Some(FromPretrainedParameters {
            revision: revision.clone().unwrap_or(String::from("main")),
            user_agent: HashMap::new(),
            auth_token,
        }),
    )
    .map_err(|e| TensorRtLlmBackendError::Tokenizer(e.to_string()))?;

    let backend = TensorRtLlmBackend::new(tokenizer, model_id, executor_worker)?;
    server::run(
        backend,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        max_input_tokens,
        max_total_tokens,
        validation_workers,
        None,
        tokenizer_name,
        tokenizer_config_path,
        revision,
        hostname,
        port,
        cors_allow_origin,
        false,
        None,
        None,
        true,
        max_client_batch_size,
        usage_stats,
    )
    .await?;
    Ok(())
}
