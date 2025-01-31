mod backend;

use backend::{LlamacppNuma, LlamacppSplitMode, LlamacppConfig, LlamacppBackend, BackendError};
use clap::{Parser};
use text_generation_router::{logging, server, usage_stats};
use thiserror::Error;
use tokenizers::{Tokenizer, FromPretrainedParameters};
use tokio::sync::oneshot::error::RecvError;
use tracing::error;

/// Backend Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Name of the model to load.
    #[clap(long, env)]
    model_id: String,

    /// Revision of the model.
    #[clap(default_value = "main", long, env)]
    revision: String,

    /// Path to the GGUF model file to be used for inference.
    #[clap(long, env)]
    model_gguf: String, // TODO Option() with hf->gguf & quantize

    /// Context size for the model.
    #[clap(default_value = "4096", long, env)]
    n_ctx: usize,

    /// Number of threads to use for inference.
    #[clap(default_value = "1", long, env)]
    n_threads: usize,

    /// Number of layers to store in VRAM.
    #[clap(default_value = "0", long, env)]
    n_gpu_layers: usize,

    /// Split the model across multiple GPUs.
    #[clap(default_value = "layer", long, env)]
    split_mode: LlamacppSplitMode,

    /// Defragment the KV cache if holes/size > threshold.
    #[clap(default_value = "-1.0", long, env)]
    defrag_threshold: f32,

    /// Setup NUMA optimizations.
    #[clap(default_value = "disabled", value_enum, long, env)]
    numa: LlamacppNuma,

    /// Whether to use memory mapping.
    #[clap(default_value = "true", long, env)]
    use_mmap: bool,

    /// Whether to use memory locking.
    #[clap(default_value = "false", long, env)]
    use_mlock: bool,

    /// Enable offloading of KQV operations to the GPU.
    #[clap(default_value = "false", long, env)]
    offload_kqv: bool,

    /// Enable flash attention for faster inference. (EXPERIMENTAL)
    #[clap(default_value = "true", long, env)]
    flash_attention: bool,

    /// TODO
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "2", long, env)]
    max_best_of: usize,
    #[clap(default_value = "4", long, env)]
    max_stop_sequences: usize,
    #[clap(default_value = "5", long, env)]
    max_top_n_tokens: u32,

    /// Maximum number of input tokens allowed per request.
    #[clap(default_value = "1024", long, env)]
    max_input_tokens: usize,

    /// Maximum total tokens (input + output) allowed per request.
    #[clap(default_value = "2048", long, env)]
    max_total_tokens: usize,

//  #[clap(default_value = "1.2", long, env)]
//  waiting_served_ratio: f32,
//  #[clap(default_value = "4096", long, env)]
//  max_batch_prefill_tokens: u32,

    /// Maximum tokens within a batch
    #[clap(default_value = "4096", long, env)]
    max_batch_total_tokens: usize,

//  #[clap(default_value = "20", long, env)]
//  max_waiting_tokens: usize,

    /// Maximum number of requests per batch
    #[clap(default_value = "1", long, env)]
    max_batch_size: usize,

    /// The IP address to listen on
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,

    /// The port to listen on.
    #[clap(default_value = "3001", long, short, env)]
    port: u16,

//  #[clap(default_value = "/tmp/text-generation-server-0", long, env)]
//  master_shard_uds_path: String,
//  #[clap(long, env)]
//  tokenizer_name: String,
//  #[clap(long, env)]
//  tokenizer_config_path: Option<String>,
//  #[clap(long, env, value_enum)]
//  trust_remote_code: bool,
//  #[clap(long, env)]
//  api_key: Option<String>,

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
    #[clap(long, env)]
    tokenizer_config_path: Option<String>,
    #[clap(long, env, default_value_t = false)]
    disable_grammar_support: bool,
    #[clap(default_value = "4", long, env)]
    max_client_batch_size: usize,
    #[clap(default_value = "on", long, env)]
    usage_stats: usage_stats::UsageStatsLevel,
    #[clap(default_value = "2000000", long, env)]
    payload_limit: usize,
}

#[tokio::main]
async fn main() -> Result<(), RouterError> {
    let args = Args::parse();

    logging::init_logging(
        args.otlp_endpoint,
        args.otlp_service_name,
        args.json_output
    );

    if args.max_input_tokens >= args.max_total_tokens {
        return Err(RouterError::ArgumentValidation(
            "`max_input_tokens` must be < `max_total_tokens`".to_string(),
        ));
    }
    if args.max_total_tokens > args.max_batch_total_tokens {
        return Err(RouterError::ArgumentValidation(
            "`max_total_tokens` must be <= `max_batch_total_tokens`".to_string(),
        ));
    }
    if args.max_batch_size * args.max_total_tokens > args.max_batch_total_tokens {
        return Err(RouterError::ArgumentValidation(
            "`max_batch_size` * `max_total_tokens` must be <= `max_batch_total_tokens`".to_string(),
        ));
    }
    if args.max_batch_total_tokens > args.n_ctx {
        return Err(RouterError::ArgumentValidation(
            "`max_batch_total_tokens` must be <= `n_ctx`".to_string(),
        ));
    }

    // TODO: check if we use the same cache of Server
    // check if llamacpp is faster
    let tokenizer = {
        let token = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();
        let params = FromPretrainedParameters {
            revision: args.revision.clone(),
            token: token,
            ..Default::default()
        };
        Tokenizer::from_pretrained(
            args.model_id.clone(),
            Some(params)
        )?
    };

    let (backend, ok) = LlamacppBackend::new(
        LlamacppConfig {
            model_gguf:             args.model_gguf,
            n_ctx:                  args.n_ctx,
            n_threads:              args.n_threads,
            n_gpu_layers:           args.n_gpu_layers,
            split_mode:             args.split_mode,
            defrag_threshold:       args.defrag_threshold,
            numa:                   args.numa,
            use_mmap:               args.use_mmap,
            use_mlock:              args.use_mlock,
            flash_attention:        args.flash_attention,
            offload_kqv:            args.offload_kqv,
            max_batch_total_tokens: args.max_batch_total_tokens,
            max_batch_size:         args.max_batch_size,
            batch_timeout:          tokio::time::Duration::from_millis(5),
        },
        tokenizer,
    );
    ok.await??;

    server::run(
        backend,
        args.max_concurrent_requests,
        args.max_best_of,
        args.max_stop_sequences,
        args.max_top_n_tokens,
        args.max_input_tokens,
        args.max_total_tokens,
        args.validation_workers,
        None, // api_key
        args.model_id, // tokenizer_name
        args.tokenizer_config_path,
        Some(args.revision),
        false, // trust_remote_code
        args.hostname,
        args.port,
        args.cors_allow_origin,
        args.ngrok,
        args.ngrok_authtoken,
        args.ngrok_edge,
        args.disable_grammar_support,
        args.max_client_batch_size,
        args.usage_stats,
        args.payload_limit,
    )
    .await?;
    Ok(())
}

#[derive(Debug, Error)]
enum RouterError {
    #[error("Argument validation error: {0}")]
    ArgumentValidation(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
    #[error("WebServer error: {0}")]
    WebServer(#[from] server::WebServerError),
    #[error("Recv error: {0}")]
    RecvError(#[from] RecvError),
}
