use clap::Parser;
use text_generation_backends_vllm::{EngineArgs, VllmBackend, VllmBackendError};
use text_generation_router::{server, usage_stats};

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
    #[clap(long, env)]
    max_input_tokens: Option<usize>,
    #[clap(long, env)]
    max_total_tokens: Option<usize>,
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
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
    #[clap(long, env)]
    tokenizer_config_path: Option<String>,
    #[clap(long, env)]
    revision: Option<String>,
    #[clap(long, env, value_enum)]
    trust_remote_code: bool,
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
    #[clap(long, env, default_value_t = false)]
    disable_grammar_support: bool,
    #[clap(default_value = "4", long, env)]
    max_client_batch_size: usize,
    #[clap(default_value = "on", long, env)]
    usage_stats: usage_stats::UsageStatsLevel,
    #[clap(default_value = "2000000", long, env)]
    payload_limit: usize,
}

impl Into<EngineArgs> for &Args {
    fn into(self) -> EngineArgs {
        EngineArgs {
            model: self.tokenizer_name.clone(),
            pipeline_parallel_size: 1, // TODO
            tensor_parallel_size: 1,   // TODO
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), VllmBackendError> {
    let args = Args::parse();
    let backend = VllmBackend::from_engine_args((&args).into())?;

    server::run(
        backend,
        args.max_concurrent_requests,
        args.max_best_of,
        args.max_stop_sequences,
        args.max_top_n_tokens,
        args.max_input_tokens.unwrap_or(1024), // TODO
        args.max_total_tokens.unwrap_or(2048), // TODO
        args.validation_workers,
        args.api_key,
        args.tokenizer_name,
        args.tokenizer_config_path,
        args.revision,
        args.trust_remote_code,
        args.hostname,
        args.port,
        args.cors_allow_origin,
        false,
        None,
        None,
        args.disable_grammar_support,
        args.max_batch_size.unwrap_or(16),
        args.usage_stats,
        args.payload_limit,
    )
    .await?;
    Ok(())
}
