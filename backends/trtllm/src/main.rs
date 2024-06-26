use clap::Parser;
use thiserror::Error;
use text_generation_router::server;
use trtllm::TensorRTError;

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
}

#[derive(Debug, Error)]
enum RouterError {
    #[error("Argument validation error: {0}")]
    ArgumentValidation(String),
    #[error("Backend failed: {0}")]
    Backend(#[from] TensorRTError),
    #[error("WebServer error: {0}")]
    WebServer(#[from] server::WebServerError),
    #[error("Tokio runtime failed to start: {0}")]
    Tokio(#[from] std::io::Error),
}


#[tokio::main]
async fn main() -> Result<(), RouterError> {
    // Get args
    let args = Args::parse();

    Ok(())
}