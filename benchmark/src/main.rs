use clap::Parser;
/// Text Generation Inference benchmarking tool
use std::path::Path;
use text_generation_client::ShardedClient;
use tokenizers::Tokenizer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
    #[clap(default_value = "1", long, env)]
    batch_size: Vec<u32>,
    #[clap(default_value = "10", long, env)]
    sequence_length: u32,
    #[clap(default_value = "64", long, env)]
    decode_length: u32,
    #[clap(default_value = "10", long, env)]
    runs: usize,
    #[clap(default_value = "2", long, env)]
    warmups: usize,
    #[clap(default_value = "/tmp/text-generation-server-0", long, env)]
    master_shard_uds_path: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get args
    let args = Args::parse();
    // Pattern match configuration
    let Args {
        tokenizer_name,
        batch_size,
        sequence_length,
        decode_length,
        runs,
        warmups,
        master_shard_uds_path,
    } = args;

    init_logging();

    // Tokenizer instance
    // This will only be used to validate payloads
    tracing::info!("Loading tokenizer");
    let local_path = Path::new(&tokenizer_name);
    let tokenizer =
        if local_path.exists() && local_path.is_dir() && local_path.join("tokenizer.json").exists()
        {
            // Load local tokenizer
            tracing::info!("Found local tokenizer");
            Tokenizer::from_file(local_path.join("tokenizer.json")).unwrap()
        } else {
            // Download and instantiate tokenizer
            // We need to download it outside of the Tokio runtime
            tracing::info!("Downloading tokenizer");
            Tokenizer::from_pretrained(tokenizer_name.clone(), None).unwrap()
        };
    tracing::info!("Tokenizer loaded");

    // Launch Tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            // Instantiate sharded client from the master unix socket
            tracing::info!("Connect to model server");
            let mut sharded_client = ShardedClient::connect_uds(master_shard_uds_path)
                .await
                .expect("Could not connect to server");
            // Clear the cache; useful if the webserver rebooted
            sharded_client
                .clear_cache(None)
                .await
                .expect("Unable to clear cache");
            tracing::info!("Connected");

            text_generation_benchmark::run(
                tokenizer_name,
                tokenizer,
                batch_size,
                sequence_length,
                decode_length,
                runs,
                warmups,
                sharded_client,
            )
            .await
            .unwrap();
        });
    Ok(())
}

/// Init logging using LOG_LEVEL
fn init_logging() {
    // STDOUT/STDERR layer
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true);

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();
}
