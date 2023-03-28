/// Text Generation Inference benchmarking tool
use std::path::Path;
use clap::Parser;
use tokenizers::Tokenizer;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use text_generation_client::ShardedClient;

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
    #[clap(default_value = "1", long, env)]
    batch_size: Vec<u32>,
    #[clap(default_value = "128", long, env)]
    sequence_length: u32,
    #[clap(default_value = "100", long, env)]
    decode_length: u32,
    #[clap(default_value = "2", long, env)]
    runs: usize,
    #[clap(default_value = "/tmp/text-generation-0", long, env)]
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
        master_shard_uds_path,
    } = args;

    // Tokenizer instance
    // This will only be used to validate payloads
    let local_path = Path::new(&tokenizer_name);
    let tokenizer =
        if local_path.exists() && local_path.is_dir() && local_path.join("tokenizer.json").exists()
        {
            // Load local tokenizer
            Tokenizer::from_file(local_path.join("tokenizer.json")).expect("unable to load local tokenizer")
        } else {
            // Download and instantiate tokenizer
            // We need to download it outside of the Tokio runtime
            Tokenizer::from_pretrained(tokenizer_name.clone(), None).expect("unable to load hub tokenizer")
        };

    // Launch Tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            init_logging();

            // Instantiate sharded client from the master unix socket
            tracing::info!("Connect to model server");
            let mut sharded_client = ShardedClient::connect_uds(master_shard_uds_path)
                .await
                .expect("Could not connect to server");
            // Clear the cache; useful if the webserver rebooted
            sharded_client
                .clear_cache()
                .await
                .expect("Unable to clear cache");
            tracing::info!("Connected");

            text_generation_benchmark::run(
                tokenizer,
                batch_size,
                sequence_length,
                decode_length,
                runs,
                sharded_client,
            ).await.unwrap();
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
