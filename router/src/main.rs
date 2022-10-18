/// Text Generation Inference webserver entrypoint
use bloom_inference_client::ShardedClient;
use clap::Parser;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;
use text_generation_router::server;
use tokenizers::Tokenizer;

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "1000", long, env)]
    max_input_length: usize,
    #[clap(default_value = "32", long, env)]
    max_batch_size: usize,
    #[clap(default_value = "5", long, env)]
    max_waiting_time: u64,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(default_value = "/tmp/bloom-inference-0", long, env)]
    master_shard_uds_path: String,
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,
}

fn main() -> Result<(), std::io::Error> {
    // Get args
    let args = Args::parse();
    // Pattern match configuration
    let Args {
        max_concurrent_requests,
        max_input_length,
        max_batch_size,
        max_waiting_time,
        port,
        master_shard_uds_path,
        tokenizer_name,
        validation_workers,
    } = args;

    if validation_workers == 1 {
        panic!("validation_workers must be > 0");
    }

    let max_waiting_time = Duration::from_secs(max_waiting_time);

    // Download and instantiate tokenizer
    // This will only be used to validate payloads
    //
    // We need to download it outside of the Tokio runtime
    let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None).unwrap();

    // Launch Tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            tracing_subscriber::fmt::init();

            // Instantiate sharded client from the master unix socket
            let sharded_client = ShardedClient::connect_uds(master_shard_uds_path)
                .await
                .expect("Could not connect to server");
            // Clear the cache; useful if the webserver rebooted
            sharded_client
                .clear_cache()
                .await
                .expect("Unable to clear cache");
            tracing::info!("Connected");

            // Binds on localhost
            let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port);

            // Run server
            server::run(
                max_concurrent_requests,
                max_input_length,
                max_batch_size,
                max_waiting_time,
                sharded_client,
                tokenizer,
                validation_workers,
                addr,
            )
            .await;
            Ok(())
        })
}
