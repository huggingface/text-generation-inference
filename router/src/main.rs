use bloom_inference_client::ShardedClient;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use text_generation_router::server;
use tokenizers::Tokenizer;
use clap::Parser;

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "32", long, short, env)]
    max_batch_size: usize,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(default_value = "/tmp/bloom-inference-0", long, env)]
    shard_uds_path: String,
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
}

fn main() -> Result<(), std::io::Error> {
    // Get args
    let args = Args::parse();
// Pattern match configuration
    let Args {
        max_batch_size,
        port,
        shard_uds_path,
        tokenizer_name,
    } = args;


    let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None).unwrap();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            tracing_subscriber::fmt::init();

            let sharded_client = ShardedClient::connect_uds(shard_uds_path)
                .await
                .expect("Could not connect to server");
            sharded_client
                .clear_cache()
                .await
                .expect("Unable to clear cache");
            tracing::info!("Connected");

            let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port);

            server::run(max_batch_size, sharded_client, tokenizer, addr).await;
            Ok(())
        })
}
