use bloom_inference_client::ShardedClient;
use std::net::SocketAddr;
use text_generation_router::server;
use tokenizers::Tokenizer;

fn main() -> Result<(), std::io::Error> {
    let tokenizer = Tokenizer::from_pretrained("bigscience/bloom", None).unwrap();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            tracing_subscriber::fmt::init();

            let sharded_client = ShardedClient::connect_uds("/tmp/bloom-inference-0".to_string())
                .await
                .expect("Could not connect to server");
            sharded_client
                .clear_cache()
                .await
                .expect("Unable to clear cache");
            tracing::info!("Connected");

            let addr = SocketAddr::from(([0, 0, 0, 0], 3000));

            server::run(sharded_client, tokenizer, addr).await;
            Ok(())
        })
}
