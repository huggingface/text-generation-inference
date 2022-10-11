use std::net::SocketAddr;
use bloom_inference_client::ShardedClient;
use std::time::Duration;
use tokenizers::Tokenizer;

mod server;
mod validation;

use validation::Validation;

mod db;

use db::Db;

mod batcher;

use batcher::Batcher;

fn main() -> Result<(), std::io::Error> {
    let tokenizer = Tokenizer::from_pretrained("bigscience/bloom", None).unwrap();

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            tracing_subscriber::fmt::init();

            let sharded_client = ShardedClient::connect_uds(
                "/tmp/bloom-inference-0".to_string(),
                Duration::from_secs(5),
            )
            .await;
            sharded_client
                .clear_cache()
                .await
                .expect("Unable to clear cache");
            tracing::info!("Connected");

            let addr = SocketAddr::from(([127, 0, 0, 1], 3000));

            server::run(sharded_client, tokenizer, addr).await;
            Ok(())
        })
}
