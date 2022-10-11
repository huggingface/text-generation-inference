use bloom_inference_client::ShardedClient;
use poem;
use poem::listener::TcpListener;
use std::time::Duration;

mod server;

mod db;
use db::Db;

mod batcher;
use batcher::Batcher;

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    tracing_subscriber::fmt::init();

    let sharded_client =
        ShardedClient::connect_uds("/tmp/bloom-inference-0".to_string(), Duration::from_secs(5))
            .await;
    sharded_client
        .clear_cache()
        .await
        .expect("Unable to clear cache");
    tracing::info!("Connected");

    let addr = "127.0.0.1:3000".to_string();
    let listener = TcpListener::bind(addr);

    server::run(sharded_client, listener).await
}
