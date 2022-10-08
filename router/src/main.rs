use tokio::time::Instant;

use poem;
use poem::middleware::AddData;
use poem::web::Data;
use poem::{handler, listener::TcpListener, post, web::Json, EndpointExt, Result, Route, Server};

use bloom_inference_client::ShardedClient;
use serde::Deserialize;
use std::time::Duration;
use poem::http::StatusCode;
use tracing::instrument;

mod db;

use db::Db;

mod infer;

use infer::Infer;

#[derive(Clone, Debug, Deserialize)]
struct GenerateParameters {
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: u32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_do_sample")]
    do_sample: bool,
    #[serde(default = "default_max_new_tokens")]
    max_new_tokens: u32,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_k() -> u32 {
    0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_do_sample() -> bool {
    false
}

fn default_max_new_tokens() -> u32 {
    20
}

#[derive(Clone, Debug, Deserialize)]
struct GenerateRequest {
    inputs: String,
    #[serde(default = "default_parameters")]
    parameters: GenerateParameters,
}

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        temperature: default_temperature(),
        top_k: default_top_k(),
        top_p: default_top_p(),
        do_sample: default_do_sample(),
        max_new_tokens: default_max_new_tokens(),
    }
}

#[handler]
#[instrument(skip(infer), fields(time, time_per_token))]
async fn generate(
    infer: Data<&Infer>,
    req: Json<GenerateRequest>,
) -> Result<Json<serde_json::Value>> {
    let start = Instant::now();

    let output = infer
        .infer(GenerateRequest {
            inputs: req.inputs.clone(),
            parameters: req.parameters.clone(),
        })
        .await;

    match output {
        Ok(generated_text) => {
            tracing::Span::current().record("time", format!("{:?}", start.elapsed()));
            tracing::Span::current().record("time_per_token", format!("{:?}", start.elapsed() / req.parameters.max_new_tokens));
            tracing::info!("response: {}", generated_text);

            Ok(Json(serde_json::json!({
                "generated_text": generated_text,
            })))
        }
        Err(_) => {
            Err(poem::Error::from_status(StatusCode::INTERNAL_SERVER_ERROR))
        }
    }
}

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

    let infer = Infer::new(sharded_client);

    let app = Route::new()
        .at("/generate", post(generate))
        .with(AddData::new(infer));
    Server::new(TcpListener::bind("127.0.0.1:3000"))
        .run(app)
        .await
}
