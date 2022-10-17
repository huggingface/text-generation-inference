use crate::{Batcher, Validation};
use axum::extract::Extension;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use bloom_inference_client::ShardedClient;
use serde::Deserialize;
use std::net::SocketAddr;
use tokenizers::Tokenizer;
use tokio::time::Instant;
use tracing::instrument;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct GenerateParameters {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_do_sample")]
    pub do_sample: bool,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: u32,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_k() -> i32 {
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

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        temperature: default_temperature(),
        top_k: default_top_k(),
        top_p: default_top_p(),
        do_sample: default_do_sample(),
        max_new_tokens: default_max_new_tokens(),
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct GenerateRequest {
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
}

#[instrument(skip(state), fields(time, time_per_token))]
async fn liveness(state: Extension<ServerState>) -> Result<(), (StatusCode, String)> {
    state
        .infer
        .infer(
            1,
            GenerateRequest {
                inputs: "liveness".to_string(),
                parameters: GenerateParameters {
                    temperature: 1.0,
                    top_k: 0,
                    top_p: 1.0,
                    do_sample: false,
                    max_new_tokens: 1,
                },
            },
        )
        .await?;
    Ok(())
}

#[instrument(skip(state), fields(time, time_per_token))]
async fn generate(
    state: Extension<ServerState>,
    req: Json<GenerateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let start = Instant::now();

    let (input_length, validated_request) = state
        .validation
        .validate(GenerateRequest {
            inputs: req.inputs.clone(),
            parameters: req.parameters.clone(),
        })
        .await?;

    let generated_text = state.infer.infer(input_length, validated_request).await?;

    tracing::Span::current().record("time", format!("{:?}", start.elapsed()));
    tracing::Span::current().record(
        "time_per_token",
        format!("{:?}", start.elapsed() / req.parameters.max_new_tokens),
    );
    tracing::info!("response: {}", generated_text);

    Ok(Json(serde_json::json!({
        "generated_text": generated_text,
    })))
}

#[derive(Clone)]
struct ServerState {
    validation: Validation,
    infer: Batcher,
}

pub async fn run(client: ShardedClient, tokenizer: Tokenizer, addr: SocketAddr) {
    client.clear_cache().await.expect("Unable to clear cache");
    tracing::info!("Connected");

    let infer = Batcher::new(client);

    let validation = Validation::new(tokenizer);

    let shared_state = ServerState { validation, infer };

    let app = Router::new()
        .route("/generate", post(generate))
        .layer(Extension(shared_state.clone()))
        .route("/health", get(liveness))
        .layer(Extension(shared_state.clone()));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
