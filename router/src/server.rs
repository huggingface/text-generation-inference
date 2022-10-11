use std::net::SocketAddr;
use axum::{Router, Json};
use axum::http::StatusCode;
use axum::extract::Extension;
use axum::routing::post;
use crate::{Batcher, ShardedClient, Validation};
use serde::Deserialize;
use tokenizers::Tokenizer;
use tokio::time::Instant;
use tracing::instrument;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct GenerateParameters {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: u32,
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
async fn generate(
    state: Extension<ServerState>,
    req: Json<GenerateRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let start = Instant::now();

    let (input_length, validated_request) = match state.validation
        .validate(GenerateRequest {
            inputs: req.inputs.clone(),
            parameters: req.parameters.clone(),
        })
        .await {
        Ok(result) => result,
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR)
    };

    let output = state.infer.infer(input_length, validated_request).await;

    match output {
        Ok(generated_text) => {
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
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

#[derive(Clone)]
struct ServerState {
    validation: Validation,
    infer: Batcher,
}

pub async fn run(
    client: ShardedClient,
    tokenizer: Tokenizer,
    addr: SocketAddr,
)  {
    client.clear_cache().await.expect("Unable to clear cache");
    tracing::info!("Connected");

    let infer = Batcher::new(client);

    let validation = Validation::new(tokenizer);

    let shared_state = ServerState {
        validation,
        infer,
    };

    let app = Router::new().route("/generate", post(generate)).layer(Extension(shared_state));

    axum::Server::bind(&addr)
        .serve(app.into_make_service()).await.unwrap();
}
