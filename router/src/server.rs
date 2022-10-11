use crate::{Batcher, ShardedClient, Validation};
use poem::http::StatusCode;
use poem::listener::TcpListener;
use poem::middleware::AddData;
use poem::web::{Data, Json};
use poem::{handler, post, EndpointExt, Route, Server};
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

#[handler]
#[instrument(skip(validation, infer), fields(time, time_per_token))]
async fn generate(
    validation: Data<&Validation>,
    infer: Data<&Batcher>,
    req: Json<GenerateRequest>,
) -> poem::Result<Json<serde_json::Value>> {
    let start = Instant::now();

    let (input_length, validated_request) = match validation
        .validate(GenerateRequest {
            inputs: req.inputs.clone(),
            parameters: req.parameters.clone(),
        })
        .await {
        Ok(result) => result,
        Err(_) => return Err(poem::Error::from_status(StatusCode::INTERNAL_SERVER_ERROR))
    };

    let output = infer.infer(input_length, validated_request).await;

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
        Err(_) => Err(poem::Error::from_status(StatusCode::INTERNAL_SERVER_ERROR)),
    }
}

pub async fn run(
    client: ShardedClient,
    tokenizer: Tokenizer,
    listener: TcpListener<String>,
) -> Result<(), std::io::Error> {
    client.clear_cache().await.expect("Unable to clear cache");
    tracing::info!("Connected");

    let infer = Batcher::new(client);

    let validation = Validation::new(tokenizer);

    let app = Route::new()
        .at("/generate", post(generate))
        .with(AddData::new(validation))
        .with(AddData::new(infer));

    Server::new(listener).run(app).await
}
