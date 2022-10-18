use crate::{
    Batcher, GenerateParameters, GenerateRequest, GenerateResponse, GeneratedText, Validation,
};
use axum::extract::Extension;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use bloom_inference_client::ShardedClient;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokenizers::Tokenizer;
use tokio::signal;
use tokio::sync::Semaphore;
use tokio::time::Instant;
use tracing::instrument;

// Server shared state
#[derive(Clone)]
struct ServerState {
    validation: Validation,
    batcher: Batcher,
    limit_concurrent_requests: Arc<Semaphore>,
}

/// Health check method
#[instrument(skip(state), fields(time, time_per_token))]
async fn health(state: Extension<ServerState>) -> Result<(), (StatusCode, String)> {
    // TODO: while this is the best health check we can do, it is a bit on the heavy side and might
    //       be a bit too slow for a health check.
    //       What we should do instead if check if the gRPC channels are still healthy.

    // Limit concurrent requests by acquiring a permit from the semaphore
    let _permit = state.limit_concurrent_requests.try_acquire().map_err(|_| {
        (
            StatusCode::TOO_MANY_REQUESTS,
            "Model is overloaded".to_string(),
        )
    })?;

    // Send a small inference request
    state
        .batcher
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

/// Generate method
#[instrument(skip(state), fields(time, time_per_token))]
async fn generate(
    state: Extension<ServerState>,
    req: Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let start = Instant::now();
    // Limit concurrent requests by acquiring a permit from the semaphore
    let _permit = state.limit_concurrent_requests.try_acquire().map_err(|_| {
        (
            StatusCode::TOO_MANY_REQUESTS,
            "Model is overloaded".to_string(),
        )
    })?;

    // Validate request
    let (input_length, validated_request) = state
        .validation
        // FIXME: can't we get rid of the cloning here??
        .validate(GenerateRequest {
            inputs: req.inputs.clone(),
            parameters: req.parameters.clone(),
        })
        .await?;

    // Inference
    let generated_text = state.batcher.infer(input_length, validated_request).await?;

    // Tracing metadata
    tracing::Span::current().record("time", format!("{:?}", start.elapsed()));
    tracing::Span::current().record(
        "time_per_token",
        format!("{:?}", start.elapsed() / req.parameters.max_new_tokens),
    );
    tracing::info!("response: {}", generated_text);

    // Send response
    let response = vec![GeneratedText { generated_text }];
    Ok(Json(response))
}

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(
    max_concurrent_requests: usize,
    max_input_length: usize,
    max_batch_size: usize,
    max_waiting_time: Duration,
    client: ShardedClient,
    tokenizer: Tokenizer,
    validation_workers: usize,
    addr: SocketAddr,
) {
    // Create state
    let batcher = Batcher::new(client, max_batch_size, max_waiting_time);
    let validation = Validation::new(validation_workers, tokenizer, max_input_length);
    let shared_state = ServerState {
        validation,
        batcher,
        limit_concurrent_requests: Arc::new(Semaphore::new(max_concurrent_requests)),
    };

    // Create router
    let app = Router::new()
        .route("/generate", post(generate))
        .layer(Extension(shared_state.clone()))
        .route("/health", get(health))
        .layer(Extension(shared_state.clone()));

    // Run server
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        // Wait until all requests are finished to shut down
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

/// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("signal received, starting graceful shutdown");
}
