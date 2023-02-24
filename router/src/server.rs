/// HTTP Server logic
use crate::infer::{InferError, InferStreamResponse};
use crate::{
    Details, ErrorResponse, FinishReason, GenerateParameters, GenerateRequest, GenerateResponse,
    Infer, PrefillToken, StreamDetails, StreamResponse, Token, Validation,
};
use axum::extract::Extension;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::opentelemetry_tracing_layer;
use futures::Stream;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::convert::Infallible;
use std::net::SocketAddr;
use text_generation_client::ShardedClient;
use tokenizers::Tokenizer;
use tokio::signal;
use tokio::time::Instant;
use tokio_stream::StreamExt;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{info_span, instrument, Instrument};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Health check method
#[instrument(skip(infer))]
async fn health(infer: Extension<Infer>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    // TODO: while this is the best health check we can do, it is a bit on the heavy side and might
    //       be a bit too slow for a health check.
    //       What we should do instead is check if the gRPC channels are still healthy.

    // Send a small inference request
    infer
        .generate(GenerateRequest {
            inputs: "liveness".to_string(),
            parameters: GenerateParameters {
                temperature: None,
                repetition_penalty: None,
                top_k: None,
                top_p: None,
                do_sample: false,
                max_new_tokens: 1,
                stop: Vec::new(),
                details: false,
                seed: None,
            },
        })
        .await?;
    Ok(())
}

/// Generate tokens
#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/generate",
    request_body = GenerateRequest,
    responses(
        (status = 200, description = "Generated Text", body = GenerateResponse),
        (status = 424, description = "Generation Error", body = ErrorResponse,
            example = json!({"error": "Request failed during generation"})),
        (status = 429, description = "Model is overloaded", body = ErrorResponse,
            example = json!({"error": "Model is overloaded"})),
        (status = 422, description = "Input validation error", body = ErrorResponse,
            example = json!({"error": "Input validation error"})),
        (status = 500, description = "Incomplete generation", body = ErrorResponse,
            example = json!({"error": "Incomplete generation"})),
    )
)]
#[instrument(
    skip(infer),
    fields(
        total_time,
        validation_time,
        queue_time,
        inference_time,
        time_per_token,
        seed,
    )
)]
async fn generate(
    infer: Extension<Infer>,
    req: Json<GenerateRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    let start_time = Instant::now();

    // Inference
    let details = req.0.parameters.details;
    let response = infer.generate(req.0).await?;

    // Token details
    let details = match details {
        true => Some(Details {
            finish_reason: FinishReason::from(response.generated_text.finish_reason),
            generated_tokens: response.generated_text.generated_tokens,
            prefill: Some(response.prefill),
            tokens: Some(response.tokens),
            seed: response.generated_text.seed,
        }),
        false => None,
    };

    // Timings
    let total_time = start_time.elapsed();
    let validation_time = response.queued - start_time;
    let queue_time = response.start - response.queued;
    let inference_time = Instant::now() - response.start;
    let time_per_token = inference_time / response.generated_text.generated_tokens;

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-time-per-token",
        time_per_token.as_millis().to_string().parse().unwrap(),
    );

    // Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));
    span.record("time_per_token", format!("{time_per_token:?}"));
    span.record("seed", format!("{:?}", response.generated_text.seed));
    tracing::info!("Output: {}", response.generated_text.text);

    // Metrics
    metrics::increment_counter!("tgi_request_success");
    metrics::histogram!("tgi_request_duration", total_time);
    metrics::histogram!("tgi_request_validation_duration", validation_time);
    metrics::histogram!("tgi_request_queue_duration", queue_time);
    metrics::histogram!("tgi_request_inference_duration", inference_time);
    metrics::histogram!("tgi_request_mean_time_per_token_duration", time_per_token);
    metrics::histogram!(
        "tgi_request_generated_tokens",
        response.generated_text.generated_tokens as f64
    );

    // Send response
    let response = GenerateResponse {
        generated_text: response.generated_text.text,
        details,
    };
    Ok((headers, Json(response)))
}

/// Generate a stream of token using Server-Sent Events
#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/generate_stream",
    request_body = GenerateRequest,
    responses(
        (status = 200, description = "Generated Text", body = StreamResponse,
            content_type="text/event-stream"),
        (status = 424, description = "Generation Error", body = ErrorResponse,
            example = json!({"error": "Request failed during generation"}),
            content_type="text/event-stream"),
        (status = 429, description = "Model is overloaded", body = ErrorResponse,
            example = json!({"error": "Model is overloaded"}),
            content_type="text/event-stream"),
        (status = 422, description = "Input validation error", body = ErrorResponse,
            example = json!({"error": "Input validation error"}),
            content_type="text/event-stream"),
        (status = 500, description = "Incomplete generation", body = ErrorResponse,
            example = json!({"error": "Incomplete generation"}),
            content_type="text/event-stream"),
    )
)]
#[instrument(
    skip(infer),
    fields(
        total_time,
        validation_time,
        queue_time,
        inference_time,
        time_per_token,
        seed,
    )
)]
async fn generate_stream(
    infer: Extension<Infer>,
    req: Json<GenerateRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let span = tracing::Span::current();
    let start_time = Instant::now();

    let stream = async_stream::stream! {
        // Inference
        let mut end_reached = false;
        let mut error = false;
        let details = req.0.parameters.details;

        match infer.generate_stream(req.0).instrument(info_span!(parent: &span, "async_stream")).await {
            Ok(mut response_stream) => {
                // Server-Sent Event stream
                while let Some(response) = response_stream.next().await {
                    match response {
                        Ok(response) => {
                            match response {
                                // Prefill is ignored
                                InferStreamResponse::Prefill(_) => {}
                                // Yield event for every new token
                                InferStreamResponse::Token(token) => {
                                    // StreamResponse
                                    let stream_token = StreamResponse {
                                        token,
                                        generated_text: None,
                                        details: None,
                                    };

                                    yield Ok(Event::default().json_data(stream_token).unwrap())
                                }
                                // Yield event for last token and compute timings
                                InferStreamResponse::End {
                                    token,
                                    generated_text,
                                    start,
                                    queued,
                                } => {
                                    // Token details
                                    let details = match details {
                                        true => Some(StreamDetails {
                                            finish_reason: FinishReason::from(generated_text.finish_reason),
                                            generated_tokens: generated_text.generated_tokens,
                                            seed: generated_text.seed,
                                        }),
                                        false => None,
                                    };

                                    // Timings
                                    let total_time = start_time.elapsed();
                                    let validation_time = queued - start_time;
                                    let queue_time = start - queued;
                                    let inference_time = Instant::now() - start;
                                    let time_per_token = inference_time / generated_text.generated_tokens;

                                    // Tracing metadata
                                    span.record("total_time", format!("{total_time:?}"));
                                    span.record("validation_time", format!("{validation_time:?}"));
                                    span.record("queue_time", format!("{queue_time:?}"));
                                    span.record("inference_time", format!("{inference_time:?}"));
                                    span.record("time_per_token", format!("{time_per_token:?}"));
                                    span.record("seed", format!("{:?}", generated_text.seed));
                                    tracing::info!(parent: &span, "Output: {}", generated_text.text);

                                    // Metrics
                                    metrics::increment_counter!("tgi_request_success");
                                    metrics::histogram!("tgi_request_duration", total_time);
                                    metrics::histogram!("tgi_request_validation_duration", validation_time);
                                    metrics::histogram!("tgi_request_queue_duration", queue_time);
                                    metrics::histogram!("tgi_request_inference_duration", inference_time);
                                    metrics::histogram!("tgi_request_mean_time_per_token_duration", time_per_token);
                                    metrics::histogram!("tgi_request_generated_tokens", generated_text.generated_tokens as f64);

                                    // StreamResponse
                                    end_reached = true;
                                    let stream_token = StreamResponse {
                                        token,
                                        generated_text: Some(generated_text.text),
                                        details
                                    };

                                    yield Ok(Event::default().json_data(stream_token).unwrap())
                                }
                            }
                        }
                        // yield error
                        Err(err) => {
                            error = true;
                            yield Ok(Event::from(err))
                        }
                    }
                }
            },
            // yield error
            Err(err) => {
                error = true;
                yield Ok(Event::from(err))
            }
        }
        // Check if generation reached the end
        // Skip if we already sent an error
        if !end_reached && !error {
            let err = InferError::IncompleteGeneration;
            metrics::increment_counter!("tgi_request_failure", "err" => "incomplete");
            tracing::error!("{err}");
            yield Ok(Event::from(err))
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Prometheus metrics scrape endpoint
#[utoipa::path(
    get,
    tag = "Text Generation Inference",
    path = "/metrics",
    responses((status = 200, description = "Prometheus Metrics", body = String))
)]
async fn metrics(prom_handle: Extension<PrometheusHandle>) -> String {
    prom_handle.render()
}

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(
    max_concurrent_requests: usize,
    max_stop_sequences: usize,
    max_input_length: usize,
    max_total_tokens: usize,
    max_batch_size: usize,
    max_waiting_tokens: usize,
    client: ShardedClient,
    tokenizer: Tokenizer,
    validation_workers: usize,
    addr: SocketAddr,
    allow_origin: Option<AllowOrigin>,
) {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
        paths(
            generate,
            generate_stream,
            metrics,
        ),
        components(
            schemas(
                GenerateRequest,
                GenerateParameters,
                PrefillToken,
                Token,
                GenerateResponse,
                Details,
                FinishReason,
                StreamResponse,
                StreamDetails,
                ErrorResponse,
            )
        ),
        tags(
            (name = "Text Generation Inference", description = "Hugging Face Text Generation Inference API")
        ),
        info(
            title = "Text Generation Inference",
            license(
                name = "Apache 2.0",
                url = "https://www.apache.org/licenses/LICENSE-2.0"
            )
        )
    )]
    struct ApiDoc;

    // Create state
    let validation = Validation::new(
        validation_workers,
        tokenizer,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
    );
    let infer = Infer::new(
        client,
        validation,
        max_batch_size,
        max_waiting_tokens,
        max_concurrent_requests,
    );

    // Prometheus handler
    let builder = PrometheusBuilder::new();
    let prom_handle = builder
        .install_recorder()
        .expect("failed to install metrics recorder");

    // CORS layer
    let allow_origin = allow_origin.unwrap_or(AllowOrigin::any());
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    // Create router
    let app = Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-doc/openapi.json", ApiDoc::openapi()))
        .route("/", post(generate))
        .route("/generate", post(generate))
        .route("/generate_stream", post(generate_stream))
        .route("/", get(health))
        .route("/health", get(health))
        .layer(Extension(infer))
        .route("/metrics", get(metrics))
        .layer(Extension(prom_handle))
        .layer(opentelemetry_tracing_layer())
        .layer(cors_layer);

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
    opentelemetry::global::shutdown_tracer_provider();
}

impl From<i32> for FinishReason {
    fn from(finish_reason: i32) -> Self {
        let finish_reason = text_generation_client::FinishReason::from_i32(finish_reason).unwrap();
        match finish_reason {
            text_generation_client::FinishReason::Length => FinishReason::Length,
            text_generation_client::FinishReason::EosToken => FinishReason::EndOfSequenceToken,
            text_generation_client::FinishReason::StopSequence => FinishReason::StopSequence,
        }
    }
}

/// Convert to Axum supported formats
impl From<InferError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: InferError) -> Self {
        let status_code = match err {
            InferError::GenerationError(_) => StatusCode::FAILED_DEPENDENCY,
            InferError::Overloaded(_) => StatusCode::TOO_MANY_REQUESTS,
            InferError::ValidationError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::IncompleteGeneration => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (
            status_code,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    }
}

impl From<InferError> for Event {
    fn from(err: InferError) -> Self {
        Event::default()
            .json_data(ErrorResponse {
                error: err.to_string(),
            })
            .unwrap()
    }
}
