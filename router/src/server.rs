use crate::config::Config;
/// HTTP Server logic
use crate::health::Health;
use crate::infer::{InferError, InferResponse, InferStreamResponse, ToolGrammar};
use crate::validation::ValidationError;
use crate::{
    BestOfSequence, Details, ErrorResponse, FinishReason, GenerateParameters, GenerateRequest,
    GenerateResponse, GrammarType, HubModelInfo, HubTokenizerConfig, Infer, Info, Message,
    PrefillToken, SimpleToken, StreamDetails, StreamResponse, Token, TokenizeResponse, Usage,
    Validation,
};
use crate::{
    ChatCompletion, ChatCompletionChoice, ChatCompletionChunk, ChatCompletionComplete,
    ChatCompletionDelta, ChatCompletionLogprob, ChatCompletionLogprobs, ChatCompletionTopLogprob,
    ChatRequest, CompatGenerateRequest, Completion, CompletionComplete, CompletionCompleteChunk,
    CompletionRequest, DeltaToolCall, Function, Tool, VertexRequest, VertexResponse,
};
use crate::{FunctionDefinition, ToolCall, ToolType};
use async_stream::__private::AsyncStream;
use axum::extract::Extension;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{http, Json, Router};
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;
use futures::stream::StreamExt;
use futures::stream::{FuturesOrdered, FuturesUnordered};
use futures::Stream;
use futures::TryStreamExt;
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use serde_json::Value;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use text_generation_client::{ShardInfo, ShardedClient};
use tokenizers::Tokenizer;
use tokio::select;
use tokio::signal;
use tokio::sync::oneshot;
use tokio::time::Instant;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{info_span, instrument, Instrument};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

/// Generate tokens if `stream == false` or a stream of token if `stream == true`
#[utoipa::path(
post,
tag = "Text Generation Inference",
path = "/",
request_body = CompatGenerateRequest,
responses(
(status = 200, description = "Generated Text",
content(
("application/json" = GenerateResponse),
("text/event-stream" = StreamResponse),
)),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(skip(infer, req))]
async fn compat_generate(
    Extension(default_return_full_text): Extension<bool>,
    infer: Extension<Infer>,
    compute_type: Extension<ComputeType>,
    Json(mut req): Json<CompatGenerateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // default return_full_text given the pipeline_tag
    if req.parameters.return_full_text.is_none() {
        req.parameters.return_full_text = Some(default_return_full_text)
    }

    // switch on stream
    if req.stream {
        Ok(generate_stream(infer, compute_type, Json(req.into()))
            .await
            .into_response())
    } else {
        let (headers, Json(generation)) = generate(infer, compute_type, Json(req.into())).await?;
        // wrap generation inside a Vec to match api-inference
        Ok((headers, Json(vec![generation])).into_response())
    }
}

/// Text Generation Inference endpoint info
#[utoipa::path(
get,
tag = "Text Generation Inference",
path = "/info",
responses((status = 200, description = "Served model info", body = Info))
)]
#[instrument]
async fn get_model_info(info: Extension<Info>) -> Json<Info> {
    Json(info.0)
}

#[utoipa::path(
get,
tag = "Text Generation Inference",
path = "/health",
responses(
(status = 200, description = "Everything is working fine"),
(status = 503, description = "Text generation inference is down", body = ErrorResponse,
example = json ! ({"error": "unhealthy", "error_type": "healthcheck"})),
)
)]
#[instrument(skip(health))]
/// Health check method
async fn health(mut health: Extension<Health>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    match health.check().await {
        true => Ok(()),
        false => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "unhealthy".to_string(),
                error_type: "healthcheck".to_string(),
            }),
        )),
    }
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
example = json ! ({"error": "Request failed during generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"})),
)
)]
#[instrument(
skip_all,
fields(
parameters = ? req.parameters,
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
    Extension(ComputeType(compute_type)): Extension<ComputeType>,
    Json(req): Json<GenerateRequest>,
) -> Result<(HeaderMap, Json<GenerateResponse>), (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    generate_internal(infer, ComputeType(compute_type), Json(req), span).await
}

async fn generate_internal(
    infer: Extension<Infer>,
    ComputeType(compute_type): ComputeType,
    Json(req): Json<GenerateRequest>,
    span: tracing::Span,
) -> Result<(HeaderMap, Json<GenerateResponse>), (StatusCode, Json<ErrorResponse>)> {
    let start_time = Instant::now();
    metrics::increment_counter!("tgi_request_count");

    // Do not long ultra long inputs, like image payloads.
    tracing::debug!("Input: {}", &req.inputs[..1000.min(req.inputs.len())]);

    let compute_characters = req.inputs.chars().count();
    let mut add_prompt = None;
    if req.parameters.return_full_text.unwrap_or(false) {
        add_prompt = Some(req.inputs.clone());
    }

    let details: bool = req.parameters.details || req.parameters.decoder_input_details;

    // Inference
    let (response, best_of_responses) = match req.parameters.best_of {
        Some(best_of) if best_of > 1 => {
            let (response, best_of_responses) = infer.generate_best_of(req, best_of).await?;
            (response, Some(best_of_responses))
        }
        _ => (infer.generate(req).await?, None),
    };

    // Token details
    let input_length = response._input_length;
    let details = match details {
        true => {
            // convert best_of_responses
            let best_of_sequences = best_of_responses.map(|responses: Vec<InferResponse>| {
                responses
                    .into_iter()
                    .map(|response: InferResponse| {
                        // Add prompt if return_full_text
                        let mut output_text = response.generated_text.text;
                        if let Some(prompt) = &add_prompt {
                            output_text = prompt.clone() + &output_text;
                        }

                        BestOfSequence {
                            generated_text: output_text,
                            finish_reason: FinishReason::from(
                                response.generated_text.finish_reason,
                            ),
                            generated_tokens: response.generated_text.generated_tokens,
                            prefill: response.prefill,
                            tokens: response.tokens,
                            top_tokens: response.top_tokens,
                            seed: response.generated_text.seed,
                        }
                    })
                    .collect()
            });

            Some(Details {
                finish_reason: FinishReason::from(response.generated_text.finish_reason),
                generated_tokens: response.generated_text.generated_tokens,
                prefill: response.prefill,
                tokens: response.tokens,
                seed: response.generated_text.seed,
                best_of_sequences,
                top_tokens: response.top_tokens,
            })
        }
        false => None,
    };

    // Timings
    let total_time = start_time.elapsed();
    let validation_time = response.queued - start_time;
    let queue_time = response.start - response.queued;
    let inference_time = Instant::now() - response.start;
    let time_per_token = inference_time / response.generated_text.generated_tokens;

    // Tracing metadata
    span.record("total_time", format!("{total_time:?}"));
    span.record("validation_time", format!("{validation_time:?}"));
    span.record("queue_time", format!("{queue_time:?}"));
    span.record("inference_time", format!("{inference_time:?}"));
    span.record("time_per_token", format!("{time_per_token:?}"));
    span.record("seed", format!("{:?}", response.generated_text.seed));

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", compute_type.parse().unwrap());
    headers.insert(
        "x-compute-time",
        total_time.as_secs_f64().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
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
    headers.insert("x-prompt-tokens", input_length.into());
    headers.insert(
        "x-generated-tokens",
        response.generated_text.generated_tokens.into(),
    );

    // Metrics
    metrics::increment_counter!("tgi_request_success");
    metrics::histogram!("tgi_request_duration", total_time.as_secs_f64());
    metrics::histogram!(
        "tgi_request_validation_duration",
        validation_time.as_secs_f64()
    );
    metrics::histogram!("tgi_request_queue_duration", queue_time.as_secs_f64());
    metrics::histogram!(
        "tgi_request_inference_duration",
        inference_time.as_secs_f64()
    );
    metrics::histogram!(
        "tgi_request_mean_time_per_token_duration",
        time_per_token.as_secs_f64()
    );
    metrics::histogram!(
        "tgi_request_generated_tokens",
        response.generated_text.generated_tokens as f64
    );

    // Send response
    let mut output_text = response.generated_text.text;
    if let Some(prompt) = add_prompt {
        output_text = prompt + &output_text;
    }

    tracing::debug!("Output: {}", output_text);
    tracing::info!("Success");

    let response = GenerateResponse {
        generated_text: output_text,
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
content_type = "text/event-stream"),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation"}),
content_type = "text/event-stream"),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded"}),
content_type = "text/event-stream"),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error"}),
content_type = "text/event-stream"),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation"}),
content_type = "text/event-stream"),
)
)]
#[instrument(
skip_all,
fields(
parameters = ? req.parameters,
total_time,
validation_time,
queue_time,
inference_time,
time_per_token,
seed,
)
)]
async fn generate_stream(
    Extension(infer): Extension<Infer>,
    Extension(compute_type): Extension<ComputeType>,
    Json(req): Json<GenerateRequest>,
) -> (
    HeaderMap,
    Sse<impl Stream<Item = Result<Event, Infallible>>>,
) {
    let span = tracing::Span::current();
    let on_message_callback = |stream_token: StreamResponse| {
        let event = Event::default();
        event.json_data(stream_token).unwrap()
    };
    let (headers, response_stream) =
        generate_stream_internal(infer, compute_type, Json(req), on_message_callback, span).await;
    let sse = Sse::new(response_stream).keep_alive(KeepAlive::default());
    (headers, sse)
}

async fn generate_stream_internal(
    infer: Infer,
    ComputeType(compute_type): ComputeType,
    Json(req): Json<GenerateRequest>,
    on_message_callback: impl Fn(StreamResponse) -> Event,
    span: tracing::Span,
) -> (HeaderMap, impl Stream<Item = Result<Event, Infallible>>) {
    let start_time = Instant::now();
    metrics::increment_counter!("tgi_request_count");

    tracing::debug!("Input: {}", req.inputs);

    let compute_characters = req.inputs.chars().count();

    let mut headers = HeaderMap::new();
    headers.insert("x-compute-type", compute_type.parse().unwrap());
    headers.insert(
        "x-compute-characters",
        compute_characters.to_string().parse().unwrap(),
    );
    headers.insert("X-Accel-Buffering", "no".parse().unwrap());

    let stream = async_stream::stream! {
        // Inference
        let mut end_reached = false;
        let mut error = false;

        let mut add_prompt = None;
        if req.parameters.return_full_text.unwrap_or(false) {
            add_prompt = Some(req.inputs.clone());
        }
        let details = req.parameters.details;

        let best_of = req.parameters.best_of.unwrap_or(1);
        if best_of != 1 {
            let err = InferError::from(ValidationError::BestOfStream);
            metrics::increment_counter!("tgi_request_failure", "err" => "validation");
            tracing::error!("{err}");
            yield Ok(Event::from(err));
        } else if req.parameters.decoder_input_details {
            let err = InferError::from(ValidationError::PrefillDetailsStream);
            metrics::increment_counter!("tgi_request_failure", "err" => "validation");
            tracing::error!("{err}");
            yield Ok(Event::from(err));
        } else {
            match infer.generate_stream(req).instrument(info_span!(parent: &span, "async_stream")).await {
                // Keep permit as long as generate_stream lives
                Ok((_permit, _input_length, mut response_stream)) => {
                    let mut index = 0;
                    // Server-Sent Event stream
                    while let Some(response) = response_stream.next().await {
                        index += 1;
                        match response {
                            Ok(response) => {
                                match response {
                                    // Prefill is ignored
                                    InferStreamResponse::Prefill(_) => {}
                                    // Yield event for every new token
                                    InferStreamResponse::Intermediate{
                                        token,
                                        top_tokens,
                                    } => {
                                        tracing::debug!(parent: &span, "Token: {:?}", token);

                                        // StreamResponse
                                        let stream_token = StreamResponse {
                                            index,
                                            token,
                                            top_tokens,
                                            generated_text: None,
                                            details: None,
                                        };
                                        let event = on_message_callback(stream_token);
                                        yield Ok(event);
                                    }
                                    // Yield event for last token and compute timings
                                    InferStreamResponse::End {
                                        token,
                                        generated_text,
                                        start,
                                        queued,
                                        top_tokens,
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

                                        // Metrics
                                        metrics::increment_counter!("tgi_request_success");
                                        metrics::histogram!("tgi_request_duration", total_time.as_secs_f64());
                                        metrics::histogram!("tgi_request_validation_duration", validation_time.as_secs_f64());
                                        metrics::histogram!("tgi_request_queue_duration", queue_time.as_secs_f64());
                                        metrics::histogram!("tgi_request_inference_duration", inference_time.as_secs_f64());
                                        metrics::histogram!("tgi_request_mean_time_per_token_duration", time_per_token.as_secs_f64());
                                        metrics::histogram!("tgi_request_generated_tokens", generated_text.generated_tokens as f64);

                                        // StreamResponse
                                        end_reached = true;

                                        let mut output_text = generated_text.text;
                                        if let Some(prompt) = add_prompt {
                                            output_text = prompt + &output_text;
                                        }

                                        tracing::debug!(parent: &span, "Output: {}", output_text);
                                        tracing::info!(parent: &span, "Success");

                                        let stream_token = StreamResponse {
                                            index,
                                            token,
                                            top_tokens,
                                            generated_text: Some(output_text),
                                            details
                                        };


                                        let event = on_message_callback(stream_token);
                                        yield Ok(event);
                                        break;
                                    }
                                }
                            }
                            // yield error
                            Err(err) => {
                                error = true;
                                yield Ok(Event::from(err));
                                break;
                            }
                        }
                    }
                },
                // yield error
                Err(err) => {
                    error = true;
                    yield Ok(Event::from(err));
                }
            }
            // Check if generation reached the end
            // Skip if we already sent an error
            if !end_reached && !error {
                let err = InferError::IncompleteGeneration;
                metrics::increment_counter!("tgi_request_failure", "err" => "incomplete");
                tracing::error!("{err}");
                yield Ok(Event::from(err));
            }
        }
    };

    (headers, stream)
}

/// Generate tokens
#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/v1/completions",
    request_body = CompletionRequest,
    responses(
    (status = 200, description = "Generated Chat Completion",
    content(
    ("application/json" = Completion),
    ("text/event-stream" = CompletionCompleteChunk),
    )),
    (status = 424, description = "Generation Error", body = ErrorResponse,
    example = json ! ({"error": "Request failed during generation"})),
    (status = 429, description = "Model is overloaded", body = ErrorResponse,
    example = json ! ({"error": "Model is overloaded"})),
    (status = 422, description = "Input validation error", body = ErrorResponse,
    example = json ! ({"error": "Input validation error"})),
    (status = 500, description = "Incomplete generation", body = ErrorResponse,
    example = json ! ({"error": "Incomplete generation"})),
    )
    )]
#[instrument(
    skip_all,
    fields(
    // parameters = ? req.parameters,
    total_time,
    validation_time,
    queue_time,
    inference_time,
    time_per_token,
    seed,
    )
    )]
async fn completions(
    Extension(infer): Extension<Infer>,
    Extension(compute_type): Extension<ComputeType>,
    Extension(info): Extension<Info>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    metrics::increment_counter!("tgi_request_count");

    let stream = req.stream;
    let max_new_tokens = req.max_tokens.or(Some(100));
    let seed = req.seed;

    // if suffix is present throw an error
    if req.suffix.is_some() {
        metrics::increment_counter!("tgi_request_failure", "err" => "validation");
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: "Suffix is not supported and can be achieved by preprocessing the prompt."
                    .to_string(),
                error_type: "suffix not supported".to_string(),
            }),
        ));
    }

    if req.prompt.len() > info.max_client_batch_size {
        metrics::increment_counter!("tgi_request_failure", "err" => "validation");
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: format!(
                    "Number of prompts exceeds the maximum allowed batch size of {}",
                    info.max_client_batch_size
                ),
                error_type: "batch size exceeded".to_string(),
            }),
        ));
    }

    let generate_requests: Vec<GenerateRequest> = req
        .prompt
        .iter()
        .map(|prompt| GenerateRequest {
            inputs: prompt.to_string(),
            parameters: GenerateParameters {
                best_of: None,
                temperature: req.temperature,
                repetition_penalty: req.repetition_penalty,
                frequency_penalty: req.frequency_penalty,
                top_k: None,
                top_p: req.top_p,
                typical_p: None,
                do_sample: true,
                max_new_tokens,
                return_full_text: None,
                stop: Vec::new(),
                truncate: None,
                watermark: false,
                details: true,
                decoder_input_details: !stream,
                seed,
                top_n_tokens: None,
                grammar: None,
            },
        })
        .collect();

    let mut x_compute_type = None;
    let mut x_compute_characters = 0u32;
    let mut x_accel_buffering = None;

    if stream {
        let mut response_streams = FuturesOrdered::new();
        for (index, generate_request) in generate_requests.into_iter().enumerate() {
            let model_id = info.model_id.clone();
            let system_fingerprint =
                format!("{}-{}", info.version, info.docker_label.unwrap_or("native"));
            let infer_clone = infer.clone();
            let compute_type_clone = compute_type.clone();
            let span_clone = span.clone();

            // Create a future for each generate_stream_internal call.
            let generate_future = async move {
                let on_message_callback = move |stream_token: StreamResponse| {
                    let event = Event::default();

                    let current_time = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                        .as_secs();

                    event
                        .json_data(CompletionCompleteChunk {
                            id: "".to_string(),
                            object: "text_completion".to_string(),
                            created: current_time,

                            choices: vec![CompletionComplete {
                                finish_reason: "".to_string(),
                                index: index as u32,
                                logprobs: None,
                                text: stream_token.token.text,
                            }],

                            model: model_id.clone(),
                            system_fingerprint: system_fingerprint.clone(),
                        })
                        .map_or_else(|_e| Event::default(), |data| data)
                };

                let (header_tx, header_rx) = oneshot::channel();
                let (sse_tx, sse_rx) = tokio::sync::mpsc::unbounded_channel();

                tokio::spawn(async move {
                    let (header_map, sse) = generate_stream_internal(
                        infer_clone.clone(),
                        compute_type_clone.clone(),
                        Json(generate_request),
                        on_message_callback,
                        span_clone.clone(),
                    )
                    .await;

                    // send and dont wait for response
                    let _ = header_tx.send(header_map);

                    // pin an emit messages to the sse_tx
                    let mut sse = Box::pin(sse);
                    while let Some(event) = sse.next().await {
                        if sse_tx.send(event).is_err() {
                            tracing::error!("Failed to send event. Receiver dropped.");
                            break;
                        }
                    }
                });

                (header_rx, sse_rx)
            };
            response_streams.push_back(generate_future);
        }

        let mut all_rxs = vec![];

        while let Some((header_rx, sse_rx)) = response_streams.next().await {
            all_rxs.push(sse_rx);

            // get the headers from the first response of each stream
            let headers = header_rx.await.map_err(|e| {
                tracing::error!("Failed to get headers: {:?}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Failed to get headers".to_string(),
                        error_type: "headers".to_string(),
                    }),
                )
            })?;
            if x_compute_type.is_none() {
                x_compute_type = headers
                    .get("x-compute-type")
                    .and_then(|v| v.to_str().ok())
                    .map(|v| v.to_string());

                x_accel_buffering = headers
                    .get("x-accel-buffering")
                    .and_then(|v| v.to_str().ok())
                    .map(|v| v.to_string());
            }
            x_compute_characters += headers
                .get("x-compute-characters")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }

        let mut headers = HeaderMap::new();
        if let Some(x_compute_type) = x_compute_type {
            headers.insert("x-compute-type", x_compute_type.parse().unwrap());
        }
        headers.insert("x-compute-characters", x_compute_characters.into());
        if let Some(x_accel_buffering) = x_accel_buffering {
            headers.insert("x-accel-buffering", x_accel_buffering.parse().unwrap());
        }

        // now sink the sse streams into a single stream and remove the ones that are done
        let stream: AsyncStream<Result<Event, Infallible>, _> = async_stream::stream! {
            loop {
                let mut i = 0;
                while i < all_rxs.len() {
                    let rx = &mut all_rxs[i];
                    select! {
                        Some(event) = rx.recv() => {
                            yield event;
                        }
                        else => {
                            all_rxs.remove(i);
                            continue; // skip the increment to handle the next element at the same index
                        }
                    }
                    i += 1; // only increment when no element was removed
                }

                if all_rxs.is_empty() {
                    break;
                }
            }
        };

        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok((headers, sse).into_response())
    } else {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();

        let responses = FuturesUnordered::new();
        for (index, generate_request) in generate_requests.into_iter().enumerate() {
            let infer_clone = infer.clone();
            let compute_type_clone = compute_type.clone();
            let span_clone = span.clone();
            let response_future = async move {
                let result = generate_internal(
                    Extension(infer_clone),
                    compute_type_clone,
                    Json(generate_request),
                    span_clone,
                )
                .await;
                result.map(|(headers, generation)| (index, headers, generation))
            };
            responses.push(response_future);
        }
        let generate_responses = responses.try_collect::<Vec<_>>().await?;

        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u32;
        let mut total_tokens = 0u32;

        let mut x_compute_time = 0u32;
        let mut x_total_time = 0u32;
        let mut x_validation_time = 0u32;
        let mut x_queue_time = 0u32;
        let mut x_inference_time = 0u32;
        let mut x_time_per_token = 0u32;
        let mut x_prompt_tokens = 0u32;
        let mut x_generated_tokens = 0u32;

        let choices = generate_responses
            .into_iter()
            .map(|(index, headers, Json(generation))| {
                let details = generation.details.ok_or((
                    // this should never happen but handle if details are missing unexpectedly
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "No details in generation".to_string(),
                        error_type: "no details".to_string(),
                    }),
                ))?;

                if x_compute_type.is_none() {
                    x_compute_type = headers
                        .get("x-compute-type")
                        .and_then(|v| v.to_str().ok())
                        .map(|v| v.to_string());
                }

                // accumulate headers and usage from each response
                x_compute_time += headers
                    .get("x-compute-time")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
                    .unwrap_or(0);
                x_compute_characters += headers
                    .get("x-compute-characters")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
                    .unwrap_or(0);
                x_total_time += headers
                    .get("x-total-time")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
                    .unwrap_or(0);
                x_validation_time += headers
                    .get("x-validation-time")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
                    .unwrap_or(0);
                x_queue_time += headers
                    .get("x-queue-time")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
                    .unwrap_or(0);
                x_inference_time += headers
                    .get("x-inference-time")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
                    .unwrap_or(0);
                x_time_per_token += headers
                    .get("x-time-per-token")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
                    .unwrap_or(0);
                x_prompt_tokens += headers
                    .get("x-prompt-tokens")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
                    .unwrap_or(0);
                x_generated_tokens += headers
                    .get("x-generated-tokens")
                    .and_then(|v| v.to_str().ok()?.parse().ok())
                    .unwrap_or(0);

                prompt_tokens += details.prefill.len() as u32;
                completion_tokens += details.generated_tokens;
                total_tokens += details.prefill.len() as u32 + details.generated_tokens;

                Ok(CompletionComplete {
                    finish_reason: details.finish_reason.to_string(),
                    index: index as u32,
                    logprobs: None,
                    text: generation.generated_text,
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|(status, Json(err))| (status, Json(err)))?;

        let response = Completion {
            id: "".to_string(),
            object: "text_completion".to_string(),
            created: current_time,
            model: info.model_id.clone(),
            system_fingerprint: format!(
                "{}-{}",
                info.version,
                info.docker_label.unwrap_or("native")
            ),
            choices,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            },
        };

        // headers similar to `generate` but aggregated
        let mut headers = HeaderMap::new();
        if let Some(x_compute_type) = x_compute_type {
            headers.insert("x-compute-type", x_compute_type.parse().unwrap());
        }
        headers.insert("x-compute-characters", x_compute_characters.into());
        headers.insert("x-total-time", x_total_time.into());
        headers.insert("x-validation-time", x_validation_time.into());
        headers.insert("x-queue-time", x_queue_time.into());
        headers.insert("x-inference-time", x_inference_time.into());
        headers.insert("x-time-per-token", x_time_per_token.into());
        headers.insert("x-prompt-tokens", x_prompt_tokens.into());
        headers.insert("x-generated-tokens", x_generated_tokens.into());
        if let Some(x_accel_buffering) = x_accel_buffering {
            headers.insert("x-accel-buffering", x_accel_buffering.parse().unwrap());
        }
        Ok((headers, Json(response)).into_response())
    }
}

/// Generate tokens
#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/v1/chat/completions",
    request_body = ChatRequest,
    responses(
    (status = 200, description = "Generated Chat Completion",
    content(
    ("application/json" = ChatCompletion),
    ("text/event-stream" = ChatCompletionChunk),
    )),
    (status = 424, description = "Generation Error", body = ErrorResponse,
    example = json ! ({"error": "Request failed during generation"})),
    (status = 429, description = "Model is overloaded", body = ErrorResponse,
    example = json ! ({"error": "Model is overloaded"})),
    (status = 422, description = "Input validation error", body = ErrorResponse,
    example = json ! ({"error": "Input validation error"})),
    (status = 500, description = "Incomplete generation", body = ErrorResponse,
    example = json ! ({"error": "Incomplete generation"})),
    )
    )]
#[instrument(
    skip_all,
    fields(
    // parameters = ? req.parameters,
    total_time,
    validation_time,
    queue_time,
    inference_time,
    time_per_token,
    seed,
    )
    )]
async fn chat_completions(
    Extension(infer): Extension<Infer>,
    Extension(compute_type): Extension<ComputeType>,
    Extension(info): Extension<Info>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    metrics::increment_counter!("tgi_request_count");

    let ChatRequest {
        logprobs,
        max_tokens,
        messages,
        presence_penalty,
        seed,
        stop,
        stream,
        tools,
        tool_choice,
        tool_prompt,
        ..
    } = req;

    let repetition_penalty = presence_penalty.map(|x| x + 2.0);
    let max_new_tokens = max_tokens.or(Some(100));
    let logprobs = logprobs.unwrap_or(false);
    let tool_prompt = tool_prompt.unwrap_or_default();
    let stop = stop.unwrap_or_default();

    // extract tool grammar if present
    let tool_grammar = match ToolGrammar::apply(tools, tool_choice) {
        Ok(grammar) => grammar,
        Err(err) => {
            metrics::increment_counter!("tgi_request_failure", "err" => "validation");
            tracing::error!("{err}");
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorResponse {
                    error: err.to_string(),
                    error_type: err.error_type().to_string(),
                }),
            ));
        }
    };

    let grammar_with_prompt = tool_grammar
        .as_ref()
        .map(|t| (GrammarType::Json(serde_json::json!(t)), tool_prompt));

    let typed_grammar = grammar_with_prompt
        .as_ref()
        .map(|(grammar, _)| grammar.clone());

    // apply chat template to flatten the request into a single input
    let inputs = match infer.apply_chat_template(messages, grammar_with_prompt) {
        Ok(inputs) => inputs,
        Err(err) => {
            metrics::increment_counter!("tgi_request_failure", "err" => "validation");
            tracing::error!("{err}");
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorResponse {
                    error: err.to_string(),
                    error_type: err.error_type().to_string(),
                }),
            ));
        }
    };

    // build the request passing some parameters
    let generate_request = GenerateRequest {
        inputs: inputs.to_string(),
        parameters: GenerateParameters {
            best_of: None,
            temperature: req.temperature,
            repetition_penalty,
            frequency_penalty: req.frequency_penalty,
            top_k: None,
            top_p: req.top_p,
            typical_p: None,
            do_sample: true,
            max_new_tokens,
            return_full_text: None,
            stop,
            truncate: None,
            watermark: false,
            details: true,
            decoder_input_details: !stream,
            seed,
            top_n_tokens: req.top_logprobs,
            grammar: typed_grammar,
        },
    };

    // static values that will be returned in all cases
    let model_id = info.model_id.clone();
    let system_fingerprint = format!("{}-{}", info.version, info.docker_label.unwrap_or("native"));

    // switch on stream
    if stream {
        // pass this callback to the stream generation and build the required event structure
        let on_message_callback = move |stream_token: StreamResponse| {
            let event = Event::default();

            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                .as_secs();

            let logprobs = logprobs.then(|| {
                ChatCompletionLogprobs::from((stream_token.token.clone(), stream_token.top_tokens))
            });

            // replace the content with the tool calls if grammar is present
            let (content, tool_calls) = if tool_grammar.is_some() {
                (None, Some(vec![stream_token.token.text]))
            } else {
                (Some(stream_token.token.text), None)
            };

            event
                .json_data(ChatCompletionChunk::new(
                    model_id.clone(),
                    system_fingerprint.clone(),
                    content,
                    tool_calls,
                    current_time,
                    logprobs,
                    stream_token.details.map(|d| d.finish_reason.to_string()),
                ))
                .map_or_else(
                    |e| {
                        println!("Failed to serialize ChatCompletionChunk: {:?}", e);
                        Event::default()
                    },
                    |data| data,
                )
        };

        let (headers, response_stream) = generate_stream_internal(
            infer,
            compute_type,
            Json(generate_request),
            on_message_callback,
            span,
        )
        .await;
        let sse = Sse::new(response_stream).keep_alive(KeepAlive::default());
        Ok((headers, sse).into_response())
    } else {
        let (headers, Json(generation)) =
            generate_internal(Extension(infer), compute_type, Json(generate_request), span).await?;

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
            .as_secs();

        let (tool_calls, output) = if tool_grammar.is_some() {
            // gen_text should be valid json
            let gen_text_value: Value =
                serde_json::from_str(&generation.generated_text).map_err(|e| {
                    (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        Json(ErrorResponse {
                            error: e.to_string(),
                            error_type: "Input validation error".to_string(),
                        }),
                    )
                })?;
            let tool_calls = vec![ToolCall {
                id: 0,
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    description: None,
                    name: gen_text_value
                        .get("function")
                        .and_then(|f| f.get("_name"))
                        .and_then(|name| name.as_str())
                        .unwrap_or("default_function_name")
                        .to_string(),
                    // Serialize the JSON object obtained from "function" to an escaped JSON string
                    arguments: gen_text_value
                        .get("function")
                        .map(|f| {
                            let mut f_cloned = f.clone();
                            if let Value::Object(ref mut props) = f_cloned {
                                props.remove("_name");
                            }
                            f_cloned
                        })
                        .unwrap_or_default(),
                },
            }];
            (Some(tool_calls), None)
        } else {
            (None, Some(generation.generated_text))
        };
        // build the complete response object with the full text
        let response = ChatCompletion::new(
            model_id,
            system_fingerprint,
            output,
            current_time,
            generation.details.unwrap(),
            logprobs,
            tool_calls,
        );

        // wrap generation inside a Vec to match api-inference
        Ok((headers, Json(response)).into_response())
    }
}

/// Generate tokens from Vertex request
#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/vertex",
    request_body = VertexRequest,
    responses(
    (status = 200, description = "Generated Text", body = VertexResponse),
    (status = 424, description = "Generation Error", body = ErrorResponse,
    example = json ! ({"error": "Request failed during generation"})),
    (status = 429, description = "Model is overloaded", body = ErrorResponse,
    example = json ! ({"error": "Model is overloaded"})),
    (status = 422, description = "Input validation error", body = ErrorResponse,
    example = json ! ({"error": "Input validation error"})),
    (status = 500, description = "Incomplete generation", body = ErrorResponse,
    example = json ! ({"error": "Incomplete generation"})),
    )
    )]
#[instrument(
    skip_all,
    fields(
        total_time,
        validation_time,
        queue_time,
        inference_time,
        time_per_token,
        seed,
    )
)]
async fn vertex_compatibility(
    Extension(infer): Extension<Infer>,
    Extension(compute_type): Extension<ComputeType>,
    Json(req): Json<VertexRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    metrics::increment_counter!("tgi_request_count");

    // check that theres at least one instance
    if req.instances.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: "Input validation error".to_string(),
                error_type: "Input validation error".to_string(),
            }),
        ));
    }

    // Process all instances
    let predictions = req
        .instances
        .iter()
        .map(|instance| {
            let generate_request = GenerateRequest {
                inputs: instance.inputs.clone(),
                parameters: GenerateParameters {
                    do_sample: true,
                    max_new_tokens: instance.parameters.as_ref().and_then(|p| p.max_new_tokens),
                    seed: instance.parameters.as_ref().and_then(|p| p.seed),
                    details: true,
                    decoder_input_details: true,
                    ..Default::default()
                },
            };

            async {
                generate_internal(
                    Extension(infer.clone()),
                    compute_type.clone(),
                    Json(generate_request),
                    span.clone(),
                )
                .await
                .map(|(_, Json(generation))| generation.generated_text)
                .map_err(|_| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "Incomplete generation".into(),
                            error_type: "Incomplete generation".into(),
                        }),
                    )
                })
            }
        })
        .collect::<FuturesUnordered<_>>()
        .try_collect::<Vec<_>>()
        .await?;

    let response = VertexResponse { predictions };
    Ok((HeaderMap::new(), Json(response)).into_response())
}

/// Tokenize inputs
#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/tokenize",
    request_body = GenerateRequest,
    responses(
    (status = 200, description = "Tokenized ids", body = TokenizeResponse),
    (status = 404, description = "No tokenizer found", body = ErrorResponse,
    example = json ! ({"error": "No fast tokenizer available"})),
    )
    )]
#[instrument(skip_all)]
async fn tokenize(
    Extension(infer): Extension<Infer>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let input = req.inputs.clone();
    let encoding = infer.tokenize(req).await?;
    if let Some(encoding) = encoding {
        let tokens: Vec<SimpleToken> = encoding
            .get_ids()
            .iter()
            .zip(encoding.get_offsets())
            .map(|(&id, &(start, stop))| {
                let text: String = input.chars().skip(start).take(stop - start).collect();
                SimpleToken {
                    id,
                    text,
                    start,
                    stop,
                }
            })
            .collect();
        Ok(Json(TokenizeResponse(tokens)))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "No fast tokenizer or tokenizer.json for this model".to_string(),
                error_type: "no fast tokenizer".to_string(),
            }),
        ))
    }
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

#[derive(Clone, Debug)]
pub(crate) struct ComputeType(String);

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_info: HubModelInfo,
    shard_info: ShardInfo,
    compat_return_full_text: bool,
    max_concurrent_requests: usize,
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_length: usize,
    max_total_tokens: usize,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    max_batch_size: Option<usize>,
    client: ShardedClient,
    tokenizer: Option<Tokenizer>,
    config: Option<Config>,
    validation_workers: usize,
    addr: SocketAddr,
    allow_origin: Option<AllowOrigin>,
    ngrok: bool,
    ngrok_authtoken: Option<String>,
    ngrok_edge: Option<String>,
    tokenizer_config: HubTokenizerConfig,
    messages_api_enabled: bool,
    grammar_support: bool,
    max_client_batch_size: usize,
) -> Result<(), axum::BoxError> {
    // OpenAPI documentation
    #[derive(OpenApi)]
    #[openapi(
    paths(
    health,
    get_model_info,
    compat_generate,
    generate,
    generate_stream,
    chat_completions,
    completions,
    tokenize,
    metrics,
    ),
    components(
    schemas(
    Info,
    CompatGenerateRequest,
    GenerateRequest,
    GrammarType,
    ChatRequest,
    Message,
    ChatCompletionComplete,
    ChatCompletionChoice,
    ChatCompletionDelta,
    ChatCompletionChunk,
    ChatCompletionLogprob,
    ChatCompletionLogprobs,
    ChatCompletionTopLogprob,
    ChatCompletion,
    CompletionRequest,
    CompletionComplete,
    CompletionCompleteChunk,
    GenerateParameters,
    PrefillToken,
    Token,
    GenerateResponse,
    TokenizeResponse,
    SimpleToken,
    BestOfSequence,
    Details,
    FinishReason,
    StreamResponse,
    StreamDetails,
    ErrorResponse,
    GrammarType,
    Usage,
    DeltaToolCall,
    ToolType,
    Tool,
    ToolCall,
    Function,
    FunctionDefinition,
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
        config,
        max_best_of,
        max_stop_sequences,
        max_top_n_tokens,
        max_input_length,
        max_total_tokens,
        grammar_support,
    );
    let generation_health = Arc::new(AtomicBool::new(false));
    let health_ext = Health::new(client.clone(), generation_health.clone());
    let infer = Infer::new(
        client,
        validation,
        waiting_served_ratio,
        max_batch_prefill_tokens,
        max_batch_total_tokens,
        max_waiting_tokens,
        max_batch_size,
        max_concurrent_requests,
        shard_info.requires_padding,
        shard_info.window_size,
        shard_info.speculate,
        generation_health,
        tokenizer_config,
    );

    // Duration buckets
    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let n_duration_buckets = 35;
    let mut duration_buckets = Vec::with_capacity(n_duration_buckets);
    // Minimum duration in seconds
    let mut value = 0.0001;
    for _ in 0..n_duration_buckets {
        // geometric sequence
        value *= 1.5;
        duration_buckets.push(value);
    }
    // Input Length buckets
    let input_length_matcher = Matcher::Full(String::from("tgi_request_input_length"));
    let input_length_buckets: Vec<f64> = (0..100)
        .map(|x| (max_input_length as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Generated tokens buckets
    let generated_tokens_matcher = Matcher::Full(String::from("tgi_request_generated_tokens"));
    let generated_tokens_buckets: Vec<f64> = (0..100)
        .map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Input Length buckets
    let max_new_tokens_matcher = Matcher::Full(String::from("tgi_request_max_new_tokens"));
    let max_new_tokens_buckets: Vec<f64> = (0..100)
        .map(|x| (max_total_tokens as f64 / 100.0) * (x + 1) as f64)
        .collect();
    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("tgi_batch_next_size"));
    let batch_size_buckets: Vec<f64> = (0..1024).map(|x| (x + 1) as f64).collect();
    // Speculated tokens buckets
    let skipped_matcher = Matcher::Full(String::from("tgi_request_skipped_tokens"));
    let skipped_buckets: Vec<f64> = (0..shard_info.speculate + 1).map(|x| x as f64).collect();

    // Prometheus handler
    let builder = PrometheusBuilder::new()
        .set_buckets_for_metric(duration_matcher, &duration_buckets)
        .unwrap()
        .set_buckets_for_metric(input_length_matcher, &input_length_buckets)
        .unwrap()
        .set_buckets_for_metric(generated_tokens_matcher, &generated_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(max_new_tokens_matcher, &max_new_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)
        .unwrap()
        .set_buckets_for_metric(skipped_matcher, &skipped_buckets)
        .unwrap();
    let prom_handle = builder
        .install_recorder()
        .expect("failed to install metrics recorder");

    // CORS layer
    let allow_origin = allow_origin.unwrap_or(AllowOrigin::any());
    let cors_layer = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([http::header::CONTENT_TYPE])
        .allow_origin(allow_origin);

    // Endpoint info
    let info = Info {
        model_id: model_info.model_id,
        model_sha: model_info.sha,
        model_dtype: shard_info.dtype,
        model_device_type: shard_info.device_type,
        model_pipeline_tag: model_info.pipeline_tag,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
        waiting_served_ratio,
        max_batch_total_tokens,
        max_waiting_tokens,
        max_batch_size,
        validation_workers,
        max_client_batch_size,
        version: env!("CARGO_PKG_VERSION"),
        sha: option_env!("VERGEN_GIT_SHA"),
        docker_label: option_env!("DOCKER_LABEL"),
    };

    // Define VertextApiDoc conditionally only if the "google" feature is enabled
    let doc = {
        // avoid `mut` if possible
        #[cfg(feature = "google")]
        {
            use crate::VertexInstance;

            #[derive(OpenApi)]
            #[openapi(
                paths(vertex_compatibility),
                components(schemas(VertexInstance, VertexRequest, VertexResponse))
            )]
            struct VertextApiDoc;

            // limiting mutability to the smallest scope necessary
            let mut doc = ApiDoc::openapi();
            doc.merge(VertextApiDoc::openapi());
            doc
        }
        #[cfg(not(feature = "google"))]
        ApiDoc::openapi()
    };

    // Configure Swagger UI
    let swagger_ui = SwaggerUi::new("/docs").url("/api-doc/openapi.json", doc);

    // Define base and health routes
    let base_routes = Router::new()
        .route("/", post(compat_generate))
        .route("/", get(health))
        .route("/info", get(get_model_info))
        .route("/generate", post(generate))
        .route("/generate_stream", post(generate_stream))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/vertex", post(vertex_compatibility))
        .route("/tokenize", post(tokenize))
        .route("/health", get(health))
        .route("/ping", get(health))
        .route("/metrics", get(metrics));

    // Conditional AWS Sagemaker route
    let aws_sagemaker_route = if messages_api_enabled {
        Router::new().route("/invocations", post(chat_completions)) // Use 'chat_completions' for OAI_ENABLED
    } else {
        Router::new().route("/invocations", post(compat_generate)) // Use 'compat_generate' otherwise
    };

    let compute_type =
        ComputeType(std::env::var("COMPUTE_TYPE").unwrap_or("gpu+optimized".to_string()));

    // Combine routes and layers
    let mut app = Router::new()
        .merge(swagger_ui)
        .merge(base_routes)
        .merge(aws_sagemaker_route);

    #[cfg(feature = "google")]
    {
        tracing::info!("Built with `google` feature");
        tracing::info!(
            "Environment variables `AIP_PREDICT_ROUTE` and `AIP_HEALTH_ROUTE` will be respected."
        );
        if let Ok(env_predict_route) = std::env::var("AIP_PREDICT_ROUTE") {
            app = app.route(&env_predict_route, post(vertex_compatibility));
        }
        if let Ok(env_health_route) = std::env::var("AIP_HEALTH_ROUTE") {
            app = app.route(&env_health_route, get(health));
        }
    }

    // add layers after routes
    app = app
        .layer(Extension(info))
        .layer(Extension(health_ext.clone()))
        .layer(Extension(compat_return_full_text))
        .layer(Extension(infer))
        .layer(Extension(compute_type))
        .layer(Extension(prom_handle.clone()))
        .layer(OtelAxumLayer::default())
        .layer(cors_layer);

    if ngrok {
        #[cfg(feature = "ngrok")]
        {
            use ngrok::config::TunnelBuilder;

            let _ = addr;

            let authtoken =
                ngrok_authtoken.expect("`ngrok-authtoken` must be set when using ngrok tunneling");

            let edge = ngrok_edge.expect("`ngrok-edge` must be set when using ngrok tunneling");

            let tunnel = ngrok::Session::builder()
                .authtoken(authtoken)
                .connect()
                .await
                .unwrap()
                .labeled_tunnel()
                .label("edge", edge);

            let listener = tunnel.listen().await.unwrap();

            // Run prom metrics and health locally too
            tokio::spawn(
                axum::Server::bind(&addr)
                    .serve(
                        Router::new()
                            .route("/health", get(health))
                            .route("/metrics", get(metrics))
                            .layer(Extension(health_ext))
                            .layer(Extension(prom_handle))
                            .into_make_service(),
                    )
                    //Wait until all requests are finished to shut down
                    .with_graceful_shutdown(shutdown_signal()),
            );

            // Run server
            axum::Server::builder(listener)
                .serve(app.into_make_service())
                //Wait until all requests are finished to shut down
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        #[cfg(not(feature = "ngrok"))]
        {
            let _ngrok_authtoken = ngrok_authtoken;
            let _ngrok_domain = ngrok_domain;
            let _ngrok_username = ngrok_username;
            let _ngrok_password = ngrok_password;

            panic!("`text-generation-router` was compiled without the `ngrok` feature");
        }
    } else {
        // Run server
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            // Wait until all requests are finished to shut down
            .with_graceful_shutdown(shutdown_signal())
            .await?;
    }
    Ok(())
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
        let finish_reason = text_generation_client::FinishReason::try_from(finish_reason).unwrap();
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
            InferError::TemplateError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            InferError::ToolError(_) => StatusCode::UNPROCESSABLE_ENTITY,
        };

        (
            status_code,
            Json(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            }),
        )
    }
}

impl From<InferError> for Event {
    fn from(err: InferError) -> Self {
        Event::default()
            .json_data(ErrorResponse {
                error: err.to_string(),
                error_type: err.error_type().to_string(),
            })
            .unwrap()
    }
}
