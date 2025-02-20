use crate::infer::Infer;
use crate::server::{chat_completions, compat_generate, completions, ComputeType};
use crate::{
    ChatCompletion, ChatCompletionChunk, ChatRequest, Chunk, CompatGenerateRequest,
    CompletionFinal, CompletionRequest, ErrorResponse, GenerateResponse, Info, StreamResponse,
};
use axum::extract::Extension;
use axum::http::StatusCode;
use axum::response::Response;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use utoipa::ToSchema;

#[derive(Clone, Deserialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum SagemakerRequest {
    Generate(CompatGenerateRequest),
    Chat(ChatRequest),
    Completion(CompletionRequest),
}

// Used for OpenAPI specs
#[allow(dead_code)]
#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum SagemakerResponse {
    Generate(GenerateResponse),
    Chat(ChatCompletion),
    Completion(CompletionFinal),
}

// Used for OpenAPI specs
#[allow(dead_code)]
#[derive(Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum SagemakerStreamResponse {
    Generate(StreamResponse),
    Chat(ChatCompletionChunk),
    Completion(Chunk),
}

/// Generate tokens from Sagemaker request
#[utoipa::path(
post,
tag = "Text Generation Inference",
path = "/invocations",
request_body = SagemakerRequest,
responses(
(status = 200, description = "Generated Chat Completion",
content(
("application/json" = SagemakerResponse),
("text/event-stream" = SagemakerStreamResponse),
)),
(status = 424, description = "Generation Error", body = ErrorResponse,
example = json ! ({"error": "Request failed during generation", "error_type": "generation"})),
(status = 429, description = "Model is overloaded", body = ErrorResponse,
example = json ! ({"error": "Model is overloaded", "error_type": "overloaded"})),
(status = 422, description = "Input validation error", body = ErrorResponse,
example = json ! ({"error": "Input validation error", "error_type": "validation"})),
(status = 500, description = "Incomplete generation", body = ErrorResponse,
example = json ! ({"error": "Incomplete generation", "error_type": "incomplete_generation"})),
)
)]
#[instrument(skip_all)]
pub(crate) async fn sagemaker_compatibility(
    default_return_full_text: Extension<bool>,
    infer: Extension<Infer>,
    compute_type: Extension<ComputeType>,
    info: Extension<Info>,
    Json(req): Json<SagemakerRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    match req {
        SagemakerRequest::Generate(req) => {
            compat_generate(default_return_full_text, infer, compute_type, Json(req)).await
        }
        SagemakerRequest::Chat(req) => chat_completions(infer, compute_type, info, Json(req)).await,
        SagemakerRequest::Completion(req) => {
            completions(infer, compute_type, info, Json(req)).await
        }
    }
}
