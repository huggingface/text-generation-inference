use crate::infer::Infer;
use crate::server::{generate_internal, ComputeType};
use crate::{ChatRequest, ErrorResponse, GenerateParameters, GenerateRequest};
use axum::extract::Extension;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use utoipa::ToSchema;

#[derive(Clone, Deserialize, ToSchema)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct GenerateVertexInstance {
    #[schema(example = "What is Deep Learning?")]
    pub inputs: String,
    #[schema(nullable = true, default = "null", example = "null")]
    pub parameters: Option<GenerateParameters>,
}

#[derive(Clone, Deserialize, ToSchema)]
#[cfg_attr(test, derive(Debug, PartialEq))]
#[serde(untagged)]
pub(crate) enum VertexInstance {
    Generate(GenerateVertexInstance),
    Chat(ChatRequest),
}

#[derive(Deserialize, ToSchema)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) struct VertexRequest {
    #[serde(rename = "instances")]
    pub instances: Vec<VertexInstance>,
}

#[derive(Clone, Deserialize, ToSchema, Serialize)]
pub(crate) struct VertexResponse {
    pub predictions: Vec<String>,
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
pub(crate) async fn vertex_compatibility(
    Extension(infer): Extension<Infer>,
    Extension(compute_type): Extension<ComputeType>,
    Json(req): Json<VertexRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let span = tracing::Span::current();
    metrics::counter!("tgi_request_count").increment(1);

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

    // Prepare futures for all instances
    let mut futures = Vec::with_capacity(req.instances.len());

    for instance in req.instances.into_iter() {
        let generate_request = match instance {
            VertexInstance::Generate(instance) => GenerateRequest {
                inputs: instance.inputs.clone(),
                add_special_tokens: true,
                parameters: GenerateParameters {
                    do_sample: true,
                    max_new_tokens: instance.parameters.as_ref().and_then(|p| p.max_new_tokens),
                    seed: instance.parameters.as_ref().and_then(|p| p.seed),
                    details: true,
                    decoder_input_details: true,
                    ..Default::default()
                },
            },
            VertexInstance::Chat(instance) => {
                let (generate_request, _using_tools): (GenerateRequest, bool) =
                    instance.try_into_generate(&infer)?;
                generate_request
            }
        };

        let infer_clone = infer.clone();
        let compute_type_clone = compute_type.clone();
        let span_clone = span.clone();

        futures.push(async move {
            generate_internal(
                Extension(infer_clone),
                compute_type_clone,
                Json(generate_request),
                span_clone,
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
        });
    }

    // execute all futures in parallel, collect results, returning early if any error occurs
    let results = futures::future::join_all(futures).await;
    let predictions: Result<Vec<_>, _> = results.into_iter().collect();
    let predictions = predictions?;

    let response = VertexResponse { predictions };
    Ok((HeaderMap::new(), Json(response)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, MessageContent};

    #[test]
    fn vertex_deserialization() {
        let string = serde_json::json!({

        "instances": [
            {
                "messages": [{"role": "user", "content": "What's Deep Learning?"}],
                "max_tokens": 128,
                "top_p": 0.95,
                "temperature": 0.7
            }
        ]

        });
        let request: VertexRequest = serde_json::from_value(string).expect("Can deserialize");
        assert_eq!(
            request,
            VertexRequest {
                instances: vec![VertexInstance::Chat(ChatRequest {
                    messages: vec![Message {
                        role: "user".to_string(),
                        content: MessageContent::SingleText("What's Deep Learning?".to_string()),
                        name: None,
                    },],
                    max_tokens: Some(128),
                    top_p: Some(0.95),
                    temperature: Some(0.7),
                    ..Default::default()
                })]
            }
        );
    }
}
