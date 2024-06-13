use crate::{
    default_parameters,
    server::{generate_internal, ComputeType},
    Deserialize, ErrorResponse, GenerateParameters, GenerateRequest, Infer, Serialize, ToSchema,
};
use axum::extract::{Extension, Path};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::FuturesUnordered;
use futures::TryStreamExt;
use reqwest::header::HeaderMap;
use reqwest::StatusCode;

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct OutputChunk {
    pub name: String,
    pub shape: Vec<usize>,
    pub datatype: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct InferenceOutput {
    pub id: String,
    pub outputs: Vec<OutputChunk>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct InferenceRequest {
    pub id: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub(crate) struct Input {
    pub name: String,
    pub shape: Vec<usize>,
    pub datatype: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub(crate) struct Output {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct LiveResponse {
    pub live: bool,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ReadyResponse {
    pub live: bool,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MetadataServerResponse {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

// Routes

#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/v2/health/live",
    responses(
        (status = 200, description = "Service is live", body = LiveReponse),
        (status = 404, description = "Service not found", body = ErrorResponse,
            example = json!({"error": "No response"}))
    )
)]
pub async fn kserve_health_live() -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let data = LiveResponse { live: true };
    Ok((HeaderMap::new(), Json(data)).into_response())
}

#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/v2/health/ready",
    responses(
        (status = 200, description = "Service is ready", body = ReadyResponse),
        (status = 404, description = "Service not found", body = ErrorResponse,
            example = json!({"error": "No response"}))
    )
)]
pub async fn kserve_health_ready() -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let data = ReadyResponse { live: true };
    Ok((HeaderMap::new(), Json(data)).into_response())
}

#[utoipa::path(
    get,
    tag = "Text Generation Inference",
    path = "/v2",
    responses(
        (status = 200, description = "Metadata retrieved", body = MetadataServerResponse),
        (status = 404, description = "Service not found", body = ErrorResponse,
            example = json!({"error": "No response"}))
    )
)]
pub async fn kerve_server_metadata() -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let data = MetadataServerResponse {
        name: "text-generation-inference".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        extensions: vec![
            "health".to_string(),
            "models".to_string(),
            "metrics".to_string(),
        ],
    };
    Ok((HeaderMap::new(), Json(data)).into_response())
}

#[utoipa::path(
    get,
    tag = "Text Generation Inference",
    path = "/v2/models/{model_name}/versions/{model_version}",
    responses(
        (status = 200, description = "Model version metadata retrieved", body = MetadataServerResponse),
        (status = 404, description = "Model or version not found", body = ErrorResponse,
            example = json!({"error": "No response"}))
    )
)]
pub async fn kserve_model_metadata(
    Path((model_name, model_version)): Path<(String, String)>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let data = MetadataServerResponse {
        name: model_name,
        version: model_version,
        extensions: vec!["infer".to_string(), "ready".to_string()],
    };
    Ok((HeaderMap::new(), Json(data)).into_response())
}

#[utoipa::path(
    post,
    tag = "Text Generation Inference",
    path = "/v2/models/{model_name}/versions/{model_version}/infer",
    request_body = Json<InferenceRequest>,
    responses(
        (status = 200, description = "Inference executed successfully", body = InferenceOutput),
        (status = 404, description = "Model or version not found", body = ErrorResponse,
            example = json!({"error": "No response"}))
    )
)]
pub async fn kserve_model_infer(
    infer: Extension<Infer>,
    Extension(compute_type): Extension<ComputeType>,
    Json(payload): Json<InferenceRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let id = payload.id.clone();
    let str_inputs = payload
        .inputs
        .iter()
        .map(|input| {
            std::str::from_utf8(&input.data).map_err(|e| {
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(ErrorResponse {
                        error: e.to_string(),
                        error_type: "utf8".to_string(),
                    }),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if str_inputs.len() != payload.outputs.len() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: "Inputs and outputs length mismatch".to_string(),
                error_type: "length mismatch".to_string(),
            }),
        ));
    }

    let output_chunks = str_inputs
        .iter()
        .zip(&payload.outputs)
        .map(|(str_input, output)| {
            let generate_request = GenerateRequest {
                inputs: str_input.to_string(),
                parameters: payload.parameters.clone(),
            };
            let infer = infer.clone();
            let compute_type = compute_type.clone();
            let span = tracing::Span::current();
            async move {
                generate_internal(infer, compute_type, Json(generate_request), span)
                    .await
                    .map(|(_, Json(generation))| {
                        let generation_as_bytes = generation.generated_text.as_bytes().to_vec();
                        OutputChunk {
                            name: output.name.clone(),
                            shape: vec![1, generation_as_bytes.len()],
                            datatype: "BYTES".to_string(),
                            data: generation_as_bytes,
                        }
                    })
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

    let inference_output = InferenceOutput {
        id: id.clone(),
        outputs: output_chunks,
    };

    Ok((HeaderMap::new(), Json(inference_output)).into_response())
}

#[utoipa::path(
    get,
    tag = "Text Generation Inference",
    path = "/v2/models/{model_name}/versions/{model_version}/ready",
    responses(
        (status = 200, description = "Model version is ready", body = ReadyResponse),
        (status = 404, description = "Model or version not found", body = ErrorResponse,
            example = json!({"error": "No response"}))
    )
)]
pub async fn kserve_model_metadata_ready(
    Path((_model_name, _model_version)): Path<(String, String)>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let data = ReadyResponse { live: true };
    Ok((HeaderMap::new(), Json(data)).into_response())
}
