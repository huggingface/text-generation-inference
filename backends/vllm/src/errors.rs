use pyo3::PyErr;
use text_generation_router::infer::InferError;
use text_generation_router::server::WebServerError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VllmBackendError {
    #[error("[Python] {0}")]
    Python(PyErr),

    #[error("[WebServer] {0}")]
    WebServer(WebServerError),
}

impl From<PyErr> for VllmBackendError {
    fn from(value: PyErr) -> Self {
        Self::Python(value)
    }
}

impl From<WebServerError> for VllmBackendError {
    fn from(value: WebServerError) -> Self {
        Self::WebServer(value)
    }
}

impl From<VllmBackendError> for InferError {
    fn from(value: VllmBackendError) -> Self {
        InferError::GenerationError(value.to_string())
    }
}
