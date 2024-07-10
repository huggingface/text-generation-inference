use thiserror::Error;

use text_generation_router::server;

#[derive(Debug, Error)]
pub enum TensorRtLlmBackendError {
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Argument validation error: {0}")]
    ArgumentValidation(String),
    #[error("WebServer error: {0}")]
    WebServer(#[from] server::WebServerError),
    #[error("Tokio runtime failed to start: {0}")]
    Tokio(#[from] std::io::Error),
}
