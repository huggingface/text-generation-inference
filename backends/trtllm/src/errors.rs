use std::path::PathBuf;
use thiserror::Error;

use text_generation_router::server;

#[derive(Debug, Error)]
pub enum TensorRtLlmBackendError {
    #[error("Provided engine folder {0} doesn't exist")]
    EngineFolderDoesntExists(PathBuf),
    #[error("Provided executorWorker binary path {0} doesn't exist")]
    ExecutorWorkerNotFound(PathBuf),
    #[error("TensorRT-LLM Runtime error: {0}")]
    Runtime(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Argument validation error: {0}")]
    ArgumentValidation(String),
    #[error("WebServer error: {0}")]
    WebServer(#[from] server::WebServerError),
    #[error("Tokio runtime failed to start: {0}")]
    Tokio(#[from] std::io::Error),
}
