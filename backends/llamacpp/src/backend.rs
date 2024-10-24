use crate::ffi::{create_llamacpp_backend, LlamaCppBackendImpl};
use async_trait::async_trait;
use cxx::UniquePtr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;
use thiserror::Error;
use tokio::task::spawn_blocking;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::info;

unsafe impl Send for LlamaCppBackendImpl {}

#[derive(Debug, Error)]
pub enum LlamaCppBackendError {
    #[error("Provided GGUF model path {0} doesn't exist")]
    ModelFileDoesntExist(String),

    #[error("Failed to initialize model from GGUF file {0}: {1}")]
    ModelInitializationFailed(PathBuf, String),
}

pub struct LlamaCppBackend {}

impl LlamaCppBackend {
    pub fn new<P: AsRef<Path> + Send>(model_path: P) -> Result<Self, LlamaCppBackendError> {
        let path = Arc::new(model_path.as_ref());
        if !path.exists() {
            return Err(LlamaCppBackendError::ModelFileDoesntExist(
                path.display().to_string(),
            ));
        }

        let mut backend = create_llamacpp_backend(path.to_str().unwrap()).map_err(|err| {
            LlamaCppBackendError::ModelInitializationFailed(
                path.to_path_buf(),
                err.what().to_string(),
            )
        })?;

        info!(
            "Successfully initialized llama.cpp backend from {}",
            path.display()
        );

        spawn_blocking(move || scheduler_loop(backend));
        Ok(Self {})
    }
}

async fn scheduler_loop(mut backend: UniquePtr<LlamaCppBackendImpl>) {}

#[async_trait]
impl Backend for LlamaCppBackend {
    fn schedule(
        &self,
        _request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        Err(InferError::GenerationError("Not implemented yet".into()))
    }

    async fn health(&self, _: bool) -> bool {
        true
    }
}
