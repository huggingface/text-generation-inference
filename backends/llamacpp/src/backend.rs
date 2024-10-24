use crate::ffi::{create_llamacpp_backend, LlamaCppBackendImpl};
use cxx::UniquePtr;
use std::path::Path;
use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;
use tokio_stream::wrappers::UnboundedReceiverStream;

pub struct TgiLlamaCppBakend {
    backend: UniquePtr<LlamaCppBackendImpl>,
}

impl TgiLlamaCppBakend {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, ()> {
        Ok(Self {
            backend: create_llamacpp_backend(model_path.as_ref().to_str().unwrap()),
        })
    }
}

impl Backend for TgiLlamaCppBakend {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        Err(InferError::GenerationError("Not implemented yet".into()))
    }

    async fn health(&self, current_health: bool) -> bool {
        todo!()
    }
}
