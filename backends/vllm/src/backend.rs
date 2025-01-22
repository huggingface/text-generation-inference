use crate::errors::VllmBackendError;
use crate::{EngineArgs, LlmEngine};
use async_trait::async_trait;
use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;
use tokio_stream::wrappers::UnboundedReceiverStream;

pub struct VllmBackend {
    engine: LlmEngine,
}

impl VllmBackend {
    pub fn from_engine_args(args: EngineArgs) -> Result<VllmBackend, VllmBackendError> {
        Ok(Self {
            engine: LlmEngine::from_engine_args(args)?,
        })
    }
}

#[async_trait]
impl Backend for VllmBackend {
    fn schedule(
        &self,
        _request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        todo!()
    }

    async fn health(&self, _current_health: bool) -> bool {
        true
    }
}
