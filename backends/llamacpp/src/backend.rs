use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;
use tokio_stream::wrappers::UnboundedReceiverStream;

pub struct TgiLlamaCppBakend {}

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
