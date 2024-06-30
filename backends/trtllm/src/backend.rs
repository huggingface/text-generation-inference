use tokio_stream::wrappers::UnboundedReceiverStream;

use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;

pub struct TensorRtLLmBackend {}

impl Backend for TensorRtLLmBackend {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        todo!()
    }

    async fn health(&self, current_health: bool) -> bool {
        todo!()
    }
}
