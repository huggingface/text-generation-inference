use std::fmt::{Display, Formatter};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;

pub struct TensorRTBackend {}

#[async_trait]
impl Backend for TensorRTBackend {
    fn schedule(
        &self,
        _request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        let (_sender, receiver) = mpsc::unbounded_channel();

        Ok(UnboundedReceiverStream::new(receiver))
    }

    async fn health(&self, current_health: bool) -> bool {
        todo!()
    }
}

impl Display for TensorRTBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorRT-LLM Backend")
    }
}
