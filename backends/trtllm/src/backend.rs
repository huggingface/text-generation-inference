use std::path::Path;

use async_trait::async_trait;
use cxx::UniquePtr;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{create_trtllm_backend, TensorRtLlmBackend};

pub struct TrtLLmBackend {
    tokenizer: Tokenizer,
    inner: UniquePtr<TensorRtLlmBackend>,
}

unsafe impl Sync for TrtLLmBackend {}
unsafe impl Send for TrtLLmBackend {}

impl TrtLLmBackend {
    pub fn new<P: AsRef<Path>>(
        tokenizer: Tokenizer,
        engine_folder: P,
    ) -> Result<Self, TensorRtLlmBackendError> {
        let engine_folder = engine_folder.as_ref();
        let inner = create_trtllm_backend(engine_folder.to_str().unwrap());

        Ok(Self { tokenizer, inner })
    }
}

#[async_trait]
impl Backend for TrtLLmBackend {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let request_id = self.inner.submit();

        Ok(UnboundedReceiverStream::new(receiver))
    }

    async fn health(&self, _current_health: bool) -> bool {
        self.inner.is_ready()
    }
}
