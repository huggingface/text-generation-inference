use std::cell::RefCell;
use std::path::Path;

use async_trait::async_trait;
use cxx::UniquePtr;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;

use text_generation_router::{FinishReason, Token};
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::{Chunk, ValidGenerateRequest, ValidParameters};

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{create_tensorrt_llm_backend, TensorRtLlmBackendImpl};

type InferResult<T> = Result<T, InferError>;

pub struct GenerationContext(mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>);

pub struct TrtLLmBackend {
    tokenizer: Tokenizer,
    inner: RefCell<UniquePtr<TensorRtLlmBackendImpl>>,
}

unsafe impl Sync for TrtLLmBackend {}
unsafe impl Send for TrtLLmBackend {}

impl TrtLLmBackend {
    pub fn new<P: AsRef<Path>>(
        tokenizer: Tokenizer,
        engine_folder: P,
    ) -> Result<Self, TensorRtLlmBackendError> {
        let engine_folder = engine_folder.as_ref();
        let inner = create_tensorrt_llm_backend(engine_folder.to_str().unwrap(), "");

        Ok(Self {
            tokenizer,
            inner: RefCell::new(inner),
        })
    }

    fn infer_text(
        &self,
        ctx: GenerationContext,
        text: &str,
        params: ValidParameters,
    ) -> InferResult<()> {
        // Keep track of processing time
        let start = Instant::now();

        // Encode the input
        let ctx = Box::new(ctx);
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| InferError::ToolError(e.to_string()))?;

        // Submit the request to the backend and retrieve the handle to query its status
        let request_id = self
            .inner
            .borrow_mut()
            .as_mut()
            .expect("Failed to retrieve pointer to TRTLLM backend")
            .submit(
                encoding.get_ids(),
                128,
                params.top_k as i32,
                params.top_p,
                params.temperature,
                params.seed,
            );

        // Stream generated tokens
        // spawn_blocking(move || {
        let num_generated_tokens = self
            .inner
            .borrow_mut()
            .as_mut()
            .expect("Failed to retrieve pointer to TRTLLM backend")
            .stream(ctx, request_id, |ctx, token, step, is_final| {
                // self.tokenizer.decode(&*[token], true).unwrap();
                let sender = ctx.0;
                let token = Token {
                    id: token,
                    text: String::from(""),
                    logprob: 1.0f32,
                    special: false,
                };

                sender
                    .send(Ok(InferStreamResponse::Intermediate {
                        token,
                        top_tokens: vec![],
                    }))
                    .unwrap()
            });

        // Notify the end
        let _ = ctx.0.send(Ok(InferStreamResponse::End {
            token: Token {
                id: 0,
                text: String::from(""),
                logprob: 1.0f32,
                special: false,
            },
            top_tokens: vec![],
            generated_text: GeneratedText {
                text: String::from(""),
                generated_tokens: num_generated_tokens,
                finish_reason: FinishReason::EndOfSequenceToken,
                seed: Some(params.seed),
            },
            start,
            queued: Instant::now(),
        }));
        // });

        Ok(())
    }
}

#[async_trait]
impl Backend for TrtLLmBackend {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> InferResult<UnboundedReceiverStream<InferResult<InferStreamResponse>>> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let ctx = GenerationContext(sender);

        // Unpack parameters
        let params = request.parameters;

        // Ensure we are running in the right conditions for the input (i.e. single textual chunk)
        let input = match request.inputs.len() {
            0 => Err(InferError::GenerationError("No input provided".into())),
            1 => Ok(request.inputs.first().unwrap()),
            _ => Err(InferError::GenerationError(format!(
                "Unsupported multi-chunks ({}) inference.",
                request.inputs.len()
            ))),
        }?;

        // Currently we handle single chunk of text
        match input {
            Chunk::Text(text) => {
                self.infer_text(ctx, &**text, params)?;
            }
            Chunk::Image(_) => panic!("Unsupported"),
        };

        Ok(UnboundedReceiverStream::new(receiver))
    }

    async fn health(&self, _current_health: bool) -> bool {
        self.inner.borrow_mut().is_ready()
    }
}
