use std::cell::RefCell;
use std::path::Path;

use async_trait::async_trait;
use cxx::UniquePtr;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;

use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::{Chunk, ValidGenerateRequest};

use crate::errors::TensorRtLlmBackendError;
use crate::ffi::{create_trtllm_backend, TensorRtLlmBackendImpl};

struct GenerationContext(mpsc::UnboundedSender<Result<InferStreamResponse, InferError>>);

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
        let inner = create_trtllm_backend(engine_folder.to_str().unwrap(), "");

        Ok(Self {
            tokenizer,
            inner: RefCell::new(inner),
        })
    }
}

#[async_trait]
impl Backend for TrtLLmBackend {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let ctx = Box::new(GenerationContext(sender));

        // Unpack parameters
        let params = request.parameters;

        // Currently we handle single chunk of text
        if request.inputs.len() == 1 {
            match request
                .inputs
                .first()
                .expect("Failed to access the first chunk")
            {
                Chunk::Text(text) => {
                    let encoding = self
                        .tokenizer
                        .encode(&**text, true)
                        .map_err(|e| InferError::ToolError(e.to_string()))?;

                    let _start = Instant::now();
                    let _request_id = self
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

                    // spawn_blocking(|| {
                    //     // Stream generated tokens
                    //     let num_generated_tokens = self
                    //         .inner
                    //         .borrow_mut()
                    //         .as_mut()
                    //         .expect("Failed to retrieve pointer to TRTLLM backend")
                    //         .stream(request_id, ctx, |token, step, is_final| {
                    //             // self.tokenizer.decode(&*[token], true).unwrap();
                    //             let token = Token {
                    //                 id: token,
                    //                 text: String::from(""),
                    //                 logprob: 1.0f32,
                    //                 special: false,
                    //             };
                    //
                    //             sender
                    //                 .send(Ok(InferStreamResponse::Intermediate {
                    //                     token,
                    //                     top_tokens: vec![],
                    //                 }))
                    //                 .unwrap()
                    //         });
                    //
                    //     // Notify the end
                    //     Ok(InferStreamResponse::End {
                    //         token: Token {
                    //             id: 0,
                    //             text: String::from(""),
                    //             logprob: 1.0f32,
                    //             special: false,
                    //         },
                    //         top_tokens: vec![],
                    //         generated_text: GeneratedText {
                    //             text: String::from(""),
                    //             generated_tokens: num_generated_tokens,
                    //             finish_reason: FinishReason::EndOfSequenceToken,
                    //             seed: Some(params.seed),
                    //         },
                    //         start,
                    //         queued: Instant::now(),
                    //     })
                    // });
                }
                Chunk::Image(_) => {}
            }
        };

        Ok(UnboundedReceiverStream::new(receiver))
    }

    async fn health(&self, _current_health: bool) -> bool {
        self.inner.borrow_mut().is_ready()
    }
}
