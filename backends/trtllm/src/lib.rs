pub use looper::TensorRtLlmBackendV2;

pub mod errors;
mod looper;
mod utils;

#[cxx::bridge(namespace = "huggingface::tgi::backends::trtllm")]
mod ffi {
    #[cxx_name = "finish_reason_t"]
    #[derive(Debug, Clone, Copy)]
    pub enum FinishReason {
        /// The request is not finished.
        #[cxx_name = "kNOT_FINISHED"]
        NotFinished = 0u8,

        /// The request finished because the end id was generated.
        #[cxx_name = "kEND_ID"]
        EndTokenId = 1u8,

        /// The request finished because a stop word was generated.
        #[cxx_name = "kSTOP_WORDS"]
        StopWords = 2u8,

        /// The request finished because the maximum number of tokens was reached.
        #[cxx_name = "kLENGTH"]
        MaxLength = 3u8,
    }

    /// Struct used as shared type between rust and C++ to represent the result
    /// of a single decoding iteration
    #[cxx_name = "generation_step_t"]
    #[derive(Debug, Clone)]
    pub struct GenerationStep {
        request_id: u64,
        token_id: u32,
        log_prob: f32,
        is_final: bool,
        finish_reason: FinishReason,
        has_error: bool,
        error_msg: String,
    }

    unsafe extern "C++" {
        include!("backends/trtllm/csrc/ffi.hpp");

        /// Represent an instance of the underlying TensorRT-LLM backend
        #[cxx_name = "tensorrt_llm_backend_t"]
        type TensorRtLlmBackendImpl;

        /// Create an instance backed behind a std::unique_ptr to manage the lifespan of the backend
        ///
        /// # Arguments
        ///
        /// * `engine_folder`: Path to the folder containing all the TRTLLM engines
        /// * `executor_worker`: Path to the TRTLLM executor worker
        ///
        /// returns: <unknown>
        ///
        /// # Examples
        ///
        /// ```
        ///
        /// ```
        fn create_backend_from_engine_folder(
            engine_folder: &str,
            executor_worker: &str,
        ) -> Result<UniquePtr<TensorRtLlmBackendImpl>>;

        fn num_tokens_ready(self: &TensorRtLlmBackendImpl) -> usize;

        fn submit(
            self: Pin<&mut TensorRtLlmBackendImpl>,
            tokens: &[u32],
            max_new_tokens: u32,
            top_k: u32,
            top_p: f32,
            temperature: f32,
            repetition_penalty: f32,
            frequency_penalty: f32,
            seed: u64,
        ) -> Result<u64>;

        fn pull_tokens(
            self: Pin<&mut TensorRtLlmBackendImpl>,
        ) -> Result<UniquePtr<CxxVector<GenerationStep>>>;

        fn cancel(self: Pin<&mut TensorRtLlmBackendImpl>, request_id: u64);
    }
}

use ffi::FinishReason;
use text_generation_router::FinishReason as InferFinishReason;

impl From<FinishReason> for InferFinishReason {
    fn from(reason: FinishReason) -> Self {
        match reason {
            FinishReason::StopWords => InferFinishReason::StopSequence,
            FinishReason::MaxLength => InferFinishReason::Length,
            FinishReason::EndTokenId => InferFinishReason::EndOfSequenceToken,
            _ => panic!("Cannot convert {reason:?} to text_generation_router::FinishReason"),
        }
    }
}
