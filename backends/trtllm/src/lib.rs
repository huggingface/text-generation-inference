pub use looper::TensorRtLlmBackendV2;

pub mod errors;
mod looper;
mod utils;

#[cxx::bridge(namespace = "huggingface::tgi::backends")]
mod ffi {
    /// Struct used as shared type between rust and C++ to represent the result
    /// of a single decoding iteration
    #[derive(Debug, Clone)]
    pub struct GenerationStep {
        request_id: u64,
        token_id: u32,
        log_prob: f32,
        is_final: bool,
        has_error: bool,
        error_msg: String,
    }

    unsafe extern "C++" {
        include!("backends/trtllm/src/ffi.cpp");

        /// Represent an instance of the underlying TensorRT-LLM backend
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
        #[rust_name = "create_tensorrt_llm_backend"]
        fn CreateTensorRtLlmBackend(
            engine_folder: &str,
            executor_worker: &str,
        ) -> Result<UniquePtr<TensorRtLlmBackendImpl>>;

        #[rust_name = "num_responses_ready"]
        fn NumResponsesReady(self: &TensorRtLlmBackendImpl) -> usize;

        #[rust_name = "submit"]
        fn Submit(
            self: Pin<&mut TensorRtLlmBackendImpl>,
            tokens: &[u32],
            max_new_tokens: u32,
            top_k: i32,
            top_p: f32,
            temperature: f32,
            repetition_penalty: f32,
            frequency_penalty: f32,
            seed: u64,
        ) -> Result<u64>;

        #[rust_name = "pull_tokens"]
        fn PullTokens(
            self: Pin<&mut TensorRtLlmBackendImpl>,
        ) -> Result<UniquePtr<CxxVector<GenerationStep>>>;
    }
}
