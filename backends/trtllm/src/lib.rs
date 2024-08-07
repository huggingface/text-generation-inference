pub use backend::{GenerationContext, TensorRtLlmBackend};

mod backend;
pub mod errors;

#[cxx::bridge(namespace = "huggingface::tgi::backends")]
mod ffi {

    /// Struct used as shared type between rust and C++ to represent the result
    /// of a single decoding iteration
    pub struct GenerationStep {
        token_id: u32,
        log_prob: f32,
        is_final: bool,
        has_error: bool,
        error_msg: String,
    }

    extern "Rust" {
        type GenerationContext;
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
        ) -> UniquePtr<TensorRtLlmBackendImpl>;

        // #[rust_name = "is_ready"]
        // fn IsReady(self: &TensorRtLlmBackendImpl) -> bool;

        #[rust_name = "num_responses_ready"]
        fn NumResponsesReady(self: &TensorRtLlmBackendImpl) -> usize;

        #[rust_name = "submit"]
        fn Submit(
            self: Pin<&mut TensorRtLlmBackendImpl>,
            tokens: &[u32],
            top_k: i32,
            top_p: f32,
            temperature: f32,
            repetition_penalty: f32,
            frequency_penalty: f32,
            seed: u64,
        ) -> u64;

        #[rust_name = "stream_tokens"]
        unsafe fn StreamTokens(
            self: Pin<&mut TensorRtLlmBackendImpl>,
            request_id: u64,
            ctx: *mut GenerationContext,
            cb: unsafe fn(*mut GenerationContext, GenerationStep),
        ) -> usize;

        // #[rust_name = "shutdown"]
        // fn Shutdown(self: Pin<&mut TensorRtLlmBackendImpl>);
    }
}
