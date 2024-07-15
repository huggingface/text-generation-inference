pub use backend::TensorRtLlmBackend;

use crate::backend::GenerationContext;

mod backend;
pub mod errors;

#[cxx::bridge(namespace = "huggingface::tgi::backends")]
mod ffi {
    extern "Rust" {
        type GenerationContext;
    }

    unsafe extern "C++" {
        include!("backends/trtllm/src/ffi.cpp");

        /// Represent an instance of the underlying TensorRT-LLM backend
        type TensorRtLlmBackendImpl;

        /// Create an instance backed behind an std::unique_ptr to manage the lifespan of the backend
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

        #[rust_name = "is_ready"]
        fn IsReady(self: &TensorRtLlmBackendImpl) -> bool;

        #[rust_name = "submit"]
        fn Submit(
            self: Pin<&mut TensorRtLlmBackendImpl>,
            tokens: &[u32],
            max_new_tokens: i32,
            top_k: i32,
            top_p: f32,
            temperature: f32,
            seed: u64,
        ) -> u64;

        #[rust_name = "stream"]
        fn Stream(
            self: Pin<&mut TensorRtLlmBackendImpl>,
            ctx: Box<GenerationContext>,
            request_id: u64,
            callback: fn(Box<GenerationContext>, u32, u32, bool),
        ) -> u32;

        #[rust_name = "shutdown"]
        fn Shutdown(self: Pin<&mut TensorRtLlmBackendImpl>);
    }
}
