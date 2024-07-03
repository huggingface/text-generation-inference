pub use backend::TrtLLmBackend;

mod backend;
pub mod errors;

#[cxx::bridge(namespace = "huggingface::tgi::backends")]
mod ffi {
    unsafe extern "C++" {
        include!("backends/trtllm/src/ffi.cpp");

        type TensorRtLlmBackend;

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
        fn create_trtllm_backend(engine_folder: &str, executor_worker: &str) -> UniquePtr<TensorRtLlmBackend>;

        #[rust_name = "is_ready"]
        fn IsReady(&self) -> bool;

        #[rust_name = "submit"]
        fn Submit(&self) -> u64;
    }
}
