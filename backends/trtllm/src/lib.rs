pub use backend::TrtLLmBackend;

mod backend;
pub mod errors;

#[cxx::bridge(namespace = "huggingface::tgi::backends")]
mod ffi {
    unsafe extern "C++" {
        include!("backends/trtllm/src/ffi.cpp");

        type TensorRtLlmBackend;

        fn create_trtllm_backend(engine_folder: &str) -> UniquePtr<TensorRtLlmBackend>;

        #[rust_name = "is_ready"]
        fn IsReady(&self) -> bool;

        #[rust_name = "submit"]
        fn Submit(&self) -> u64;
    }
}
