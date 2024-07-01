pub use backend::TrtLLmBackend;

mod backend;
pub mod errors;

#[cxx::bridge(namespace = "huggingface::tgi::backends")]
mod ffi {
    unsafe extern "C++" {
        include!("backends/trtllm/src/ffi.cpp");

        type TensorRtLlmBackend;

        fn create_trtllm_backend(engine_folder: &str) -> UniquePtr<TensorRtLlmBackend>;
    }
}
