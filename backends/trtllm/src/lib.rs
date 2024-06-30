mod backend;

#[cxx::bridge(namespace = "huggingface::tgi::backends")]
mod ffi {
    unsafe extern "C++" {
        include!("backends/trtllm/include/backend.h");

        type TensorRtLlmBackendImpl;

    }
}
