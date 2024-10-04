pub mod backend;

#[cxx::bridge(namespace = "huggingface::tgi::backends::llama")]
mod ffi {
    unsafe extern "C++" {
        include!("backends/llamacpp/csrc/backend.cpp");

        /// Represent an instance of the llama.cpp backend instance on C++ side
        type LlamaCppBackendImpl;
    }
}
