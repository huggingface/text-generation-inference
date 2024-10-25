pub mod backend;

#[cxx::bridge(namespace = "huggingface::tgi::backends::llamacpp::impl")]
mod ffi {
    unsafe extern "C++" {
        include!("backends/llamacpp/csrc/ffi.hpp");

        /// Represent an instance of the llama.cpp backend instance on C++ side
        type LlamaCppBackendImpl;

        #[rust_name = "create_llamacpp_backend"]
        fn CreateLlamaCppBackendImpl(
            modelPath: &str,
            n_threads: u16,
        ) -> Result<UniquePtr<LlamaCppBackendImpl>>;
    }
}
