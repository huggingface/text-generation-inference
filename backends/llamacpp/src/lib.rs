use crate::backend::InferContext;
use crate::ffi::SamplingParams;

pub mod backend;

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            top_k: u32::MAX,
            top_p: 1.0f32,
            frequency_penalty: 0.0f32,
            repetition_penalty: 0.0f32,
            seed: 2014u64,
        }
    }
}

#[cxx::bridge(namespace = "huggingface::tgi::backends::llamacpp")]
mod ffi {
    #[derive(Debug, Copy, Clone)]
    struct GenerationParams {
        max_new_tokens: u32,
        ignore_eos_token: bool,
    }

    #[derive(Debug, Copy, Clone)]
    struct SamplingParams {
        top_k: u32,
        top_p: f32,
        frequency_penalty: f32,
        repetition_penalty: f32,
        seed: u64,
    }

    extern "Rust" {
        type InferContext<'a>;
    }

    unsafe extern "C++" {
        include!("backends/llamacpp/csrc/ffi.hpp");

        #[cxx_name = "generation_params_t"]
        type GenerationParams;

        #[cxx_name = "sampling_params_t"]
        type SamplingParams;

        /// Represent an instance of the llama.cpp backend instance on C++ side
        #[cxx_name = "llama_cpp_worker_frontend_t"]
        type LlamaCppWorkerFrontend;

        fn create_worker_frontend(modelPath: &str) -> Result<UniquePtr<LlamaCppWorkerFrontend>>;

        unsafe fn stream(
            self: Pin<&mut LlamaCppWorkerFrontend>,
            tokens: &[u32],
            generation_params: GenerationParams,
            sampling_params: &SamplingParams,
            stream: *mut InferContext,
            callback: unsafe fn(*mut InferContext, u32, f32, bool, usize) -> bool,
        ) -> Result<usize>;
    }
}
