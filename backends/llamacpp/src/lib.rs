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
            temperature: 1.0f32,
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
        temperature: f32,
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

        /// Create a new llama.cpp worker
        ///
        /// # Arguments
        ///
        /// * `modelPath`: Path to the GGUF model file to load
        /// * `num_threads`: Number of threads the worker is allowed to spawn to run computations
        ///
        /// returns: Result<<unknown>, <unknown>>
        ///
        /// # Examples
        ///
        /// ```
        ///
        /// ```
        fn create_worker_frontend(
            modelPath: &str,
            num_threads: u32,
        ) -> Result<UniquePtr<LlamaCppWorkerFrontend>>;

        /// Define the NUMA cores affinity on which the current thread is allowed to be scheduled.
        ///
        /// # Arguments
        ///
        /// * `affinity`: Set of CPU cores allowed for scheduling
        ///
        /// returns: ()
        ///
        /// # Examples
        ///
        /// ```
        /// // Bind the current thread for execution on cores 0, 1, 2, 3
        /// set_numa_core_affinity(&[0, 1, 2, 3]);
        /// ```
        fn set_numa_core_affinity(affinity: &[usize]);

        /// Force llama.cpp to reevaluate the allowed NUMA context (core and memory affinity) for
        /// its internal threads scheduling.
        /// This method can potentially cause llama.cpp / ggml to reallocate its internal threadpool to
        /// match the new affinity constraints
        ///
        /// returns: ()
        ///
        /// # Examples
        ///
        /// ```
        /// set_numa_core_affinity(&[0, 1, 2, 3]);
        /// update_numa_affinity();
        /// ```
        fn update_numa_affinity();

        /// Generate new tokens from the provided prompt input `tokens` and generation and sampling parameters,
        /// streaming back each generated individual token through the `callback`.
        ///
        /// # Arguments
        ///
        /// * `tokens`: Prompt input tokenized from the request's text input
        /// * `generation_params`: Parameters controling the generation loop
        /// * `sampling_params`: Parameters controling the sampling from the token distribution
        /// * `stream`: Opaque structure mapping HTTP client transport to stream back token
        /// * `callback`: Function pointer called everytime a new token is generated
        ///
        /// returns: Result<usize, <unknown>>
        ///
        /// # Examples
        ///
        /// ```
        ///
        /// ```
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
