use pyo3::{prelude::*, wrap_pyfunction};
use std::thread;
use text_generation_launcher::{internal_main, internal_main_args as internal_main_args_launcher};
use text_generation_router::internal_main_args as internal_main_args_router;
use tokio::runtime::Runtime;

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    model_id,
    revision,
    validation_workers,
    sharded,
    num_shard,
    _quantize,
    speculate,
    _dtype,
    trust_remote_code,
    max_concurrent_requests,
    max_best_of,
    max_stop_sequences,
    max_top_n_tokens,
    max_input_tokens,
    max_input_length,
    max_total_tokens,
    waiting_served_ratio,
    max_batch_prefill_tokens,
    max_batch_total_tokens,
    max_waiting_tokens,
    max_batch_size,
    cuda_graphs,
    hostname,
    port,
    shard_uds_path,
    master_addr,
    master_port,
    huggingface_hub_cache,
    weights_cache_override,
    disable_custom_kernels,
    cuda_memory_fraction,
    _rope_scaling,
    rope_factor,
    json_output,
    otlp_endpoint,
    cors_allow_origin,
    watermark_gamma,
    watermark_delta,
    ngrok,
    ngrok_authtoken,
    ngrok_edge,
    tokenizer_config_path,
    disable_grammar_support,
    env,
    max_client_batch_size,
))]
fn rust_launcher(
    py: Python<'_>,
    model_id: String,
    revision: Option<String>,
    validation_workers: usize,
    sharded: Option<bool>,
    num_shard: Option<usize>,
    _quantize: Option<String>, // Option<Quantization>,
    speculate: Option<usize>,
    _dtype: Option<String>, // Option<Dtype>,
    trust_remote_code: bool,
    max_concurrent_requests: usize,
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_tokens: Option<usize>,
    max_input_length: Option<usize>,
    max_total_tokens: Option<usize>,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: Option<u32>,
    max_batch_total_tokens: Option<u32>,
    max_waiting_tokens: usize,
    max_batch_size: Option<usize>,
    cuda_graphs: Option<Vec<usize>>,
    hostname: String,
    port: u16,
    shard_uds_path: String,
    master_addr: String,
    master_port: usize,
    huggingface_hub_cache: Option<String>,
    weights_cache_override: Option<String>,
    disable_custom_kernels: bool,
    cuda_memory_fraction: f32,
    _rope_scaling: Option<f32>, // Option<RopeScaling>,
    rope_factor: Option<f32>,
    json_output: bool,
    otlp_endpoint: Option<String>,
    cors_allow_origin: Vec<String>,
    watermark_gamma: Option<f32>,
    watermark_delta: Option<f32>,
    ngrok: bool,
    ngrok_authtoken: Option<String>,
    ngrok_edge: Option<String>,
    tokenizer_config_path: Option<String>,
    disable_grammar_support: bool,
    env: bool,
    max_client_batch_size: usize,
) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        internal_main(
            model_id,
            revision,
            validation_workers,
            sharded,
            num_shard,
            None,
            speculate,
            None,
            trust_remote_code,
            max_concurrent_requests,
            max_best_of,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_tokens,
            max_input_length,
            max_total_tokens,
            waiting_served_ratio,
            max_batch_prefill_tokens,
            max_batch_total_tokens,
            max_waiting_tokens,
            max_batch_size,
            cuda_graphs,
            hostname,
            port,
            shard_uds_path,
            master_addr,
            master_port,
            huggingface_hub_cache,
            weights_cache_override,
            disable_custom_kernels,
            cuda_memory_fraction,
            None,
            rope_factor,
            json_output,
            otlp_endpoint,
            cors_allow_origin,
            watermark_gamma,
            watermark_delta,
            ngrok,
            ngrok_authtoken,
            ngrok_edge,
            tokenizer_config_path,
            disable_grammar_support,
            env,
            max_client_batch_size,
        )
        .unwrap();

        Ok(Python::with_gil(|py| py.None()))
    })
}

/// Asynchronous sleep function.
#[pyfunction]
fn rust_sleep(py: Python<'_>) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async {
        tokio::time::sleep(std::time::Duration::from_secs(20)).await;
        Ok(Python::with_gil(|py| py.None()))
    })
}

#[pyfunction]
fn rust_router(_py: Python<'_>) -> PyResult<String> {
    let handle = thread::spawn(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async { internal_main_args_router().await })
    });
    match handle.join() {
        Ok(thread_output) => match thread_output {
            Ok(_) => println!("Inner server exited successfully"),
            Err(e) => println!("Inner server exited with error: {:?}", e),
        },
        Err(e) => {
            println!("Server exited with error: {:?}", e);
        }
    }
    Ok("Completed".to_string())
}

#[pyfunction]
fn rust_launcher_cli(_py: Python<'_>) -> PyResult<String> {
    match internal_main_args_launcher() {
        Ok(_) => println!("Server exited successfully"),
        Err(e) => println!("Server exited with error: {:?}", e),
    }
    Ok("Completed".to_string())
}

#[pymodule]
fn _tgi(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_sleep, m)?)?;
    m.add_function(wrap_pyfunction!(rust_router, m)?)?;
    m.add_function(wrap_pyfunction!(rust_launcher, m)?)?;
    m.add_function(wrap_pyfunction!(rust_launcher_cli, m)?)?;
    Ok(())
}
