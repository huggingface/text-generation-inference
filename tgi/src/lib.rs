use pyo3::{prelude::*, wrap_pyfunction};
use text_generation_launcher::{launcher_main, launcher_main_without_server};
use text_generation_router::internal_main;

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
        launcher_main(
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
fn fully_packaged(
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
        use std::thread;
        use tokio::runtime::Runtime;

        let model_id_clone = model_id.clone();
        let max_concurrent_requests_clone = max_concurrent_requests;
        let max_best_of_clone = max_best_of;
        let max_stop_sequences_clone = max_stop_sequences;
        let max_top_n_tokens_clone = max_top_n_tokens;
        let max_input_tokens_clone = max_input_tokens.unwrap_or(1024);
        let max_total_tokens_clone = max_total_tokens.unwrap_or(2048);
        let waiting_served_ratio_clone = waiting_served_ratio;

        let max_batch_prefill_tokens_clone = max_batch_prefill_tokens.unwrap_or(4096);
        let max_batch_total_tokens_clone = max_batch_total_tokens;
        let max_waiting_tokens_clone = max_waiting_tokens;
        let max_batch_size_clone = max_batch_size;
        let hostname_clone = hostname.clone();
        let port_clone = port;

        // TODO: fix this
        let _shard_uds_path_clone = shard_uds_path.clone();

        let tokenizer_config_path = tokenizer_config_path.clone();
        let revision = revision.clone();
        let validation_workers = validation_workers;
        let json_output = json_output;

        let otlp_endpoint = otlp_endpoint.clone();
        let cors_allow_origin = cors_allow_origin.clone();
        let ngrok = ngrok;
        let ngrok_authtoken = ngrok_authtoken.clone();
        let ngrok_edge = ngrok_edge.clone();
        let messages_api_enabled = true;
        let disable_grammar_support = disable_grammar_support;
        let max_client_batch_size = max_client_batch_size;

        let ngrok_clone = ngrok;
        let ngrok_authtoken_clone = ngrok_authtoken.clone();
        let ngrok_edge_clone = ngrok_edge.clone();
        let messages_api_enabled_clone = messages_api_enabled;
        let disable_grammar_support_clone = disable_grammar_support;
        let max_client_batch_size_clone = max_client_batch_size;

        let tokenizer_config_path_clone = tokenizer_config_path.clone();
        let revision_clone = revision.clone();
        let validation_workers_clone = validation_workers;
        let json_output_clone = json_output;
        let otlp_endpoint_clone = otlp_endpoint.clone();

        let webserver_callback = move || {
            let handle = thread::spawn(move || {
                let rt = Runtime::new().unwrap();
                rt.block_on(async {
                    internal_main(
                        max_concurrent_requests_clone,
                        max_best_of_clone,
                        max_stop_sequences_clone,
                        max_top_n_tokens_clone,
                        max_input_tokens_clone,
                        max_total_tokens_clone,
                        waiting_served_ratio_clone,
                        max_batch_prefill_tokens_clone,
                        max_batch_total_tokens_clone,
                        max_waiting_tokens_clone,
                        max_batch_size_clone,
                        hostname_clone,
                        port_clone,
                        "/tmp/text-generation-server-0".to_string(),
                        model_id_clone,
                        tokenizer_config_path_clone,
                        revision_clone,
                        validation_workers_clone,
                        json_output_clone,
                        otlp_endpoint_clone,
                        None,
                        ngrok_clone,
                        ngrok_authtoken_clone,
                        ngrok_edge_clone,
                        messages_api_enabled_clone,
                        disable_grammar_support_clone,
                        max_client_batch_size_clone,
                    )
                    .await
                })
            });
            match handle.join() {
                Ok(_) => println!("Server exited successfully"),
                Err(e) => println!("Server exited with error: {:?}", e),
            }
            Ok(())
        };

        // parse the arguments and run the main function
        launcher_main_without_server(
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
            Box::new(webserver_callback),
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

// TODO: remove hardcoding
#[pyfunction]
fn rust_server(py: Python<'_>) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async {
        let _ = internal_main(
            128,                                             // max_concurrent_requests: usize,
            2,                                               // max_best_of: usize,
            4,                                               // max_stop_sequences: usize,
            5,                                               // max_top_n_tokens: u32,
            1024,                                            // max_input_tokens: usize,
            2048,                                            // max_total_tokens: usize,
            1.2,                                             // waiting_served_ratio: f32,
            4096,                                            // max_batch_prefill_tokens: u32,
            None,                                            // max_batch_total_tokens: Option<u32>,
            20,                                              // max_waiting_tokens: usize,
            None,                                            // max_batch_size: Option<usize>,
            "0.0.0.0".to_string(),                           // hostname: String,
            3000,                                            // port: u16,
            "/tmp/text-generation-server-0".to_string(),     // master_shard_uds_path: String,
            "llava-hf/llava-v1.6-mistral-7b-hf".to_string(), // tokenizer_name: String,
            None,  // tokenizer_config_path: Option<String>,
            None,  // revision: Option<String>,
            2,     // validation_workers: usize,
            false, // json_output: bool,
            None,  // otlp_endpoint: Option<String>,
            None,  // cors_allow_origin: Option<Vec<String>>,
            false, // ngrok: bool,
            None,  // ngrok_authtoken: Option<String>,
            None,  // ngrok_edge: Option<String>,
            false, // messages_api_enabled: bool,
            false, // disable_grammar_support: bool,
            4,     // max_client_batch_size: usize,
        )
        .await;
        Ok(Python::with_gil(|py| py.None()))
    })
}

#[pymodule]
fn tgi(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_sleep, m)?)?;
    m.add_function(wrap_pyfunction!(rust_server, m)?)?;
    m.add_function(wrap_pyfunction!(rust_launcher, m)?)?;
    m.add_function(wrap_pyfunction!(fully_packaged, m)?)?;
    Ok(())
}
