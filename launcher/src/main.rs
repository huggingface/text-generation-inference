/// Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

use clap::{Parser, ValueEnum};
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use serde::Deserialize;
use std::env;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Lines, Read};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::{mpsc, Arc};
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{fs, io};
use tracing_subscriber::EnvFilter;

mod env_runtime;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Quantization {
    /// 4 bit quantization. Requires a specific GTPQ quantized model:
    ///   https://hf.co/models?search=awq.
    /// Should replace GPTQ models whereever possible because of the better latency
    Awq,
    /// 8 bit quantization, doesn't require specific model.
    /// Should be a drop-in replacement to bitsandbytes with much better performance.
    /// Kernels are from https://github.com/NetEase-FuXi/EETQ.git
    Eetq,
    /// 4 bit quantization. Requires a specific GTPQ quantized model: https://hf.co/models?search=gptq.
    /// text-generation-inference will use exllama (faster) kernels whereever possible, and use
    /// triton kernel (wider support) when it's not.
    /// AWQ has faster kernels.
    Gptq,
    /// Bitsandbytes 8bit. Can be applied on any model, will cut the memory requirement in half,
    /// but it is known that the model will be much slower to run than the native f16.
    #[deprecated(
        since = "1.1.0",
        note = "Use `eetq` instead, which provides better latencies overall and is drop-in in most cases"
    )]
    Bitsandbytes,
    /// Bitsandbytes 4bit. Can be applied on any model, will cut the memory requirement by 4x,
    /// but it is known that the model will be much slower to run than the native f16.
    BitsandbytesNF4,
    /// Bitsandbytes 4bit. nf4 should be preferred in most cases but maybe this one has better
    /// perplexity performance for you model
    BitsandbytesFP4,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            Quantization::Bitsandbytes => {
                write!(f, "bitsandbytes")
            }
            Quantization::BitsandbytesNF4 => {
                write!(f, "bitsandbytes-nf4")
            }
            Quantization::BitsandbytesFP4 => {
                write!(f, "bitsandbytes-fp4")
            }
            Quantization::Gptq => {
                write!(f, "gptq")
            }
            Quantization::Awq => {
                write!(f, "awq")
            }
            Quantization::Eetq => {
                write!(f, "eetq")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    Float16,
    #[clap(name = "bfloat16")]
    BFloat16,
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            Dtype::Float16 => {
                write!(f, "float16")
            }
            Dtype::BFloat16 => {
                write!(f, "bfloat16")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RopeScaling {
    Linear,
    Dynamic,
}

impl std::fmt::Display for RopeScaling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            RopeScaling::Linear => {
                write!(f, "linear")
            }
            RopeScaling::Dynamic => {
                write!(f, "dynamic")
            }
        }
    }
}

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The name of the model to load.
    /// Can be a MODEL_ID as listed on <https://hf.co/models> like
    /// `gpt2` or `OpenAssistant/oasst-sft-1-pythia-12b`.
    /// Or it can be a local directory containing the necessary files
    /// as saved by `save_pretrained(...)` methods of transformers
    #[clap(default_value = "bigscience/bloom-560m", long, env)]
    model_id: String,

    /// The actual revision of the model if you're referring to a model
    /// on the hub. You can use a specific commit id or a branch like `refs/pr/2`.
    #[clap(long, env)]
    revision: Option<String>,

    /// The number of tokenizer workers used for payload validation and truncation inside the
    /// router.
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,

    /// Whether to shard the model across multiple GPUs
    /// By default text-generation-inference will use all available GPUs to run
    /// the model. Setting it to `false` deactivates `num_shard`.
    #[clap(long, env)]
    sharded: Option<bool>,

    /// The number of shards to use if you don't want to use all GPUs on a given machine.
    /// You can use `CUDA_VISIBLE_DEVICES=0,1 text-generation-launcher... --num_shard 2`
    /// and `CUDA_VISIBLE_DEVICES=2,3 text-generation-launcher... --num_shard 2` to
    /// launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance.
    #[clap(long, env)]
    num_shard: Option<usize>,

    /// Whether you want the model to be quantized.
    #[clap(long, env, value_enum)]
    quantize: Option<Quantization>,

    /// The dtype to be forced upon the model. This option cannot be used with `--quantize`.
    #[clap(long, env, value_enum)]
    dtype: Option<Dtype>,

    /// Whether you want to execute hub modelling code. Explicitly passing a `revision` is
    /// encouraged when loading a model with custom code to ensure no malicious code has been
    /// contributed in a newer revision.
    #[clap(long, env, value_enum)]
    trust_remote_code: bool,

    /// The maximum amount of concurrent requests for this particular deployment.
    /// Having a low limit will refuse clients requests instead of having them
    /// wait for too long and is usually good to handle backpressure correctly.
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,

    /// This is the maximum allowed value for clients to set `best_of`.
    /// Best of makes `n` generations at the same time, and return the best
    /// in terms of overall log probability over the entire generated sequence
    #[clap(default_value = "2", long, env)]
    max_best_of: usize,

    /// This is the maximum allowed value for clients to set `stop_sequences`.
    /// Stop sequences are used to allow the model to stop on more than just
    /// the EOS token, and enable more complex "prompting" where users can preprompt
    /// the model in a specific way and define their "own" stop token aligned with
    /// their prompt.
    #[clap(default_value = "4", long, env)]
    max_stop_sequences: usize,

    /// This is the maximum allowed value for clients to set `top_n_tokens`.
    /// `top_n_tokens is used to return information about the the `n` most likely
    /// tokens at each generation step, instead of just the sampled token. This
    /// information can be used for downstream tasks like for classification or
    /// ranking.
    #[clap(default_value = "5", long, env)]
    max_top_n_tokens: u32,

    /// This is the maximum allowed input length (expressed in number of tokens)
    /// for users. The larger this value, the longer prompt users can send which
    /// can impact the overall memory required to handle the load.
    /// Please note that some models have a finite range of sequence they can handle.
    #[clap(default_value = "1024", long, env)]
    max_input_length: usize,

    /// This is the most important value to set as it defines the "memory budget"
    /// of running clients requests.
    /// Clients will send input sequences and ask to generate `max_new_tokens`
    /// on top. with a value of `1512` users can send either a prompt of
    /// `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for
    /// `1511` max_new_tokens.
    /// The larger this value, the larger amount each request will be in your RAM
    /// and the less effective batching can be.
    #[clap(default_value = "2048", long, env)]
    max_total_tokens: usize,

    /// This represents the ratio of waiting queries vs running queries where
    /// you want to start considering pausing the running queries to include the waiting
    /// ones into the same batch.
    /// `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's
    /// only 10 queries left in the current batch we check if we can fit those 12
    /// waiting queries into the batching strategy, and if yes, then batching happens
    /// delaying the 10 running queries by a `prefill` run.
    ///
    /// This setting is only applied if there is room in the batch
    /// as defined by `max_batch_total_tokens`.
    #[clap(default_value = "1.2", long, env)]
    waiting_served_ratio: f32,

    /// Limits the number of tokens for the prefill operation.
    /// Since this operation take the most memory and is compute bound, it is interesting
    /// to limit the number of requests that can be sent.
    #[clap(default_value = "4096", long, env)]
    max_batch_prefill_tokens: u32,

    /// **IMPORTANT** This is one critical control to allow maximum usage
    /// of the available hardware.
    ///
    /// This represents the total amount of potential tokens within a batch.
    /// When using padding (not recommended) this would be equivalent of
    /// `batch_size` * `max_total_tokens`.
    ///
    /// However in the non-padded (flash attention) version this can be much finer.
    ///
    /// For `max_batch_total_tokens=1000`, you could fit `10` queries of `total_tokens=100`
    /// or a single query of `1000` tokens.
    ///
    /// Overall this number should be the largest possible amount that fits the
    /// remaining memory (after the model is loaded). Since the actual memory overhead
    /// depends on other parameters like if you're using quantization, flash attention
    /// or the model implementation, text-generation-inference cannot infer this number
    /// automatically.
    #[clap(long, env)]
    max_batch_total_tokens: Option<u32>,

    /// This setting defines how many tokens can be passed before forcing the waiting
    /// queries to be put on the batch (if the size of the batch allows for it).
    /// New queries require 1 `prefill` forward, which is different from `decode`
    /// and therefore you need to pause the running batch in order to run `prefill`
    /// to create the correct values for the waiting queries to be able to join the batch.
    ///
    /// With a value too small, queries will always "steal" the compute to run `prefill`
    /// and running queries will be delayed by a lot.
    ///
    /// With a value too big, waiting queries could wait for a very long time
    /// before being allowed a slot in the running batch. If your server is busy
    /// that means that requests that could run in ~2s on an empty server could
    /// end up running in ~20s because the query had to wait for 18s.
    ///
    /// This number is expressed in number of tokens to make it a bit more
    /// "model" agnostic, but what should really matter is the overall latency
    /// for end users.
    #[clap(default_value = "20", long, env)]
    max_waiting_tokens: usize,

    /// The IP address to listen on
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,

    /// The port to listen on.
    #[clap(default_value = "3000", long, short, env)]
    port: u16,

    /// The name of the socket for gRPC communication between the webserver
    /// and the shards.
    #[clap(default_value = "/tmp/text-generation-server", long, env)]
    shard_uds_path: String,

    /// The address the master shard will listen on. (setting used by torch distributed)
    #[clap(default_value = "localhost", long, env)]
    master_addr: String,

    /// The address the master port will listen on. (setting used by torch distributed)
    #[clap(default_value = "29500", long, env)]
    master_port: usize,

    /// The location of the huggingface hub cache.
    /// Used to override the location if you want to provide a mounted disk for instance
    #[clap(long, env)]
    huggingface_hub_cache: Option<String>,

    /// The location of the huggingface hub cache.
    /// Used to override the location if you want to provide a mounted disk for instance
    #[clap(long, env)]
    weights_cache_override: Option<String>,

    /// For some models (like bloom), text-generation-inference implemented custom
    /// cuda kernels to speed up inference. Those kernels were only tested on A100.
    /// Use this flag to disable them if you're running on different hardware and
    /// encounter issues.
    #[clap(long, env)]
    disable_custom_kernels: bool,

    /// Limit the CUDA available memory.
    /// The allowed value equals the total visible memory multiplied by cuda-memory-fraction.
    #[clap(default_value = "1.0", long, env)]
    cuda_memory_fraction: f32,

    /// Rope scaling will only be used for RoPE models
    /// and allow rescaling the position rotary to accomodate for
    /// larger prompts.
    ///
    /// Goes together with `rope_factor`.
    ///
    /// `--rope-factor 2.0` gives linear scaling with a factor of 2.0
    /// `--rope-scaling dynamic` gives dynamic scaling with a factor of 1.0
    /// `--rope-scaling linear` gives linear scaling with a factor of 1.0 (Nothing will be changed
    /// basically)
    ///
    /// `--rope-scaling linear --rope-factor` fully describes the scaling you want
    #[clap(long, env)]
    rope_scaling: Option<RopeScaling>,

    /// Rope scaling will only be used for RoPE models
    /// See `rope_scaling`
    #[clap(long, env)]
    rope_factor: Option<f32>,

    /// Outputs the logs in JSON format (useful for telemetry)
    #[clap(long, env)]
    json_output: bool,

    #[clap(long, env)]
    otlp_endpoint: Option<String>,

    #[clap(long, env)]
    cors_allow_origin: Vec<String>,
    #[clap(long, env)]
    watermark_gamma: Option<f32>,
    #[clap(long, env)]
    watermark_delta: Option<f32>,

    /// Enable ngrok tunneling
    #[clap(long, env)]
    ngrok: bool,

    /// ngrok authentication token
    #[clap(long, env)]
    ngrok_authtoken: Option<String>,

    /// ngrok edge
    #[clap(long, env)]
    ngrok_edge: Option<String>,

    /// Display a lot of information about your runtime environment
    #[clap(long, short, action)]
    env: bool,
}

#[derive(Debug)]
enum ShardStatus {
    Ready,
    Failed(usize),
}

#[allow(clippy::too_many_arguments)]
fn shard_manager(
    model_id: String,
    revision: Option<String>,
    quantize: Option<Quantization>,
    dtype: Option<Dtype>,
    max_total_tokens: usize,
    trust_remote_code: bool,
    uds_path: String,
    rank: usize,
    world_size: usize,
    master_addr: String,
    master_port: usize,
    huggingface_hub_cache: Option<String>,
    weights_cache_override: Option<String>,
    disable_custom_kernels: bool,
    watermark_gamma: Option<f32>,
    watermark_delta: Option<f32>,
    cuda_memory_fraction: f32,
    rope_scaling: Option<RopeScaling>,
    rope_factor: Option<f32>,
    otlp_endpoint: Option<String>,
    status_sender: mpsc::Sender<ShardStatus>,
    shutdown: Arc<AtomicBool>,
    _shutdown_sender: mpsc::Sender<()>,
) {
    // Enter shard-manager tracing span
    let _span = tracing::span!(tracing::Level::INFO, "shard-manager", rank = rank).entered();

    // Get UDS path
    let uds_string = format!("{uds_path}-{rank}");
    let uds = Path::new(&uds_string);
    // Clean previous runs
    if uds.exists() {
        fs::remove_file(uds).unwrap();
    }

    // Process args
    let mut shard_args = vec![
        "serve".to_string(),
        model_id,
        "--uds-path".to_string(),
        uds_path,
        "--logger-level".to_string(),
        "INFO".to_string(),
        "--json-output".to_string(),
    ];

    // Activate trust remote code
    if trust_remote_code {
        shard_args.push("--trust-remote-code".to_string());
    }

    // Activate tensor parallelism
    if world_size > 1 {
        shard_args.push("--sharded".to_string());
    }

    if let Some(quantize) = quantize {
        shard_args.push("--quantize".to_string());
        shard_args.push(quantize.to_string())
    }

    if let Some(dtype) = dtype {
        shard_args.push("--dtype".to_string());
        shard_args.push(dtype.to_string())
    }

    // Model optional revision
    if let Some(revision) = revision {
        shard_args.push("--revision".to_string());
        shard_args.push(revision)
    }

    let rope = match (rope_scaling, rope_factor) {
        (None, None) => None,
        (Some(scaling), None) => Some((scaling, 1.0)),
        (Some(scaling), Some(factor)) => Some((scaling, factor)),
        (None, Some(factor)) => Some((RopeScaling::Linear, factor)),
    };
    // OpenTelemetry
    if let Some(otlp_endpoint) = otlp_endpoint {
        shard_args.push("--otlp-endpoint".to_string());
        shard_args.push(otlp_endpoint);
    }

    // Copy current process env
    let mut envs: Vec<(OsString, OsString)> = env::vars_os().collect();

    // Max total tokens
    envs.push(("MAX_TOTAL_TOKENS".into(), max_total_tokens.to_string().into()));

    // Torch Distributed Env vars
    if world_size == 1 {
        envs.push(("RANK".into(), rank.to_string().into()));
    }
    envs.push(("WORLD_SIZE".into(), world_size.to_string().into()));
    envs.push(("MASTER_ADDR".into(), master_addr.into()));
    envs.push(("MASTER_PORT".into(), master_port.to_string().into()));
    envs.push(("NCCL_ASYNC_ERROR_HANDLING".into(), "1".into()));

    // CUDA memory fraction
    envs.push((
        "CUDA_MEMORY_FRACTION".into(),
        cuda_memory_fraction.to_string().into(),
    ));

    // Safetensors load fast
    envs.push(("SAFETENSORS_FAST_GPU".into(), "1".into()));

    // Enable hf transfer for insane download speeds
    let enable_hf_transfer = env::var("HF_HUB_ENABLE_HF_TRANSFER").unwrap_or("1".to_string());
    envs.push((
        "HF_HUB_ENABLE_HF_TRANSFER".into(),
        enable_hf_transfer.into(),
    ));

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        envs.push(("HUGGING_FACE_HUB_TOKEN".into(), api_token.into()))
    };

    // Detect rope scaling
    // Sending as env instead of CLI args to not bloat everything
    // those only can be used by RoPE models, so passing information around
    // for all models will complexify code unnecessarily
    if let Some((scaling, factor)) = rope {
        envs.push(("ROPE_SCALING".into(), scaling.to_string().into()));
        envs.push(("ROPE_FACTOR".into(), factor.to_string().into()));
    }

    // If huggingface_hub_cache is some, pass it to the shard
    // Useful when running inside a docker container
    if let Some(huggingface_hub_cache) = huggingface_hub_cache {
        envs.push(("HUGGINGFACE_HUB_CACHE".into(), huggingface_hub_cache.into()));
    };

    // If weights_cache_override is some, pass it to the shard
    // Useful when running inside a HuggingFace Inference Endpoint
    if let Some(weights_cache_override) = weights_cache_override {
        envs.push((
            "WEIGHTS_CACHE_OVERRIDE".into(),
            weights_cache_override.into(),
        ));
    };

    // If disable_custom_kernels is true, pass it to the shard as an env var
    if disable_custom_kernels {
        envs.push(("DISABLE_CUSTOM_KERNELS".into(), "True".into()))
    }

    // Watermark Gamma
    if let Some(watermark_gamma) = watermark_gamma {
        envs.push(("WATERMARK_GAMMA".into(), watermark_gamma.to_string().into()))
    }

    // Watermark Delta
    if let Some(watermark_delta) = watermark_delta {
        envs.push(("WATERMARK_DELTA".into(), watermark_delta.to_string().into()))
    }

    // Start process
    tracing::info!("Starting shard");
    let mut p = match Command::new("text-generation-server")
        .args(shard_args)
        .envs(envs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                tracing::error!("text-generation-server not found in PATH");
                tracing::error!("Please install it with `make install-server`")
            }
            {
                tracing::error!("{}", err);
            }

            status_sender.send(ShardStatus::Failed(rank)).unwrap();
            return;
        }
    };

    // Redirect STDOUT to the console
    let shard_stdout_reader = BufReader::new(p.stdout.take().unwrap());
    let shard_stderr_reader = BufReader::new(p.stderr.take().unwrap());

    //stdout tracing thread
    thread::spawn(move || {
        log_lines(shard_stdout_reader.lines());
    });

    let mut ready = false;
    let start_time = Instant::now();
    let mut wait_time = Instant::now();
    loop {
        // Process exited
        if let Some(exit_status) = p.try_wait().unwrap() {
            // We read stderr in another thread as it seems that lines() can block in some cases
            let (err_sender, err_receiver) = mpsc::channel();
            thread::spawn(move || {
                for line in shard_stderr_reader.lines().flatten() {
                    err_sender.send(line).unwrap_or(());
                }
            });
            let mut err = String::new();
            while let Ok(line) = err_receiver.recv_timeout(Duration::from_millis(10)) {
                err = err + "\n" + &line;
            }

            tracing::error!("Shard complete standard error output:\n{err}");

            if let Some(signal) = exit_status.signal() {
                tracing::error!("Shard process was signaled to shutdown with signal {signal}");
            }

            status_sender.send(ShardStatus::Failed(rank)).unwrap();
            return;
        }

        // We received a shutdown signal
        if shutdown.load(Ordering::SeqCst) {
            terminate("Shard", p, Duration::from_secs(30)).unwrap();
            tracing::info!("Shard terminated");
            return;
        }

        // Shard is ready
        if uds.exists() && !ready {
            tracing::info!("Shard ready in {:?}", start_time.elapsed());
            sleep(Duration::from_millis(2000));
            status_sender.send(ShardStatus::Ready).unwrap();
            ready = true;
        } else if !ready && wait_time.elapsed() > Duration::from_secs(10) {
            tracing::info!("Waiting for shard to be ready...");
            wait_time = Instant::now();
        }
        sleep(Duration::from_millis(100));
    }
}

fn shutdown_shards(shutdown: Arc<AtomicBool>, shutdown_receiver: &mpsc::Receiver<()>) {
    tracing::info!("Shutting down shards");
    // Update shutdown value to true
    // This will be picked up by the shard manager
    shutdown.store(true, Ordering::SeqCst);

    // Wait for shards to shutdown
    // This will block till all shutdown_sender are dropped
    let _ = shutdown_receiver.recv();
}

fn num_cuda_devices() -> Option<usize> {
    let devices = match env::var("CUDA_VISIBLE_DEVICES") {
        Ok(devices) => devices,
        Err(_) => env::var("NVIDIA_VISIBLE_DEVICES").ok()?,
    };
    let n_devices = devices.split(',').count();
    Some(n_devices)
}

#[derive(Deserialize)]
#[serde(rename_all = "UPPERCASE")]
enum PythonLogLevelEnum {
    Trace,
    Debug,
    Info,
    Success,
    Warning,
    Error,
    Critical,
}

#[derive(Deserialize)]
struct PythonLogLevel {
    name: PythonLogLevelEnum,
}

#[derive(Deserialize)]
struct PythonLogRecord {
    level: PythonLogLevel,
}

#[derive(Deserialize)]
struct PythonLogMessage {
    text: String,
    record: PythonLogRecord,
}

impl PythonLogMessage {
    fn trace(&self) {
        match self.record.level.name {
            PythonLogLevelEnum::Trace => tracing::trace!("{}", self.text),
            PythonLogLevelEnum::Debug => tracing::debug!("{}", self.text),
            PythonLogLevelEnum::Info => tracing::info!("{}", self.text),
            PythonLogLevelEnum::Success => tracing::info!("{}", self.text),
            PythonLogLevelEnum::Warning => tracing::warn!("{}", self.text),
            PythonLogLevelEnum::Error => tracing::error!("{}", self.text),
            PythonLogLevelEnum::Critical => tracing::error!("{}", self.text),
        }
    }
}

impl TryFrom<&String> for PythonLogMessage {
    type Error = serde_json::Error;

    fn try_from(value: &String) -> Result<Self, Self::Error> {
        serde_json::from_str::<Self>(value)
    }
}

fn log_lines<S: Sized + BufRead>(lines: Lines<S>) {
    for line in lines.flatten() {
        match PythonLogMessage::try_from(&line) {
            Ok(log) => log.trace(),
            Err(_) => tracing::debug!("{line}"),
        }
    }
}

fn find_num_shards(
    sharded: Option<bool>,
    num_shard: Option<usize>,
) -> Result<usize, LauncherError> {
    // get the number of shards given `sharded` and `num_shard`
    let num_shard = match (sharded, num_shard) {
        (Some(true), None) => {
            // try to default to the number of available GPUs
            tracing::info!("Parsing num_shard from CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES");
            let n_devices = num_cuda_devices()
                .expect("--num-shard and CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES are not set");
            if n_devices <= 1 {
                return Err(LauncherError::NotEnoughCUDADevices(format!(
                    "`sharded` is true but only found {n_devices} CUDA devices"
                )));
            }
            n_devices
        }
        (Some(true), Some(num_shard)) => {
            // we can't have only one shard while sharded
            if num_shard <= 1 {
                return Err(LauncherError::ArgumentValidation(
                    "`sharded` is true but `num_shard` <= 1".to_string(),
                ));
            }
            num_shard
        }
        (Some(false), Some(num_shard)) => num_shard,
        (Some(false), None) => 1,
        (None, None) => num_cuda_devices().unwrap_or(1),
        (None, Some(num_shard)) => num_shard,
    };
    if num_shard < 1 {
        return Err(LauncherError::ArgumentValidation(
            "`num_shard` cannot be < 1".to_string(),
        ));
    }
    Ok(num_shard)
}

#[derive(Debug)]
enum LauncherError {
    ArgumentValidation(String),
    NotEnoughCUDADevices(String),
    DownloadError,
    ShardCannotStart,
    ShardDisconnected,
    ShardFailed,
    WebserverFailed,
    WebserverCannotStart,
}

fn download_convert_model(args: &Args, running: Arc<AtomicBool>) -> Result<(), LauncherError> {
    // Enter download tracing span
    let _span = tracing::span!(tracing::Level::INFO, "download").entered();

    let mut download_args = vec![
        "download-weights".to_string(),
        args.model_id.to_string(),
        "--extension".to_string(),
        ".safetensors".to_string(),
        "--logger-level".to_string(),
        "INFO".to_string(),
        "--json-output".to_string(),
    ];

    // Model optional revision
    if let Some(revision) = &args.revision {
        download_args.push("--revision".to_string());
        download_args.push(revision.to_string())
    }

    // Trust remote code for automatic peft fusion
    if args.trust_remote_code {
        download_args.push("--trust-remote-code".to_string());
    }

    // Copy current process env
    let mut envs: Vec<(OsString, OsString)> = env::vars_os().collect();

    // If huggingface_hub_cache is set, pass it to the download process
    // Useful when running inside a docker container
    if let Some(ref huggingface_hub_cache) = args.huggingface_hub_cache {
        envs.push(("HUGGINGFACE_HUB_CACHE".into(), huggingface_hub_cache.into()));
    };

    // Enable hf transfer for insane download speeds
    let enable_hf_transfer = env::var("HF_HUB_ENABLE_HF_TRANSFER").unwrap_or("1".to_string());
    envs.push((
        "HF_HUB_ENABLE_HF_TRANSFER".into(),
        enable_hf_transfer.into(),
    ));

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        envs.push(("HUGGING_FACE_HUB_TOKEN".into(), api_token.into()))
    };

    // If args.weights_cache_override is some, pass it to the download process
    // Useful when running inside a HuggingFace Inference Endpoint
    if let Some(weights_cache_override) = &args.weights_cache_override {
        envs.push((
            "WEIGHTS_CACHE_OVERRIDE".into(),
            weights_cache_override.into(),
        ));
    };

    // Start process
    tracing::info!("Starting download process.");
    let mut download_process = match Command::new("text-generation-server")
        .args(download_args)
        .envs(envs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                tracing::error!("text-generation-server not found in PATH");
                tracing::error!("Please install it with `make install-server`")
            } else {
                tracing::error!("{}", err);
            }

            return Err(LauncherError::DownloadError);
        }
    };

    // Redirect STDOUT to the console
    let download_stdout = download_process.stdout.take().unwrap();
    let stdout = BufReader::new(download_stdout);

    thread::spawn(move || {
        log_lines(stdout.lines());
    });

    loop {
        if let Some(status) = download_process.try_wait().unwrap() {
            if status.success() {
                tracing::info!("Successfully downloaded weights.");
                break;
            }

            let mut err = String::new();
            download_process
                .stderr
                .take()
                .unwrap()
                .read_to_string(&mut err)
                .unwrap();
            if let Some(signal) = status.signal() {
                tracing::error!(
                    "Download process was signaled to shutdown with signal {signal}: {err}"
                );
            } else {
                tracing::error!("Download encountered an error: {err}");
            }

            return Err(LauncherError::DownloadError);
        }
        if !running.load(Ordering::SeqCst) {
            terminate("download", download_process, Duration::from_secs(10)).unwrap();
            return Ok(());
        }
        sleep(Duration::from_millis(100));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn spawn_shards(
    num_shard: usize,
    args: &Args,
    shutdown: Arc<AtomicBool>,
    shutdown_receiver: &mpsc::Receiver<()>,
    shutdown_sender: mpsc::Sender<()>,
    status_receiver: &mpsc::Receiver<ShardStatus>,
    status_sender: mpsc::Sender<ShardStatus>,
    running: Arc<AtomicBool>,
) -> Result<(), LauncherError> {
    // Start shard processes
    for rank in 0..1 {
        let model_id = args.model_id.clone();
        let revision = args.revision.clone();
        let uds_path = args.shard_uds_path.clone();
        let master_addr = args.master_addr.clone();
        let huggingface_hub_cache = args.huggingface_hub_cache.clone();
        let weights_cache_override = args.weights_cache_override.clone();
        let status_sender = status_sender.clone();
        let shutdown = shutdown.clone();
        let shutdown_sender = shutdown_sender.clone();
        let otlp_endpoint = args.otlp_endpoint.clone();
        let quantize = args.quantize;
        let dtype = args.dtype;
        let max_total_tokens = args.max_total_tokens;
        let trust_remote_code = args.trust_remote_code;
        let master_port = args.master_port;
        let disable_custom_kernels = args.disable_custom_kernels;
        let watermark_gamma = args.watermark_gamma;
        let watermark_delta = args.watermark_delta;
        let cuda_memory_fraction = args.cuda_memory_fraction;
        let rope_scaling = args.rope_scaling;
        let rope_factor = args.rope_factor;
        thread::spawn(move || {
            shard_manager(
                model_id,
                revision,
                quantize,
                dtype,
                max_total_tokens,
                trust_remote_code,
                uds_path,
                rank,
                num_shard,
                master_addr,
                master_port,
                huggingface_hub_cache,
                weights_cache_override,
                disable_custom_kernels,
                watermark_gamma,
                watermark_delta,
                cuda_memory_fraction,
                rope_scaling,
                rope_factor,
                otlp_endpoint,
                status_sender,
                shutdown,
                shutdown_sender,
            )
        });
    }
    drop(shutdown_sender);

    // Wait for shard to start
    let mut shard_ready = 0;
    while running.load(Ordering::SeqCst) {
        match status_receiver.try_recv() {
            Ok(ShardStatus::Ready) => {
                shard_ready += 1;
                if shard_ready == 1 {
                    break;
                }
            }
            Err(TryRecvError::Empty) => {
                sleep(Duration::from_millis(100));
            }
            Ok(ShardStatus::Failed(rank)) => {
                tracing::error!("Shard {rank} failed to start");
                shutdown_shards(shutdown, shutdown_receiver);
                return Err(LauncherError::ShardCannotStart);
            }
            Err(TryRecvError::Disconnected) => {
                tracing::error!("Shard status channel disconnected");
                shutdown_shards(shutdown, shutdown_receiver);
                return Err(LauncherError::ShardDisconnected);
            }
        }
    }
    Ok(())
}

fn spawn_webserver(
    args: Args,
    shutdown: Arc<AtomicBool>,
    shutdown_receiver: &mpsc::Receiver<()>,
) -> Result<Child, LauncherError> {
    // All shard started
    // Start webserver
    tracing::info!("Starting Webserver");
    let mut router_args = vec![
        "--max-concurrent-requests".to_string(),
        args.max_concurrent_requests.to_string(),
        "--max-best-of".to_string(),
        args.max_best_of.to_string(),
        "--max-stop-sequences".to_string(),
        args.max_stop_sequences.to_string(),
        "--max-top-n-tokens".to_string(),
        args.max_top_n_tokens.to_string(),
        "--max-input-length".to_string(),
        args.max_input_length.to_string(),
        "--max-total-tokens".to_string(),
        args.max_total_tokens.to_string(),
        "--max-batch-prefill-tokens".to_string(),
        args.max_batch_prefill_tokens.to_string(),
        "--waiting-served-ratio".to_string(),
        args.waiting_served_ratio.to_string(),
        "--max-waiting-tokens".to_string(),
        args.max_waiting_tokens.to_string(),
        "--validation-workers".to_string(),
        args.validation_workers.to_string(),
        "--hostname".to_string(),
        args.hostname.to_string(),
        "--port".to_string(),
        args.port.to_string(),
        "--master-shard-uds-path".to_string(),
        format!("{}-0", args.shard_uds_path),
        "--tokenizer-name".to_string(),
        args.model_id,
    ];

    // Model optional max batch total tokens
    if let Some(max_batch_total_tokens) = args.max_batch_total_tokens {
        router_args.push("--max-batch-total-tokens".to_string());
        router_args.push(max_batch_total_tokens.to_string());
    }

    // Model optional revision
    if let Some(ref revision) = args.revision {
        router_args.push("--revision".to_string());
        router_args.push(revision.to_string())
    }

    if args.json_output {
        router_args.push("--json-output".to_string());
    }

    // OpenTelemetry
    if let Some(otlp_endpoint) = args.otlp_endpoint {
        router_args.push("--otlp-endpoint".to_string());
        router_args.push(otlp_endpoint);
    }

    // CORS origins
    for origin in args.cors_allow_origin.into_iter() {
        router_args.push("--cors-allow-origin".to_string());
        router_args.push(origin);
    }

    // Ngrok
    if args.ngrok {
        router_args.push("--ngrok".to_string());
        router_args.push("--ngrok-authtoken".to_string());
        router_args.push(args.ngrok_authtoken.unwrap());
        router_args.push("--ngrok-edge".to_string());
        router_args.push(args.ngrok_edge.unwrap());
    }

    // Copy current process env
    let mut envs: Vec<(OsString, OsString)> = env::vars_os().collect();

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        envs.push(("HUGGING_FACE_HUB_TOKEN".into(), api_token.into()))
    };

    let mut webserver = match Command::new("text-generation-router")
        .args(router_args)
        .envs(envs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            tracing::error!("Failed to start webserver: {}", err);
            if err.kind() == io::ErrorKind::NotFound {
                tracing::error!("text-generation-router not found in PATH");
                tracing::error!("Please install it with `make install-router`")
            } else {
                tracing::error!("{}", err);
            }

            shutdown_shards(shutdown, shutdown_receiver);
            return Err(LauncherError::WebserverCannotStart);
        }
    };

    // Redirect STDOUT and STDERR to the console
    let webserver_stdout = webserver.stdout.take().unwrap();
    let webserver_stderr = webserver.stderr.take().unwrap();

    thread::spawn(move || {
        let stdout = BufReader::new(webserver_stdout);
        let stderr = BufReader::new(webserver_stderr);
        for line in stdout.lines() {
            println!("{}", line.unwrap());
        }
        for line in stderr.lines() {
            println!("{}", line.unwrap());
        }
    });
    Ok(webserver)
}

fn terminate(process_name: &str, mut process: Child, timeout: Duration) -> io::Result<ExitStatus> {
    tracing::info!("Terminating {process_name}");

    let terminate_time = Instant::now();
    signal::kill(Pid::from_raw(process.id() as i32), Signal::SIGTERM).unwrap();

    tracing::info!("Waiting for {process_name} to gracefully shutdown");

    while terminate_time.elapsed() < timeout {
        if let Some(status) = process.try_wait()? {
            tracing::info!("{process_name} terminated");
            return Ok(status);
        }
        sleep(Duration::from_millis(100));
    }

    tracing::info!("Killing {process_name}");

    process.kill()?;
    let exit_status = process.wait()?;

    tracing::info!("{process_name} killed");
    Ok(exit_status)
}

fn main() -> Result<(), LauncherError> {
    // Pattern match configuration
    let args: Args = Args::parse();

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    if args.json_output {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .compact()
            .init();
    }

    if args.env {
        let env_runtime = env_runtime::Env::new();
        tracing::info!("{}", env_runtime);
    }

    tracing::info!("{:?}", args);

    // Validate args
    if args.max_input_length >= args.max_total_tokens {
        return Err(LauncherError::ArgumentValidation(
            "`max_input_length` must be < `max_total_tokens`".to_string(),
        ));
    }
    if args.max_input_length as u32 > args.max_batch_prefill_tokens {
        return Err(LauncherError::ArgumentValidation(format!(
            "`max_batch_prefill_tokens` must be >= `max_input_length`. Given: {} and {}",
            args.max_batch_prefill_tokens, args.max_input_length
        )));
    }

    if args.validation_workers == 0 {
        return Err(LauncherError::ArgumentValidation(
            "`validation_workers` must be > 0".to_string(),
        ));
    }
    if args.trust_remote_code {
        tracing::warn!(
            "`trust_remote_code` is set. Trusting that model `{}` do not contain malicious code.",
            args.model_id
        );
    }

    let num_shard = find_num_shards(args.sharded, args.num_shard)?;
    if num_shard > 1 {
        tracing::info!("Sharding model on {num_shard} processes");
    }

    if let Some(ref max_batch_total_tokens) = args.max_batch_total_tokens {
        if args.max_batch_prefill_tokens > *max_batch_total_tokens {
            return Err(LauncherError::ArgumentValidation(format!(
                "`max_batch_prefill_tokens` must be <= `max_batch_total_tokens`. Given: {} and {}",
                args.max_batch_prefill_tokens, max_batch_total_tokens
            )));
        }
        if args.max_total_tokens as u32 > *max_batch_total_tokens {
            return Err(LauncherError::ArgumentValidation(format!(
                "`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {} and {}",
                args.max_total_tokens, max_batch_total_tokens
            )));
        }
    }

    if args.ngrok {
        if args.ngrok_authtoken.is_none() {
            return Err(LauncherError::ArgumentValidation(
                "`ngrok-authtoken` must be set when using ngrok tunneling".to_string(),
            ));
        }

        if args.ngrok_edge.is_none() {
            return Err(LauncherError::ArgumentValidation(
                "`ngrok-edge` must be set when using ngrok tunneling".to_string(),
            ));
        }
    }

    // Signal handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Download and convert model weights
    download_convert_model(&args, running.clone())?;

    if !running.load(Ordering::SeqCst) {
        // Launcher was asked to stop
        return Ok(());
    }

    // Shared shutdown bool
    let shutdown = Arc::new(AtomicBool::new(false));
    // Shared shutdown channel
    // When shutting down, the main thread will wait for all senders to be dropped
    let (shutdown_sender, shutdown_receiver) = mpsc::channel();

    // Shared channel to track shard status
    let (status_sender, status_receiver) = mpsc::channel();

    spawn_shards(
        num_shard,
        &args,
        shutdown.clone(),
        &shutdown_receiver,
        shutdown_sender,
        &status_receiver,
        status_sender,
        running.clone(),
    )?;

    // We might have received a termination signal
    if !running.load(Ordering::SeqCst) {
        shutdown_shards(shutdown, &shutdown_receiver);
        return Ok(());
    }

    let mut webserver =
        spawn_webserver(args, shutdown.clone(), &shutdown_receiver).map_err(|err| {
            shutdown_shards(shutdown.clone(), &shutdown_receiver);
            err
        })?;

    // Default exit code
    let mut exit_code = Ok(());

    while running.load(Ordering::SeqCst) {
        if let Ok(ShardStatus::Failed(rank)) = status_receiver.try_recv() {
            tracing::error!("Shard {rank} crashed");
            exit_code = Err(LauncherError::ShardFailed);
            break;
        };

        match webserver.try_wait().unwrap() {
            Some(_) => {
                tracing::error!("Webserver Crashed");
                shutdown_shards(shutdown, &shutdown_receiver);
                return Err(LauncherError::WebserverFailed);
            }
            None => {
                sleep(Duration::from_millis(100));
            }
        };
    }

    // Graceful termination
    terminate("webserver", webserver, Duration::from_secs(90)).unwrap();
    shutdown_shards(shutdown, &shutdown_receiver);

    exit_code
}
