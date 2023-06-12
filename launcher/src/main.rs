use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::env;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::Arc;
use std::sync::{mpsc, Mutex};
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{fs, io};
use subprocess::{ExitStatus, Popen, PopenConfig, PopenError, Redirection};

mod env_runtime;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Quantization {
    Bitsandbytes,
    Gptq,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            Quantization::Bitsandbytes => {
                write!(f, "bitsandbytes")
            }
            Quantization::Gptq => {
                write!(f, "gptq")
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

    /// Whether you want the model to be quantized. This will use `bitsandbytes` for
    /// quantization on the fly, or `gptq`.
    #[clap(long, env, value_enum)]
    quantize: Option<Quantization>,

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

    /// This is the maximum allowed input length (expressed in number of tokens)
    /// for users. The larger this value, the longer prompt users can send which
    /// can impact the overall memory required to handle the load.
    /// Please note that some models have a finite range of sequence they can handle.
    #[clap(default_value = "1000", long, env)]
    max_input_length: usize,

    /// This is the most important value to set as it defines the "memory budget"
    /// of running clients requests.
    /// Clients will send input sequences and ask to generate `max_new_tokens`
    /// on top. with a value of `1512` users can send either a prompt of
    /// `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for
    /// `1511` max_new_tokens.
    /// The larger this value, the larger amount each request will be in your RAM
    /// and the less effective batching can be.
    #[clap(default_value = "1512", long, env)]
    max_total_tokens: usize,

    /// The maximum allowed batch size during dynamic batching.
    /// Using `max_batch_total_tokens` should be favored in general
    /// as it's a finer way to control RAM usage.
    #[clap(long, env)]
    max_batch_size: Option<usize>,

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
    /// So you don't have to control that finely
    /// `max_batch_size` or `max_total_tokens`. In fact you could mostly relax them if you
    /// want maximum flexibility. However, for your users if they are asking for the full amount of
    /// total tokens, they are likely to wait for a very long time to get a spot
    /// in the batch (since they are going to be alone) so setting `max_batch_size`
    /// and `max_total_tokens` can still be useful to prevent those long waiting times.
    ///
    /// Overall this number should be the largest possible amount that fits the
    /// remaining memory (after the model is loaded). Since the actual memory overhead
    /// depends on other parameters like if you're using quantization, flash attention
    /// or the model implementation, text-generation-inference cannot infer this number
    /// automatically.
    #[clap(default_value = "32000", long, env)]
    max_batch_total_tokens: u32,

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
    #[clap(default_value = "3000", long, short, env)]

    /// The port to listen on.
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

    /// Display a lot of information about your runtime environment
    #[clap(long, short, action)]
    env: bool,
}

#[derive(Debug)]
enum ShardStatus {
    Ready,
    Failed((usize, String)),
}

#[allow(clippy::too_many_arguments)]
fn shard_manager(
    model_id: String,
    revision: Option<String>,
    quantize: Option<Quantization>,
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
    otlp_endpoint: Option<String>,
    status_sender: mpsc::Sender<ShardStatus>,
    shutdown: Arc<Mutex<bool>>,
    _shutdown_sender: mpsc::Sender<()>,
) {
    // Get UDS path
    let uds_string = format!("{uds_path}-{rank}");
    let uds = Path::new(&uds_string);
    // Clean previous runs
    fs::remove_file(uds).unwrap_or_default();

    // Process args
    let mut shard_argv = vec![
        "text-generation-server".to_string(),
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
        shard_argv.push("--trust-remote-code".to_string());
    }

    // Activate tensor parallelism
    if world_size > 1 {
        shard_argv.push("--sharded".to_string());
    }

    if let Some(quantize) = quantize {
        shard_argv.push("--quantize".to_string());
        shard_argv.push(quantize.to_string())
    }

    // Model optional revision
    if let Some(revision) = revision {
        shard_argv.push("--revision".to_string());
        shard_argv.push(revision)
    }

    // OpenTelemetry
    if let Some(otlp_endpoint) = otlp_endpoint {
        shard_argv.push("--otlp-endpoint".to_string());
        shard_argv.push(otlp_endpoint);
    }

    // Copy current process env
    let mut env: Vec<(OsString, OsString)> = env::vars_os().collect();

    // Torch Distributed Env vars
    env.push(("RANK".into(), rank.to_string().into()));
    env.push(("WORLD_SIZE".into(), world_size.to_string().into()));
    env.push(("MASTER_ADDR".into(), master_addr.into()));
    env.push(("MASTER_PORT".into(), master_port.to_string().into()));
    env.push(("NCCL_ASYNC_ERROR_HANDLING".into(), "1".into()));

    // Safetensors load fast
    env.push(("SAFETENSORS_FAST_GPU".into(), "1".into()));

    // Enable hf transfer for insane download speeds
    let enable_hf_transfer = env::var("HF_HUB_ENABLE_HF_TRANSFER").unwrap_or("1".to_string());
    env.push((
        "HF_HUB_ENABLE_HF_TRANSFER".into(),
        enable_hf_transfer.into(),
    ));

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        env.push(("HUGGING_FACE_HUB_TOKEN".into(), api_token.into()))
    };

    // If huggingface_hub_cache is some, pass it to the shard
    // Useful when running inside a docker container
    if let Some(huggingface_hub_cache) = huggingface_hub_cache {
        env.push(("HUGGINGFACE_HUB_CACHE".into(), huggingface_hub_cache.into()));
    };

    // If weights_cache_override is some, pass it to the shard
    // Useful when running inside a HuggingFace Inference Endpoint
    if let Some(weights_cache_override) = weights_cache_override {
        env.push((
            "WEIGHTS_CACHE_OVERRIDE".into(),
            weights_cache_override.into(),
        ));
    };

    // If disable_custom_kernels is true, pass it to the shard as an env var
    if disable_custom_kernels {
        env.push(("DISABLE_CUSTOM_KERNELS".into(), "True".into()))
    }

    // Watermark Gamma
    if let Some(watermark_gamma) = watermark_gamma {
        env.push(("WATERMARK_GAMMA".into(), watermark_gamma.to_string().into()))
    }

    // Watermark Delta
    if let Some(watermark_delta) = watermark_delta {
        env.push(("WATERMARK_DELTA".into(), watermark_delta.to_string().into()))
    }

    // Start process
    tracing::info!("Starting shard {rank}");
    let mut p = match Popen::create(
        &shard_argv,
        PopenConfig {
            stdout: Redirection::Pipe,
            stderr: Redirection::Pipe,
            // Needed for the shutdown procedure
            setpgid: true,
            // NCCL env vars
            env: Some(env),
            ..Default::default()
        },
    ) {
        Ok(p) => p,
        Err(err) => {
            if let PopenError::IoError(ref err) = err {
                if err.kind() == io::ErrorKind::NotFound {
                    tracing::error!("text-generation-server not found in PATH");
                    tracing::error!("Please install it with `make install-server`")
                }
            }
            status_sender
                .send(ShardStatus::Failed((rank, err.to_string())))
                .unwrap();
            return;
        }
    };

    // Redirect STDOUT to the console
    let shard_stdout = p.stdout.take().unwrap();

    thread::spawn(move || {
        // Enter shard-manager tracing span
        let stdout = BufReader::new(shard_stdout);
        let _span = tracing::span!(tracing::Level::INFO, "shard-manager", rank = rank).entered();
        for line in stdout.lines() {
            // Parse loguru logs
            if let Ok(log) = serde_json::from_str::<PythonLogMessage>(&line.unwrap()) {
                log.trace();
            }
        }
    });

    let mut ready = false;
    let start_time = Instant::now();
    let mut wait_time = Instant::now();
    loop {
        // Process exited
        if let Some(exit_status) = p.poll() {
            let mut err = String::new();
            p.stderr.take().unwrap().read_to_string(&mut err).unwrap();

            if let ExitStatus::Signaled(signal) = exit_status {
                tracing::error!("Shard process was signaled to shutdown with signal {signal}");
            }

            status_sender
                .send(ShardStatus::Failed((rank, err)))
                .unwrap();
            return;
        }

        // We received a shutdown signal
        if *shutdown.lock().unwrap() {
            p.terminate().unwrap();
            let _ = p.wait_timeout(Duration::from_secs(90));
            tracing::info!("Shard {rank} terminated");
            return;
        }

        // Shard is ready
        if uds.exists() && !ready {
            tracing::info!("Shard {rank} ready in {:?}", start_time.elapsed());
            status_sender.send(ShardStatus::Ready).unwrap();
            ready = true;
        } else if !ready && wait_time.elapsed() > Duration::from_secs(10) {
            tracing::info!("Waiting for shard {rank} to be ready...");
            wait_time = Instant::now();
        }
        sleep(Duration::from_millis(100));
    }
}

fn shutdown_shards(shutdown: Arc<Mutex<bool>>, shutdown_receiver: &mpsc::Receiver<()>) {
    tracing::info!("Shutting down shards");
    // Update shutdown value to true
    // This will be picked up by the shard manager
    {
        let mut shutdown = shutdown.lock().unwrap();
        *shutdown = true;
    }

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

fn find_num_shards(sharded: Option<bool>, num_shard: Option<usize>) -> usize {
    // get the number of shards given `sharded` and `num_shard`
    let num_shard = match (sharded, num_shard) {
        (Some(true), None) => {
            // try to default to the number of available GPUs
            tracing::info!("Parsing num_shard from CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES");
            let n_devices = num_cuda_devices()
                .expect("--num-shard and CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES are not set");
            if n_devices <= 1 {
                panic!("`sharded` is true but only found {n_devices} CUDA devices");
            }
            n_devices
        }
        (Some(true), Some(num_shard)) => {
            // we can't have only one shard while sharded
            if num_shard <= 1 {
                panic!("`sharded` is true but `num_shard` <= 1");
            }
            num_shard
        }
        (Some(false), Some(num_shard)) => num_shard,
        (Some(false), None) => 1,
        (None, None) => num_cuda_devices().unwrap_or(1),
        (None, Some(num_shard)) => num_shard,
    };
    if num_shard < 1 {
        panic!("`num_shard` cannot be < 1");
    }
    num_shard
}

#[derive(Debug)]
enum LauncherError {
    DownloadError,
    ShardCannotStart,
    ShardDisconnected,
    ShardFailed,
    WebserverFailed,
    WebserverCannotStart,
}

fn download_convert_model(args: &Args, running: Arc<AtomicBool>) -> Result<(), LauncherError> {
    let mut download_argv = vec![
        "text-generation-server".to_string(),
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
        download_argv.push("--revision".to_string());
        download_argv.push(revision.to_string())
    }

    // Copy current process env
    let mut env: Vec<(OsString, OsString)> = env::vars_os().collect();

    // If huggingface_hub_cache is set, pass it to the download process
    // Useful when running inside a docker container
    if let Some(ref huggingface_hub_cache) = args.huggingface_hub_cache {
        env.push(("HUGGINGFACE_HUB_CACHE".into(), huggingface_hub_cache.into()));
    };

    // Enable hf transfer for insane download speeds
    let enable_hf_transfer = env::var("HF_HUB_ENABLE_HF_TRANSFER").unwrap_or("1".to_string());
    env.push((
        "HF_HUB_ENABLE_HF_TRANSFER".into(),
        enable_hf_transfer.into(),
    ));

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        env.push(("HUGGING_FACE_HUB_TOKEN".into(), api_token.into()))
    };

    // If args.weights_cache_override is some, pass it to the download process
    // Useful when running inside a HuggingFace Inference Endpoint
    if let Some(weights_cache_override) = &args.weights_cache_override {
        env.push((
            "WEIGHTS_CACHE_OVERRIDE".into(),
            weights_cache_override.into(),
        ));
    };

    // Start process
    tracing::info!("Starting download process.");
    let mut download_process = match Popen::create(
        &download_argv,
        PopenConfig {
            stdout: Redirection::Pipe,
            stderr: Redirection::Pipe,
            // Needed for the shutdown procedure
            setpgid: true,
            env: Some(env),
            ..Default::default()
        },
    ) {
        Ok(p) => p,
        Err(err) => {
            if let PopenError::IoError(ref err) = err {
                if err.kind() == io::ErrorKind::NotFound {
                    tracing::error!("text-generation-server not found in PATH");
                    tracing::error!("Please install it with `make install-server`")
                }
            }
            return Err(LauncherError::DownloadError);
        }
    };

    // Redirect STDOUT to the console
    let download_stdout = download_process.stdout.take().unwrap();
    thread::spawn(move || {
        // Enter download tracing span
        let stdout = BufReader::new(download_stdout);
        let _span = tracing::span!(tracing::Level::INFO, "download").entered();
        for line in stdout.lines() {
            // Parse loguru logs
            if let Ok(log) = serde_json::from_str::<PythonLogMessage>(&line.unwrap()) {
                log.trace();
            }
        }
    });

    loop {
        if let Some(status) = download_process.poll() {
            match status {
                ExitStatus::Exited(exit_code) => {
                    if exit_code == 0 {
                        tracing::info!("Successfully downloaded weights.");
                        break;
                    } else {
                        let mut err = String::new();
                        download_process
                            .stderr
                            .take()
                            .unwrap()
                            .read_to_string(&mut err)
                            .unwrap();
                        tracing::error!("Download encountered an error: {err}");
                        return Err(LauncherError::DownloadError);
                    }
                }
                ExitStatus::Signaled(signal) => {
                    let mut err = String::new();
                    download_process
                        .stderr
                        .take()
                        .unwrap()
                        .read_to_string(&mut err)
                        .unwrap();
                    tracing::error!(
                        "Download process was signaled to shutdown with signal {signal}: {err}"
                    );
                    return Err(LauncherError::DownloadError);
                }
                e => {
                    tracing::error!("Download process exited with an unknown status.: {e:?}");
                    return Err(LauncherError::DownloadError);
                }
            }
        }
        if !running.load(Ordering::SeqCst) {
            download_process.terminate().unwrap();
            tracing::info!("Waiting for download process to gracefully shutdown");
            download_process
                .wait_timeout(Duration::from_secs(90))
                .unwrap();
            tracing::info!("Download process terminated");
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
    shutdown: Arc<Mutex<bool>>,
    shutdown_receiver: &mpsc::Receiver<()>,
    shutdown_sender: mpsc::Sender<()>,
    status_receiver: &mpsc::Receiver<ShardStatus>,
    status_sender: mpsc::Sender<ShardStatus>,
    running: Arc<AtomicBool>,
) -> Result<(), LauncherError> {
    if args.trust_remote_code {
        tracing::warn!(
            "`trust_remote_code` is set. Trusting that model `{}` do not contain malicious code.",
            args.model_id
        );
        if args.revision.is_none() {
            tracing::warn!("Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.");
        }
    }

    // Start shard processes
    for rank in 0..num_shard {
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
        let trust_remote_code = args.trust_remote_code;
        let master_port = args.master_port;
        let disable_custom_kernels = args.disable_custom_kernels;
        let watermark_gamma = args.watermark_gamma;
        let watermark_delta = args.watermark_delta;
        thread::spawn(move || {
            shard_manager(
                model_id,
                revision,
                quantize,
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
                if shard_ready == num_shard {
                    break;
                }
            }
            Err(TryRecvError::Empty) => {
                sleep(Duration::from_millis(100));
            }
            Ok(ShardStatus::Failed((rank, err))) => {
                tracing::error!("Shard {} failed to start:\n{}", rank, err);
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
    shutdown: Arc<Mutex<bool>>,
    shutdown_receiver: &mpsc::Receiver<()>,
) -> Result<Popen, LauncherError> {
    // All shard started
    // Start webserver
    tracing::info!("Starting Webserver");
    let mut argv = vec![
        "text-generation-router".to_string(),
        "--max-concurrent-requests".to_string(),
        args.max_concurrent_requests.to_string(),
        "--max-best-of".to_string(),
        args.max_best_of.to_string(),
        "--max-stop-sequences".to_string(),
        args.max_stop_sequences.to_string(),
        "--max-input-length".to_string(),
        args.max_input_length.to_string(),
        "--max-total-tokens".to_string(),
        args.max_total_tokens.to_string(),
        "--waiting-served-ratio".to_string(),
        args.waiting_served_ratio.to_string(),
        "--max-waiting-tokens".to_string(),
        args.max_waiting_tokens.to_string(),
        "--port".to_string(),
        args.port.to_string(),
        "--master-shard-uds-path".to_string(),
        format!("{}-0", args.shard_uds_path),
        "--tokenizer-name".to_string(),
        args.model_id,
    ];

    // Deprecate max_batch_size
    if let Some(max_batch_size) = args.max_batch_size {
        argv.push("--max-batch-size".to_string());
        argv.push(max_batch_size.to_string())
    } else {
        argv.push("--max-batch-total-tokens".to_string());
        argv.push(args.max_batch_total_tokens.to_string())
    }

    // Model optional revision
    if let Some(ref revision) = args.revision {
        argv.push("--revision".to_string());
        argv.push(revision.to_string())
    }

    if args.json_output {
        argv.push("--json-output".to_string());
    }

    // OpenTelemetry
    if let Some(otlp_endpoint) = args.otlp_endpoint {
        argv.push("--otlp-endpoint".to_string());
        argv.push(otlp_endpoint);
    }

    // CORS origins
    for origin in args.cors_allow_origin.into_iter() {
        argv.push("--cors-allow-origin".to_string());
        argv.push(origin);
    }

    // Copy current process env
    let mut env: Vec<(OsString, OsString)> = env::vars_os().collect();

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        env.push(("HUGGING_FACE_HUB_TOKEN".into(), api_token.into()))
    };

    let mut webserver = match Popen::create(
        &argv,
        PopenConfig {
            stdout: Redirection::Pipe,
            stderr: Redirection::Pipe,
            // Needed for the shutdown procedure
            setpgid: true,
            env: Some(env),
            ..Default::default()
        },
    ) {
        Ok(p) => p,
        Err(err) => {
            tracing::error!("Failed to start webserver: {}", err);
            if let PopenError::IoError(err) = err {
                if err.kind() == io::ErrorKind::NotFound {
                    tracing::error!("text-generation-router not found in PATH");
                    tracing::error!("Please install it with `make install-router`")
                }
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

fn main() -> Result<(), LauncherError> {
    // Pattern match configuration
    let args = Args::parse();

    if args.json_output {
        tracing_subscriber::fmt().json().init();
    } else {
        tracing_subscriber::fmt().compact().init();
    }

    if args.env {
        let env_runtime = env_runtime::Env::new();
        tracing::info!("{}", env_runtime);
    }

    tracing::info!("{:?}", args);

    let num_shard = find_num_shards(args.sharded, args.num_shard);
    if num_shard > 1 {
        tracing::info!("Sharding model on {num_shard} processes");
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

    // Shared shutdown bool
    let shutdown = Arc::new(Mutex::new(false));
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

    let mut webserver = spawn_webserver(args, shutdown.clone(), &shutdown_receiver)?;

    // Default exit code
    let mut exit_code = Ok(());

    while running.load(Ordering::SeqCst) {
        if let Ok(ShardStatus::Failed((rank, err))) = status_receiver.try_recv() {
            tracing::error!("Shard {rank} failed:\n{err}");
            exit_code = Err(LauncherError::ShardFailed);
            break;
        };

        match webserver.poll() {
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
    webserver.terminate().unwrap();
    tracing::info!("Waiting for webserver to gracefully shutdown");
    webserver.wait_timeout(Duration::from_secs(90)).unwrap();
    tracing::info!("Webserver terminated");
    shutdown_shards(shutdown, &shutdown_receiver);

    exit_code
}
