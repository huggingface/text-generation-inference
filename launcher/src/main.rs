use clap::Parser;
use serde::Deserialize;
use std::env;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::Arc;
use std::sync::{mpsc, Mutex};
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{fs, io};
use subprocess::{ExitStatus, Popen, PopenConfig, PopenError, Redirection};

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(default_value = "bigscience/bloom-560m", long, env)]
    model_id: String,
    #[clap(long, env)]
    revision: Option<String>,
    #[clap(long, env)]
    sharded: Option<bool>,
    #[clap(long, env)]
    num_shard: Option<usize>,
    #[clap(long, env)]
    quantize: bool,
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "2", long, env)]
    max_best_of: usize,
    #[clap(default_value = "4", long, env)]
    max_stop_sequences: usize,
    #[clap(default_value = "1000", long, env)]
    max_input_length: usize,
    #[clap(default_value = "1512", long, env)]
    max_total_tokens: usize,
    #[clap(default_value = "32", long, env)]
    max_batch_size: usize,
    #[clap(default_value = "20", long, env)]
    max_waiting_tokens: usize,
    #[clap(default_value = "3000", long, short, env)]
    port: u16,
    #[clap(default_value = "/tmp/text-generation-server", long, env)]
    shard_uds_path: String,
    #[clap(default_value = "localhost", long, env)]
    master_addr: String,
    #[clap(default_value = "29500", long, env)]
    master_port: usize,
    #[clap(long, env)]
    huggingface_hub_cache: Option<String>,
    #[clap(long, env)]
    weights_cache_override: Option<String>,
    #[clap(long, env)]
    disable_custom_kernels: bool,
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
}

fn main() -> ExitCode {
    // Pattern match configuration
    let args = Args::parse();

    if args.json_output {
        tracing_subscriber::fmt().json().init();
    } else {
        tracing_subscriber::fmt().compact().init();
    }

    tracing::info!("{:?}", args);

    let Args {
        model_id,
        revision,
        sharded,
        num_shard,
        quantize,
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
        max_batch_size,
        max_waiting_tokens,
        port,
        shard_uds_path,
        master_addr,
        master_port,
        huggingface_hub_cache,
        weights_cache_override,
        disable_custom_kernels,
        json_output,
        otlp_endpoint,
        cors_allow_origin,
        watermark_gamma,
        watermark_delta,
    } = args;

    // get the number of shards given `sharded` and `num_shard`
    let num_shard = if let Some(sharded) = sharded {
        // sharded is set
        match sharded {
            // sharded is set and true
            true => {
                match num_shard {
                    None => {
                        // try to default to the number of available GPUs
                        tracing::info!("Parsing num_shard from CUDA_VISIBLE_DEVICES");
                        let n_devices = num_cuda_devices()
                            .expect("--num-shard and CUDA_VISIBLE_DEVICES are not set");
                        if n_devices <= 1 {
                            panic!("`sharded` is true but only found {n_devices} CUDA devices");
                        }
                        n_devices
                    }
                    Some(num_shard) => {
                        // we can't have only one shard while sharded
                        if num_shard <= 1 {
                            panic!("`sharded` is true but `num_shard` <= 1");
                        }
                        num_shard
                    }
                }
            }
            // sharded is set and false
            false => {
                let num_shard = num_shard.unwrap_or(1);
                // we can't have more than one shard while not sharded
                if num_shard != 1 {
                    panic!("`sharded` is false but `num_shard` != 1");
                }
                num_shard
            }
        }
    } else {
        match num_shard {
            // get num_shard from CUDA_VISIBLE_DEVICES or default to a single shard
            None => num_cuda_devices().unwrap_or(1),
            Some(num_shard) => num_shard,
        }
    };
    if num_shard < 1 {
        panic!("`num_shard` cannot be < 1");
    }

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

    // Check if model_id is a local model
    let local_path = Path::new(&model_id);
    let is_local_model = local_path.exists() && local_path.is_dir();

    // Download weights for sharded models
    if !is_local_model && weights_cache_override.is_none() && num_shard > 1 {
        let mut download_argv = vec![
            "text-generation-server".to_string(),
            "download-weights".to_string(),
            model_id.clone(),
            "--extension".to_string(),
            ".safetensors".to_string(),
            "--logger-level".to_string(),
            "INFO".to_string(),
            "--json-output".to_string(),
        ];

        // Model optional revision
        if let Some(ref revision) = revision {
            download_argv.push("--revision".to_string());
            download_argv.push(revision.to_string())
        }

        // Copy current process env
        let mut env: Vec<(OsString, OsString)> = env::vars_os().collect();

        // If huggingface_hub_cache is set, pass it to the shard
        // Useful when running inside a docker container
        if let Some(ref huggingface_hub_cache) = huggingface_hub_cache {
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
                return ExitCode::FAILURE;
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
                            return ExitCode::FAILURE;
                        }
                    }
                    _ => {
                        tracing::error!("Download process exited with an unknown status.");
                        return ExitCode::FAILURE;
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
                return ExitCode::SUCCESS;
            }
            sleep(Duration::from_millis(100));
        }
    }

    // Shared shutdown bool
    let shutdown = Arc::new(Mutex::new(false));
    // Shared shutdown channel
    // When shutting down, the main thread will wait for all senders to be dropped
    let (shutdown_sender, shutdown_receiver) = mpsc::channel();

    // Shared channel to track shard status
    let (status_sender, status_receiver) = mpsc::channel();

    // Start shard processes
    for rank in 0..num_shard {
        let model_id = model_id.clone();
        let revision = revision.clone();
        let uds_path = shard_uds_path.clone();
        let master_addr = master_addr.clone();
        let huggingface_hub_cache = huggingface_hub_cache.clone();
        let weights_cache_override = weights_cache_override.clone();
        let status_sender = status_sender.clone();
        let shutdown = shutdown.clone();
        let shutdown_sender = shutdown_sender.clone();
        let otlp_endpoint = otlp_endpoint.clone();
        thread::spawn(move || {
            shard_manager(
                model_id,
                revision,
                quantize,
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
                shutdown_shards(shutdown, &shutdown_receiver);
                return ExitCode::FAILURE;
            }
            Err(TryRecvError::Disconnected) => {
                tracing::error!("Shard status channel disconnected");
                shutdown_shards(shutdown, &shutdown_receiver);
                return ExitCode::FAILURE;
            }
        }
    }

    // We might have received a termination signal
    if !running.load(Ordering::SeqCst) {
        shutdown_shards(shutdown, &shutdown_receiver);
        return ExitCode::SUCCESS;
    }

    // All shard started
    // Start webserver
    tracing::info!("Starting Webserver");
    let mut argv = vec![
        "text-generation-router".to_string(),
        "--max-concurrent-requests".to_string(),
        max_concurrent_requests.to_string(),
        "--max-best-of".to_string(),
        max_best_of.to_string(),
        "--max-stop-sequences".to_string(),
        max_stop_sequences.to_string(),
        "--max-input-length".to_string(),
        max_input_length.to_string(),
        "--max-total-tokens".to_string(),
        max_total_tokens.to_string(),
        "--max-batch-size".to_string(),
        max_batch_size.to_string(),
        "--max-waiting-tokens".to_string(),
        max_waiting_tokens.to_string(),
        "--port".to_string(),
        port.to_string(),
        "--master-shard-uds-path".to_string(),
        format!("{shard_uds_path}-0"),
        "--tokenizer-name".to_string(),
        model_id,
    ];

    // Model optional revision
    if let Some(ref revision) = revision {
        argv.push("--revision".to_string());
        argv.push(revision.to_string())
    }

    if json_output {
        argv.push("--json-output".to_string());
    }

    // OpenTelemetry
    if let Some(otlp_endpoint) = otlp_endpoint {
        argv.push("--otlp-endpoint".to_string());
        argv.push(otlp_endpoint);
    }

    // CORS origins
    for origin in cors_allow_origin.into_iter() {
        argv.push("--cors-allow-origin".to_string());
        argv.push(origin);
    }

    let mut webserver = match Popen::create(
        &argv,
        PopenConfig {
            stdout: Redirection::Pipe,
            stderr: Redirection::Pipe,
            // Needed for the shutdown procedure
            setpgid: true,
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

            shutdown_shards(shutdown, &shutdown_receiver);
            return ExitCode::FAILURE;
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

    // Default exit code
    let mut exit_code = ExitCode::SUCCESS;

    while running.load(Ordering::SeqCst) {
        if let Ok(ShardStatus::Failed((rank, err))) = status_receiver.try_recv() {
            tracing::error!("Shard {rank} failed:\n{err}");
            exit_code = ExitCode::FAILURE;
            break;
        };

        match webserver.poll() {
            Some(_) => {
                tracing::error!("Webserver Crashed");
                shutdown_shards(shutdown, &shutdown_receiver);
                return ExitCode::FAILURE;
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

#[derive(Debug)]
enum ShardStatus {
    Ready,
    Failed((usize, String)),
}

#[allow(clippy::too_many_arguments)]
fn shard_manager(
    model_id: String,
    revision: Option<String>,
    quantize: bool,
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

    // Activate tensor parallelism
    if world_size > 1 {
        shard_argv.push("--sharded".to_string());
    }

    if quantize {
        shard_argv.push("--quantize".to_string())
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
        if p.poll().is_some() {
            let mut err = String::new();
            p.stderr.take().unwrap().read_to_string(&mut err).unwrap();
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
    if let Ok(cuda_visible_devices) = env::var("CUDA_VISIBLE_DEVICES") {
        let n_devices = cuda_visible_devices.split(',').count();
        return Some(n_devices);
    }
    None
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
