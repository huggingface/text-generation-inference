use clap::Parser;
use serde_json::Value;
use std::env;
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
    #[clap(default_value = "1", long, env)]
    num_shard: usize,
    #[clap(long, env)]
    quantize: bool,
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,
    #[clap(default_value = "1000", long, env)]
    max_input_length: usize,
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
        num_shard,
        quantize,
        max_concurrent_requests,
        max_input_length,
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
    } = args;

    // Signal handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Download weights
    if weights_cache_override.is_none() {
        let mut download_argv = vec![
            "text-generation-server".to_string(),
            "download-weights".to_string(),
            model_id.clone(),
            "--logger-level".to_string(),
            "INFO".to_string(),
            "--json-output".to_string(),
        ];
        if num_shard == 1 {
            download_argv.push("--extension".to_string());
            download_argv.push(".bin".to_string());
        } else {
            download_argv.push("--extension".to_string());
            download_argv.push(".safetensors".to_string());
        }

        // Model optional revision
        if let Some(ref revision) = revision {
            download_argv.push("--revision".to_string());
            download_argv.push(revision.to_string())
        }

        let mut env = Vec::new();

        // If the HUGGINGFACE_HUB_CACHE env var is set, pass it to the shard
        // Useful when running inside a docker container
        if let Some(ref huggingface_hub_cache) = huggingface_hub_cache {
            env.push(("HUGGINGFACE_HUB_CACHE".into(), huggingface_hub_cache.into()));
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
                if let Ok(value) = serde_json::from_str::<Value>(&line.unwrap()) {
                    if let Some(text) = value.get("text") {
                        // Format escaped newlines
                        tracing::info!("{}", text.to_string().replace("\\n", ""));
                    }
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
    } else {
        tracing::info!(
            "weights_cache_override is set to {:?}.",
            weights_cache_override
        );
        tracing::info!("Skipping download.")
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
        "--max-input-length".to_string(),
        max_input_length.to_string(),
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

    if json_output {
        argv.push("--json-output".to_string());
    }

    // OpenTelemetry
    if let Some(otlp_endpoint) = otlp_endpoint {
        argv.push("--otlp-endpoint".to_string());
        argv.push(otlp_endpoint);
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
        "ERROR".to_string(),
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

    let mut env = vec![
        ("RANK".into(), rank.to_string().into()),
        ("WORLD_SIZE".into(), world_size.to_string().into()),
        ("MASTER_ADDR".into(), master_addr.into()),
        ("MASTER_PORT".into(), master_port.to_string().into()),
        ("SAFETENSORS_FAST_GPU".into(), "1".into()),
        ("NCCL_ASYNC_ERROR_HANDLING".into(), "1".into()),
    ];

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

    // If the NCCL_SHM_DISABLE env var is set, pass it to the shard
    // needed when running NCCL inside a docker container and when you can't increase shm size
    if let Ok(nccl_shm_disalbe) = env::var("NCCL_SHM_DISABLE") {
        env.push(("NCCL_SHM_DISABLE".into(), nccl_shm_disalbe.into()));
    };

    // If the CUDA_VISIBLE_DEVICES env var is set, pass it to the shard
    if let Ok(cuda_visible_devices) = env::var("CUDA_VISIBLE_DEVICES") {
        env.push(("CUDA_VISIBLE_DEVICES".into(), cuda_visible_devices.into()));
    };

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
            if let Ok(value) = serde_json::from_str::<Value>(&line.unwrap()) {
                if let Some(text) = value.get("text") {
                    // Format escaped newlines
                    tracing::error!("{}", text.to_string().replace("\\n", "\n"));
                }
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
