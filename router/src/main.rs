/// Text Generation Inference webserver entrypoint
use axum::http::HeaderValue;
use clap::Parser;
use opentelemetry::sdk::propagation::TraceContextPropagator;
use opentelemetry::sdk::trace;
use opentelemetry::sdk::trace::Sampler;
use opentelemetry::sdk::Resource;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use text_generation_client::ShardedClient;
use text_generation_router::server;
use tokenizers::Tokenizer;
use tower_http::cors::AllowOrigin;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
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
    #[clap(default_value = "/tmp/text-generation-0", long, env)]
    master_shard_uds_path: String,
    #[clap(default_value = "bigscience/bloom", long, env)]
    tokenizer_name: String,
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,
    #[clap(long, env)]
    json_output: bool,
    #[clap(long, env)]
    otlp_endpoint: Option<String>,
    #[clap(long, env)]
    cors_allow_origin: Option<Vec<String>>,
}

fn main() -> Result<(), std::io::Error> {
    // Get args
    let args = Args::parse();
    // Pattern match configuration
    let Args {
        max_concurrent_requests,
        max_best_of,
        max_stop_sequences,
        max_input_length,
        max_total_tokens,
        max_batch_size,
        max_waiting_tokens,
        port,
        master_shard_uds_path,
        tokenizer_name,
        validation_workers,
        json_output,
        otlp_endpoint,
        cors_allow_origin,
    } = args;

    if validation_workers == 0 {
        panic!("validation_workers must be > 0");
    }

    // CORS allowed origins
    // map to go inside the option and then map to parse from String to HeaderValue
    // Finally, convert to AllowOrigin
    let cors_allow_origin: Option<AllowOrigin> = cors_allow_origin.map(|cors_allow_origin| {
        AllowOrigin::list(
            cors_allow_origin
                .iter()
                .map(|origin| origin.parse::<HeaderValue>().unwrap()),
        )
    });

    // Tokenizer instance
    // This will only be used to validate payloads
    let local_path = Path::new(&tokenizer_name);
    let tokenizer =
        if local_path.exists() && local_path.is_dir() && local_path.join("tokenizer.json").exists()
        {
            // Load local tokenizer
            Tokenizer::from_file(local_path.join("tokenizer.json")).unwrap()
        } else {
            // Download and instantiate tokenizer
            // We need to download it outside of the Tokio runtime
            Tokenizer::from_pretrained(tokenizer_name.clone(), None).unwrap()
        };

    // Launch Tokio runtime
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            init_logging(otlp_endpoint, json_output);

            // Get pipeline tag
            let model_info = reqwest::get(format!(
                "https://huggingface.co/api/models/{tokenizer_name}"
            ))
            .await
            .expect("Could not connect to hf.co")
            .text()
            .await
            .expect("error when retrieving model info from hf.co");
            let model_info: serde_json::Value =
                serde_json::from_str(&model_info).expect("unable to parse model info");

            // if pipeline-tag == text-generation we default to return_full_text = true
            let compat_return_full_text = match model_info.get("pipeline_tag") {
                None => {
                    tracing::warn!("no pipeline tag found for model {tokenizer_name}");
                    false
                }
                Some(pipeline_tag) => pipeline_tag.as_str() == Some("text-generation"),
            };

            // Instantiate sharded client from the master unix socket
            let mut sharded_client = ShardedClient::connect_uds(master_shard_uds_path)
                .await
                .expect("Could not connect to server");
            // Clear the cache; useful if the webserver rebooted
            sharded_client
                .clear_cache()
                .await
                .expect("Unable to clear cache");
            tracing::info!("Connected");

            // Binds on localhost
            let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), port);

            // Run server
            server::run(
                compat_return_full_text,
                max_concurrent_requests,
                max_best_of,
                max_stop_sequences,
                max_input_length,
                max_total_tokens,
                max_batch_size,
                max_waiting_tokens,
                sharded_client,
                tokenizer,
                validation_workers,
                addr,
                cors_allow_origin,
            )
            .await;
            Ok(())
        })
}

/// Init logging using env variables LOG_LEVEL and LOG_FORMAT:
///     - otlp_endpoint is an optional URL to an Open Telemetry collector
///     - LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
///     - LOG_FORMAT may be TEXT or JSON (default to TEXT)
fn init_logging(otlp_endpoint: Option<String>, json_output: bool) {
    let mut layers = Vec::new();

    // STDOUT/STDERR layer
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true);

    let fmt_layer = match json_output {
        true => fmt_layer.json().flatten_event(true).boxed(),
        false => fmt_layer.boxed(),
    };
    layers.push(fmt_layer);

    // OpenTelemetry tracing layer
    if let Some(otlp_endpoint) = otlp_endpoint {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(otlp_endpoint),
            )
            .with_trace_config(
                trace::config()
                    .with_resource(Resource::new(vec![KeyValue::new(
                        "service.name",
                        "text-generation-inference.router",
                    )]))
                    .with_sampler(Sampler::AlwaysOn),
            )
            .install_batch(opentelemetry::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            axum_tracing_opentelemetry::init_propagator().unwrap();
        };
    }

    // Filter events with LOG_LEVEL
    let env_filter =
        EnvFilter::try_from_env("LOG_LEVEL").unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();
}
