use axum::{extract::Request, middleware::Next, response::Response};
use opentelemetry::sdk::propagation::TraceContextPropagator;
use opentelemetry::sdk::trace;
use opentelemetry::sdk::trace::Sampler;
use opentelemetry::sdk::Resource;
use opentelemetry::trace::{SpanContext, SpanId, TraceContextExt, TraceFlags, TraceId};
use opentelemetry::Context;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{filter::LevelFilter, EnvFilter, Layer};

struct TraceParent {
    #[allow(dead_code)]
    version: u8,
    trace_id: TraceId,
    parent_id: SpanId,
    trace_flags: TraceFlags,
}

fn parse_traceparent(header_value: &str) -> Option<TraceParent> {
    let parts: Vec<&str> = header_value.split('-').collect();
    if parts.len() != 4 {
        return None;
    }

    let version = u8::from_str_radix(parts[0], 16).ok()?;
    if version == 0xff {
        return None;
    }

    let trace_id = TraceId::from_hex(parts[1]).ok()?;
    let parent_id = SpanId::from_hex(parts[2]).ok()?;
    let trace_flags = u8::from_str_radix(parts[3], 16).ok()?;

    Some(TraceParent {
        version,
        trace_id,
        parent_id,
        trace_flags: TraceFlags::new(trace_flags),
    })
}

pub async fn trace_context_middleware(mut request: Request, next: Next) -> Response {
    let context = request
        .headers()
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_traceparent)
        .map(|traceparent| {
            Context::new().with_remote_span_context(SpanContext::new(
                traceparent.trace_id,
                traceparent.parent_id,
                traceparent.trace_flags,
                true,
                Default::default(),
            ))
        });

    request.extensions_mut().insert(context);

    next.run(request).await
}

/// Init logging using env variables LOG_LEVEL and LOG_FORMAT:
///     - otlp_endpoint is an optional URL to an Open Telemetry collector
///     - otlp_service_name service name to appear in APM
///     - LOG_LEVEL may be TRACE, DEBUG, INFO, WARN or ERROR (default to INFO)
///     - LOG_FORMAT may be TEXT or JSON (default to TEXT)
///     - LOG_COLORIZE may be "false" or "true" (default to "true" or ansi supported platforms)
pub fn init_logging(otlp_endpoint: Option<String>, otlp_service_name: String, json_output: bool) {
    let mut layers = Vec::new();

    // STDOUT/STDERR layer
    let ansi = std::env::var("LOG_COLORIZE") != Ok("1".to_string());
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_ansi(ansi)
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
                        otlp_service_name,
                    )]))
                    .with_sampler(Sampler::AlwaysOn),
            )
            .install_batch(opentelemetry::runtime::Tokio);

        if let Ok(tracer) = tracer {
            layers.push(tracing_opentelemetry::layer().with_tracer(tracer).boxed());
            init_tracing_opentelemetry::init_propagator().unwrap();
        };
    }

    // Filter events with LOG_LEVEL
    let varname = "LOG_LEVEL";
    let env_filter = if let Ok(log_level) = std::env::var(varname) {
        // Override to avoid simple logs to be spammed with tokio level informations
        let log_level = match &log_level[..] {
            "warn" => "text_generation_launcher=warn,text_generation_router=warn",
            "info" => "text_generation_launcher=info,text_generation_router=info",
            "debug" => "text_generation_launcher=debug,text_generation_router=debug",
            log_level => log_level,
        };
        EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .parse_lossy(log_level)
    } else {
        EnvFilter::new("info")
    };

    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .init();
}
