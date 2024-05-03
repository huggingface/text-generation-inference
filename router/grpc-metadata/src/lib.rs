//! A crate to extract and inject a OpenTelemetry context from and to a gRPC request.
//! Inspired by: https://github.com/open-telemetry/opentelemetry-rust gRPC examples

use opentelemetry::global;
use opentelemetry::propagation::Injector;
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Inject context in the metadata of a gRPC request.
struct MetadataInjector<'a>(pub &'a mut tonic::metadata::MetadataMap);

impl<'a> Injector for MetadataInjector<'a> {
    /// Set a key and value in the MetadataMap.  Does nothing if the key or value are not valid inputs
    fn set(&mut self, key: &str, value: String) {
        if let Ok(key) = tonic::metadata::MetadataKey::from_bytes(key.as_bytes()) {
            if let Ok(val) = value.parse() {
                self.0.insert(key, val);
            }
        }
    }
}

/// Get a context from the global context and inject the span into a gRPC request's metadata.
fn inject(metadata: &mut tonic::metadata::MetadataMap) {
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(
            &tracing::Span::current().context(),
            &mut MetadataInjector(metadata),
        )
    })
}

pub trait InjectTelemetryContext {
    fn inject_context(self) -> Self;
}

impl<T> InjectTelemetryContext for tonic::Request<T> {
    fn inject_context(mut self) -> Self {
        inject(self.metadata_mut());
        self
    }
}
