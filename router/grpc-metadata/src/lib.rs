//! A crate to extract and inject a OpenTelemetry context from and to a gRPC request.
//! Inspired by: https://github.com/open-telemetry/opentelemetry-rust gRPC examples

use opentelemetry::global;
use opentelemetry::propagation::{Extractor, Injector};
use tracing_opentelemetry::OpenTelemetrySpanExt;

/// Extract context metadata from a gRPC request's metadata
struct MetadataExtractor<'a>(pub &'a tonic::metadata::MetadataMap);

impl<'a> Extractor for MetadataExtractor<'a> {
    /// Get a value for a key from the MetadataMap.  If the value can't be converted to &str, returns None
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|metadata| metadata.to_str().ok())
    }

    /// Collect all the keys from the MetadataMap.
    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .map(|key| match key {
                tonic::metadata::KeyRef::Ascii(v) => v.as_str(),
                tonic::metadata::KeyRef::Binary(v) => v.as_str(),
            })
            .collect::<Vec<_>>()
    }
}

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
