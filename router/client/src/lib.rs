//! Text Generation gRPC client library

mod client;
#[allow(clippy::derive_partial_eq_without_eq)]
mod pb;
mod sharded_client;

use base64::{engine::general_purpose::STANDARD, Engine};
pub use client::Client;
pub use pb::generate::v2::input_chunk::Chunk;
pub use pb::generate::v2::HealthResponse;
pub use pb::generate::v2::Image;
pub use pb::generate::v2::InfoResponse as ShardInfo;
pub use pb::generate::v2::{
    Batch, CachedBatch, FinishReason, GeneratedText, Generation, GrammarType, Input, InputChunk,
    NextTokenChooserParameters, Request, StoppingCriteriaParameters, Tokens,
};
pub use sharded_client::ShardedClient;
use thiserror::Error;
use tonic::transport;
use tonic::Status;

#[derive(Error, Debug, Clone)]
pub enum ClientError {
    #[error("Could not connect to Text Generation server: {0}")]
    Connection(String),
    #[error("Server error: {0}")]
    Generation(String),
    #[error("Sharded results are empty")]
    EmptyResults,
}

impl From<Status> for ClientError {
    fn from(err: Status) -> Self {
        let err = Self::Generation(err.message().to_string());
        tracing::error!("{err}");
        err
    }
}

impl From<transport::Error> for ClientError {
    fn from(err: transport::Error) -> Self {
        let err = Self::Connection(err.to_string());
        tracing::error!("{err}");
        err
    }
}

pub type Result<T> = std::result::Result<T, ClientError>;

// Small convenience re-wrapping of `Chunk`.
impl From<Chunk> for InputChunk {
    fn from(chunk: Chunk) -> Self {
        InputChunk { chunk: Some(chunk) }
    }
}

/// Convert input chunks to a stringly-typed input for backwards
/// compat for backends that haven't implemented chunked inputs.
pub trait ChunksToString {
    /// Convert chunks to string.
    fn chunks_to_string(&self) -> String;
}

impl ChunksToString for Vec<InputChunk> {
    fn chunks_to_string(&self) -> String {
        let mut output = String::new();
        self.iter().for_each(|c| match &c.chunk {
            Some(Chunk::Text(text)) => output.push_str(text),
            Some(Chunk::Image(Image { data, mimetype })) => {
                let encoded = STANDARD.encode(data);
                output.push_str(&format!("![](data:{};base64,{})", mimetype, encoded))
            }
            // We don't create empty chunks, so this should be unreachable.
            None => unreachable!("Chunks should never be empty"),
        });
        output
    }
}
