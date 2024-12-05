//! Text Generation gRPC client library

pub mod v2;
pub mod v3;

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine};
use thiserror::Error;
use tonic::transport;
use tonic::Status;

pub use v3::{Chunk, Image, Input, InputChunk};

#[async_trait]
pub trait Health {
    /// Check if a generate server is healthy by asking it to allocate a tensor on device
    async fn device_health(&self) -> Result<()>;

    /// Check if a generate server is healthy by doing a forward pass.
    /// EXPENSIVE
    async fn model_health(&self) -> Result<()>;
}

#[derive(Debug)]
pub struct ShardInfo {
    pub requires_padding: bool,
    pub dtype: String,
    pub device_type: String,
    pub window_size: Option<u32>,
    pub speculate: u32,
}

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

static WARMUP_IMAGE_BASE64 : &str = "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";

pub type Result<T> = std::result::Result<T, ClientError>;
