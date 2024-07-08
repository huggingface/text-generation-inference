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

static WARMUP_IMAGE_BASE64 :&str = "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAABg2lDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV/TSotUROxQxCFDdbKLijjWKhShQqgVWnUwufQLmrQkKS6OgmvBwY/FqoOLs64OroIg+AHi7OCk6CIl/i8ptIjx4Lgf7+497t4BQqvKNDOQADTdMjKppJjLr4rBVwQQwhAERGVm1uckKQ3P8XUPH1/v4jzL+9yfY0AtmAzwicQJVjcs4g3imU2rznmfOMLKskp8Tjxh0AWJH7muuPzGueSwwDMjRjYzTxwhFks9rPQwKxsa8TRxTNV0yhdyLquctzhr1Qbr3JO/MFzQV5a5TnMUKSxiCRJEKGiggiosxGnVSTGRof2kh3/E8UvkUshVASPHAmrQIDt+8D/43a1ZnJp0k8JJoO/Ftj/GgOAu0G7a9vexbbdPAP8zcKV3/bUWMPtJerOrxY6AwW3g4rqrKXvA5Q4QfarLhuxIfppCsQi8n9E35YHhW6B/ze2ts4/TByBLXaVvgINDYLxE2ese7w719vbvmU5/PycecohsjayNAAAACXBIWXMAAC4jAAAuIwF4pT92AAAAB3RJTUUH6AQIEQMnlTSSjwAAABl0RVh0Q29tbWVudABDcmVhdGVkIHdpdGggR0lNUFeBDhcAAAASSURBVDjLY2AYBaNgFIyCoQsABMQAAeRw1DoAAAAASUVORK5CYII=";

pub type Result<T> = std::result::Result<T, ClientError>;
