//! Text Generation gRPC client library

mod client;
#[allow(clippy::derive_partial_eq_without_eq)]
mod pb;
mod sharded_client;

pub use client::Client;
pub use pb::generate::v2::HealthResponse;
pub use pb::generate::v2::InfoResponse as ShardInfo;
pub use pb::generate::v2::{
    Batch, CachedBatch, FinishReason, GeneratedText, Generation, GrammarType,
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
