//! BLOOM Inference gRPC client library

mod client;
mod pb;
mod sharded_client;

pub use client::Client;
pub use pb::generate::v1::{
    Batch, BatchCached, CacheEntry, FinishedGeneration, LogitsWarperParameters, Request,
};
pub use sharded_client::ShardedClient;
use thiserror::Error;
pub use tonic::transport::Uri;
use tonic::Status;

#[derive(Error, Debug, Clone)]
#[error("Text generation client error: {msg:?}")]
pub struct ClientError {
    msg: String,
    // source: Status,
}

impl From<Status> for ClientError {
    fn from(err: Status) -> Self {
        Self {
            msg: err.to_string(),
            // source: err,
        }
    }
}

pub type Result<T> = std::result::Result<T, ClientError>;
