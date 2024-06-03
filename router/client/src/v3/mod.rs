#[allow(clippy::derive_partial_eq_without_eq)]
mod pb;

mod client;
mod sharded_client;

pub use client::Client;
pub use pb::generate::v3::HealthResponse;
pub use pb::generate::v3::{
    Batch, CachedBatch, FinishReason, GeneratedText, Generation, GrammarType, InfoResponse,
    NextTokenChooserParameters, Request, StoppingCriteriaParameters, Tokens,
};
pub use sharded_client::ShardedClient;
