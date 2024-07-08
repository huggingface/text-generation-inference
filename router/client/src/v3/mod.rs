#[allow(clippy::derive_partial_eq_without_eq)]
mod pb;

mod client;
mod sharded_client;

pub use client::Client;
pub use pb::generate::v3::{
    input_chunk::Chunk, Batch, CachedBatch, FinishReason, GeneratedText, Generation, GrammarType,
    HealthResponse, Image, InfoResponse, Input, InputChunk, NextTokenChooserParameters, Request,
    StoppingCriteriaParameters, Tokens,
};
pub use sharded_client::ShardedClient;
