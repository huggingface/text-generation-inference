/// Single shard Client
use crate::pb::generate::v1::text_generation_service_client::TextGenerationServiceClient;
use crate::pb::generate::v1::*;
use crate::{ClientError, Result};
use grpc_metadata::InjectTelemetryContext;
use tonic::transport::{Channel, Uri};
use tracing::instrument;
use crate::pb::generate::v1::model_info_response::ModelType;

/// Text Generation Inference gRPC client
#[derive(Clone)]
pub struct Client {
    stub: TextGenerationServiceClient<Channel>,
}

impl Client {
    /// Returns a client connected to the given url
    pub async fn connect(uri: Uri) -> Result<Self> {
        let channel = Channel::builder(uri).connect().await?;

        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let channel = Channel::from_shared("http://[::]:50051".to_string())
            .unwrap()
            .connect_with_connector(tower::service_fn(move |_: Uri| {
                tokio::net::UnixStream::connect(path.clone())
            }))
            .await?;

        Ok(Self {
            stub: TextGenerationServiceClient::new(channel),
        })
    }

    /// Returns a list of uris or unix sockets of all shards
    #[instrument(skip(self))]
    pub async fn service_discovery(&mut self) -> Result<Vec<String>> {
        let request = tonic::Request::new(ServiceDiscoveryRequest {}).inject_context();
        let response = self.stub.service_discovery(request).await?;
        let urls = response
            .into_inner()
            .urls
            .into_iter()
            // Remove unix socket prefix
            .map(|url| match url.strip_prefix("unix://") {
                None => url,
                Some(stripped_url) => stripped_url.to_string(),
            })
            .collect();
        Ok(urls)
    }

    /// Clear the past generations cache
    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self) -> Result<()> {
        let request = tonic::Request::new(ClearCacheRequest {}).inject_context();
        self.stub.clear_cache(request).await?;
        Ok(())
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns Generation for each request in batch
    /// and the next cached batch
    #[instrument(skip_all, fields(id = &batch.id, size = &batch.size))]
    pub async fn prefill(&mut self, batch: Batch) -> Result<Vec<Generation>> {
        let request = tonic::Request::new(PrefillRequest { batch: Some(batch) }).inject_context();
        let response = self.stub.prefill(request).await?.into_inner();
        Ok(response.generations)
    }

    /// Generate one token for each request in the given cached batches
    ///
    /// Returns Generation for each request in batches
    /// and the next cached batch
    #[instrument(skip_all, fields(size = _size))]
    pub async fn decode(
        &mut self,
        batches: Vec<CachedBatch>,
        _size: u32,
    ) -> Result<(Vec<Generation>, Option<u64>)> {
        let request = tonic::Request::new(DecodeRequest { batches }).inject_context();
        let response = self.stub.decode(request).await?.into_inner();
        Ok((response.generations, response.batch_id))
    }

    /// Get shard model info
    #[instrument(skip(self))]
    pub async fn model_info(&mut self) -> Result<(ModelType, u32, bool)> {
        let request = tonic::Request::new(ModelInfoRequest {}).inject_context();
        let response = self.stub.model_info(request).await?.into_inner();
        ModelType::from_i32(response.model_type)
            .map(|mt| (mt, response.eos_token, response.skip_special_tokens))
            .ok_or(ClientError::Generation("Unrecognized model type".to_string()))
    }
}
