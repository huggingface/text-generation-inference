use crate::pb::generate::v1::text_generation_service_client::TextGenerationServiceClient;
use crate::pb::generate::v1::*;
use crate::Result;
use tonic::transport::{Channel, Uri};
use tracing::*;

/// BLOOM Inference gRPC client
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

    #[instrument(skip(self))]
    pub async fn service_discovery(&mut self) -> Result<Vec<String>> {
        let request = tonic::Request::new(ServiceDiscoveryRequest {});
        let response = self
            .stub
            .service_discovery(request)
            .instrument(info_span!("service_discovery"))
            .await?;
        let urls = response
            .into_inner()
            .urls
            .into_iter()
            .map(|url| match url.strip_prefix("unix://") {
                None => url,
                Some(stripped_url) => stripped_url.to_string(),
            })
            .collect();
        Ok(urls)
    }

    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self) -> Result<()> {
        let request = tonic::Request::new(ClearCacheRequest {});
        self.stub
            .clear_cache(request)
            .instrument(info_span!("clear_cache"))
            .await?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn generate(&mut self, batch: Batch) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let request = tonic::Request::new(GenerateRequest { batch: Some(batch) });
        let response = self
            .stub
            .generate(request)
            .instrument(info_span!("generate"))
            .await?
            .into_inner();
        Ok((response.generated_texts, response.batch))
    }

    #[instrument(skip(self))]
    pub async fn generate_with_cache(
        &mut self,
        batches: Vec<Batch>,
    ) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let request = tonic::Request::new(GenerateWithCacheRequest { batches });
        let response = self
            .stub
            .generate_with_cache(request)
            .instrument(info_span!("generate_with_cache"))
            .await?
            .into_inner();
        Ok((response.generated_texts, response.batch))
    }

    #[instrument(skip(self))]
    pub async fn generate_until_finished(
        &mut self,
        batch: Batch,
    ) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let request = tonic::Request::new(GenerateUntilFinishedRequest { batch: Some(batch) });
        let response = self
            .stub
            .generate_until_finished(request)
            .instrument(info_span!("generate_until_finished"))
            .await?
            .into_inner();
        Ok((response.generated_texts, response.batch))
    }

    #[instrument(skip(self))]
    pub async fn generate_until_finished_with_cache(
        &mut self,
        batches: Vec<Batch>,
    ) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let request = tonic::Request::new(GenerateUntilFinishedWithCacheRequest { batches });
        let response = self
            .stub
            .generate_until_finished_with_cache(request)
            .instrument(info_span!("generate_until_finished_with_cache"))
            .await?
            .into_inner();
        Ok((response.generated_texts, response.batch))
    }
}
