use crate::pb::generate::v1::text_generation_client::TextGenerationClient;
use crate::pb::generate::v1::*;
use crate::Result;
use std::time::Duration;
use tonic::transport::{Channel, Uri};
use tower::timeout::Timeout;
use tracing::*;

/// BLOOM Inference gRPC client
#[derive(Clone)]
pub struct Client {
    stub: TextGenerationClient<Timeout<Channel>>,
}

impl Client {
    /// Returns a client connected to the given url. Requests exceeding timeout will fail.
    pub async fn connect(uri: Uri, timeout: Duration) -> Self {
        let channel = Channel::builder(uri)
            .connect()
            .await
            .expect("Transport error");
        let timeout_channel = Timeout::new(channel, timeout);

        Self {
            stub: TextGenerationClient::new(timeout_channel),
        }
    }

    /// Returns a client connected to the given unix socket. Requests exceeding timeout will fail.
    pub async fn connect_uds(path: String, timeout: Duration) -> Self {
        let channel = Channel::from_shared(format!("http://[::]:50051"))
            .unwrap()
            .connect_with_connector(tower::service_fn(move |_: Uri| {
                tokio::net::UnixStream::connect(path.clone())
            }))
            .await
            .expect("Transport error");
        let timeout_channel = Timeout::new(channel, timeout);

        Self {
            stub: TextGenerationClient::new(timeout_channel),
        }
    }

    #[instrument(skip(self))]
    pub async fn service_discovery(&mut self) -> Result<Vec<String>> {
        let request = tonic::Request::new(Empty {});
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
        let request = tonic::Request::new(Empty {});
        self.stub
            .clear_cache(request)
            .instrument(info_span!("clear_cache"))
            .await?;
        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn generate(
        &mut self,
        request: Batch,
    ) -> Result<(Vec<FinishedGeneration>, Option<CacheEntry>)> {
        let request = tonic::Request::new(request);
        let response = self
            .stub
            .generate(request)
            .instrument(info_span!("generate"))
            .await?
            .into_inner();
        Ok((response.finished, response.cache_entry))
    }

    #[instrument(skip(self))]
    pub async fn generate_with_cache(
        &mut self,
        request: BatchCached,
    ) -> Result<(Vec<FinishedGeneration>, Option<CacheEntry>)> {
        let request = tonic::Request::new(request);
        let response = self
            .stub
            .generate_with_cache(request)
            .instrument(info_span!("generate_with_cache"))
            .await?
            .into_inner();
        Ok((response.finished, response.cache_entry))
    }
}
