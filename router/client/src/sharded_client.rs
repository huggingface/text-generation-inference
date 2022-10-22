/// Multi shard Client
use crate::Result;
use crate::{Batch, Client, GeneratedText};
use futures::future::join_all;
use futures::future::select_all;
use tonic::transport::Uri;

/// Text Generation Inference gRPC multi client
pub struct ShardedClient {
    clients: Vec<Client>,
}

impl ShardedClient {
    fn new(clients: Vec<Client>) -> Self {
        Self {
            clients,
        }
    }

    /// Create a new ShardedClient from a master client. The master client will communicate with
    /// the other shards and returns all uris/unix sockets with the `service_discovery` gRPC method.
    async fn from_master_client(mut master_client: Client) -> Result<Self> {
        // Get all uris/unix sockets from the master client
        let uris = master_client.service_discovery().await.unwrap();
        let futures = uris.into_iter().map(Client::connect_uds);
        let clients: Result<Vec<Client>> = join_all(futures).await.into_iter().collect();
        Ok(Self::new(clients?))
    }

    /// Returns a client connected to the given uri
    pub async fn connect(uri: Uri) -> Result<Self> {
        let master_client = Client::connect(uri).await?;
        Self::from_master_client(master_client).await
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let master_client = Client::connect_uds(path).await?;
        Self::from_master_client(master_client).await
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns a list of generated texts of request that met their stopping criteria
    /// and the next cached batch
    pub async fn generate(&mut self, batch: Batch) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.generate(batch.clone())))
            .collect();
        // As soon as we receive one response, we can return as all shards will return the same
        let (result, _, _) = select_all(futures).await;
        result
    }

    /// Generate one token for each request in the given cached batch
    ///
    /// Returns a list of generated texts of request that met their stopping criteria
    /// and the next cached batch
    pub async fn generate_with_cache(
        &mut self,
        batches: Vec<Batch>,
    ) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.generate_with_cache(batches.clone())))
            .collect();
        // As soon as we receive one response, we can return as all shards will return the same
        let (result, _, _) = select_all(futures).await;
        result
    }

    /// Clear the past generations cache
    pub async fn clear_cache(&mut self) -> Result<()> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.clear_cache())
            .collect();
        join_all(futures).await.into_iter().collect()
    }
}
