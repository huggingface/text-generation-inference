/// Multi shard Client
use crate::Result;
use crate::{Batch, Client, Generation};
use futures::future::join_all;
use futures::future::select_all;
use tonic::transport::Uri;
use tracing::instrument;
use crate::pb::generate::v1::CachedBatch;
use crate::pb::generate::v1::model_info_response::ModelType;

/// Text Generation Inference gRPC multi client
pub struct ShardedClient {
    clients: Vec<Client>,
}

impl ShardedClient {
    fn new(clients: Vec<Client>) -> Self {
        Self { clients }
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

    /// Clear the past generations cache
    #[instrument(skip(self))]
    pub async fn clear_cache(&mut self) -> Result<()> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| client.clear_cache())
            .collect();
        join_all(futures).await.into_iter().collect()
    }

    /// Generate one token for each request in the given batch
    ///
    /// Returns Generation for each request in batch
    /// and the next cached batch
    #[instrument(skip_all, fields(id = &batch.id, size = &batch.size))]
    pub async fn prefill(&mut self, batch: &Batch) -> Result<Vec<Generation>> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.prefill(batch.clone())))
            .collect();
        // As soon as we receive one response, we can return as all shards will return the same
        let (result, _, _) = select_all(futures).await;
        result
    }

    /// Generate one token for each request in the given cached batches
    ///
    /// Returns Generation for each request in batches
    /// and the next cached batch
    #[instrument(skip_all, fields(size))]
    pub async fn decode(
        &mut self,
        batches: Vec<CachedBatch>,
        size: u32,
    ) -> Result<(Vec<Generation>, Option<u64>)> {
        let futures: Vec<_> = self
            .clients
            .iter_mut()
            .map(|client| Box::pin(client.decode(batches.clone(), size)))
            .collect();
        let all_complete = batches.iter().all(|cb| cb.status.is_none());
        if all_complete {
            // Ensure that none of the shard requests are cancelled
            join_all(futures).await.pop().unwrap()
        } else {
            // As soon as we receive one response, we can return as all shards will return the same
            let (result, _, _) = select_all(futures).await;
            result
        }
    }

    /// Get shard model info
    pub async fn model_info(&mut self) -> Result<(bool, u32, bool)> {
        self.clients[0].model_info().await
            .map(|(mt, eos, sst)| (mt == ModelType::Seq2seqLm, eos, sst))
    }
}
