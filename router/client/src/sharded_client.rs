/// Multi shard Client
use crate::Result;
use crate::{Batch, Client, GeneratedText};
use futures::future::join_all;
use tokio::sync::{broadcast, mpsc};
use tonic::transport::Uri;

/// List of all available commands that can be sent through the command channel
#[derive(Clone, Debug)]
enum Command {
    Generate(
        Batch,
        mpsc::Sender<Result<(Vec<GeneratedText>, Option<Batch>)>>,
    ),
    GenerateWithCache(
        Vec<Batch>,
        mpsc::Sender<Result<(Vec<GeneratedText>, Option<Batch>)>>,
    ),
    ClearCache(mpsc::Sender<Result<()>>),
}

/// Tokio task that handles the communication with a single shard
///
/// We subscribe on a broadcast channel to receive commands that will be sent by
/// the ShardedClient.
///
/// Each command is fan out to all shards.
///
/// The result of the command is sent back to the ShardedClient through a mpsc channel (multi
/// producer = the shards, single consumer = the ShardedClient).
async fn client_task(mut client: Client, mut request_subscriber: broadcast::Receiver<Command>) {
    while let Ok(message) = request_subscriber.recv().await {
        match message {
            Command::Generate(batch, response_tx) => {
                let result = client.generate(batch).await;
                // We can unwrap_or(()) here because the only error that can happen is if the
                // receiver is dropped, which means that the ShardedClient already received a
                // response from another shard
                response_tx.try_send(result).unwrap_or(());
            }
            Command::GenerateWithCache(batches, response_tx) => {
                let result = client.generate_with_cache(batches).await;
                response_tx.try_send(result).unwrap_or(());
            }
            Command::ClearCache(response_tx) => {
                let result = client.clear_cache().await;
                response_tx.try_send(result).unwrap_or(());
            }
        };
    }
}

/// Text Generation Inference gRPC multi client
pub struct ShardedClient {
    _clients: Vec<Client>,
    request_tx: broadcast::Sender<Command>,
}

impl ShardedClient {
    fn new(clients: Vec<Client>) -> Self {
        // The broadcast channel to communicate with the shards
        // We use a capacity of one as the shards are not asynchronous and can only process one
        // command at a time
        let (request_tx, _) = broadcast::channel(1);

        // Spawn client tasks
        for client in clients.iter() {
            let request_subscriber = request_tx.subscribe();
            tokio::spawn(client_task(client.clone(), request_subscriber));
        }

        Self {
            _clients: clients,
            request_tx,
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
    pub async fn generate(&self, batch: Batch) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        // Create a channel to receive the response from the shards
        // We will only ever receive one message on this channel
        let (response_tx, mut response_rx) = mpsc::channel(1);
        self.request_tx
            .send(Command::Generate(batch, response_tx))
            .unwrap();
        // As soon as we receive one response, we can return as all shards will return the same
        response_rx.recv().await.unwrap()
    }

    /// Generate one token for each request in the given cached batch
    ///
    /// Returns a list of generated texts of request that met their stopping criteria
    /// and the next cached batch
    pub async fn generate_with_cache(
        &self,
        batches: Vec<Batch>,
    ) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        // Create a channel to receive the response from the shards
        // We will only ever receive one message on this channel
        let (response_tx, mut response_rx) = mpsc::channel(1);
        self.request_tx
            .send(Command::GenerateWithCache(batches, response_tx))
            .unwrap();
        // As soon as we receive one response, we can return as all shards will return the same
        response_rx.recv().await.unwrap()
    }

    /// Clear the past generations cache
    pub async fn clear_cache(&self) -> Result<()> {
        // Create a channel to receive the response from the shards
        // We will only ever receive one message on this channel
        let (response_tx, mut response_rx) = mpsc::channel(1);
        self.request_tx
            .send(Command::ClearCache(response_tx))
            .unwrap();
        response_rx.recv().await.unwrap()
    }
}
