use crate::Result;
use crate::{Batch, Client, GeneratedText};
use futures::future::join_all;
use tokio::sync::{broadcast, mpsc};
use tonic::transport::Uri;

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
    GenerateUntilFinished(
        Batch,
        mpsc::Sender<Result<(Vec<GeneratedText>, Option<Batch>)>>,
    ),
    GenerateUntilFinishedWithCache(
        Vec<Batch>,
        mpsc::Sender<Result<(Vec<GeneratedText>, Option<Batch>)>>,
    ),
    ClearCache(mpsc::Sender<Result<()>>),
}

async fn client_task(mut client: Client, mut request_subscriber: broadcast::Receiver<Command>) {
    while let Ok(message) = request_subscriber.recv().await {
        match message {
            Command::Generate(batch, response_tx) => {
                let result = client.generate(batch).await;
                response_tx.try_send(result).unwrap_or(());
            }
            Command::GenerateWithCache(batches, response_tx) => {
                let result = client.generate_with_cache(batches).await;
                response_tx.try_send(result).unwrap_or(());
            }
            Command::GenerateUntilFinished(batch, response_tx) => {
                let result = client.generate_until_finished(batch).await;
                response_tx.try_send(result).unwrap_or(());
            }
            Command::GenerateUntilFinishedWithCache(batches, response_tx) => {
                let result = client.generate_until_finished_with_cache(batches).await;
                response_tx.try_send(result).unwrap_or(());
            }
            Command::ClearCache(response_tx) => {
                let result = client.clear_cache().await;
                response_tx.try_send(result).unwrap_or(());
            }
        };
    }
}

pub struct ShardedClient {
    request_tx: broadcast::Sender<Command>,
}

impl ShardedClient {
    fn new(mut clients: Vec<Client>) -> Self {
        let (request_tx, _) = broadcast::channel(1);

        for client in clients.drain(..) {
            let request_subscriber = request_tx.subscribe();
            tokio::spawn(client_task(client, request_subscriber));
        }

        Self { request_tx }
    }

    async fn from_master_client(mut master_client: Client) -> Result<Self> {
        let uris = master_client.service_discovery().await.unwrap();
        let futures = uris.into_iter().map(|path| Client::connect_uds(path));
        let clients: Result<Vec<Client>> = join_all(futures).await.into_iter().collect();
        Ok(Self::new(clients?))
    }

    /// Returns a client connected to the given url
    pub async fn connect(uri: Uri) -> Result<Self> {
        let master_client = Client::connect(uri).await?;
        Self::from_master_client(master_client).await
    }

    /// Returns a client connected to the given unix socket
    pub async fn connect_uds(path: String) -> Result<Self> {
        let master_client = Client::connect_uds(path).await?;
        Self::from_master_client(master_client).await
    }

    pub async fn generate(&self, batch: Batch) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let (response_tx, mut response_rx) = mpsc::channel(1);
        self.request_tx
            .send(Command::Generate(batch, response_tx))
            .unwrap();
        response_rx.recv().await.unwrap()
    }

    pub async fn generate_with_cache(
        &self,
        batches: Vec<Batch>,
    ) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let (response_tx, mut response_rx) = mpsc::channel(1);
        self.request_tx
            .send(Command::GenerateWithCache(batches, response_tx))
            .unwrap();
        response_rx.recv().await.unwrap()
    }

    pub async fn generate_until_finished(
        &self,
        batch: Batch,
    ) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let (response_tx, mut response_rx) = mpsc::channel(1);
        self.request_tx
            .send(Command::GenerateUntilFinished(batch, response_tx))
            .unwrap();
        response_rx.recv().await.unwrap()
    }

    pub async fn generate_until_finished_with_cache(
        &self,
        batches: Vec<Batch>,
    ) -> Result<(Vec<GeneratedText>, Option<Batch>)> {
        let (response_tx, mut response_rx) = mpsc::channel(1);
        self.request_tx
            .send(Command::GenerateUntilFinishedWithCache(
                batches,
                response_tx,
            ))
            .unwrap();
        response_rx.recv().await.unwrap()
    }

    pub async fn clear_cache(&self) -> Result<()> {
        let (response_tx, mut response_rx) = mpsc::channel(1);
        self.request_tx
            .send(Command::ClearCache(response_tx))
            .unwrap();
        response_rx.recv().await.unwrap()
    }
}
