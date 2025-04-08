use axum::{
    body::Body,
    extract::{Request, State},
    http::uri::Uri,
    response::{IntoResponse, Response},
    Json,
};
use futures_util::stream::StreamExt;
use hyper::StatusCode;
use hyper_util::{client::legacy::connect::HttpConnector, rt::TokioExecutor};
use rand::{rng, Rng};
use serde::Deserialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{mpsc, oneshot};

mod trie;

use crate::trie::Trie;

const FACTOR_KEY: &str = "TGI_KVROUTER_FACTOR";
type Client = hyper_util::client::legacy::Client<HttpConnector, Body>;

pub struct ContentAware {
    trie: Trie,
}

impl Default for ContentAware {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentAware {
    pub fn new() -> Self {
        let trie = Trie::new();
        Self { trie }
    }
}

impl LoadBalancer for ContentAware {
    fn next(&mut self, key: &[u8], n_backends: usize) -> usize {
        let trie = &mut self.trie;
        let (start, stop) = trie.insert(key);
        let n = trie.count();
        eprintln!(
            "Start {start} Stop {stop} N {n} : Key {}",
            String::from_utf8_lossy(key)
        );
        let mut rng = rng();
        let x: f32 = rng.random();
        println!("Random number is {x:.2}");
        let start = (start as f32) / (n as f32);
        let stop = (stop as f32) / (n as f32);
        let rescaled_x = x * (stop - start) + start;
        assert!(rescaled_x >= start);
        assert!(rescaled_x <= stop);
        assert!(rescaled_x >= 0.0);
        assert!(rescaled_x <= 1.0);
        println!("Start {start:.2} stop {stop:.2}: rescaled {rescaled_x:.2}");
        let n: usize = (rescaled_x * (n_backends as f32)) as usize;
        n
    }
}

pub struct RoundRobin {
    current: AtomicUsize,
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRobin {
    pub fn new() -> Self {
        let current = AtomicUsize::new(0);
        Self { current }
    }
}

impl LoadBalancer for RoundRobin {
    fn next(&mut self, _key: &[u8], _n_backends: usize) -> usize {
        self.current.fetch_add(1, Ordering::Relaxed)
    }
}

pub struct OverloadHandler<T: LoadBalancer> {
    load_balancer: T,
    backends: Vec<String>,
    inqueue: Vec<AtomicUsize>,
    inflight: Vec<AtomicUsize>,
    factor: f32,
    rx: Rcv,
}

impl<T: LoadBalancer> OverloadHandler<T> {
    pub async fn new(load_balancer: T, backends: Vec<String>, rx: Rcv) -> Self {
        let inflight = backends.iter().map(|_| AtomicUsize::new(0)).collect();
        let inqueue = backends.iter().map(|_| AtomicUsize::new(0)).collect();
        let factor: f32 = std::env::var(FACTOR_KEY)
            .unwrap_or("1.5".to_string())
            .parse()
            .unwrap_or(1.5);
        Self {
            load_balancer,
            backends,
            factor,
            inflight,
            inqueue,
            rx,
        }
    }

    async fn next(&mut self, key: &[u8]) -> Option<String> {
        if self.backends.is_empty() {
            return None;
        }
        // Get the backend URL
        let index = self.load_balancer.next(key, self.backends.len());
        let n = self.backends.len();
        let mut index = index % n;

        let mut inflight = self.inflight[index].load(Ordering::Relaxed);
        let mut inqueue = self.inqueue[index].load(Ordering::Relaxed);

        for i in 0..n {
            if (inqueue as f32) <= self.factor * inflight as f32 {
                break;
            }
            if i == 0 {
                eprintln!(
                    "Backend overloaded (queue: {inqueue} inflight {inflight}), jumping ahead"
                );
            }
            index += 1;
            index %= self.backends.len();
            inflight = self.inflight[index].load(Ordering::Relaxed);
            inqueue = self.inflight[index].load(Ordering::Relaxed);
        }
        let backend = &self.backends[index];
        self.inflight[index].fetch_add(1, Ordering::Relaxed);
        self.inqueue[index].fetch_add(1, Ordering::Relaxed);
        Some(backend.to_string())
    }

    pub async fn run(&mut self) {
        while let Some(msg) = self.rx.recv().await {
            eprintln!("Msg {msg:?}");
            match msg {
                Msg::Next(key, sx) => {
                    let Some(backend) = self.next(&key).await else {
                        drop(sx);
                        return;
                    };
                    eprintln!("Sending back backend {backend}");
                    if let Err(err) = sx.send(backend) {
                        eprintln!("Cannot send back result: {err}");
                    }
                }
                Msg::Dequeue(backend) => {
                    let index = self.backends.iter().position(|b| b == &backend);
                    if let Some(index) = index {
                        self.inqueue[index].fetch_sub(1, Ordering::Relaxed);
                    }
                }
                Msg::Deflight(backend) => {
                    let index = self.backends.iter().position(|b| b == &backend);
                    if let Some(index) = index {
                        self.inflight[index].fetch_sub(1, Ordering::Relaxed);
                    }
                }
                Msg::AddBackend(backend) => {
                    self.backends.push(backend);
                    self.backends.sort();
                }
                Msg::RemoveBackend(backend) => {
                    self.backends.retain(|b| *b == backend);
                    self.backends.sort();
                }
                Msg::SetBackends(backends) => {
                    self.backends = backends;
                }
            }
        }
    }
}

pub trait LoadBalancer {
    fn next(&mut self, key: &[u8], n_backends: usize) -> usize;
}

#[derive(Debug)]
pub enum Msg {
    Next(Vec<u8>, oneshot::Sender<String>),
    Dequeue(String),
    Deflight(String),
    AddBackend(String),
    RemoveBackend(String),
    SetBackends(Vec<String>),
}

type Snd = mpsc::Sender<Msg>;
type Rcv = mpsc::Receiver<Msg>;

#[derive(Clone)]
pub struct Communicator {
    sender: Snd,
    client: Client,
}

impl Communicator {
    pub fn new(sender: Snd) -> Self {
        let client = hyper_util::client::legacy::Client::<(), ()>::builder(TokioExecutor::new())
            .build(HttpConnector::new());
        Self { sender, client }
    }

    async fn dequeue(&self, backend: String) -> Result<(), mpsc::error::SendError<Msg>> {
        self.sender.send(Msg::Dequeue(backend)).await
    }

    async fn deflight(&self, backend: String) -> Result<(), mpsc::error::SendError<Msg>> {
        self.sender.send(Msg::Deflight(backend)).await
    }

    async fn next(&self, key: Vec<u8>) -> Result<String, mpsc::error::SendError<Msg>> {
        let (sx, rx) = oneshot::channel();
        self.sender.send(Msg::Next(key, sx)).await?;
        let backend = rx
            .await
            .map_err(|_| mpsc::error::SendError(Msg::AddBackend("todo".to_string())))?;
        Ok(backend)
    }
}

pub async fn handler(
    State(state): State<Communicator>,
    req: Request,
) -> Result<Response<Body>, StatusCode> {
    // Get the next backend index
    let (parts, body) = req.into_parts();
    let mut response_stream = body.into_data_stream();
    let event = response_stream.next().await;
    let key = if let Some(Ok(event)) = &event {
        event.to_vec()
    } else {
        vec![]
    };
    let backend = state.next(key).await.map_err(|_| StatusCode::BAD_GATEWAY)?;
    let response_stream = async_stream::stream! {
        let mut response_stream = Box::pin(response_stream);
        if let Some(event) = event{
            yield event;
        }
        while let Some(raw_event) = response_stream.next().await {
            yield raw_event;
        }
    };
    let body = Body::from_stream(response_stream);
    let mut req = Request::from_parts(parts, body);
    let path = req.uri().path();
    let path_query = req
        .uri()
        .path_and_query()
        .map(|v| v.as_str())
        .unwrap_or(path);

    let uri = format!("{backend}{path_query}");
    eprintln!("Inflight {uri}");
    *req.uri_mut() = Uri::try_from(uri).unwrap();

    let response = state
        .client
        .request(req)
        .await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    let response = response.into_response();
    let (parts, body) = response.into_parts();
    let response_stream = body.into_data_stream();
    let response_stream = async_stream::stream! {
        let mut response_stream = Box::pin(response_stream);
        let mut start = true;
        while let Some(raw_event) = response_stream.next().await {
            if start{
                eprintln!("Not inqueue");

                state.dequeue(backend.to_string()).await.unwrap();
                start = false;
            }
            yield raw_event;
        }
        eprintln!("Not inflight");
        state.deflight(backend.to_string()).await.unwrap();
    };

    let body = Body::from_stream(response_stream);

    Ok(Response::from_parts(parts, body))
}

#[derive(Deserialize)]
pub struct SetBackends {
    backends: Vec<String>,
}

pub async fn set_backends_handler(
    State(state): State<Communicator>,
    Json(SetBackends { backends }): Json<SetBackends>,
) -> impl IntoResponse {
    let _ = state.sender.send(Msg::SetBackends(backends)).await;
    StatusCode::OK
}
