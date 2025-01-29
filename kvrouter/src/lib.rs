use axum::{
    body::Body,
    extract::{Request, State},
    http::uri::Uri,
    response::{IntoResponse, Response},
};
use futures_util::stream::StreamExt;
use hyper_util::{client::legacy::connect::HttpConnector, rt::TokioExecutor};
use rand::{rng, Rng};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

mod trie;

use crate::trie::Trie;

const FACTOR_KEY: &str = "TGI_KVROUTER_FACTOR";
type Client = hyper_util::client::legacy::Client<HttpConnector, Body>;

#[derive(Clone)]
pub struct ContentAware {
    trie: Arc<Mutex<Trie>>,
}

impl Default for ContentAware {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentAware {
    pub fn new() -> Self {
        let trie = Arc::new(Mutex::new(Trie::new()));
        Self { trie }
    }
}

impl LoadBalancer for ContentAware {
    fn next(&mut self, key: &[u8], n_backends: usize) -> usize {
        let mut trie = self.trie.lock().unwrap();
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

#[derive(Clone)]
pub struct RoundRobin {
    current: Arc<AtomicUsize>,
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRobin {
    pub fn new() -> Self {
        let current = Arc::new(AtomicUsize::new(0));
        Self { current }
    }
}

impl LoadBalancer for RoundRobin {
    fn next(&mut self, _key: &[u8], _n_backends: usize) -> usize {
        self.current.fetch_add(1, Ordering::Relaxed)
    }
}

#[derive(Clone)]
pub struct OverloadHandler<T: LoadBalancer> {
    client: Client,
    load_balancer: T,
    backends: Arc<Vec<String>>,
    inqueue: Arc<Vec<AtomicUsize>>,
    inflight: Arc<Vec<AtomicUsize>>,
    factor: f32,
}

impl<T: LoadBalancer> OverloadHandler<T> {
    pub fn new(load_balancer: T, backends: Vec<String>) -> Self {
        let client = hyper_util::client::legacy::Client::<(), ()>::builder(TokioExecutor::new())
            .build(HttpConnector::new());
        let inflight = Arc::new(backends.iter().map(|_| AtomicUsize::new(0)).collect());
        let inqueue = Arc::new(backends.iter().map(|_| AtomicUsize::new(0)).collect());
        let factor: f32 = std::env::var(FACTOR_KEY)
            .unwrap_or("1.5".to_string())
            .parse()
            .unwrap_or(1.5);
        let backends = Arc::new(backends);
        Self {
            load_balancer,
            backends,
            client,
            factor,
            inflight,
            inqueue,
        }
    }

    fn next(&mut self, key: &[u8]) -> usize {
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
        index
    }
}

pub trait LoadBalancer {
    fn next(&mut self, key: &[u8], n_backends: usize) -> usize;
}

pub async fn handler<T: LoadBalancer>(
    State(mut state): State<OverloadHandler<T>>,
    req: Request,
) -> Response<Body> {
    // Get the next backend index
    let limit = 1024 * 1024;
    let (parts, body) = req.into_parts();
    // TODO
    let bytes = axum::body::to_bytes(body, limit).await.unwrap();
    let index = state.next(&bytes);
    let backend = &state.backends[index];
    state.inflight[index].fetch_add(1, Ordering::Relaxed);
    state.inqueue[index].fetch_add(1, Ordering::Relaxed);

    let body: Body = bytes.into();
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
        // TODO
        .unwrap();
    //.map_err(|_| StatusCode::BAD_GATEWAY)?;
    let response = response.into_response();
    let (parts, body) = response.into_parts();
    let response_stream = body.into_data_stream();
    let response_stream = async_stream::stream! {
        let mut response_stream = Box::pin(response_stream);
        let mut start = true;
        while let Some(raw_event) = response_stream.next().await {
            if start{
                eprintln!("Not inqueue");

                state.inqueue[index].fetch_sub(1, Ordering::Relaxed);
                start = false;
            }
            yield raw_event;
        }
        eprintln!("Not inflight");
        state.inflight[index].fetch_sub(1, Ordering::Relaxed);
    };

    let body = Body::from_stream(response_stream);

    Response::from_parts(parts, body)
}
