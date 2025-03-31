use std::sync::Arc;

use axum::{
    routing::Router,
    routing::{get, post},
};
use hyper::StatusCode;
use kvrouter::{
    handler, set_backends_handler, Communicator, ContentAware, OverloadHandler, RoundRobin,
};
use tokio::sync::RwLock;

#[tokio::main]
async fn main() {
    // List of backend servers
    let backends = Arc::new(RwLock::new(vec![
        // "http://localhost:8000".to_string(),
        // "http://localhost:8001".to_string(),
        // "http://localhost:8002".to_string(),
        // "http://localhost:8003".to_string(),
    ]));

    // Create a new instance of the RoundRobinRouter

    println!("Using Content aware");
    // Create the Axum router

    let (sx, rx) = tokio::sync::mpsc::channel(100);
    let communicator = Communicator::new(sx);
    tokio::task::spawn(async move {
        if std::env::var("TGI_KVROUTER_LB").unwrap_or("".to_string()) == *"roundrobin" {
            println!("Using round robin");
            let lb = RoundRobin::new();
            let mut router = OverloadHandler::new(lb, backends, rx).await;
            router.run().await;
        } else {
            let lb = ContentAware::new();
            let mut router = OverloadHandler::new(lb, backends, rx).await;
            router.run().await;
        };
    });
    let app = Router::new()
        .route("/{*key}", get(handler))
        .route("/{*key}", post(handler))
        .route("/_kvrouter/health", get(|| async { StatusCode::OK }))
        .route("/_kvrouter/set-backends", post(set_backends_handler))
        .with_state(communicator);

    // run it
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
