use axum::{
    routing::Router,
    routing::{get, post},
};
use hyper::StatusCode;
use kvrouter::{
    handler, set_backends_handler, Communicator, ContentAware, OverloadHandler, RoundRobin,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // List of backend servers
    let backends = vec![
        // "http://localhost:8000".to_string(),
        // "http://localhost:8001".to_string(),
        // "http://localhost:8002".to_string(),
        // "http://localhost:8003".to_string(),
    ];

    // Create a new instance of the RoundRobinRouter

    println!("Using Content aware");
    // Create the Axum router

    let (sx, rx) = tokio::sync::mpsc::channel(100);
    let communicator = Communicator::new(sx);
    let host = std::env::var("TGI_KVROUTER_HOST").unwrap_or("127.0.0.1".to_string());
    let port: u16 = std::env::var("TGI_KVROUTER_PORT")
        .unwrap_or("3000".to_string())
        .parse()?;
    tokio::task::spawn(async move {
        if std::env::var("TGI_KVROUTER_LB").unwrap_or("".to_string()) == *"roundrobin" {
            println!("Using round robin");
            let lb = RoundRobin::new();
            let mut router = OverloadHandler::new(lb, backends, rx);
            router.run().await;
        } else {
            let lb = ContentAware::new();
            let mut router = OverloadHandler::new(lb, backends, rx);
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
    let listener = tokio::net::TcpListener::bind((host, port)).await.unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
    Ok(())
}
