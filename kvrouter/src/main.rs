use axum::{
    routing::Router,
    routing::{get, post},
};
use kvrouter::{handler, RoundRobin};

#[tokio::main]
async fn main() {
    // List of backend servers
    let backends = vec![
        "http://localhost:8000".to_string(),
        "http://localhost:8001".to_string(),
    ];

    // Create a new instance of the RoundRobinRouter
    let router = RoundRobin::new(backends);

    // Create the Axum router
    let app = Router::new()
        .route("/{*key}", get(handler))
        .route("/{*key}", post(handler))
        .with_state(router);

    // run it
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
