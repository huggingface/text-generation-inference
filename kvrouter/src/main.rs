use axum::{
    routing::Router,
    routing::{get, post},
};
use kvrouter::{handler, ContentAware, OverloadHandler, RoundRobin};

#[tokio::main]
async fn main() {
    // List of backend servers
    let backends = vec![
        "http://localhost:8000".to_string(),
        "http://localhost:8001".to_string(),
        "http://localhost:8002".to_string(),
        "http://localhost:8003".to_string(),
    ];

    // Create a new instance of the RoundRobinRouter
    if std::env::var("TGI_KVROUTER_LB").unwrap_or("".to_string()) == *"roundrobin" {
        println!("Using round robin");
        let lb = RoundRobin::new();
        // Create the Axum router
        let router = OverloadHandler::new(lb, backends);
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
    } else {
        println!("Using Content aware");
        let lb = ContentAware::new();
        // Create the Axum router
        let router = OverloadHandler::new(lb, backends);
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
    };
}
