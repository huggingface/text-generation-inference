use pyo3::prelude::*;

#[pyo3_asyncio::tokio::main(flavor = "multi_thread")]
async fn main() {
    println!("Hello, world!");
}
