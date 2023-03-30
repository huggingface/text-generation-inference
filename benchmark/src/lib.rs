extern crate core;

mod ui;
mod utils;
mod generation;

use crate::ui::UI;
use tokenizers::Tokenizer;
use tokio::sync::{broadcast, mpsc};
use text_generation_client::ShardedClient;


pub async fn run(
    tokenizer_name: String,
    tokenizer: Tokenizer,
    batch_size: Vec<u32>,
    sequence_length: u32,
    decode_length: u32,
    n_runs: usize,
    warmups: usize,
    client: ShardedClient,
) -> Result<(), Box<dyn std::error::Error>> {
    let (run_sender, run_receiver) = mpsc::channel(8);
    let (shutdown_sender, shutdown_receiver) = broadcast::channel(1);
    let (shutdown_guard_sender, mut shutdown_guard_receiver) = mpsc::channel(1);

    tokio::spawn(
        generation::generation_task(tokenizer, batch_size.clone(), sequence_length, decode_length, n_runs, warmups, client, run_sender, shutdown_receiver, shutdown_guard_sender.clone()),
    );

    tokio::spawn(
        UI {
            tokenizer_name,
            decode_length,
            sequence_length,
            n_run: n_runs,
            batch_size: batch_size,
            receiver: run_receiver,
            shutdown_sender,
            _shutdown_guard_sender: shutdown_guard_sender.clone()
        }
            .draw(),
    );

    drop (shutdown_guard_sender);

    // Wait for tasks to shutdown
    let _ = shutdown_guard_receiver.recv().await;

    Ok(())
}
