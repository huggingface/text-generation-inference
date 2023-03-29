extern crate core;

mod ui;
mod utils;

use crate::ui::UI;
use std::time::{Duration, Instant};
use text_generation_client::{
    Batch, ClientError, NextTokenChooserParameters, Request, ShardedClient,
    StoppingCriteriaParameters,
};
use tokenizers::{Tokenizer, TruncationDirection};
use tokio::sync::{broadcast, mpsc};

const LOREM_IPSUM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

#[derive(Debug, Clone)]
pub(crate) struct Prefill {
    batch_size: u32,
    sequence_length: u32,
    latency: Duration,
}

#[derive(Debug, Clone)]
pub(crate) struct Decode {
    batch_size: u32,
    sequence_length: u32,
    decode_length: u32,
    latency: Duration,
}

#[derive(Debug)]
pub(crate) struct Run {
    prefill: Prefill,
    decode: Decode,
}

#[derive(Debug)]
pub(crate) enum Message {
    Prefill(Prefill),
    Decode(Decode),
    IncreaseRun,
    IncreaseBatch,
}

pub async fn run(
    tokenizer_name: String,
    tokenizer: Tokenizer,
    batch_size: Vec<u32>,
    sequence_length: u32,
    decode_length: u32,
    n_runs: usize,
    warmups: usize,
    mut client: ShardedClient,
) -> Result<(), Box<dyn std::error::Error>> {
    let (ui_sender, ui_receiver) = mpsc::channel(8);
    let (shutdown_sender, mut shutdown_receiver) = broadcast::channel(1);

    tokio::spawn(
        UI {
            tokenizer_name,
            decode_length,
            sequence_length,
            n_run: n_runs,
            batch_size: batch_size.clone(),
            receiver: ui_receiver,
            shutdown_sender,
        }
        .draw(),
    );

    let mut runs = Vec::with_capacity(batch_size.len() * n_runs);
    let sequence = create_sequence(sequence_length, tokenizer);

    for b in batch_size {
        for _ in 0..warmups {
            let (_, decode_batch) = tokio::select! {
                res = run_prefill(sequence.clone(), sequence_length, 1, decode_length, &mut client) => res?,
                _ = shutdown_receiver.recv() => {
                    return Ok(());
                }
            };
            let _ = tokio::select! {
                res = run_decode(decode_batch, sequence_length, &mut client) => res?,
                _ = shutdown_receiver.recv() => {
                    return Ok(());
                }
            };
        }

        for _ in 0..n_runs {
            let (prefill, decode_batch) = tokio::select! {
                res = run_prefill(sequence.clone(), sequence_length, b, decode_length, &mut client) => res?,
                _ = shutdown_receiver.recv() => {
                    return Ok(());
                }
            };
            ui_sender
                .send(Message::Prefill(prefill.clone()))
                .await
                .unwrap();

            let decode = tokio::select! {
                res = run_decode(decode_batch, sequence_length, &mut client) => res?,
                _ = shutdown_receiver.recv() => {
                    return Ok(());
                }
            };

            ui_sender
                .send(Message::Decode(decode.clone()))
                .await
                .unwrap();
            runs.push(Run { prefill, decode });

            ui_sender.send(Message::IncreaseRun).await.unwrap();
        }
        ui_sender.send(Message::IncreaseBatch).await.unwrap();
    }

    // Signal the UI that we are done
    drop(ui_sender);

    // Wait for UI shutdown signal
    let _ = shutdown_receiver.recv().await;

    Ok(())
}

async fn run_prefill(
    sequence: String,
    sequence_length: u32,
    batch_size: u32,
    decode_length: u32,
    client: &mut ShardedClient,
) -> Result<(Prefill, Batch), ClientError> {
    let requests = (0..batch_size)
        .map(|id| Request {
            id: id.into(),
            inputs: sequence.clone(),
            parameters: Some(NextTokenChooserParameters {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                typical_p: 1.0,
                do_sample: false,
                seed: 0,
                repetition_penalty: 1.0,
                watermark: false,
            }),
            stopping_parameters: Some(StoppingCriteriaParameters {
                max_new_tokens: decode_length,
                stop_sequences: vec![],
                ignore_eos_token: true,
            }),
        })
        .collect();

    let batch = Batch {
        id: 0,
        requests,
        size: batch_size,
    };

    let start_time = Instant::now();
    let (_, decode_batch) = client.prefill(batch.clone()).await?;
    let elasped = start_time.elapsed();

    let decode_batch = decode_batch.expect("decode_batch is None. This is a bug.");

    let step = Prefill {
        batch_size,
        sequence_length,
        latency: elasped,
    };

    Ok((step, decode_batch))
}

async fn run_decode(
    batch: Batch,
    sequence_length: u32,
    client: &mut ShardedClient,
) -> Result<Decode, ClientError> {
    let batch_size = batch.size;
    let mut decode_length = 0;
    let start_time = Instant::now();

    let mut next_batch = Some(batch);
    while let Some(batch) = next_batch {
        let result = client.decode(vec![batch]).await?;
        next_batch = result.1;
        decode_length += 1;
    }
    let elapsed = start_time.elapsed();
    let step = Decode {
        batch_size,
        sequence_length,
        decode_length,
        latency: elapsed,
    };
    Ok(step)
}

fn create_sequence(sequence_length: u32, tokenizer: Tokenizer) -> String {
    let lorem_ipsum_length = tokenizer.encode(LOREM_IPSUM, true).unwrap().len();
    // Repeat lorem ipsum to cover sequence length
    let string_sequence =
        LOREM_IPSUM.repeat((0..sequence_length).step_by(lorem_ipsum_length).len());
    // Encode sequence
    let mut encoding = tokenizer.encode(string_sequence, true).unwrap();
    // Truncate to sequence_length
    encoding.truncate(sequence_length as usize, 0, TruncationDirection::Left);
    // Decode
    tokenizer
        .decode(Vec::from(encoding.get_ids()), false)
        .unwrap()
}
