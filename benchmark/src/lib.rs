mod ui;

use std::time::{Duration, Instant};
use tokenizers::{Tokenizer, TruncationDirection};
use tokio::time;
use text_generation_client::{ShardedClient, Request, Batch, StoppingCriteriaParameters, NextTokenChooserParameters, ClientError};
use tokio::sync::mpsc;
use crate::ui::UI;

const LOREM_IPSUM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

#[derive(Debug)]
pub(crate) enum Step {
    Prefill,
    Decode,
}

#[derive(Debug)]
pub(crate) struct Run {
    step: Step,
    batch_size: u32,
    sequence_length: u32,
    decode_length: u32,
    time: Duration,
}

pub async fn run(
    tokenizer: Tokenizer,
    batch_size: Vec<u32>,
    sequence_length: u32,
    decode_length: u32,
    runs: usize,
    mut client: ShardedClient,
) -> Result<(), Box<dyn std::error::Error>> {
    let (sender, receiver) = mpsc::channel(8);


    tokio::spawn(
        UI {
            n_run: runs,
            n_batch: batch_size.len(),
            n_batch_done: 0,
            run_receiver: receiver,
        }.draw()
    );

    let sequence = create_sequence(sequence_length, tokenizer);


    for b in batch_size {
        for _ in 0..runs {
            let (run, decode_batch) = run_prefill(sequence.clone(), sequence_length, b, decode_length, &mut client).await?;
            sender.send(run).await.unwrap();

            let run = run_decode(decode_batch, sequence_length, &mut client).await?;
            sender.send(run).await.unwrap();

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    drop(sender);

    tokio::time::sleep(Duration::from_millis(100)).await;

    Ok(())
}


async fn run_prefill(
    sequence: String,
    sequence_length: u32,
    batch_size: u32,
    decode_length: u32,
    client: &mut ShardedClient) -> Result<(Run, Batch), ClientError> {
    let requests = (0..batch_size).map(|id| {
        Request {
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
        }
    }).collect();

    let batch = Batch {
        id: 0,
        requests,
        size: batch_size,
    };

    let start_time = Instant::now();
    let (_, decode_batch) = client.prefill(batch.clone()).await?;
    let elasped = start_time.elapsed();

    let decode_batch = decode_batch.expect("decode_batch is None. This is a bug.");

    let run = Run {
        step: Step::Prefill,
        batch_size,
        sequence_length,
        decode_length: 1,
        time: elasped,
    };

    Ok((run, decode_batch))
}

async fn run_decode(batch: Batch, sequence_length: u32, client: &mut ShardedClient) -> Result<Run, ClientError> {
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
    let run = Run {
        step: Step::Decode,
        batch_size,
        sequence_length,
        decode_length,
        time: elapsed,
    };
    Ok(run)
}

fn create_sequence(sequence_length: u32, tokenizer: Tokenizer) -> String {
    let lorem_ipsum_length = tokenizer.encode(LOREM_IPSUM, true).unwrap().len();
    // Repeat lorem ipsum to cover sequence length
    let string_sequence = LOREM_IPSUM.repeat((0..sequence_length).step_by(lorem_ipsum_length).len());
    // Encode sequence
    let mut encoding = tokenizer.encode(string_sequence, true).unwrap();
    // Truncate to sequence_length
    encoding.truncate(sequence_length as usize, 0, TruncationDirection::Left);
    // Decode
    tokenizer.decode(Vec::from(encoding.get_ids()), false).unwrap()
}