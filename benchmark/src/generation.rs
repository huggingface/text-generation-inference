use std::time::{Duration, Instant};
use text_generation_client::{
    Batch, CachedBatch, ClientError, NextTokenChooserParameters, Request, ShardedClient,
    StoppingCriteriaParameters,
};
use tokenizers::{Tokenizer, TruncationDirection};
use tokio::sync::{broadcast, mpsc};

const LOREM_IPSUM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

#[derive(Debug, Clone)]
pub(crate) struct Prefill {
    pub(crate) latency: Duration,
    pub(crate) throughput: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct Decode {
    pub(crate) latency: Duration,
    pub(crate) token_latency: Duration,
    pub(crate) throughput: f64,
}

#[derive(Debug)]
pub(crate) enum Message {
    Warmup,
    Prefill(Prefill),
    Decode(Decode),
    EndRun,
    EndBatch,
}

/// Benchmarking task
#[allow(clippy::too_many_arguments)]
pub(crate) async fn generation_task(
    tokenizer: Tokenizer,
    batch_size: Vec<u32>,
    sequence_length: u32,
    decode_length: u32,
    n_runs: usize,
    warmups: usize,
    parameters: NextTokenChooserParameters,
    client: ShardedClient,
    run_sender: mpsc::Sender<Result<Message, ClientError>>,
    mut shutdown_receiver: broadcast::Receiver<()>,
    _shutdown_guard_sender: mpsc::Sender<()>,
) {
    // End task if a message is received on shutdown_receiver
    // _shutdown_guard_sender will be dropped once the task is finished
    tokio::select! {
        res = generate_runs(tokenizer, batch_size, sequence_length, decode_length, n_runs, warmups, parameters, client, run_sender.clone())  => {
            if let Err(err) = res {
                run_sender.send(Err(err)).await.unwrap_or(());
            }
        },
        _ = shutdown_receiver.recv() => {}
    }
}

/// Benchmark prefill/decode
#[allow(clippy::too_many_arguments)]
async fn generate_runs(
    tokenizer: Tokenizer,
    batch_size: Vec<u32>,
    sequence_length: u32,
    decode_length: u32,
    n_runs: usize,
    warmups: usize,
    parameters: NextTokenChooserParameters,
    mut client: ShardedClient,
    run_sender: mpsc::Sender<Result<Message, ClientError>>,
) -> Result<(), ClientError> {
    // Create a dummy sequence
    let sequence = create_sequence(sequence_length, tokenizer);

    for b in batch_size {
        // Warmups on batch size
        for _ in 0..warmups {
            let (_, decode_batch) = prefill(
                sequence.clone(),
                sequence_length,
                b,
                decode_length,
                parameters.clone(),
                &mut client,
            )
            .await?;
            let _ = decode(decode_batch, &mut client).await?;
            // Send warmup message
            run_sender.send(Ok(Message::Warmup)).await.unwrap_or(());
        }

        for _ in 0..n_runs {
            let (prefill, decode_batch) = prefill(
                sequence.clone(),
                sequence_length,
                b,
                decode_length,
                parameters.clone(),
                &mut client,
            )
            .await?;
            // Send prefill message
            run_sender
                .send(Ok(Message::Prefill(prefill)))
                .await
                .unwrap_or(());

            let decode = decode(decode_batch, &mut client).await?;

            // Send decode message
            run_sender
                .send(Ok(Message::Decode(decode)))
                .await
                .unwrap_or(());

            // Send run ended message
            run_sender.send(Ok(Message::EndRun)).await.unwrap_or(());
        }
        // Batch ended
        run_sender.send(Ok(Message::EndBatch)).await.unwrap_or(());
    }
    Ok(())
}

// Run a prefill step
async fn prefill(
    sequence: String,
    sequence_length: u32,
    batch_size: u32,
    decode_length: u32,
    parameters: NextTokenChooserParameters,
    client: &mut ShardedClient,
) -> Result<(Prefill, CachedBatch), ClientError> {
    // Create requests
    let requests = (0..batch_size)
        .map(|id| Request {
            id: id.into(),
            prefill_logprobs: false,
            inputs: sequence.clone(),
            truncate: sequence_length,
            parameters: Some(parameters.clone()),
            stopping_parameters: Some(StoppingCriteriaParameters {
                max_new_tokens: decode_length,
                stop_sequences: vec![],
                ignore_eos_token: true, // Will not stop even if a eos token is generated
            }),
        })
        .collect();

    let batch = Batch {
        id: 0,
        requests,
        size: batch_size,
        max_tokens: batch_size * (sequence_length + decode_length),
    };

    // Run prefill
    let start_time = Instant::now();
    let (_, decode_batch) = client.prefill(batch.clone()).await?;

    // Get latency
    let latency = start_time.elapsed();

    // Compute throughput from latency and batch size
    let throughput = batch_size as f64 / latency.as_secs_f64();

    // Decode batch cannot be empty
    let decode_batch = decode_batch.expect("decode_batch is None. This is a bug.");

    let step = Prefill {
        latency,
        throughput,
    };

    Ok((step, decode_batch))
}

/// Run a full decode
async fn decode(batch: CachedBatch, client: &mut ShardedClient) -> Result<Decode, ClientError> {
    let mut decode_length = 0;
    let batch_size = batch.size;

    let start_time = Instant::now();

    // Full decode over decode length
    let mut next_batch = Some(batch);
    while let Some(batch) = next_batch {
        let result = client.decode(vec![batch]).await?;
        next_batch = result.1;
        decode_length += 1;
    }

    // Get latency
    let latency = start_time.elapsed();
    let token_latency = latency / decode_length;

    // Compute throughput from latency, batch size and decode length
    let throughput = (batch_size * decode_length) as f64 / latency.as_secs_f64();

    let step = Decode {
        latency,
        token_latency,
        throughput,
    };
    Ok(step)
}

/// Create a dummy sequence of the correct length
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
