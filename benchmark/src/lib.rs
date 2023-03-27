mod ui;

use std::time::Duration;
use tokenizers::{Tokenizer, TruncationDirection};
use tokio::time;
use text_generation_client::{ShardedClient, Request, Batch, StoppingCriteriaParameters, NextTokenChooserParameters};
use time::Instant;
use tokio::sync::mpsc;
use crate::ui::UI;

const LOREM_IPSUM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

enum Step {
    Prefill,
    Decode,
}

struct Run {
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
    // mut client: ShardedClient,
) -> Result<(), Box<dyn std::error::Error>> {
    // let prefill_runs = benchmark_prefill(&tokenizer, &batch_size, &sequence_length, &decode_length, runs, &mut client).await;

    let (sender, receiver) = mpsc::channel(8);


    tokio::spawn(
        UI {
            n_run: runs,
            n_batch: batch_size.len(),
            n_batch_done: 0,
            run_receiver: receiver,
        }.draw()
    );


    for n in 0..runs {
        sender.send(()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
    }


    Ok(())
}

//
// async fn benchmark_prefill(tokenizer: &Tokenizer,
//                            batch_size: &Vec<u32>,
//                            sequence_length: u32,
//                            decode_length: u32,
//                            runs: u32,
//                            client: &mut ShardedClient) -> Vec<Run> {
//     let mut results = Vec::new();
//
//     let lorem_ipsum_length = tokenizer.encode(LOREM_IPSUM, true).unwrap().len();
//
//     for s in sequence_length {
//         let sequence = create_sequence(s, lorem_ipsum_length, tokenizer);
//         for b in batch_size {
//             for d in decode_length {
//                 let requests = (0..*b).map(|id| {
//                     Request {
//                         id: id.into(),
//                         inputs: sequence.clone(),
//                         input_length: *s,
//                         parameters: Some(NextTokenChooserParameters {
//                             temperature: 1.0,
//                             top_k: 0,
//                             top_p: 1.0,
//                             typical_p: 1.0,
//                             do_sample: false,
//                             seed: 0,
//                             repetition_penalty: 1.0,
//                             watermark: false,
//                         }),
//                         stopping_parameters: Some(StoppingCriteriaParameters {
//                             max_new_tokens: *d,
//                             stop_sequences: vec![],
//                             ignore_eos_token: true,
//                         }),
//                     }
//                 }).collect();
//
//                 let batch = Batch {
//                     id: 0,
//                     requests,
//                     size: *b,
//                 };
//
//                 for _ in 0..runs {
//                     let start_time = Instant::now();
//                     client.prefill(batch.clone()).await.unwrap();
//                     let elasped = start_time.elapsed();
//
//                     client.clear_cache().await.unwrap();
//
//                     results.push(Run {
//                         step: Step::Prefill,
//                         batch_size: *b,
//                         sequence_length: *s,
//                         decode_length: *d,
//                         time: elasped,
//                     });
//                 }
//             }
//         }
//     }
//     results
// }
//
// fn create_sequence(sequence_length: &u32, lorem_ipsum_length: usize, tokenizer: &Tokenizer) -> String {
//     // Repeat lorem ipsum to cover sequence length
//     let string_sequence = LOREM_IPSUM.repeat((0..*sequence_length).step_by(lorem_ipsum_length).len());
//     // Encode sequence
//     let mut encoding = tokenizer.encode(string_sequence, true).unwrap();
//     // Truncate to sequence_length
//     encoding.truncate(*sequence_length as usize, 0, TruncationDirection::Left);
//     // Decode
//     tokenizer.decode(Vec::from(encoding.get_ids()), false).unwrap()
// }