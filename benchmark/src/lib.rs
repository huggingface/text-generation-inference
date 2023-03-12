use std::time::Duration;
use tokenizers::{Tokenizer, TruncationDirection};
use tokio::time;
use text_generation_client::{ShardedClient, Request, Batch, StoppingCriteriaParameters, NextTokenChooserParameters};
use time::Instant;
use plotters::prelude::*;
use itertools::Itertools;

const LOREM_IPSUM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
const OUT_FILE_NAME: &'static str = "errorbar.png";

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
    sequence_length: Vec<u32>,
    decode_length: Vec<u32>,
    runs: u32,
    mut client: ShardedClient,
) -> Result<(), Box<dyn std::error::Error>> {
    // let prefill_runs = benchmark_prefill(&tokenizer, &batch_size, &sequence_length, &decode_length, runs, &mut client).await;
    let mut runs: Vec<(f64, f64)> = Vec::new();
    for i in 0..10{
        for j in 0..10 {
            runs.push((i as f64, j as f64));
        }
    }

    let data = runs;
    // let down_sampled = down_sample(&data[..]);

    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Linear Function with Noise", ("sans-serif", 60))
        .margin(10)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(-10f64..10f64, -10f64..10f64)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(data, &GREEN.mix(0.3)))?
        .label("Raw Data");

    // chart.draw_series(LineSeries::new(
    //     down_sampled.iter().map(|(x, _, y, _)| (*x, *y)),
    //     &BLUE,
    // ))?;

    // chart
    //     .draw_series(
    //         down_sampled.iter().map(|(x, yl, ym, yh)| {
    //             ErrorBar::new_vertical(*x, *yl, *ym, *yh, BLUE.filled(), 20)
    //         }),
    //     )?
    //     .label("Down-sampled")
    //     .legend(|(x, y)| PathElement::new(vec![(x, y), (x, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(WHITE.filled())
        .draw()?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUT_FILE_NAME);
    Ok(())

}

// fn down_sample(data: &[(f64, f64)]) -> Vec<(f64, f64, f64, f64)> {
//     let down_sampled: Vec<_> = data
//         .iter()
//         .into_iter()
//         .map(|(x, g)| {
//             let mut g: Vec<_> = g.map(|(_, y)| *y).collect();
//             g.sort_by(|a, b| a.partial_cmp(b).unwrap());
//             (
//                 x,
//                 g[0],
//                 g.iter().sum::<f64>() / g.len() as f64,
//                 g[g.len() - 1],
//             )
//         })
//         .collect();
//     down_sampled
// }

async fn benchmark_prefill(tokenizer: &Tokenizer,
                           batch_size: &Vec<u32>,
                           sequence_length: &Vec<u32>,
                           decode_length: &Vec<u32>,
                           runs: u32,
                           client: &mut ShardedClient) -> Vec<Run> {
    let mut results = Vec::new();

    let lorem_ipsum_length = tokenizer.encode(LOREM_IPSUM, true).unwrap().len();

    for s in sequence_length {
        let sequence = create_sequence(s, lorem_ipsum_length, tokenizer);
        for b in batch_size {
            for d in decode_length {
                let requests = (0..*b).map(|id| {
                    Request {
                        id: id.into(),
                        inputs: sequence.clone(),
                        input_length: *s,
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
                            max_new_tokens: *d,
                            stop_sequences: vec![],
                            ignore_eos_token: true,
                        }),
                    }
                }).collect();

                let batch = Batch {
                    id: 0,
                    requests,
                    size: *b,
                };

                for _ in 0..runs {
                    let start_time = Instant::now();
                    client.prefill(batch.clone()).await.unwrap();
                    let elasped = start_time.elapsed();

                    client.clear_cache().await.unwrap();

                    results.push(Run {
                        step: Step::Prefill,
                        batch_size: *b,
                        sequence_length: *s,
                        decode_length: *d,
                        time: elasped,
                    });
                }
            }
        }
    }
    results
}

fn create_sequence(sequence_length: &u32, lorem_ipsum_length: usize, tokenizer: &Tokenizer) -> String {
    // Repeat lorem ipsum to cover sequence length
    let string_sequence = LOREM_IPSUM.repeat((0..*sequence_length).step_by(lorem_ipsum_length).len());
    // Encode sequence
    let mut encoding = tokenizer.encode(string_sequence, true).unwrap();
    // Truncate to sequence_length
    encoding.truncate(*sequence_length as usize, 0, TruncationDirection::Left);
    // Decode
    tokenizer.decode(Vec::from(encoding.get_ids()), false).unwrap()
}