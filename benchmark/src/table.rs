use crate::app::Data;
use tabled::settings::Merge;
use tabled::{builder::Builder, settings::Style, Table};

#[allow(clippy::too_many_arguments)]
pub(crate) fn parameters_table(
    tokenizer_name: String,
    sequence_length: u32,
    decode_length: u32,
    top_n_tokens: Option<u32>,
    n_runs: usize,
    warmups: usize,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    typical_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    watermark: bool,
    do_sample: bool,
) -> Table {
    let mut builder = Builder::default();

    builder.set_header(["Parameter", "Value"]);

    builder.push_record(["Model", &tokenizer_name]);
    builder.push_record(["Sequence Length", &sequence_length.to_string()]);
    builder.push_record(["Decode Length", &decode_length.to_string()]);
    builder.push_record(["Top N Tokens", &format!("{top_n_tokens:?}")]);
    builder.push_record(["N Runs", &n_runs.to_string()]);
    builder.push_record(["Warmups", &warmups.to_string()]);
    builder.push_record(["Temperature", &format!("{temperature:?}")]);
    builder.push_record(["Top K", &format!("{top_k:?}")]);
    builder.push_record(["Top P", &format!("{top_p:?}")]);
    builder.push_record(["Typical P", &format!("{typical_p:?}")]);
    builder.push_record(["Repetition Penalty", &format!("{repetition_penalty:?}")]);
    builder.push_record(["Frequency Penalty", &format!("{frequency_penalty:?}")]);
    builder.push_record(["Watermark", &watermark.to_string()]);
    builder.push_record(["Do Sample", &do_sample.to_string()]);

    let mut table = builder.build();
    table.with(Style::markdown());
    table
}

pub(crate) fn latency_table(data: &Data) -> Table {
    let mut builder = Builder::default();

    builder.set_header([
        "Step",
        "Batch Size",
        "Average",
        "Lowest",
        "Highest",
        "p50",
        "p90",
        "p99",
    ]);

    add_latencies(
        &mut builder,
        "Prefill",
        &data.batch_size,
        &data.prefill_latencies,
    );
    add_latencies(
        &mut builder,
        "Decode (token)",
        &data.batch_size,
        &data.decode_token_latencies,
    );
    add_latencies(
        &mut builder,
        "Decode (total)",
        &data.batch_size,
        &data.decode_latencies,
    );

    let mut table = builder.build();
    table.with(Style::markdown()).with(Merge::vertical());
    table
}

pub(crate) fn throughput_table(data: &Data) -> Table {
    let mut builder = Builder::default();

    builder.set_header(["Step", "Batch Size", "Average", "Lowest", "Highest"]);

    add_throuhgputs(
        &mut builder,
        "Prefill",
        &data.batch_size,
        &data.prefill_throughputs,
    );
    add_throuhgputs(
        &mut builder,
        "Decode",
        &data.batch_size,
        &data.decode_throughputs,
    );

    let mut table = builder.build();
    table.with(Style::markdown()).with(Merge::vertical());
    table
}

fn add_latencies(
    builder: &mut Builder,
    step: &'static str,
    batch_size: &[u32],
    batch_latencies: &[Vec<f64>],
) {
    for (i, b) in batch_size.iter().enumerate() {
        let latencies = &batch_latencies[i];
        let (avg, min, max) = avg_min_max(latencies);

        let row = [
            step,
            &b.to_string(),
            &format_value(avg, "ms"),
            &format_value(min, "ms"),
            &format_value(max, "ms"),
            &format_value(px(latencies, 50), "ms"),
            &format_value(px(latencies, 90), "ms"),
            &format_value(px(latencies, 99), "ms"),
        ];

        builder.push_record(row);
    }
}

fn add_throuhgputs(
    builder: &mut Builder,
    step: &'static str,
    batch_size: &[u32],
    batch_throughputs: &[Vec<f64>],
) {
    for (i, b) in batch_size.iter().enumerate() {
        let throughputs = &batch_throughputs[i];
        let (avg, min, max) = avg_min_max(throughputs);

        let row = [
            step,
            &b.to_string(),
            &format_value(avg, "tokens/secs"),
            &format_value(min, "tokens/secs"),
            &format_value(max, "tokens/secs"),
        ];

        builder.push_record(row);
    }
}

fn avg_min_max(data: &[f64]) -> (f64, f64, f64) {
    let average = data.iter().sum::<f64>() / data.len() as f64;
    let min = data
        .iter()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&f64::NAN);
    let max = data
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(&f64::NAN);
    (average, *min, *max)
}

fn px(data: &[f64], p: u32) -> f64 {
    let i = (f64::from(p) / 100.0 * data.len() as f64) as usize;
    *data.get(i).unwrap_or(&f64::NAN)
}

fn format_value(value: f64, unit: &'static str) -> String {
    format!("{:.2} {unit}", value)
}
