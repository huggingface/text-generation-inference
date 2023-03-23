use float_eq::assert_float_eq;
use serde::Deserialize;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use subprocess::{Popen, PopenConfig, Redirection};

#[derive(Deserialize)]
pub struct Token {
    id: u32,
    text: String,
    logprob: Option<f32>,
    special: bool,
}

#[derive(Deserialize)]
struct Details {
    finish_reason: String,
    generated_tokens: u32,
    tokens: Vec<Token>,
}

#[derive(Deserialize)]
struct GeneratedText {
    generated_text: String,
    details: Details,
}

fn start_launcher(model_id: String, num_shard: usize, port: usize, master_port: usize) -> Popen {
    let argv = vec![
        "text-generation-launcher".to_string(),
        "--model-id".to_string(),
        model_id.clone(),
        "--num-shard".to_string(),
        num_shard.to_string(),
        "--port".to_string(),
        port.to_string(),
        "--master-port".to_string(),
        master_port.to_string(),
        "--shard-uds-path".to_string(),
        format!("/tmp/test-{}-{}-{}", num_shard, port, master_port),
    ];

    let mut launcher = Popen::create(
        &argv,
        PopenConfig {
            stdout: Redirection::Pipe,
            stderr: Redirection::Merge,
            ..Default::default()
        },
    )
    .expect("Could not start launcher");

    // Redirect STDOUT and STDERR to the console
    // (STDERR is merged into STDOUT)
    let launcher_stdout = launcher.stdout.take().unwrap();

    thread::spawn(move || {
        let stdout = BufReader::new(launcher_stdout);
        for line in stdout.lines() {
            println!("{}", line.unwrap());
        }
    });

    for _ in 0..60 {
        let health = reqwest::blocking::get(format!("http://localhost:{}/health", port));
        if health.is_ok() {
            return launcher;
        }
        sleep(Duration::from_secs(2));
    }

    launcher.terminate().unwrap();
    launcher.wait().unwrap();
    panic!("failed to launch {}", model_id)
}

fn test_model(
    model_id: String,
    num_shard: usize,
    port: usize,
    master_port: usize,
) -> GeneratedText {
    let mut launcher = start_launcher(model_id, num_shard, port, master_port);

    let data = r#"
        {
            "inputs": "Test request",
            "parameters": {
                "details": true
            }
        }"#;
    let req: Value = serde_json::from_str(data).unwrap();

    let client = reqwest::blocking::Client::new();
    let res = client
        .post(format!("http://localhost:{}/generate", port))
        .json(&req)
        .send();

    launcher.terminate().unwrap();
    launcher.wait().unwrap();

    let result: GeneratedText = res.unwrap().json().unwrap();
    result
}

fn read_json(name: &str) -> GeneratedText {
    let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("tests/");
    d.push(name);

    let file = File::open(d).unwrap();
    let reader = BufReader::new(file);

    let result: GeneratedText = serde_json::from_reader(reader).unwrap();
    result
}

fn compare_results(result: GeneratedText, expected: GeneratedText) {
    assert_eq!(result.generated_text, expected.generated_text);
    assert_eq!(result.details.finish_reason, expected.details.finish_reason);
    assert_eq!(
        result.details.generated_tokens,
        expected.details.generated_tokens
    );

    for (token, expected_token) in result
        .details
        .tokens
        .into_iter()
        .zip(expected.details.tokens.into_iter())
    {
        assert_eq!(token.id, expected_token.id);
        assert_eq!(token.text, expected_token.text);
        assert_eq!(token.special, expected_token.special);
        if let Some(logprob) = token.logprob {
            let expected_logprob = expected_token.logprob.unwrap();
            assert_float_eq!(logprob, expected_logprob, abs <= 0.001);
        } else {
            assert_eq!(token.logprob, expected_token.logprob);
        }
    }
}

#[test]
fn test_bloom_560m() {
    let expected = read_json("bloom_560m.json");

    let result = test_model("bigscience/bloom-560m".to_string(), 1, 3000, 29500);
    compare_results(result, expected);
}

#[test]
fn test_bloom_560m_distributed() {
    let expected = read_json("bloom_560m.json");

    let result = test_model("bigscience/bloom-560m".to_string(), 2, 3001, 29501);
    compare_results(result, expected);
}

#[test]
fn test_mt0_base() {
    let expected = read_json("mt0_base.json");

    let result = test_model("bigscience/mt0-base".to_string(), 1, 3002, 29502);
    compare_results(result, expected);
}
