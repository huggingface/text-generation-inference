use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use subprocess::{Popen, PopenConfig, Redirection};

fn start_launcher(model_name: String, num_shard: usize, port: usize, master_port: usize) -> Popen {
    let argv = vec![
        "text-generation-launcher".to_string(),
        "--model-name".to_string(),
        model_name.clone(),
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
            stderr: Redirection::Pipe,
            ..Default::default()
        },
    )
    .expect("Could not start launcher");

    // Redirect STDOUT and STDERR to the console
    let launcher_stdout = launcher.stdout.take().unwrap();
    let launcher_stderr = launcher.stderr.take().unwrap();

    thread::spawn(move || {
        let stdout = BufReader::new(launcher_stdout);
        let stderr = BufReader::new(launcher_stderr);
        for line in stdout.lines() {
            println!("{}", line.unwrap());
        }
        for line in stderr.lines() {
            println!("{}", line.unwrap());
        }
    });

    for _ in 0..30 {
        let health = reqwest::blocking::get(format!("http://localhost:{}/health", port));
        if health.is_ok() {
            return launcher;
        }
        sleep(Duration::from_secs(2));
    }

    launcher.terminate().unwrap();
    launcher.wait().unwrap();
    panic!("failed to launch {}", model_name)
}

fn test_model(model_name: String, num_shard: usize, port: usize, master_port: usize) -> Value {
    let mut launcher = start_launcher(model_name, num_shard, port, master_port);

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

    let result: Value = res.unwrap().json().unwrap();
    result
}

#[test]
fn test_bloom_560m() {
    let result = test_model("bigscience/bloom-560m".to_string(), 1, 3000, 29500);
    println!("{}", result);
}

#[test]
fn test_bloom_560m_distributed() {
    let result = test_model("bigscience/bloom-560m".to_string(), 2, 3001, 29501);
    println!("{}", result);
}

#[test]
fn test_mt0_base() {
    let result = test_model("bigscience/mt0-base".to_string(), 1, 3002, 29502);
    println!("{}", result);
}
