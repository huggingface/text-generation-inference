use crate::config::Config;
use reqwest::header::HeaderMap;
use serde::Serialize;
use std::{fmt, process::Command, time::Duration};
use uuid::Uuid;

const TELEMETRY_URL: &str = "https://huggingface.co/api/telemetry/tgi";

#[derive(Debug, Clone, Serialize)]
pub struct UserAgent {
    pub uid: String,
    pub args: Args,
    pub env: Env,
}

impl UserAgent {
    pub fn new(reduced_args: Args) -> Self {
        Self {
            uid: Uuid::new_v4().to_string(),
            args: reduced_args,
            env: Env::new(),
        }
    }
}

#[derive(Serialize, Debug)]
pub enum EventType {
    Start,
    Stop,
    Error(String),
}

#[derive(Debug, Serialize)]
pub struct UsageStatsEvent {
    user_agent: UserAgent,
    event_type: EventType,
}

impl UsageStatsEvent {
    pub fn new(user_agent: UserAgent, event_type: EventType) -> Self {
        Self {
            user_agent,
            event_type,
        }
    }
    pub async fn send(&self) {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse().unwrap());
        let body = serde_json::to_string(&self).unwrap();
        let client = reqwest::Client::new();
        let _ = client
            .post(TELEMETRY_URL)
            .body(body)
            .timeout(Duration::from_secs(5))
            .send()
            .await;
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Args {
    model_config: Option<Config>,
    tokenizer_config: Option<String>,
    max_concurrent_requests: usize,
    max_best_of: usize,
    max_stop_sequences: usize,
    max_top_n_tokens: u32,
    max_input_tokens: usize,
    max_total_tokens: usize,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: Option<u32>,
    max_waiting_tokens: usize,
    max_batch_size: Option<usize>,
    revision: Option<String>,
    validation_workers: usize,
    messages_api_enabled: bool,
    disable_grammar_support: bool,
    max_client_batch_size: usize,
    disable_usage_stats: bool,
    disable_crash_reports: bool,
}

impl Args {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_config: Option<Config>,
        tokenizer_config: Option<String>,
        max_concurrent_requests: usize,
        max_best_of: usize,
        max_stop_sequences: usize,
        max_top_n_tokens: u32,
        max_input_tokens: usize,
        max_total_tokens: usize,
        waiting_served_ratio: f32,
        max_batch_prefill_tokens: u32,
        max_batch_total_tokens: Option<u32>,
        max_waiting_tokens: usize,
        max_batch_size: Option<usize>,
        revision: Option<String>,
        validation_workers: usize,
        messages_api_enabled: bool,
        disable_grammar_support: bool,
        max_client_batch_size: usize,
        disable_usage_stats: bool,
        disable_crash_reports: bool,
    ) -> Self {
        Self {
            model_config,
            tokenizer_config,
            max_concurrent_requests,
            max_best_of,
            max_stop_sequences,
            max_top_n_tokens,
            max_input_tokens,
            max_total_tokens,
            waiting_served_ratio,
            max_batch_prefill_tokens,
            max_batch_total_tokens,
            max_waiting_tokens,
            max_batch_size,
            revision,
            validation_workers,
            messages_api_enabled,
            disable_grammar_support,
            max_client_batch_size,
            disable_usage_stats,
            disable_crash_reports,
        }
    }
}

/// This is more or less a copy of the code from the `text-generation-launcher` crate to avoid a dependency
#[derive(Serialize, Debug, Clone)]
pub struct Env {
    git_sha: &'static str,
    docker_label: &'static str,
    nvidia_env: String,
    xpu_env: String,
    system_env: SystemInfo,
}

#[derive(Serialize, Debug, Clone)]
pub struct SystemInfo {
    cpu_count: usize,
    cpu_type: String,
    total_memory: u64,
    architecture: String,
    platform: String,
}

impl SystemInfo {
    fn new() -> Self {
        let mut system = sysinfo::System::new_all();
        system.refresh_all();

        let cpu_count = system.cpus().len();
        let cpu_type = system.cpus()[0].brand().to_string();
        let total_memory = system.total_memory();
        let architecture = std::env::consts::ARCH.to_string();
        let platform = format!(
            "{}-{}-{}",
            std::env::consts::OS,
            std::env::consts::FAMILY,
            std::env::consts::ARCH
        );
        Self {
            cpu_count,
            cpu_type,
            total_memory,
            architecture,
            platform,
        }
    }
}

impl fmt::Display for SystemInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "CPU Count: {}", self.cpu_count)?;
        writeln!(f, "CPU Type: {}", self.cpu_type)?;
        writeln!(f, "Total Memory: {}", self.total_memory)?;
        writeln!(f, "Architecture: {}", self.architecture)?;
        writeln!(f, "Platform: {}", self.platform)?;
        Ok(())
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Env {
    pub fn new() -> Self {
        let nvidia_env = nvidia_smi();
        let xpu_env = xpu_smi();
        let system_env = SystemInfo::new();

        Self {
            system_env,
            nvidia_env: nvidia_env.unwrap_or("N/A".to_string()),
            xpu_env: xpu_env.unwrap_or("N/A".to_string()),
            git_sha: option_env!("VERGEN_GIT_SHA").unwrap_or("N/A"),
            docker_label: option_env!("DOCKER_LABEL").unwrap_or("N/A"),
        }
    }
}

impl fmt::Display for Env {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Runtime environment:")?;
        writeln!(f, "Commit sha: {}", self.git_sha)?;
        writeln!(f, "Docker label: {}", self.docker_label)?;
        writeln!(f, "nvidia-smi:\n{}", self.nvidia_env)?;
        write!(f, "xpu-smi:\n{}\n", self.xpu_env)?;
        write!(f, "System:\n{}", self.system_env)?;

        Ok(())
    }
}

fn nvidia_smi() -> Option<String> {
    let output = Command::new("nvidia-smi").output().ok()?;
    let nvidia_smi = String::from_utf8(output.stdout).ok()?;
    let output = nvidia_smi.replace('\n', "\n   ");
    Some(output.trim().to_string())
}

fn xpu_smi() -> Option<String> {
    let output = Command::new("xpu-smi").arg("discovery").output().ok()?;
    let xpu_smi = String::from_utf8(output.stdout).ok()?;
    let output = xpu_smi.replace('\n', "\n   ");
    Some(output.trim().to_string())
}

pub fn is_container() -> io::Result<bool> {
    let path = Path::new("/proc/self/cgroup");
    let file = File::open(&path)?;
    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        // Check for common container runtimes
        if line.contains("/docker/") || line.contains("/docker-") ||
           line.contains("/kubepods/") || line.contains("/kubepods-") ||
           line.contains("containerd") || line.contains("crio") ||
           line.contains("podman") {
            return Ok(true);
        }
    }
    Ok(false)
}
