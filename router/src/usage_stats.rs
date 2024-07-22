use crate::config::Config;
use csv::ReaderBuilder;
use reqwest::header::HeaderMap;
use serde::Serialize;
use std::{
    fs::File,
    io::{self, BufRead},
    path::Path,
    process::Command,
    time::Duration,
};
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
    Error,
}

#[derive(Debug, Serialize)]
pub struct UsageStatsEvent {
    user_agent: UserAgent,
    event_type: EventType,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_reason: Option<String>,
}

impl UsageStatsEvent {
    pub fn new(user_agent: UserAgent, event_type: EventType, error_reason: Option<String>) -> Self {
        Self {
            user_agent,
            event_type,
            error_reason,
        }
    }
    pub async fn send(&self) {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse().unwrap());
        let body = serde_json::to_string(&self).unwrap();
        let client = reqwest::Client::new();
        let _ = client
            .post(TELEMETRY_URL)
            .headers(headers)
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
    nvidia_info: Option<Vec<NvidiaSmiInfo>>,
    xpu_info: Option<Vec<XpuSmiInfo>>,
    system_env: SystemInfo,
}

#[derive(Debug, Serialize, Clone)]
struct NvidiaSmiInfo {
    name: String,
    pci_bus_id: String,
    driver_version: String,
    pstate: String,
    pcie_link_gen_max: String,
    pcie_link_gen_current: String,
    temperature_gpu: String,
    utilization_gpu: String,
    utilization_memory: String,
    memory_total: String,
    memory_free: String,
    memory_used: String,
    reset_status_reset_required: String,
    reset_status_drain_and_reset_recommended: String,
    compute_cap: String,
    ecc_errors_corrected_volatile_total: String,
    mig_mode_current: String,
    power_draw_instant: String,
    power_limit: String,
}

impl NvidiaSmiInfo {
    fn new() -> Option<Vec<NvidiaSmiInfo>> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.gpucurrent,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,reset_status.reset_required,reset_status.drain_and_reset_recommended,compute_cap,ecc.errors.corrected.volatile.total,mig.mode.current,power.draw.instant,power.limit",
                "--format=csv"
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8(output.stdout).ok()?;

        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(stdout.as_bytes());

        let mut infos = Vec::new();

        for result in rdr.records() {
            let record = result.ok()?;
            infos.push(NvidiaSmiInfo {
                name: record[0].to_string(),
                pci_bus_id: record[1].to_string(),
                driver_version: record[2].to_string(),
                pstate: record[3].to_string(),
                pcie_link_gen_max: record[4].to_string(),
                pcie_link_gen_current: record[5].to_string(),
                temperature_gpu: record[6].to_string(),
                utilization_gpu: record[7].to_string(),
                utilization_memory: record[8].to_string(),
                memory_total: record[9].to_string(),
                memory_free: record[10].to_string(),
                memory_used: record[11].to_string(),
                reset_status_reset_required: record[12].to_string(),
                reset_status_drain_and_reset_recommended: record[13].to_string(),
                compute_cap: record[14].to_string(),
                ecc_errors_corrected_volatile_total: record[15].to_string(),
                mig_mode_current: record[16].to_string(),
                power_draw_instant: record[17].to_string(),
                power_limit: record[18].to_string(),
            });
        }

        Some(infos)
    }
}

#[derive(Debug, Serialize, Clone)]
struct XpuSmiInfo {
    device_id: usize,
    gpu_utilization: f32,
    gpu_power: f32,
    gpu_core_temperature: f32,
    gpu_memory_bandwidth_utilization: f32,
}

impl XpuSmiInfo {
    /// based on this https://github.com/intel/xpumanager/blob/master/doc/smi_user_guide.md#dump-the-device-statistics-in-csv-format
    fn new() -> Option<Vec<XpuSmiInfo>> {
        let output = Command::new("xpu-smi")
            .args([
                "dump", "-d", "-1", "-m",
                "0,1,3,17", // Metrics IDs: GPU Utilization, GPU Power, GPU Core Temperature, GPU Memory Bandwidth Utilization
                "-n", "1", "-j",
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8(output.stdout).ok()?;
        let mut infos = Vec::new();

        let json_data: serde_json::Value = match serde_json::from_str(&stdout) {
            Ok(data) => data,
            Err(_) => return None,
        };

        if let Some(metrics_data) = json_data.as_array() {
            for entry in metrics_data {
                let device_id = entry["deviceId"].as_u64()? as usize;
                let gpu_utilization = entry["metrics"][0].as_f64()? as f32;
                let gpu_power = entry["metrics"][1].as_f64()? as f32;
                let gpu_core_temperature = entry["metrics"][2].as_f64()? as f32;
                let gpu_memory_bandwidth_utilization = entry["metrics"][3].as_f64()? as f32;

                infos.push(XpuSmiInfo {
                    device_id,
                    gpu_utilization,
                    gpu_power,
                    gpu_core_temperature,
                    gpu_memory_bandwidth_utilization,
                });
            }
        }

        Some(infos)
    }
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

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

impl Env {
    pub fn new() -> Self {
        Self {
            system_env: SystemInfo::new(),
            nvidia_info: NvidiaSmiInfo::new(),
            xpu_info: XpuSmiInfo::new(),
            git_sha: option_env!("VERGEN_GIT_SHA").unwrap_or("N/A"),
            docker_label: option_env!("DOCKER_LABEL").unwrap_or("N/A"),
        }
    }
}

pub fn is_container() -> io::Result<bool> {
    let path = Path::new("/proc/self/cgroup");
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        // Check for common container runtimes
        if line.contains("/docker/")
            || line.contains("/docker-")
            || line.contains("/kubepods/")
            || line.contains("/kubepods-")
            || line.contains("containerd")
            || line.contains("crio")
            || line.contains("podman")
        {
            return Ok(true);
        }
    }
    Ok(false)
}
