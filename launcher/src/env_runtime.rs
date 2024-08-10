use std::fmt;
use std::process::Command;

pub(crate) struct Env {
    cargo_target: &'static str,
    cargo_version: &'static str,
    git_sha: &'static str,
    docker_label: &'static str,
    nvidia_env: String,
    xpu_env: String,
    hpu_env: String,
}

impl Env {
    pub fn new() -> Self {
        let nvidia_env = nvidia_smi();
        let xpu_env = xpu_smi();
        let hpu_env = hl_smi();

        Self {
            nvidia_env: nvidia_env.unwrap_or("N/A".to_string()),
            xpu_env: xpu_env.unwrap_or("N/A".to_string()),
            hpu_env: hpu_env.unwrap_or("N/A".to_string()),
            cargo_target: env!("VERGEN_CARGO_TARGET_TRIPLE"),
            cargo_version: env!("VERGEN_RUSTC_SEMVER"),
            git_sha: option_env!("VERGEN_GIT_SHA").unwrap_or("N/A"),
            docker_label: option_env!("DOCKER_LABEL").unwrap_or("N/A"),
        }
    }
}

impl fmt::Display for Env {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Runtime environment:")?;

        writeln!(f, "Target: {}", self.cargo_target)?;
        writeln!(f, "Cargo version: {}", self.cargo_version)?;
        writeln!(f, "Commit sha: {}", self.git_sha)?;
        writeln!(f, "Docker label: {}", self.docker_label)?;
        writeln!(f, "nvidia-smi:\n{}", self.nvidia_env)?;
        writeln!(f, "xpu-smi:\n{}", self.xpu_env)?;
        write!(f, "hpu-smi:\n{}", self.hpu_env)?;

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

fn hl_smi() -> Option<String> {
    let output = Command::new("hl-smi").output().ok()?;
    let hl_smi = String::from_utf8(output.stdout).ok()?;
    let output = hl_smi.replace('\n', "\n   ");
    Some(output.trim().to_string())
}
