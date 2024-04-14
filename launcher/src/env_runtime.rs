use std::fmt;
use std::process::Command;

pub(crate) struct Env {
    cargo_target: &'static str,
    cargo_version: &'static str,
    git_sha: &'static str,
    docker_label: &'static str,
    nvidia_env: String,
    npu_env: String,
}

impl Env {
    pub fn new() -> Self {
        let nvidia_env = nvidia_smi();
        let npu_env = npu_smi();

        Self {
            nvidia_env: nvidia_env.unwrap_or("N/A".to_string()),
            npu_env: npu_env.unwrap_or("N/A".to_string()),
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
        write!(f, "npu-smi:\n{}", self.npu_env)?;

        Ok(())
    }
}

fn nvidia_smi() -> Option<String> {
    let output = Command::new("nvidia-smi").output().ok()?;
    let nvidia_smi = String::from_utf8(output.stdout).ok()?;
    let output = nvidia_smi.replace('\n', "\n   ");
    Some(output.trim().to_string())
}

fn npu_smi() -> Option<String> {
    let output = Command::new("npu-smi info").output().ok()?;
    let npu_smi = String::from_utf8(output.stdout).ok()?;
    let output = npu_smi.replace('\n', "\n   ");
    Some(output.trim().to_string())
}
