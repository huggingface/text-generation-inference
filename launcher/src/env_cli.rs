use std::process::Command;

pub fn nvidia_smi() -> Option<String> {
    let output = Command::new("nvidia-smi").output().ok()?;
    let nvidia_smi = String::from_utf8(output.stdout).ok()?;
    let output = nvidia_smi.replace("\n", "\n   ");
    Some(output.trim().to_string())
}

pub fn docker_image() -> Option<String> {
    let output = Command::new("docker")
        .args(&[
            "image",
            "inspect",
            "--format",
            "{{.RepoDigests}}",
            "ghcr.io/huggingface/text-generation-inference:latest",
        ])
        .output()
        .ok()?;
    let output = String::from_utf8(output.stdout).ok()?;
    Some(output.trim().to_string())
}

pub fn print_env() {
    println!("Cargo version: {}", crate::versions::CARGO_VERSION);
    println!("Commit SHA: {}", crate::versions::GIT_HASH);
    println!(
        "Docker image sha: {}",
        docker_image().unwrap_or("N/A".to_string())
    );
    let nvidia_smi = nvidia_smi().unwrap_or("N/A".to_string());
    println!("Nvidia-smi:\n     {}", nvidia_smi);
    println!("Command line used: <FILL IN>");
    println!("OS: <FILL IN>");
}
