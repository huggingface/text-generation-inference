use std::fs::File;
use std::io::Write;
use std::process::Command;

pub fn cargo_version() -> Option<String> {
    let output = Command::new("cargo").args(&["version"]).output().ok()?;
    let output = String::from_utf8(output.stdout).ok()?;
    Some(output.trim().to_string())
}

fn main() {
    let output = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap().trim().to_string();

    let cargo_version = cargo_version().unwrap_or("N/A".to_string());
    let mut file = File::create("src/versions.rs").unwrap();
    file.write(format!("pub static GIT_HASH: &str = \"{git_hash}\";\n").as_bytes())
        .unwrap();
    file.write(format!(r#"pub static CARGO_VERSION: &str = "{cargo_version}";"#).as_bytes())
        .unwrap();
}
