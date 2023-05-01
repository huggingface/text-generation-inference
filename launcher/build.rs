use std::fs::File;
use std::io::Write;
use std::process::Command;

fn main() {
    let output = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap().trim().to_string();
    let mut file = File::create("src/versions.rs").unwrap();
    file.write(format!(r#"pub static GIT_HASH: &str = "{git_hash}";"#).as_bytes())
        .unwrap();
}
