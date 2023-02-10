use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../../proto/generate.proto");
    fs::create_dir("src/pb").unwrap_or(());

    let mut config = prost_build::Config::new();
    config.protoc_arg("--experimental_allow_proto3_optional");

    tonic_build::configure()
        .build_client(true)
        .build_server(false)
        .out_dir("src/pb")
        .include_file("mod.rs")
        .compile_with_config(config, &["../../proto/generate.proto"], &["../../proto"])
        .unwrap_or_else(|e| panic!("protobuf compilation failed: {e}"));

    Ok(())
}
