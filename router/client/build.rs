use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../../proto/");

    fs::create_dir_all("src/v2/pb").unwrap_or(());
    let mut config = prost_build::Config::new();
    config.protoc_arg("--experimental_allow_proto3_optional");

    tonic_build::configure()
        .build_client(true)
        .build_server(false)
        .out_dir("src/v2/pb")
        .include_file("mod.rs")
        .compile_with_config(config, &["../../proto/generate.proto"], &["../../proto"])
        .map_err(|e| match e.kind(){
            std::io::ErrorKind::NotFound => {panic!("`protoc` not found, install libprotoc")},
            std::io::ErrorKind::Other => {panic!("`protoc` version unsupported, upgrade protoc: https://github.com/protocolbuffers/protobuf/releases")},
            e => {e}
        }).unwrap_or_else(|e| panic!("protobuf compilation failed: {e}"));

    fs::create_dir_all("src/v3/pb").unwrap_or(());
    let mut config = prost_build::Config::new();
    config.protoc_arg("--experimental_allow_proto3_optional");

    tonic_build::configure()
        .build_client(true)
        .build_server(false)
        .out_dir("src/v3/pb")
        .include_file("mod.rs")
        .compile_with_config(config, &["../../proto/v3/generate.proto"], &["../../proto"])
        .unwrap_or_else(|e| panic!("protobuf compilation failed: {e}"));

    Ok(())
}
