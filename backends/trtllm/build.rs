use std::env;
use std::path::PathBuf;

use cxx_build::CFG;

const ADDITIONAL_BACKEND_LINK_LIBRARIES: [&str; 2] = ["spdlog", "fmt"];

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_profile = env::var("PROFILE").unwrap();

    // Build the backend implementation through CMake
    let backend_path = cmake::Config::new(".")
        .uses_cxx11()
        .generator("Ninja")
        .profile(match build_profile.as_ref() {
            "release" => "Release",
            _ => "Debug",
        })
        .build_target("tgi_trtllm_backend_impl")
        .build();

    // Build the FFI layer calling the backend above
    CFG.include_prefix = "backends/trtllm";
    cxx_build::bridge("src/lib.rs")
        .static_flag(true)
        .file("src/ffi.cpp")
        .std("c++20")
        .compile("tgi_trtllm_backend");

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=include/backend.h");
    println!("cargo:rerun-if-changed=lib/backend.cpp");
    println!("cargo:rerun-if-changed=src/ffi.cpp");

    // Additional transitive CMake dependencies
    for dependency in ADDITIONAL_BACKEND_LINK_LIBRARIES {
        let dep_folder = out_dir
            .join("build")
            .join("_deps")
            .join(format!("{}-build", dependency));

        let dep_name = match build_profile.as_ref() {
            "debug" => format!("{}d", dependency),
            _ => String::from(dependency),
        };
        println!("cargo:warning={}", dep_folder.display());
        println!("cargo:rustc-link-search=native={}", dep_folder.display());
        println!("cargo:rustc-link-lib=static={}", dep_name);
    }

    // Emit linkage information
    // - tgi_trtllm_backend (i.e. FFI layer - src/ffi.cpp)
    println!(r"cargo:rustc-link-search=native={}", backend_path.display());
    println!("cargo:rustc-link-lib=static=tgi_trtllm_backend");

    // - tgi_trtllm_backend_impl (i.e. C++ code base to run inference include/backend.h)
    println!(
        r"cargo:rustc-link-search=native={}/build",
        backend_path.display()
    );
    println!("cargo:rustc-link-lib=static=tgi_trtllm_backend_impl");
}
