use std::env;
use std::path::PathBuf;

use cxx_build::CFG;

const ADDITIONAL_BACKEND_LINK_LIBRARIES: [&str; 2] = ["spdlog", "fmt"];

// fn build_tensort_llm<P: AsRef<Path>>(tensorrt_llm_root_dir: P, is_debug: bool) -> PathBuf {
//     let build_wheel_path = tensorrt_llm_root_dir
//         .as_ref()
//         .join("/scripts")
//         .join("build_wheel.py");
//
//     let build_wheel_path_str = build_wheel_path.display().to_string();
//     let mut build_wheel_args = vec![
//         build_wheel_path_str.as_ref(),
//         "--cpp_only",
//         "--extra-cmake-vars BUILD_TESTS=OFF",
//         "--extra-cmake-vars BUILD_BENCHMARKS=OFF",
//     ];
//
//     if is_debug {
//         build_wheel_args.push("--fast_build");
//     }
//
//     let out = Command::new("python3")
//         .args(build_wheel_args)
//         .output()
//         .expect("Failed to compile TensorRT-LLM");
//     PathBuf::new().join(tensorrt_llm_root_dir)
// }

fn main() {
    // Misc variables
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_profile = env::var("PROFILE").unwrap();
    let is_debug = match build_profile.as_ref() {
        "debug" => true,
        _ => false,
    };

    // Compile TensorRT-LLM (as of today, it cannot be compiled from CMake)
    // let trtllm_path = build_tensort_llm(
    //     backend_path.join("build").join("_deps").join("trtllm-src"),
    //     is_debug,
    // );

    // Build the backend implementation through CMake
    let backend_path = cmake::Config::new(".")
        .uses_cxx11()
        .generator("Ninja")
        .profile(match is_debug {
            true => "Debug",
            false => "Release",
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
