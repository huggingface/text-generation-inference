use std::env;
use std::path::PathBuf;

use cxx_build::CFG;

const ADDITIONAL_BACKEND_LINK_LIBRARIES: [&str; 2] = ["spdlog", "fmt"];

fn main() {
    // Misc variables
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_profile = env::var("PROFILE").unwrap();
    let is_debug = match build_profile.as_ref() {
        "debug" => true,
        _ => false,
    };

    // Build the backend implementation through CMake
    let backend_path = cmake::Config::new(".")
        .uses_cxx11()
        .generator("Ninja")
        .profile(match is_debug {
            true => "Debug",
            false => "Release",
        })
        .define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc")
        .build();

    // Additional transitive CMake dependencies
    let deps_folder = out_dir.join("build").join("_deps");

    for dependency in ADDITIONAL_BACKEND_LINK_LIBRARIES {
        let dep_name = match build_profile.as_ref() {
            "debug" => format!("{}d", dependency),
            _ => String::from(dependency),
        };
        let dep_path = deps_folder.join(format!("{}-build", dependency));
        println!("cargo:rustc-link-search={}", dep_path.display());
        println!("cargo:rustc-link-lib=static={}", dep_name);
    }

    // Build the FFI layer calling the backend above
    CFG.include_prefix = "backends/trtllm";
    cxx_build::bridge("src/lib.rs")
        .static_flag(true)
        .include(deps_folder.join("fmt-src").join("include"))
        .include(deps_folder.join("spdlog-src").join("include"))
        .include(deps_folder.join("json-src").join("include"))
        .include(deps_folder.join("trtllm-src").join("cpp").join("include"))
        .include("/usr/local/cuda/include")
        .include("/usr/local/tensorrt/include")
        .file("src/ffi.cpp")
        .std("c++20")
        .compile("tgi_trtllm_backend");

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=include/backend.h");
    println!("cargo:rerun-if-changed=lib/backend.cpp");
    println!("cargo:rerun-if-changed=include/ffi.h");
    println!("cargo:rerun-if-changed=src/ffi.cpp");

    // Emit linkage information
    // - tgi_trtllm_backend (i.e. FFI layer - src/ffi.cpp)
    let trtllm_lib_path = deps_folder
        .join("trtllm-src")
        .join("cpp")
        .join("tensorrt_llm");

    let trtllm_executor_linker_search_path =
        trtllm_lib_path.join("executor").join("x86_64-linux-gnu");

    // TRTLLM libtensorrt_llm_nvrtc_wrapper.so
    let trtllm_nvrtc_linker_search_path = trtllm_lib_path
        .join("kernels")
        .join("decoderMaskedMultiheadAttention")
        .join("decoderXQAImplJIT")
        .join("nvrtcWrapper")
        .join("x86_64-linux-gnu");

    println!(r"cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!(r"cargo:rustc-link-search=native=/usr/local/cuda/lib64/stubs");
    println!(r"cargo:rustc-link-search=native=/usr/local/tensorrt/lib");
    println!(r"cargo:rustc-link-search=native={}", backend_path.display());
    // println!(
    //     r"cargo:rustc-link-search=native={}/build",
    //     backend_path.display()
    // );
    println!(
        r"cargo:rustc-link-search=native={}",
        backend_path.join("lib").display()
    );
    println!(
        r"cargo:rustc-link-search=native={}",
        trtllm_executor_linker_search_path.display()
    );
    println!(
        r"cargo:rustc-link-search=native={}",
        trtllm_nvrtc_linker_search_path.display()
    );
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=mpi");
    println!("cargo:rustc-link-lib=dylib=nvidia-ml");
    println!("cargo:rustc-link-lib=dylib=nvinfer");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_tensorrt_llm");
    println!("cargo:rustc-link-lib=dylib=tensorrt_llm_nvrtc_wrapper");
    println!("cargo:rustc-link-lib=static=tensorrt_llm_executor_static");
    println!("cargo:rustc-link-lib=dylib=tensorrt_llm");
    println!("cargo:rustc-link-lib=static=tgi_trtllm_backend_impl");
    println!("cargo:rustc-link-lib=static=tgi_trtllm_backend");
}
