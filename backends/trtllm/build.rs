use std::env;
use std::path::PathBuf;

use cxx_build::CFG;
use pkg_config;

const ADDITIONAL_BACKEND_LINK_LIBRARIES: [&str; 2] = ["spdlog", "fmt"];
const CUDA_ARCH_LIST: Option<&str> = option_env!("CUDA_ARCH_LIST");
const CUDA_REQUIRED_VERSION: &str = "12.4";
const MPI_REQUIRED_VERSION: &str = "4.1";
const INSTALL_PREFIX: Option<&str> = option_env!("CMAKE_INSTALL_PREFIX");
const TENSORRT_ROOT_DIR: Option<&str> = option_env!("TENSORRT_ROOT_DIR");

macro_rules! probe {
    ($name: literal, $version: expr) => {
        if let Err(_) = pkg_config::probe_library($name) {
            pkg_config::probe_library(&format!("cuda-{}", $version))
                .expect(&format!("Failed to locate {}", $name));
        }
    };
}

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
        .env("OPT_LEVEL", "3")
        .out_dir(INSTALL_PREFIX.unwrap_or("/usr/local/tgi"))
        .define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc")
        .define(
            "TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST",
            CUDA_ARCH_LIST.unwrap_or("90-real"), // Hopper by default
        )
        .define(
            "TGI_TRTLLM_BACKEND_TRT_ROOT",
            TENSORRT_ROOT_DIR.unwrap_or("/usr/local/tensorrt"),
        )
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

    // Emit linkage search path
    probe!("ompi", MPI_REQUIRED_VERSION);

    // Probe CUDA & co. with pkg-config
    probe!("cuda", CUDA_REQUIRED_VERSION);
    probe!("cudart", CUDA_REQUIRED_VERSION);
    probe!("cublas", CUDA_REQUIRED_VERSION);
    probe!("nvidia-ml", CUDA_REQUIRED_VERSION);

    // TensorRT
    println!(r"cargo:rustc-link-search=native=/usr/local/tensorrt/lib");
    println!(r"cargo:rustc-link-search=native={}", backend_path.display());

    // TensorRT-LLM
    println!(
        r"cargo:rustc-link-search=native={}",
        backend_path.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=tensorrt_llm");
    println!("cargo:rustc-link-lib=static=tensorrt_llm_executor_static");
    println!("cargo:rustc-link-lib=dylib=nvinfer_plugin_tensorrt_llm");
    println!("cargo:rustc-link-lib=dylib=tensorrt_llm_nvrtc_wrapper");

    // Backend
    println!("cargo:rustc-link-lib=static=tgi_trtllm_backend_impl");
    println!("cargo:rustc-link-lib=static=tgi_trtllm_backend");
}
