use cxx_build::CFG;
use pkg_config;
use std::env;
use std::env::consts::ARCH;
use std::path::{absolute, PathBuf};

const ADDITIONAL_BACKEND_LINK_LIBRARIES: [&str; 2] = ["spdlog", "fmt"];
const CUDA_ARCH_LIST: Option<&str> = option_env!("CUDA_ARCH_LIST");
const CUDA_REQUIRED_VERSION: &str = "12.6";
const MPI_REQUIRED_VERSION: &str = "4.1";
const INSTALL_PREFIX: Option<&str> = option_env!("CMAKE_INSTALL_PREFIX");
const TENSORRT_ROOT_DIR: Option<&str> = option_env!("TENSORRT_ROOT_DIR");
const NCCL_ROOT_DIR: Option<&str> = option_env!("NCCL_ROOT_DIR");

// Dependencies
const BACKEND_DEPS: [&str; 2] = ["tgi_trtllm_backend_impl", "tgi_trtllm_backend"];
const CUDA_TRANSITIVE_DEPS: [&str; 4] = ["cuda", "cudart", "cublas", "nvidia-ml"];
const TENSORRT_LLM_TRANSITIVE_DEPS: [(&str, &str); 5] = [
    ("dylib", "tensorrt_llm"),
    ("static", "tensorrt_llm_executor_static"),
    ("dylib", "tensorrt_llm_nvrtc_wrapper"),
    ("dylib", "nvinfer_plugin_tensorrt_llm"),
    ("dylib", "decoder_attention"),
];

macro_rules! probe {
    ($name: expr, $version: expr) => {
        if let Err(_) = pkg_config::probe_library($name) {
            pkg_config::probe_library(&format!("{}-{}", $name, $version))
                .expect(&format!("Failed to locate {}", $name));
        }
    };
}

fn build_backend(is_debug: bool, opt_level: &str, out_dir: &PathBuf) -> (PathBuf, PathBuf) {
    // Build the backend implementation through CMake
    let install_path = INSTALL_PREFIX.unwrap_or("/usr/local/tgi");
    let tensorrt_path = TENSORRT_ROOT_DIR.unwrap_or("/usr/local/tensorrt");
    let cuda_arch_list = CUDA_ARCH_LIST.unwrap_or("75-real;80-real;86-real;89-real;90-real");

    let mut install_path = PathBuf::from(install_path);
    if !install_path.is_absolute() {
        install_path = absolute(out_dir).expect("cannot happen").join(install_path);
    }

    let _ = cmake::Config::new(".")
        .uses_cxx11()
        .generator("Ninja")
        .profile(match is_debug {
            true => "Debug",
            false => "Release",
        })
        .env("OPT_LEVEL", opt_level)
        .define("CMAKE_INSTALL_PREFIX", &install_path)
        .define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc")
        .define("TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST", cuda_arch_list)
        .define("TGI_TRTLLM_BACKEND_TRT_ROOT", tensorrt_path)
        .build();

    // Additional transitive CMake dependencies
    let deps_folder = out_dir.join("build").join("_deps");
    for dependency in ADDITIONAL_BACKEND_LINK_LIBRARIES {
        let dep_name = match is_debug {
            true => format!("{}d", dependency),
            false => String::from(dependency),
        };
        let dep_path = deps_folder.join(format!("{}-build", dependency));
        println!("cargo:rustc-link-search={}", dep_path.display());
        println!("cargo:rustc-link-lib=static={}", dep_name);
    }

    // Emit linkage information from the artifacts we just built
    let install_lib_path = install_path.join("lib");

    println!(
        r"cargo:warning=Adding link search path: {}",
        install_lib_path.display()
    );
    println!(r"cargo:rustc-link-search={}", install_lib_path.display());

    (PathBuf::from(install_path), deps_folder)
}

fn build_ffi_layer(deps_folder: &PathBuf, is_debug: bool) {
    let ndebug = match is_debug {
        true => "1",
        false => "0",
    };

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
        .define("NDEBUG", ndebug)
        .compile("tgi_trtllm_backend");

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=cmake/trtllm.cmake");
    println!("cargo:rerun-if-changed=cmake/json.cmake");
    println!("cargo:rerun-if-changed=cmake/fmt.cmake");
    println!("cargo:rerun-if-changed=cmake/spdlog.cmake");
    println!("cargo:rerun-if-changed=include/backend.h");
    println!("cargo:rerun-if-changed=lib/backend.cpp");
    println!("cargo:rerun-if-changed=include/ffi.h");
    println!("cargo:rerun-if-changed=src/ffi.cpp");
}

fn main() {
    // Misc variables
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_profile = env::var("PROFILE").unwrap();
    let (is_debug, opt_level) = match build_profile.as_ref() {
        "debug" => (true, "0"),
        _ => (false, "3"),
    };

    // Build the backend
    let (_backend_path, deps_folder) = build_backend(is_debug, opt_level, &out_dir);

    // Build the FFI layer calling the backend above
    build_ffi_layer(&deps_folder, is_debug);

    // Emit linkage search path
    probe!("ompi", MPI_REQUIRED_VERSION);

    // Probe CUDA & co. with pkg-config
    CUDA_TRANSITIVE_DEPS.iter().for_each(|name| {
        probe!(name, CUDA_REQUIRED_VERSION);
    });

    // NCCL is slightly trickier because it might not have a pkgconfig installed
    let nccl_library_path_default = format!("/usr/local/{}-linux-gnu", ARCH);
    let nccl_library_path = NCCL_ROOT_DIR.unwrap_or(&nccl_library_path_default);
    println!(r"cargo:rustc-link-search=native={}", nccl_library_path);
    println!("cargo:rustc-link-lib=dylib=nccl");

    // TensorRT
    let tensort_library_path = TENSORRT_ROOT_DIR.unwrap_or("/usr/local/tensorrt/lib");
    println!(r"cargo:rustc-link-search=native={}", tensort_library_path);
    println!("cargo:rustc-link-lib=dylib=nvinfer");

    // TensorRT-LLM
    TENSORRT_LLM_TRANSITIVE_DEPS
        .iter()
        .for_each(|(link_type, name)| {
            println!("cargo:rustc-link-lib={}={}", link_type, name);
        });

    // Backend
    BACKEND_DEPS.iter().for_each(|name| {
        println!("cargo:rustc-link-lib=static={}", name);
    });
}
