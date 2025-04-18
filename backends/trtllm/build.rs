use cxx_build::CFG;
use pkg_config;
use std::env;
use std::env::consts::ARCH;
use std::path::{absolute, PathBuf};
use std::sync::LazyLock;

const ADDITIONAL_BACKEND_LINK_LIBRARIES: [&str; 1] = ["spdlog"];
const CUDA_ARCH_LIST: Option<&str> = option_env!("CUDA_ARCH_LIST");
const CUDA_REQUIRED_VERSION: &str = "12.8";
const MPI_REQUIRED_VERSION: &str = "4.1";
const INSTALL_PREFIX: Option<&str> = option_env!("CMAKE_INSTALL_PREFIX");
const TENSORRT_ROOT_DIR: Option<&str> = option_env!("TENSORRT_ROOT_DIR");
const NCCL_ROOT_DIR: Option<&str> = option_env!("NCCL_ROOT_DIR");

const IS_GHA_BUILD: LazyLock<bool> = LazyLock::new(|| {
    option_env!("SCCACHE_GHA_ENABLED").map_or(false, |value| match value.to_lowercase().as_str() {
        "on" => true,
        "true" => true,
        "1" => true,
        _ => false,
    })
});

// Dependencies
const BACKEND_DEPS: &str = "tgi_trtllm_backend_impl";
const CUDA_TRANSITIVE_DEPS: [&str; 4] = ["cuda", "cudart", "cublas", "nvidia-ml"];
const TENSORRT_LLM_TRANSITIVE_DEPS: [(&str, &str); 5] = [
    ("dylib", "tensorrt_llm"),
    ("dylib", "tensorrt_llm_nvrtc_wrapper"),
    ("dylib", "nvinfer_plugin_tensorrt_llm"),
    ("dylib", "decoder_attention_0"),
    ("dylib", "decoder_attention_1"),
];

macro_rules! probe {
    ($name: expr, $version: expr) => {
        if let Err(_) = pkg_config::probe_library($name) {
            pkg_config::probe_library(&format!("{}-{}", $name, $version))
                .expect(&format!("Failed to locate {}", $name));
        }
    };
}

fn get_compiler_flag(
    switch: bool,
    true_case: &'static str,
    false_case: &'static str,
) -> &'static str {
    match switch {
        true => true_case,
        false => false_case,
    }
}

fn get_library_architecture() -> &'static str {
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let env = env::var("CARGO_CFG_TARGET_ENV").unwrap();

    match os.as_str() {
        "linux" => {
            if env != "gnu" {
                panic!("unsupported linux ABI {env}, only 'gnu' is supported")
            }

            match arch.as_str() {
                "x86_64" => "x86_64-linux-gnu",
                "aarch64" => "aarch64-linux-gnu",
                _ => panic!("unsupported linux architecture {arch}"),
            }
        }
        "windows" => {
            if env != "msvc" {
                panic!("unsupported windows ABI {env}, only 'msvc' is supported")
            }

            match arch.as_str() {
                "x86_64" => "x86_64-windows-msvc",
                _ => panic!("unsupported windows architecture {arch}"),
            }
        }
        _ => panic!("unsupported OS {os}"),
    }
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

    let mut config = cmake::Config::new(".");
    config
        .uses_cxx11()
        .generator("Ninja")
        .profile(match is_debug {
            true => "Debug",
            false => "Release",
        })
        .env("OPT_LEVEL", opt_level)
        .define("CMAKE_INSTALL_PREFIX", &install_path)
        .define("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc")
        .define("CMAKE_LIBRARY_ARCHITECTURE", get_library_architecture())
        .define("TGI_TRTLLM_BACKEND_TARGET_CUDA_ARCH_LIST", cuda_arch_list)
        .define(
            "TGI_TRTLLM_BACKEND_DEBUG",
            get_compiler_flag(is_debug, "ON", "OFF"),
        )
        .define("TGI_TRTLLM_BACKEND_TRT_ROOT", tensorrt_path);

    if is_debug || *IS_GHA_BUILD {
        config.define("TGI_TRTLLM_BACKEND_BUILD_TESTS", "ON");
    }

    if option_env!("USE_LLD_LINKER").is_some() {
        println!("cargo:warning=Using lld linker");
        config.define("TGI_TRTLLM_BACKEND_BUILD_USE_LLD", "ON");
    }

    if (is_debug && option_env!("ENABLE_ASAN").is_some()) || *IS_GHA_BUILD {
        println!("cargo:warning=Enabling Address Sanitizer");
        config.define("TGI_TRTLLM_BACKEND_ENABLE_ASAN", "ON");
    }

    if (is_debug && option_env!("ENABLE_UBSAN").is_some()) || *IS_GHA_BUILD {
        println!("cargo:warning=Enabling Undefined Sanitizer");
        config.define("TGI_TRTLLM_BACKEND_ENABLE_UBSAN", "ON");
    }

    if let Some(nvcc_host_compiler) = option_env!("CMAKE_CUDA_HOST_COMPILER") {
        config.define("CMAKE_CUDA_HOST_COMPILER", nvcc_host_compiler);
    }

    if let Some(wrapper) = option_env!("RUSTC_WRAPPER") {
        println!("cargo:warning=Using caching tool: {wrapper}");
        config.define("CMAKE_C_COMPILER_LAUNCHER", wrapper);
        config.define("CMAKE_CXX_COMPILER_LAUNCHER", wrapper);
        config.define("CMAKE_CUDA_COMPILER_LAUNCHER", wrapper);
    }

    // Allow to override which Python to use ...
    if let Some(python3) = option_env!("Python3_EXECUTABLE") {
        config.define("Python3_EXECUTABLE", python3);
    }

    config.build();

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
    for path in ["lib", "lib64"] {
        let install_lib_path = install_path.join(path);
        println!(
            r"cargo:warning=Adding link search path: {}",
            install_lib_path.display()
        );
        println!(r"cargo:rustc-link-search={}", install_lib_path.display());
    }
    (PathBuf::from(install_path), deps_folder)
}

fn build_ffi_layer(deps_folder: &PathBuf, is_debug: bool) {
    CFG.include_prefix = "backends/trtllm";
    cxx_build::bridge("src/lib.rs")
        .static_flag(true)
        .std("c++23")
        .include(deps_folder.join("spdlog-src").join("include"))
        .include(deps_folder.join("json-src").join("include"))
        .include(deps_folder.join("trtllm-src").join("cpp").join("include"))
        .include("/usr/local/cuda/include")
        .include("/usr/local/tensorrt/include")
        .include("csrc/")
        .file("csrc/ffi.hpp")
        .define(
            "TGI_TRTLLM_BACKEND_DEBUG",
            get_compiler_flag(is_debug, "ON", "OFF"),
        )
        .compile("tgi_trtllm_backend");

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=cmake/trtllm.cmake");
    println!("cargo:rerun-if-changed=cmake/json.cmake");
    println!("cargo:rerun-if-changed=cmake/spdlog.cmake");
    println!("cargo:rerun-if-changed=csrc/backend.hpp");
    println!("cargo:rerun-if-changed=csrc/backend.cpp");
    println!("cargo:rerun-if-changed=csrc/hardware.hpp");
    println!("cargo:rerun-if-changed=csrc/ffi.hpp");
}

fn main() {
    // Misc variables
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_profile = env::var("PROFILE").unwrap();
    let (is_debug, opt_level) = match build_profile.as_ref() {
        "debug" => (true, "0"),
        "dev" => (true, "0"),
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
    println!("cargo:rustc-link-lib=static={}", &BACKEND_DEPS);
}
