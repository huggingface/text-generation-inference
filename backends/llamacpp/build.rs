use cxx_build::CFG;
use std::env;
use std::path::{Path, PathBuf};

const CMAKE_LLAMA_CPP_DEFAULT_CUDA_ARCHS: &str = "75-real;80-real;86-real;89-real;90-real";
const CMAKE_LLAMA_CPP_TARGET: &str = "tgi_llamacpp_backend_impl";
const CMAKE_LLAMA_CPP_FFI_TARGET: &str = "tgi_llamacpp_backend";
const MPI_REQUIRED_VERSION: &str = "4.1";

const BACKEND_DEPS: [&str; 2] = [CMAKE_LLAMA_CPP_TARGET, CMAKE_LLAMA_CPP_FFI_TARGET];

macro_rules! probe {
    ($name: expr, $version: expr) => {
        if let Err(_) = pkg_config::probe_library($name) {
            pkg_config::probe_library(&format!("{}-{}", $name, $version))
                .expect(&format!("Failed to locate {}", $name));
        }
    };
}

fn build_backend(
    is_debug: bool,
    opt_level: &str,
    out_dir: &Path,
    install_path: &PathBuf,
) -> PathBuf {
    let build_cuda = option_env!("LLAMA_CPP_BUILD_CUDA").unwrap_or("OFF");
    let cuda_archs =
        option_env!("LLAMA_CPP_TARGET_CUDA_ARCHS").unwrap_or(CMAKE_LLAMA_CPP_DEFAULT_CUDA_ARCHS);

    let _ = cmake::Config::new(".")
        .uses_cxx11()
        .generator("Ninja")
        .profile(match is_debug {
            true => "Debug",
            false => "Release",
        })
        .env("OPT_LEVEL", opt_level)
        .define("CMAKE_INSTALL_PREFIX", &install_path)
        .define("LLAMA_CPP_BUILD_CUDA", build_cuda)
        .define("LLAMA_CPP_TARGET_CUDA_ARCHS", cuda_archs)
        .build();

    let lib_path = install_path.join("lib64");
    println!("cargo:rustc-link-search=native={}", lib_path.display());

    let deps_folder = out_dir.join("build").join("_deps");
    deps_folder
}

fn build_ffi_layer(deps_folder: &Path, install_prefix: &Path) {
    println!("cargo:warning={}", deps_folder.display());
    CFG.include_prefix = "backends/llamacpp";
    cxx_build::bridge("src/lib.rs")
        .static_flag(true)
        .std("c++23")
        .include(deps_folder.join("spdlog-src").join("include")) // Why spdlog doesnt install headers?
        // .include(deps_folder.join("fmt-src").join("include")) // Why spdlog doesnt install headers?
        // .include(deps_folder.join("llama-src").join("include")) // Why spdlog doesnt install headers?
        .include(deps_folder.join("llama-src").join("ggml").join("include")) // Why spdlog doesnt install headers?
        .include(deps_folder.join("llama-src").join("common").join("include")) // Why spdlog doesnt install headers?
        .include(install_prefix.join("include"))
        .include("csrc")
        .file("csrc/ffi.hpp")
        .compile(CMAKE_LLAMA_CPP_FFI_TARGET);
}

fn main() {
    // Misc variables
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_profile = env::var("PROFILE").unwrap();
    let (is_debug, opt_level) = match build_profile.as_ref() {
        "debug" => (true, "0"),
        _ => (false, "3"),
    };

    let install_path = env::var("CMAKE_INSTALL_PREFIX")
        .map(|val| PathBuf::from(val))
        .unwrap_or(out_dir.join("dist"));

    // Build the backend
    let deps_path = build_backend(is_debug, opt_level, out_dir.as_path(), &install_path);

    // Build the FFI layer calling the backend above
    build_ffi_layer(&deps_path, &install_path);

    // Emit linkage search path
    probe!("ompi", MPI_REQUIRED_VERSION);

    // Backend
    BACKEND_DEPS.iter().for_each(|name| {
        println!("cargo:rustc-link-lib=static={}", name);
    });

    // Linkage info
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=fmtd");
    println!("cargo:rustc-link-lib=static=spdlogd");
    println!("cargo:rustc-link-lib=static=common");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=llama");

    // Rerun if one of these file change
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=csrc/backend.hpp");
    println!("cargo:rerun-if-changed=csrc/backend.cpp");
    println!("cargo:rerun-if-changed=csrc/ffi.hpp");
}
