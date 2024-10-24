use cxx_build::CFG;
use std::env;
use std::path::PathBuf;

const CMAKE_LLAMA_CPP_DEFAULT_CUDA_ARCHS: &str = "75-real;80-real;86-real;89-real;90-real";
const CMAKE_LLAMA_CPP_TARGET: &str = "tgi_llama_cpp_backend_impl";
const ADDITIONAL_BACKEND_LINK_LIBRARIES: [&str; 2] = ["spdlog", "fmt"];
const MPI_REQUIRED_VERSION: &str = "4.1";

macro_rules! probe {
    ($name: expr, $version: expr) => {
        if let Err(_) = pkg_config::probe_library($name) {
            pkg_config::probe_library(&format!("{}-{}", $name, $version))
                .expect(&format!("Failed to locate {}", $name));
        }
    };
}

fn build_backend(is_debug: bool, opt_level: &str, out_dir: &PathBuf) -> PathBuf {
    let install_path = env::var("CMAKE_INSTALL_PREFIX")
        .map(|val| PathBuf::from(val))
        .unwrap_or(out_dir.join("dist"));

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

    let deps_folder = out_dir.join("build").join("_deps");
    deps_folder
}

fn build_ffi_layer(deps_folder: &PathBuf) {
    println!("cargo:warning={}", &deps_folder.display());
    CFG.include_prefix = "backends/llamacpp";
    cxx_build::bridge("src/lib.rs")
        .static_flag(true)
        .std("c++23")
        .include(deps_folder.join("fmt-src").join("include"))
        .include(deps_folder.join("spdlog-src").join("include"))
        .include(deps_folder.join("llama-src").join("common"))
        .include(deps_folder.join("llama-src").join("ggml").join("include"))
        .include(deps_folder.join("llama-src").join("include"))
        .include("csrc/backend.hpp")
        .file("csrc/ffi.cpp")
        .compile(CMAKE_LLAMA_CPP_TARGET);

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=csrc/backend.hpp");
    println!("cargo:rerun-if-changed=csrc/backend.cpp");
    println!("cargo:rerun-if-changed=csrc/ffi.hpp");
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
    let deps_folder = build_backend(is_debug, opt_level, &out_dir);

    // Build the FFI layer calling the backend above
    build_ffi_layer(&deps_folder);

    // Emit linkage search path
    probe!("ompi", MPI_REQUIRED_VERSION);

    // Backend
    // BACKEND_DEPS.iter().for_each(|name| {
    //     println!("cargo:rustc-link-lib=static={}", name);
    // });
}
