use cxx_build::CFG;

fn main() {
    let backend_path = cmake::Config::new(".")
        .uses_cxx11()
        .generator("Ninja")
        .build_target("tgi_trtllm_backend_impl")
        .build();

    CFG.include_prefix = "backends/trtllm";
    cxx_build::bridge("src/lib.rs")
        .file("src/ffi.cpp")
        .std("c++20")
        .compile("tgi_trtllm_backend");

    println!("cargo:rerun-if-changed=include/backend.h");
    println!("cargo:rerun-if-changed=lib/backend.cpp");
    // println!("cargo:rustc-link-lib=tgi_trtllm_backend_impl");
}
