use bindgen::callbacks::{ItemInfo, ParseCallbacks};
use std::env;
use std::path::PathBuf;

#[derive(Debug)]
struct PrefixStripper;

impl ParseCallbacks for PrefixStripper {
    fn generated_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        item_info.name.strip_prefix("llama_").map(str::to_string)
    }
}

fn main() {
    if let Some(cuda_version) = option_env!("CUDA_VERSION") {
        let mut version: Vec<&str> = cuda_version.split('.').collect();
        if version.len() > 2 {
            version.pop();
        }
        let cuda_version = format!("cuda-{}", version.join("."));
        pkg_config::Config::new().probe(&cuda_version).unwrap();
    }
    let llama = pkg_config::Config::new().probe("llama").unwrap();

    for path in &llama.link_paths {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    }
    let bindings = bindgen::Builder::default()
        .clang_args(
            llama
                .include_paths
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .header_contents("llama_bindings.h", "#include <llama.h>")
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(PrefixStripper))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("llamacpp.rs"))
        .expect("Couldn't write bindings!");
}
