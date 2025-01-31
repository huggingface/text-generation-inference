use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

fn inject_transient_dependencies(lib_search_path: Option<&str>, lib_target_hardware: &str) {
    let hardware_targets = HashMap::from([("cpu", None), ("cuda", Some(vec!["cuda"]))]);

    if let Some(lib_search_path) = lib_search_path {
        lib_search_path.split(":").for_each(|path| {
            println!("cargo:rustc-link-search=dependency={path}");
        });
    }

    if let Some(hardware_transient_deps) = hardware_targets.get(lib_target_hardware) {
        if let Some(additional_transient_deps) = hardware_transient_deps {
            additional_transient_deps.iter().for_each(|dep| {
                println!("cargo:rustc-link-lib={dep}");
            });
        }
    }
}

fn main() {
    let lib_search_path = option_env!("TGI_LLAMA_LD_LIBRARY_PATH");
    let lib_target_hardware = option_env!("TGI_LLAMA_HARDWARE_TARGET").unwrap_or("cpu");

    let bindings = bindgen::Builder::default()
        .header("src/wrapper.h")
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    pkg_config::Config::new().probe("llama").unwrap();

    inject_transient_dependencies(lib_search_path, lib_target_hardware);
}
