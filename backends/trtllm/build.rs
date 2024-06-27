use std::env;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Error};
use git2::{Error as GitError, Repository, SubmoduleUpdateOptions};
use git2::build::CheckoutBuilder;

// const ENV_TGI_TENSORRT_LLM_VERSION: &str = "TGI_TRTLLM_VERSION";
const ENV_TGI_TENSORRT_LLM_VERSION_COMMIT: &str = "TGI_TRTLLM_VERSION_COMMIT";

const TENSORRT_LLM_REPOSITORY_URL: &str = "https://github.com/nvidia/tensorrt-llm";
const TENSORRT_LLM_REPOSITORY_COMMIT_HASH: &str = "9691e12bce7ae1c126c435a049eb516eb119486c";

macro_rules! log {
    ($fmt:expr) => (println!(concat!("trtllm-sys/build.rs:{}: ", $fmt), line!()));
    ($fmt:expr, $($arg:tt)*) => (println!(concat!("trtllm-sys/build.rs:{}: ", $fmt),
    line!(), $($arg)*));
}

fn get_trtllm_snapshot<P: AsRef<Path>>(
    origin: &str,
    version: &str,
    dest: P,
) -> Result<(), GitError> {
    let dest = dest.as_ref();

    let repo = if dest.join(".git").exists() {
        Repository::open(dest)
    } else {
        Repository::init(dest)
    }?;

    let mut remote = repo.remote_anonymous(origin)?;
    remote.fetch(&["main"], None, None)?;

    log!("Successfully fetched main on remote");

    let mut fetch_head = repo.find_reference("FETCH_HEAD")?;
    let commit = repo.find_commit_by_prefix(version)?;

    log!("Targeting commit {:?}", commit);

    let target = fetch_head.set_target(
        commit.id(),
        &format!("Fast-Forward to {TENSORRT_LLM_REPOSITORY_COMMIT_HASH}"),
    )?;

    repo.set_head(target.name().unwrap())?;
    repo.checkout_head(Some(CheckoutBuilder::default().force()))?;

    for mut submodule in repo.submodules()? {
        submodule.sync()?;
        submodule.update(true, Some(SubmoduleUpdateOptions::new().allow_fetch(true)))?
    }

    Ok(())
}

fn cmake_backend_build<P: AsRef<Path>>(trtllm_sources_dir: P, dest: P, profile: &str) {
    cmake::Config::new(PathBuf::from("csrc"))
        .profile(match profile {
            "release" => "Release",
            _ => "DebWithRelInfo",
        })
        .generator("Ninja")
        .define(
            "TGI_TRTLLM_BACKEND_TRTLLM_ROOT_DIR",
            trtllm_sources_dir.as_ref().as_os_str(),
        )
        .uses_cxx11()
        .build_target("tgi_trtllm_backend")
        .out_dir(dest)
        .build();
}

fn main() -> Result<(), Error> {
    env::set_var(
        ENV_TGI_TENSORRT_LLM_VERSION_COMMIT,
        TENSORRT_LLM_REPOSITORY_COMMIT_HASH,
    );

    let tensorrt_llm_sources_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("tensorrt-llm");
    let backend_out_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("tgi-trtllm-backend");
    let build_profile: String = env::var("PROFILE").unwrap();

    // First we need to retrieve TensorRT-LLM sources to build the library to link against
    get_trtllm_snapshot(
        TENSORRT_LLM_REPOSITORY_URL,
        TENSORRT_LLM_REPOSITORY_COMMIT_HASH,
        &tensorrt_llm_sources_dir,
    )
    .map_err(|e| anyhow!(e))?;

    cmake_backend_build(&tensorrt_llm_sources_dir, &backend_out_dir, &build_profile);

    Ok(())
}
