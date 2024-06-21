import huggingface_hub
import argparse
import shutil
import time

REQUIRED_MODELS = {
    "bigscience/bloom-560m": "main",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "main",
    "abhinavkulkarni/codellama-CodeLlama-7b-Python-hf-w4-g128-awq": "main",
    "tiiuae/falcon-7b": "main",
    "TechxGenus/gemma-2b-GPTQ": "main",
    "google/gemma-2b": "main",
    "openai-community/gpt2": "main",
    "turboderp/Llama-3-8B-Instruct-exl2": "2.5bpw",
    "huggingface/llama-7b-gptq": "main",
    "neuralmagic/llama-2-7b-chat-marlin": "main",
    "huggingface/llama-7b": "main",
    "FasterDecoding/medusa-vicuna-7b-v1.3": "refs/pr/1",
    "mistralai/Mistral-7B-Instruct-v0.1": "main",
    "OpenAssistant/oasst-sft-1-pythia-12b": "main",
    "stabilityai/stablelm-tuned-alpha-3b": "main",
    "google/paligemma-3b-pt-224": "main",
    "microsoft/phi-2": "main",
    "Qwen/Qwen1.5-0.5B": "main",
    "bigcode/starcoder": "main",
    "Narsil/starcoder-gptq": "main",
    "bigcode/starcoder2-3b": "main",
    "HuggingFaceM4/idefics-9b-instruct": "main",
    "HuggingFaceM4/idefics2-8b": "main",
    "llava-hf/llava-v1.6-mistral-7b-hf": "main",
    "state-spaces/mamba-130m": "main",
    "mosaicml/mpt-7b": "main",
    "bigscience/mt0-base": "main",
    "google/flan-t5-xxl": "main",
}


def cleanup_cache(token: str):
    # Retrieve the size per model for all models used in the CI.
    size_per_model = {}
    extension_per_model = {}
    for model_id, revision in REQUIRED_MODELS.items():
        model_size = 0
        all_files = huggingface_hub.list_repo_files(
            model_id,
            repo_type="model",
            revision=revision,
            token=token,
        )

        extension = None
        if any(".safetensors" in filename for filename in all_files):
            extension = ".safetensors"
        elif any(".pt" in filename for filename in all_files):
            extension = ".pt"
        elif any(".bin" in filename for filename in all_files):
            extension = ".bin"

        extension_per_model[model_id] = extension

        for filename in all_files:
            if filename.endswith(extension):
                file_url = huggingface_hub.hf_hub_url(
                    model_id, filename, revision=revision
                )
                file_metadata = huggingface_hub.get_hf_file_metadata(
                    file_url, token=token
                )
                model_size += file_metadata.size * 1e-9  # in GB

        size_per_model[model_id] = model_size

    total_required_size = sum(size_per_model.values())
    print(f"Total required disk: {total_required_size:.2f} GB")

    cached_dir = huggingface_hub.scan_cache_dir()

    cache_size_per_model = {}
    cached_required_size_per_model = {}
    cached_shas_per_model = {}

    # Retrieve the SHAs and model ids of other non-necessary models in the cache.
    for repo in cached_dir.repos:
        if repo.repo_id in REQUIRED_MODELS:
            cached_required_size_per_model[repo.repo_id] = (
                repo.size_on_disk * 1e-9
            )  # in GB
        elif repo.repo_type == "model":
            cache_size_per_model[repo.repo_id] = repo.size_on_disk * 1e-9  # in GB

            shas = []
            for rev in repo.revisions:
                shas.append(rev.commit_hash)
            cached_shas_per_model[repo.repo_id] = shas

    total_required_cached_size = sum(cached_required_size_per_model.values())
    total_other_cached_size = sum(cache_size_per_model.values())
    total_non_cached_required_size = total_required_size - total_required_cached_size

    print(
        f"Total HF cached models size: {total_other_cached_size + total_required_cached_size:.2f} GB"
    )
    print(
        f"Total non-necessary HF cached models size: {total_other_cached_size:.2f} GB"
    )

    free_memory = shutil.disk_usage("/data").free * 1e-9
    print(f"Free memory: {free_memory:.2f} GB")

    if free_memory + total_other_cached_size < total_non_cached_required_size * 1.05:
        raise ValueError(
            "Not enough space on device to execute the complete CI, please clean up the CI machine"
        )

    while free_memory < total_non_cached_required_size * 1.05:
        if len(cache_size_per_model) == 0:
            raise ValueError("This should not happen.")

        largest_model_id = max(cache_size_per_model, key=cache_size_per_model.get)

        print("Removing", largest_model_id)
        for sha in cached_shas_per_model[largest_model_id]:
            huggingface_hub.scan_cache_dir().delete_revisions(sha).execute()

        del cache_size_per_model[largest_model_id]

        free_memory = shutil.disk_usage("/data").free * 1e-9

    return extension_per_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache cleaner")
    parser.add_argument(
        "--token", help="Hugging Face Hub token.", required=True, type=str
    )
    args = parser.parse_args()

    start = time.time()
    extension_per_model = cleanup_cache(args.token)
    end = time.time()

    print(f"Cache cleanup done in {end - start:.2f} s")

    print("Downloading required models")
    start = time.time()
    for model_id, revision in REQUIRED_MODELS.items():
        print(f"Downloading {model_id}'s *{extension_per_model[model_id]}...")
        huggingface_hub.snapshot_download(
            model_id,
            repo_type="model",
            revision=revision,
            token=args.token,
            allow_patterns=f"*{extension_per_model[model_id]}",
        )
    end = time.time()

    print(f"Models download done in {end - start:.2f} s")
