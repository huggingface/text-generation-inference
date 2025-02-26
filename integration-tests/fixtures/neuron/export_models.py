import copy
import logging
import sys
from tempfile import TemporaryDirectory

import huggingface_hub
import pytest
import docker
import hashlib
import os
import tempfile

from docker.errors import NotFound


TEST_ORGANIZATION = "optimum-internal-testing"
TEST_CACHE_REPO_ID = f"{TEST_ORGANIZATION}/neuron-testing-cache"
HF_TOKEN = huggingface_hub.get_token()


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


# All model configurations below will be added to the neuron_model_config fixture
MODEL_CONFIGURATIONS = {
    "gpt2": {
        "model_id": "gpt2",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 1024,
            "num_cores": 2,
            "auto_cast_type": "fp16",
        },
    },
    "llama": {
        "model_id": "unsloth/Llama-3.2-1B-Instruct",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 2048,
            "num_cores": 2,
            "auto_cast_type": "fp16",
        },
    },
    "mistral": {
        "model_id": "optimum/mistral-1.1b-testing",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 4096,
            "num_cores": 2,
            "auto_cast_type": "bf16",
        },
    },
    "qwen2": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 4096,
            "num_cores": 2,
            "auto_cast_type": "fp16",
        },
    },
    "granite": {
        "model_id": "ibm-granite/granite-3.1-2b-instruct",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 4096,
            "num_cores": 2,
            "auto_cast_type": "bf16",
        },
    },
}


def get_neuron_backend_hash():
    import subprocess

    res = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    root_dir = res.stdout.split("\n")[0]

    def get_sha(path):
        res = subprocess.run(
            ["git", "ls-tree", "HEAD", f"{root_dir}/{path}"],
            capture_output=True,
            text=True,
        )
        # Output of the command is in the form '040000 tree|blob <SHA>\t<path>\n'
        sha = res.stdout.split("\t")[0].split(" ")[-1]
        return sha.encode()

    # We hash both the neuron backends directory and Dockerfile and create a smaller hash out of that
    m = hashlib.sha256()
    m.update(get_sha("backends/neuron"))
    m.update(get_sha("Dockerfile.neuron"))
    return m.hexdigest()[:10]


def get_neuron_model_name(config_name: str):
    return f"neuron-tgi-testing-{config_name}-{get_neuron_backend_hash()}"


def get_tgi_docker_image():
    docker_image = os.getenv("DOCKER_IMAGE", None)
    if docker_image is None:
        client = docker.from_env()
        images = client.images.list(filters={"reference": "text-generation-inference"})
        if not images:
            raise ValueError(
                "No text-generation-inference image found on this host to run tests."
            )
        docker_image = images[0].tags[0]
    return docker_image


def maybe_export_model(config_name, model_config):
    """Export a neuron model for the specified test configuration.

    If the neuron model has not already been compiled and pushed to the hub, it is
    exported by a custom image built on the fly from the base TGI image.
    This makes sure the exported model and image are aligned and avoids introducing
    neuron specific imports in the test suite.

    Args:
        config_name (`str`):
            Used to identify test configurations
        model_config (`str`):
            The model configuration for export (includes the original model id)
    """
    neuron_model_name = get_neuron_model_name(config_name)
    neuron_model_id = f"{TEST_ORGANIZATION}/{neuron_model_name}"
    hub = huggingface_hub.HfApi()
    if hub.repo_exists(neuron_model_id):
        logger.info(
            f"Skipping model export for config {config_name} as {neuron_model_id} already exists"
        )
        return neuron_model_id

    client = docker.from_env()

    env = {"LOG_LEVEL": "info", "CUSTOM_CACHE_REPO": TEST_CACHE_REPO_ID}
    if HF_TOKEN is not None:
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        env["HF_TOKEN"] = HF_TOKEN

    # Create a sub-image to export the model to workaround docker dind issues preventing
    # to share a volume from the container running tests
    model_id = model_config["model_id"]
    export_kwargs = model_config["export_kwargs"]
    base_image = get_tgi_docker_image()
    export_image = f"neuron-tgi-tests-{config_name}-export-img"
    logger.info(f"Building temporary image {export_image} from {base_image}")
    with tempfile.TemporaryDirectory() as context_dir:
        # Create entrypoint
        model_path = "/data/neuron_model"
        export_command = (
            f"optimum-cli export neuron -m {model_id} --task text-generation"
        )
        for kwarg, value in export_kwargs.items():
            export_command += f" --{kwarg} {str(value)}"
        export_command += f" {model_path}"
        entrypoint_content = f"""#!/bin/sh
        {export_command}
        huggingface-cli repo create --organization {TEST_ORGANIZATION} {neuron_model_name}
        huggingface-cli upload {TEST_ORGANIZATION}/{neuron_model_name} {model_path} --exclude *.bin *.safetensors
        optimum-cli neuron cache synchronize --repo_id {TEST_CACHE_REPO_ID}
        """
        with open(os.path.join(context_dir, "entrypoint.sh"), "wb") as f:
            f.write(entrypoint_content.encode("utf-8"))
            f.flush()
        # Create Dockerfile
        docker_content = f"""
        FROM {base_image}
        COPY entrypoint.sh /export-entrypoint.sh
        RUN chmod +x /export-entrypoint.sh
        ENTRYPOINT ["/export-entrypoint.sh"]
        """
        with open(os.path.join(context_dir, "Dockerfile"), "wb") as f:
            f.write(docker_content.encode("utf-8"))
            f.flush()
        image, logs = client.images.build(
            path=context_dir, dockerfile=f.name, tag=export_image
        )
        logger.info("Successfully built image %s", image.id)
        logger.debug("Build logs %s", logs)

    try:
        client.containers.run(
            export_image,
            environment=env,
            auto_remove=True,
            detach=False,
            devices=["/dev/neuron0"],
            shm_size="1G",
        )
        logger.info(f"Successfully exported model for config {config_name}")
    except Exception as e:
        logger.exception(f"An exception occurred while running container: {e}.")
        pass
    finally:
        # Cleanup the export image
        logger.info("Cleaning image %s", image.id)
        try:
            image.remove(force=True)
        except NotFound:
            pass
        except Exception as e:
            logger.error("Error while removing image %s, skipping", image.id)
            logger.exception(e)
    return neuron_model_id


def maybe_export_models():
    for config_name, model_config in MODEL_CONFIGURATIONS.items():
        maybe_export_model(config_name, model_config)


@pytest.fixture(scope="session", params=MODEL_CONFIGURATIONS.keys())
def neuron_model_config(request):
    """Expose a pre-trained neuron model

    The fixture first makes sure the following model artifacts are present on the hub:
    - exported neuron model under optimum-internal-testing/neuron-testing-<name>-<version>,
    - cached artifacts under optimum-internal-testing/neuron-testing-cache.
    If not, it will export the model and push it to the hub.

    It then fetches the model locally and return a dictionary containing:
    - a configuration name,
    - the original model id,
    - the export parameters,
    - the neuron model id,
    - the neuron model local path.

    For each exposed model, the local directory is maintained for the duration of the
    test session and cleaned up afterwards.
    The hub model artifacts are never cleaned up and persist accross sessions.
    They must be cleaned up manually when the optimum-neuron version changes.

    """
    config_name = request.param
    model_config = copy.deepcopy(MODEL_CONFIGURATIONS[request.param])
    # Export the model first (only if needed)
    neuron_model_id = maybe_export_model(config_name, model_config)
    with TemporaryDirectory() as neuron_model_path:
        logger.info(f"Fetching {neuron_model_id} from the HuggingFace hub")
        hub = huggingface_hub.HfApi()
        hub.snapshot_download(
            neuron_model_id, etag_timeout=30, local_dir=neuron_model_path
        )
        # Add dynamic parameters to the model configuration
        model_config["neuron_model_path"] = neuron_model_path
        model_config["neuron_model_id"] = neuron_model_id
        # Also add model configuration name to allow tests to adapt their expectations
        model_config["name"] = config_name
        # Yield instead of returning to keep a reference to the temporary directory.
        # It will go out of scope and be released only once all tests needing the fixture
        # have been completed.
        logger.info(f"{config_name} ready for testing ...")
        yield model_config
        logger.info(f"Done with {config_name}")


@pytest.fixture(scope="module")
def neuron_model_path(neuron_model_config):
    yield neuron_model_config["neuron_model_path"]


if __name__ == "__main__":
    maybe_export_models()
