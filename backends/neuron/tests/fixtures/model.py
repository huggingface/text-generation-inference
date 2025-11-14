import copy
import logging
import subprocess
import sys
from tempfile import TemporaryDirectory

import os
import pytest
from transformers import AutoTokenizer


from optimum.neuron.cache import synchronize_hub_cache


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


OPTIMUM_CACHE_REPO_ID = "optimum-internal-testing/neuron-testing-cache"


# All model configurations below will be added to the neuron_model_config fixture
MODEL_CONFIGURATIONS = {
    "llama": {
        "model_id": "unsloth/Llama-3.2-1B-Instruct",
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
            "auto_cast_type": "bf16",
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


def export_model(model_id, export_kwargs, neuron_model_path):
    export_command = [
        "optimum-cli",
        "export",
        "neuron",
        "-m",
        model_id,
        "--task",
        "text-generation",
    ]
    for kwarg, value in export_kwargs.items():
        export_command.append(f"--{kwarg}")
        export_command.append(str(value))
    export_command.append(neuron_model_path)
    logger.info(f"Exporting {model_id} with {export_kwargs}")
    try:
        subprocess.run(export_command, check=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to export model: {e}")


@pytest.fixture(scope="session", params=MODEL_CONFIGURATIONS.keys())
def neuron_model_config(request):
    """Expose a pre-trained neuron model

    The fixture exports a model locally and returns a dictionary containing:
    - a configuration name,
    - the original model id,
    - the export parameters,
    - the neuron model local path.

    For each exposed model, the local directory is maintained for the duration of the
    test session and cleaned up afterwards.

    """
    config_name = request.param
    model_config = copy.deepcopy(MODEL_CONFIGURATIONS[request.param])
    model_id = model_config["model_id"]
    export_kwargs = model_config["export_kwargs"]
    with TemporaryDirectory() as neuron_model_path:
        export_model(model_id, export_kwargs, neuron_model_path)
        synchronize_hub_cache(cache_repo_id=OPTIMUM_CACHE_REPO_ID)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(neuron_model_path)
        del tokenizer
        # Add dynamic parameters to the model configuration
        model_config["neuron_model_path"] = neuron_model_path
        # Also add model configuration name to allow tests to adapt their expectations
        model_config["name"] = config_name
        # Yield instead of returning to keep a reference to the temporary directory.
        # It will go out of scope and be released only once all tests needing the fixture
        # have been completed.
        logger.info(f"{config_name} ready for testing ...")
        os.environ["CUSTOM_CACHE_REPO"] = OPTIMUM_CACHE_REPO_ID
        yield model_config
        logger.info(f"Done with {config_name}")


@pytest.fixture(scope="module")
def neuron_model_path(neuron_model_config):
    yield neuron_model_config["neuron_model_path"]
