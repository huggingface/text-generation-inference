import copy
import logging
import subprocess
import sys
from tempfile import TemporaryDirectory

import huggingface_hub
import pytest
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils import synchronize_hub_cache
from optimum.neuron.version import __sdk_version__ as sdk_version
from optimum.neuron.version import __version__ as version


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)

OPTIMUM_CACHE_REPO_ID = "optimum-internal-testing/neuron-testing-cache"

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
        "model_id": "NousResearch/Hermes-2-Theta-Llama-3-8B",
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


def get_hub_neuron_model_id(config_name: str):
    return (
        f"optimum-internal-testing/neuron-testing-{version}-{sdk_version}-{config_name}"
    )


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

    The fixture first makes sure the following model artifacts are present on the hub:
    - exported neuron model under optimum-internal-testing/neuron-testing-<version>-<name>,
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
    model_id = model_config["model_id"]
    export_kwargs = model_config["export_kwargs"]
    neuron_model_id = get_hub_neuron_model_id(config_name)
    with TemporaryDirectory() as neuron_model_path:
        hub = huggingface_hub.HfApi()
        if hub.repo_exists(neuron_model_id):
            logger.info(f"Fetching {neuron_model_id} from the HuggingFace hub")
            hub.snapshot_download(neuron_model_id, local_dir=neuron_model_path)
        else:
            export_model(model_id, export_kwargs, neuron_model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(neuron_model_path)
            del tokenizer
            # Create the test model on the hub
            hub.create_repo(neuron_model_id, private=True)
            hub.upload_folder(
                folder_path=neuron_model_path,
                repo_id=neuron_model_id,
                ignore_patterns=[NeuronModelForCausalLM.CHECKPOINT_DIR + "/*"],
            )
            # Make sure it is cached
            synchronize_hub_cache(cache_repo_id=OPTIMUM_CACHE_REPO_ID)
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
