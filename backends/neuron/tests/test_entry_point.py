import os
import pytest
from tempfile import TemporaryDirectory

from optimum.neuron.models.inference.nxd.backend.config import NxDNeuronConfig
from optimum.neuron.utils import map_torch_dtype

from text_generation_server.tgi_env import (
    get_neuron_config_for_model,
    lookup_compatible_cached_model,
    neuron_config_to_env,
)


def test_get_neuron_config_for_model(neuron_model_config):
    neuron_model_path = neuron_model_config["neuron_model_path"]
    export_kwargs = neuron_model_config["export_kwargs"]
    os.environ["MAX_BATCH_SIZE"] = str(export_kwargs["batch_size"])
    os.environ["MAX_TOTAL_TOKENS"] = str(export_kwargs["sequence_length"])
    os.environ["HF_AUTO_CAST_TYPE"] = export_kwargs["auto_cast_type"]
    os.environ["HF_NUM_CORES"] = str(export_kwargs["num_cores"])
    neuron_config = get_neuron_config_for_model(neuron_model_path)
    assert neuron_config is not None
    assert neuron_config.batch_size == export_kwargs["batch_size"]
    assert neuron_config.sequence_length == export_kwargs["sequence_length"]
    assert neuron_config.tp_degree == export_kwargs["num_cores"]
    if isinstance(neuron_config, NxDNeuronConfig):
        assert map_torch_dtype(neuron_config.torch_dtype) == map_torch_dtype(
            export_kwargs["auto_cast_type"]
        )
    else:
        assert map_torch_dtype(neuron_config.auto_cast_type) == map_torch_dtype(
            export_kwargs["auto_cast_type"]
        )


@pytest.mark.parametrize("model_id", ["unsloth/Llama-3.2-1B-Instruct"])
def test_lookup_compatible_cached_model(model_id: str):
    neuron_config = lookup_compatible_cached_model(model_id, None)
    assert neuron_config is not None


def test_neuron_config_to_env(neuron_model_config) -> None:
    neuron_model_path = neuron_model_config["neuron_model_path"]
    neuron_config = get_neuron_config_for_model(neuron_model_path)
    with TemporaryDirectory() as temp_dir:
        os.environ["ENV_FILEPATH"] = os.path.join(temp_dir, "env.sh")
        neuron_config_to_env(neuron_config)
        with open(os.environ["ENV_FILEPATH"], "r") as env_file:
            env_content = env_file.read()
            assert f"export MAX_BATCH_SIZE={neuron_config.batch_size}" in env_content
            assert (
                f"export MAX_TOTAL_TOKENS={neuron_config.sequence_length}"
                in env_content
            )
            assert f"export HF_NUM_CORES={neuron_config.tp_degree}" in env_content
            if hasattr(neuron_config, "torch_dtype"):
                auto_cast_type = str(map_torch_dtype(neuron_config.torch_dtype)).split(
                    "."
                )[-1]
            else:
                auto_cast_type = neuron_config.auto_cast_type
            assert f"export HF_AUTO_CAST_TYPE={auto_cast_type}" in env_content
