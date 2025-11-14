import os
import pytest

from text_generation_server.generator import NeuronGenerator
from text_generation_server.model import fetch_model, is_cached


@pytest.fixture(scope="module")
def cached_model_id(neuron_model_config) -> str:
    """
    Fixture to provide a cached model ID for testing.
    This assumes the model is already cached in the local environment.
    """
    export_kwargs = neuron_model_config["export_kwargs"]
    os.environ["MAX_BATCH_SIZE"] = str(export_kwargs["batch_size"])
    os.environ["MAX_TOTAL_TOKENS"] = str(export_kwargs["sequence_length"])
    os.environ["HF_AUTO_CAST_TYPE"] = export_kwargs["auto_cast_type"]
    os.environ["HF_NUM_CORES"] = str(export_kwargs["num_cores"])
    yield neuron_model_config["model_id"]
    os.environ.pop("MAX_BATCH_SIZE", None)
    os.environ.pop("MAX_TOTAL_TOKENS", None)
    os.environ.pop("HF_AUTO_CAST_TYPE", None)
    os.environ.pop("HF_NUM_CORES", None)


def test_model_is_cached(cached_model_id):
    assert is_cached(cached_model_id), f"Model {cached_model_id} is not cached"


def test_fetch_cached_model(cached_model_id: str):
    model_path = fetch_model(cached_model_id)
    assert os.path.exists(
        model_path
    ), f"Model {cached_model_id} was not fetched successfully"
    assert os.path.isdir(model_path), f"Model {cached_model_id} is not a directory"


def test_generator_from_cached_model(cached_model_id: str):
    generator = NeuronGenerator.from_pretrained(model_id=cached_model_id)
    assert generator is not None, "Generator could not be created from cached model"
    assert generator.model is not None, "Generator model is not initialized"
    assert generator.tokenizer is not None, "Generator tokenizer is not initialized"
