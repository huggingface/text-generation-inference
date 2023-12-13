import os
import requests
import tempfile

import pytest

import huggingface_hub.constants
from huggingface_hub import hf_api

import text_generation_server.utils.hub
from text_generation_server.utils.hub import (
    weight_hub_files,
    download_weights,
    weight_files,
    EntryNotFoundError,
    LocalEntryNotFoundError,
    RevisionNotFoundError,
)


@pytest.fixture()
def offline():
    current_value = text_generation_server.utils.hub.HF_HUB_OFFLINE
    text_generation_server.utils.hub.HF_HUB_OFFLINE = True
    yield "offline"
    text_generation_server.utils.hub.HF_HUB_OFFLINE = current_value


@pytest.fixture()
def fresh_cache():
    with tempfile.TemporaryDirectory() as d:
        current_value = huggingface_hub.constants.HUGGINGFACE_HUB_CACHE
        huggingface_hub.constants.HUGGINGFACE_HUB_CACHE = d
        text_generation_server.utils.hub.HUGGINGFACE_HUB_CACHE = d
        os.environ['HUGGINGFACE_HUB_CACHE'] = d
        yield
        huggingface_hub.constants.HUGGINGFACE_HUB_CACHE = current_value
        os.environ['HUGGINGFACE_HUB_CACHE'] = current_value
        text_generation_server.utils.hub.HUGGINGFACE_HUB_CACHE = current_value


@pytest.fixture()
def prefetched():
    model_id = "bert-base-uncased"
    huggingface_hub.snapshot_download(
        repo_id=model_id,
        revision="main",
        local_files_only=False,
        repo_type="model",
        allow_patterns=["*.safetensors"]
    )
    yield model_id


def test_weight_hub_files_offline_error(offline, fresh_cache):
    # If the model is not prefetched then it will raise an error
    with pytest.raises(EntryNotFoundError):
        weight_hub_files("gpt2")


def test_weight_hub_files_offline_ok(prefetched, offline):
    # If the model is prefetched then we should be able to get the weight files from local cache
    filenames = weight_hub_files(prefetched)
    assert filenames == ['model.safetensors']


def test_weight_hub_files():
    filenames = weight_hub_files("bigscience/bloom-560m")
    assert filenames == ["model.safetensors"]


def test_weight_hub_files_llm():
    filenames = weight_hub_files("bigscience/bloom")
    assert filenames == [f"model_{i:05d}-of-00072.safetensors" for i in range(1, 73)]


def test_weight_hub_files_empty():
    with pytest.raises(EntryNotFoundError):
        weight_hub_files("bigscience/bloom", extension=".errors")


def test_download_weights():
    model_id = "bigscience/bloom-560m"
    filenames = weight_hub_files(model_id)
    files = download_weights(filenames, model_id)
    local_files = weight_files("bigscience/bloom-560m")
    assert files == local_files


def test_weight_files_revision_error():
    with pytest.raises(RevisionNotFoundError):
        weight_files("bigscience/bloom-560m", revision="error")


def test_weight_files_not_cached_error(fresh_cache):
    with pytest.raises(LocalEntryNotFoundError):
        weight_files("bert-base-uncased")
