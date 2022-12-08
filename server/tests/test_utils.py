import pytest

from text_generation.utils import (
    weight_hub_files,
    download_weights,
    weight_files,
    LocalEntryNotFoundError,
)


def test_weight_hub_files():
    filenames = weight_hub_files("bigscience/bloom-560m")
    assert filenames == ["model.safetensors"]


def test_weight_hub_files_llm():
    filenames = weight_hub_files("bigscience/bloom")
    assert filenames == [f"model_{i:05d}-of-00072.safetensors" for i in range(1, 73)]


def test_weight_hub_files_empty():
    filenames = weight_hub_files("bigscience/bloom", ".errors")
    assert filenames == []


def test_download_weights():
    files = download_weights("bigscience/bloom-560m")
    local_files = weight_files("bigscience/bloom-560m")
    assert files == local_files


def test_weight_files_error():
    with pytest.raises(LocalEntryNotFoundError):
        weight_files("bert-base-uncased")
