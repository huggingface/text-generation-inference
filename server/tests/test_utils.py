import pytest

from huggingface_hub.utils import RevisionNotFoundError

from text_generation.utils import (
    weight_hub_files,
    download_weights,
    weight_files,
    StopSequenceCriteria,
    StoppingCriteria,
    LocalEntryNotFoundError,
)


def test_stop_sequence_criteria():
    criteria = StopSequenceCriteria("/test;")

    assert not criteria("/")
    assert not criteria("/test")
    assert criteria("/test;")
    assert not criteria("/test; ")


def test_stopping_criteria():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(65827, "/test") == (False, None)
    assert criteria(30, ";") == (True, "stop_sequence")


def test_stopping_criteria_eos():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(1, "") == (False, None)
    assert criteria(0, "") == (True, "eos_token")


def test_stopping_criteria_max():
    criteria = StoppingCriteria(0, [StopSequenceCriteria("/test;")], max_new_tokens=5)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (False, None)
    assert criteria(1, "") == (True, "length")


def test_weight_hub_files():
    filenames = weight_hub_files("bigscience/bloom-560m")
    assert filenames == ["model.safetensors"]


def test_weight_hub_files_llm():
    filenames = weight_hub_files("bigscience/bloom")
    assert filenames == [f"model_{i:05d}-of-00072.safetensors" for i in range(1, 73)]


def test_weight_hub_files_empty():
    filenames = weight_hub_files("bigscience/bloom", extension=".errors")
    assert filenames == []


def test_download_weights():
    files = download_weights("bigscience/bloom-560m")
    local_files = weight_files("bigscience/bloom-560m")
    assert files == local_files


def test_weight_files_error():
    with pytest.raises(RevisionNotFoundError):
        weight_files("bigscience/bloom-560m", revision="error")
    with pytest.raises(LocalEntryNotFoundError):
        weight_files("bert-base-uncased")
