import pytest

from text_generation.utils import (
    weight_hub_files,
    download_weights,
    weight_files,
    StopSequenceCriteria,
    StoppingCriteria,
    LocalEntryNotFoundError,
)


def test_stop_sequence_criteria():
    criteria = StopSequenceCriteria([1, 2, 3])

    assert not criteria(1)
    assert criteria.current_token_idx == 1
    assert not criteria(2)
    assert criteria.current_token_idx == 2
    assert criteria(3)
    assert criteria.current_token_idx == 3


def test_stop_sequence_criteria_reset():
    criteria = StopSequenceCriteria([1, 2, 3])

    assert not criteria(1)
    assert criteria.current_token_idx == 1
    assert not criteria(2)
    assert criteria.current_token_idx == 2
    assert not criteria(4)
    assert criteria.current_token_idx == 0


def test_stop_sequence_criteria_empty():
    with pytest.raises(ValueError):
        StopSequenceCriteria([])


def test_stopping_criteria():
    criteria = StoppingCriteria([StopSequenceCriteria([1, 2, 3])], max_new_tokens=5)
    assert criteria([1]) == (False, None)
    assert criteria([1, 2]) == (False, None)
    assert criteria([1, 2, 3]) == (True, "stop_sequence")


def test_stopping_criteria_max():
    criteria = StoppingCriteria([StopSequenceCriteria([1, 2, 3])], max_new_tokens=5)
    assert criteria([1]) == (False, None)
    assert criteria([1, 1]) == (False, None)
    assert criteria([1, 1, 1]) == (False, None)
    assert criteria([1, 1, 1, 1]) == (False, None)
    assert criteria([1, 1, 1, 1, 1]) == (True, "length")


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
