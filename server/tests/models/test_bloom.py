import pytest
import torch

from copy import copy

from text_generation.pb import generate_pb2
from text_generation.models.causal_lm import CausalLMBatch
from text_generation.models.bloom import BloomCausalLMBatch, BLOOM


@pytest.fixture
def default_pb_request(default_pb_parameters, default_pb_stop_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="Test",
        input_length=1,
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )


@pytest.fixture
def default_pb_batch(default_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_pb_request], size=1)


@pytest.fixture
def default_bloom_batch(default_pb_batch, bloom_560m_tokenizer):
    return BloomCausalLMBatch.from_pb(
        default_pb_batch, bloom_560m_tokenizer, torch.device("cpu")
    )


@pytest.fixture
def default_multi_requests_bloom_batch(default_pb_request, bloom_560m_tokenizer):
    req_0 = copy(default_pb_request)
    req_1 = default_pb_request
    req_1.id = 1
    req_1.stopping_parameters.max_new_tokens = 5

    batch_pb = generate_pb2.Batch(id=0, requests=[req_0, req_1], size=2)
    return BloomCausalLMBatch.from_pb(
        batch_pb, bloom_560m_tokenizer, torch.device("cpu")
    )


@pytest.fixture(scope="session")
def default_bloom():
    return BLOOM("bigscience/bloom-560m")


def test_batch_from_pb(default_pb_batch, default_bloom_batch):
    batch = default_bloom_batch

    assert batch.batch_id == default_pb_batch.id
    assert batch.requests == default_pb_batch.requests

    assert len(batch.input_ids) == default_pb_batch.size
    assert batch.input_ids[0][-1] == 10264
    assert torch.all(batch.input_ids[0][:-1] == 3)

    assert batch.attention_mask[0][-1] == 1
    assert torch.all(batch.attention_mask[0][:-1] == 0)

    assert batch.past_key_values is None

    assert torch.equal(batch.input_ids, batch.all_input_ids[:, :, 0])

    assert batch.input_lengths == [1]

    assert batch.size == default_pb_batch.size
    assert len(batch.next_token_choosers) == len(batch.stopping_criterias) == batch.size

    assert batch.max_sequence_length == batch.input_lengths[0]


def test_batch_concatenate_no_prefill(default_bloom_batch):
    with pytest.raises(ValueError):
        BloomCausalLMBatch.concatenate([default_bloom_batch, default_bloom_batch])


def test_causal_lm_batch_type(default_bloom):
    assert default_bloom.batch_type == BloomCausalLMBatch


def test_causal_lm_generate_token(default_bloom, default_bloom_batch):
    sequence_length = len(default_bloom_batch.all_input_ids[0])
    generated_texts, next_batch = default_bloom.generate_token(default_bloom_batch)

    assert generated_texts == []
    assert isinstance(next_batch, CausalLMBatch)
    assert not next_batch.keys_head_dim_last

    assert len(next_batch.all_input_ids) == next_batch.size
    assert (
        len(next_batch.all_input_ids[0])
        == len(next_batch.attention_mask[0])
        == sequence_length + 1
    )
    assert torch.all(next_batch.all_input_ids[0][-2:] == 10264)
    assert torch.all(next_batch.all_input_ids[0][:-2] == 3)

    assert torch.all(next_batch.attention_mask[0][-2:] == 1)
    assert torch.all(next_batch.attention_mask[0][:-2] == 0)

    assert next_batch.input_ids.shape == (next_batch.size, 1)
    assert next_batch.input_ids[0, 0] == 10264

    assert next_batch.input_lengths == [2]
    assert next_batch.max_sequence_length == next_batch.input_lengths[0]

    assert next_batch.past_key_values is not None
    assert all(
        [p[0].shape == (16, 64, sequence_length) for p in next_batch.past_key_values]
    )
    assert all(
        [p[1].shape == (16, sequence_length, 64) for p in next_batch.past_key_values]
    )


def test_causal_lm_generate_token_completion(default_bloom, default_bloom_batch):
    next_batch = default_bloom_batch
    for _ in range(default_bloom_batch.stopping_criterias[0].max_new_tokens - 1):
        generated_texts, next_batch = default_bloom.generate_token(next_batch)
        assert generated_texts == []

    generated_texts, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is None

    assert len(generated_texts) == 1
    assert generated_texts[0].output == "TestTestTestTestTestTestTestTestTestTestTest"
    assert generated_texts[0].request == default_bloom_batch.requests[0]
    assert (
        generated_texts[0].tokens
        == default_bloom_batch.stopping_criterias[0].max_new_tokens
    )


def test_causal_lm_generate_token_completion_multi(
    default_bloom, default_multi_requests_bloom_batch
):
    next_batch = default_multi_requests_bloom_batch

    for i in range(
        default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens - 1
    ):
        generated_texts, next_batch = default_bloom.generate_token(next_batch)
        assert generated_texts == []

    generated_texts, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is not None

    assert len(generated_texts) == 1
    assert generated_texts[0].output == "TestTestTestTestTestTest"
    assert generated_texts[0].request == default_multi_requests_bloom_batch.requests[1]
    assert (
        generated_texts[0].tokens
        == default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens
    )

    for _ in range(
        default_multi_requests_bloom_batch.stopping_criterias[0].max_new_tokens
        - default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens
        - 1
    ):
        generated_texts, next_batch = default_bloom.generate_token(next_batch)
        assert generated_texts == []

    generated_texts, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is None

    assert len(generated_texts) == 1
    assert generated_texts[0].output == "TestTestTestTestTestTestTestTestTestTestTest"
    assert generated_texts[0].request == default_multi_requests_bloom_batch.requests[0]
    assert (
        generated_texts[0].tokens
        == default_multi_requests_bloom_batch.stopping_criterias[0].max_new_tokens
    )


def test_batch_concatenate(
    default_bloom, default_bloom_batch, default_multi_requests_bloom_batch
):
    next_batch_0 = default_bloom_batch
    _, next_batch_0 = default_bloom.generate_token(next_batch_0)
    _, next_batch_0 = default_bloom.generate_token(next_batch_0)

    next_batch_1 = default_multi_requests_bloom_batch
    _, next_batch_1 = default_bloom.generate_token(next_batch_1)

    next_batch = BloomCausalLMBatch.concatenate([next_batch_0, next_batch_1])

    assert torch.equal(next_batch.all_input_ids[0], next_batch_0.all_input_ids[0])
    assert torch.equal(next_batch.all_input_ids[1], next_batch_1.all_input_ids[0])
    assert torch.equal(next_batch.all_input_ids[2], next_batch_1.all_input_ids[1])

    assert torch.all(next_batch.attention_mask[0] == 1)
    assert torch.all(next_batch.attention_mask[1:, -2:] == 1)
    assert torch.all(next_batch.attention_mask[1:, :-2] == 0)

    assert next_batch.batch_id == 0
    assert torch.all(next_batch.input_ids == 10264)

    assert next_batch.input_lengths == [3, 2, 2]
    assert next_batch.max_sequence_length == 3

    assert next_batch.requests[0] == next_batch_0.requests[0]
    assert next_batch.requests[1:] == next_batch_1.requests

    assert next_batch.next_token_choosers[0] == next_batch_0.next_token_choosers[0]
    assert next_batch.next_token_choosers[1:] == next_batch_1.next_token_choosers

    assert next_batch.stopping_criterias[0] == next_batch_0.stopping_criterias[0]
    assert next_batch.stopping_criterias[1:] == next_batch_1.stopping_criterias

    assert next_batch.past_key_values is not None
    assert all([p[0].shape == (3, 16, 64, 2) for p in next_batch.past_key_values])
    assert all([p[1].shape == (3, 16, 2, 64) for p in next_batch.past_key_values])

    for i, past in enumerate(next_batch.past_key_values):
        assert torch.equal(next_batch_0.past_key_values[i][0][:, :, -2:], past[0][0])
        assert torch.equal(
            next_batch_1.past_key_values[i][0][:, :, -1:],
            past[0][1:, :, :, -1].reshape(-1, 64, 1),
        )

        assert torch.equal(next_batch_0.past_key_values[i][1][:, -2:, :], past[1][0])
        assert torch.equal(
            next_batch_1.past_key_values[i][1][:, -1:, :],
            past[1][1:, :, -1, :].reshape(-1, 1, 64),
        )

    for _ in range(
        default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens - 2
    ):
        generated_texts, next_batch = default_bloom.generate_token(next_batch)
        assert generated_texts == []

    generated_texts, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is not None

    assert len(generated_texts) == 1
    assert generated_texts[0].output == "TestTestTestTestTestTest"
    assert generated_texts[0].request == default_multi_requests_bloom_batch.requests[1]
    assert (
        generated_texts[0].tokens
        == default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens
    )

    for _ in range(
        default_bloom_batch.stopping_criterias[0].max_new_tokens
        - default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens
        - 2
    ):
        generated_texts, next_batch = default_bloom.generate_token(next_batch)
        assert generated_texts == []

    generated_texts, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is not None

    assert len(generated_texts) == 1
    assert generated_texts[0].output == "TestTestTestTestTestTestTestTestTestTestTest"
    assert generated_texts[0].request == default_bloom_batch.requests[0]
    assert (
        generated_texts[0].tokens
        == default_bloom_batch.stopping_criterias[0].max_new_tokens
    )

    for _ in range(
        default_multi_requests_bloom_batch.stopping_criterias[0].max_new_tokens
        - default_bloom_batch.stopping_criterias[0].max_new_tokens
        - default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens
        - 4
    ):
        generated_texts, next_batch = default_bloom.generate_token(next_batch)
        assert generated_texts == []

    generated_texts, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is None

    assert len(generated_texts) == 1
    assert generated_texts[0].output == "TestTestTestTestTestTestTestTestTestTestTest"
    assert generated_texts[0].request == default_multi_requests_bloom_batch.requests[0]
    assert (
        generated_texts[0].tokens
        == default_multi_requests_bloom_batch.stopping_criterias[0].max_new_tokens
    )
