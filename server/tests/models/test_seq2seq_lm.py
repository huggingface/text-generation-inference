import pytest
import torch

from copy import copy

from transformers import AutoTokenizer

from text_generation_server.pb import generate_pb2
from text_generation_server.models.seq2seq_lm import Seq2SeqLM, Seq2SeqLMBatch


@pytest.fixture(scope="session")
def mt0_small_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "bigscience/mt0-small", padding_side="left"
    )
    tokenizer.bos_token_id = 0
    return tokenizer


@pytest.fixture(scope="session")
def default_seq2seq_lm():
    return Seq2SeqLM("bigscience/mt0-small")


@pytest.fixture
def default_pb_request(default_pb_parameters, default_pb_stop_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="Test",
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )


@pytest.fixture
def default_pb_batch(default_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_pb_request], size=1)


@pytest.fixture
def default_seq2seq_lm_batch(default_pb_batch, mt0_small_tokenizer):
    return Seq2SeqLMBatch.from_pb(
        default_pb_batch, mt0_small_tokenizer, torch.device("cpu")
    )


@pytest.fixture
def default_multi_requests_seq2seq_lm_batch(default_pb_request, mt0_small_tokenizer):
    req_0 = copy(default_pb_request)
    req_1 = default_pb_request
    req_1.id = 1
    req_1.stopping_parameters.max_new_tokens = 5

    batch_pb = generate_pb2.Batch(id=0, requests=[req_0, req_1], size=2)
    return Seq2SeqLMBatch.from_pb(batch_pb, mt0_small_tokenizer, torch.device("cpu"))


def test_batch_from_pb(default_pb_batch, default_seq2seq_lm_batch):
    batch = default_seq2seq_lm_batch
    sequence_length = len(default_seq2seq_lm_batch.input_ids[0])

    assert batch.batch_id == default_pb_batch.id
    assert batch.requests == default_pb_batch.requests

    assert batch.input_ids.shape == (default_pb_batch.size, sequence_length)
    assert batch.input_ids[0][-2] == 4268
    assert batch.input_ids[0][-1] == 1
    assert torch.all(batch.input_ids[0][:-2] == 0)

    assert torch.all(batch.attention_mask[0][-2:] == 1)
    assert torch.all(batch.attention_mask[0][:-2] == 0)

    assert batch.decoder_input_ids.shape == (default_pb_batch.size, 1)
    assert batch.decoder_attention_mask is None
    assert batch.encoder_last_hidden_state is None

    assert batch.past_key_values is None

    assert batch.input_lengths == [2]
    assert batch.decoder_input_lengths == [1]

    assert batch.size == default_pb_batch.size
    assert len(batch.next_token_choosers) == len(batch.stopping_criterias) == batch.size

    assert batch.max_input_length == batch.input_lengths[0]
    assert batch.max_decoder_input_length == batch.decoder_input_lengths[0]


def test_batch_concatenate_no_prefill(default_seq2seq_lm_batch):
    with pytest.raises(ValueError):
        Seq2SeqLMBatch.concatenate([default_seq2seq_lm_batch, default_seq2seq_lm_batch])


def test_seq2seq_lm_batch_type(default_seq2seq_lm):
    assert default_seq2seq_lm.batch_type == Seq2SeqLMBatch


def test_seq2seq_lm_generate_token(default_seq2seq_lm, default_seq2seq_lm_batch):
    sequence_length = len(default_seq2seq_lm_batch.input_ids[0])
    generations, next_batch = default_seq2seq_lm.generate_token(
        default_seq2seq_lm_batch
    )

    assert len(generations) == len(next_batch)
    assert isinstance(next_batch, Seq2SeqLMBatch)

    assert next_batch.input_ids is None
    assert torch.equal(
        next_batch.attention_mask, default_seq2seq_lm_batch.attention_mask
    )
    assert next_batch.input_lengths == default_seq2seq_lm_batch.input_lengths
    assert next_batch.max_input_length == default_seq2seq_lm_batch.max_input_length
    assert (
        next_batch.next_token_choosers == default_seq2seq_lm_batch.next_token_choosers
    )
    assert next_batch.stopping_criterias == default_seq2seq_lm_batch.stopping_criterias

    assert next_batch.decoder_input_ids.shape == (next_batch.size, 2)
    assert next_batch.decoder_input_ids[0, 0] == 0
    assert next_batch.decoder_input_ids[0, 1] == 259
    assert next_batch.decoder_attention_mask is None
    assert next_batch.encoder_last_hidden_state.shape == (1, sequence_length, 512)

    assert next_batch.decoder_input_lengths == [2]
    assert next_batch.max_decoder_input_length == 2

    assert next_batch.past_key_values is not None
    assert all(
        [p[0].shape == (next_batch.size, 6, 1, 64) for p in next_batch.past_key_values]
    )
    assert all(
        [p[1].shape == (next_batch.size, 6, 1, 64) for p in next_batch.past_key_values]
    )
    assert all(
        [
            p[2].shape == (next_batch.size, 6, sequence_length, 64)
            for p in next_batch.past_key_values
        ]
    )
    assert all(
        [
            p[3].shape == (next_batch.size, 6, sequence_length, 64)
            for p in next_batch.past_key_values
        ]
    )
    assert all([generation.generated_text is None for generation in generations])
    assert all([len(generation.prefill_tokens) == 1 for generation in generations])
    assert all([generation.token_id.item() == 259 for generation in generations])
    assert all([generation.token_text == " " for generation in generations])
    assert generations[0].request_id == 0


def test_seq2seq_lm_generate_token_completion(
    default_seq2seq_lm, default_seq2seq_lm_batch
):
    next_batch = default_seq2seq_lm_batch
    for _ in range(6):
        generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
        assert len(generations) == len(next_batch)

    generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
    assert next_batch is None

    assert len(generations) == 1
    assert generations[0].generated_text.text == "a few weeks"
    assert generations[0].request_id == default_seq2seq_lm_batch.requests[0].id
    assert generations[0].generated_text.generated_tokens == 7


def test_seq2seq_lm_generate_token_completion_multi(
    default_seq2seq_lm, default_multi_requests_seq2seq_lm_batch
):
    next_batch = default_multi_requests_seq2seq_lm_batch

    for i in range(4):
        generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
        assert len(generations) == len(next_batch)

    generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
    assert next_batch is not None

    assert len(generations) == 2
    assert generations[1].generated_text.text == "a few "
    assert (
        generations[1].request_id
        == default_multi_requests_seq2seq_lm_batch.requests[1].id
    )
    assert generations[1].generated_text.generated_tokens == 5

    generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
    assert len(generations) == len(next_batch)

    generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
    assert next_batch is None

    assert len(generations) == 1
    assert generations[0].generated_text.text == "a few weeks"
    assert (
        generations[0].request_id
        == default_multi_requests_seq2seq_lm_batch.requests[0].id
    )
    assert generations[0].generated_text.generated_tokens == 7


def test_batch_concatenate(
    default_seq2seq_lm,
    default_seq2seq_lm_batch,
    default_multi_requests_seq2seq_lm_batch,
):
    next_batch_0 = default_seq2seq_lm_batch
    _, next_batch_0 = default_seq2seq_lm.generate_token(next_batch_0)
    _, next_batch_0 = default_seq2seq_lm.generate_token(next_batch_0)

    next_batch_1 = default_multi_requests_seq2seq_lm_batch
    _, next_batch_1 = default_seq2seq_lm.generate_token(next_batch_1)

    next_batch = Seq2SeqLMBatch.concatenate([next_batch_0, next_batch_1])

    assert next_batch.batch_id == 0

    assert torch.equal(
        next_batch.decoder_input_ids[0], next_batch_0.decoder_input_ids[0]
    )
    assert torch.all(next_batch.decoder_input_ids[1:, 0] == 0)
    assert torch.equal(
        next_batch.decoder_input_ids[1:, -2:], next_batch_1.decoder_input_ids
    )

    assert torch.all(next_batch.decoder_attention_mask[0, :3] == 1)
    assert torch.all(next_batch.decoder_attention_mask[0, 3:] == 0)
    assert torch.all(next_batch.decoder_attention_mask[1:, 0] == 0)
    assert torch.all(next_batch.decoder_attention_mask[1:, 1:3] == 1)

    assert torch.equal(
        next_batch.encoder_last_hidden_state[0],
        next_batch_0.encoder_last_hidden_state[0, -2:],
    )
    assert torch.equal(
        next_batch.encoder_last_hidden_state[1:],
        next_batch_1.encoder_last_hidden_state[:, -2:],
    )

    assert next_batch.input_lengths == [2, 2, 2]
    assert next_batch.decoder_input_lengths == [3, 2, 2]
    assert next_batch.max_input_length == 2
    assert next_batch.max_decoder_input_length == 3

    assert next_batch.requests[0] == next_batch_0.requests[0]
    assert next_batch.requests[1:] == next_batch_1.requests

    assert next_batch.next_token_choosers[0] == next_batch_0.next_token_choosers[0]
    assert next_batch.next_token_choosers[1:] == next_batch_1.next_token_choosers

    assert next_batch.stopping_criterias[0] == next_batch_0.stopping_criterias[0]
    assert next_batch.stopping_criterias[1:] == next_batch_1.stopping_criterias

    assert next_batch.past_key_values is not None
    assert all(
        [p[0].shape == (next_batch.size, 6, 2, 64) for p in next_batch.past_key_values]
    )
    assert all(
        [p[1].shape == (next_batch.size, 6, 2, 64) for p in next_batch.past_key_values]
    )
    assert all(
        [p[2].shape == (next_batch.size, 6, 2, 64) for p in next_batch.past_key_values]
    )
    assert all(
        [p[3].shape == (next_batch.size, 6, 2, 64) for p in next_batch.past_key_values]
    )

    for i, past in enumerate(next_batch.past_key_values):
        assert torch.equal(next_batch_0.past_key_values[i][0][0, :, -2:, :], past[0][0])
        assert torch.equal(
            next_batch_1.past_key_values[i][0][:, :, -1:, :], past[0][1:, :, -1:, :]
        )

        assert torch.equal(next_batch_0.past_key_values[i][1][0, :, -2:, :], past[1][0])
        assert torch.equal(
            next_batch_1.past_key_values[i][1][:, :, -1:, :], past[1][1:, :, -1:, :]
        )

        assert torch.equal(next_batch_0.past_key_values[i][2][0, :, -2:, :], past[2][0])
        assert torch.equal(
            next_batch_1.past_key_values[i][2][:, :, -2:, :], past[2][1:]
        )

        assert torch.equal(next_batch_0.past_key_values[i][3][0, :, -2:, :], past[3][0])
        assert torch.equal(
            next_batch_1.past_key_values[i][3][:, :, -2:, :], past[3][1:]
        )

    for _ in range(3):
        generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
        assert len(generations) == len(next_batch)

    generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
    assert next_batch is not None

    assert len(generations) == 3
    assert generations[2].generated_text.text == "a few "
    assert (
        generations[2].request_id
        == default_multi_requests_seq2seq_lm_batch.requests[1].id
    )
    assert generations[2].generated_text.generated_tokens == 5

    generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
    assert next_batch is not None

    assert len(generations) == 2
    assert generations[0].generated_text.text == "a few weeks"
    assert generations[0].request_id == default_seq2seq_lm_batch.requests[0].id
    assert generations[0].generated_text.generated_tokens == 7

    generations, next_batch = default_seq2seq_lm.generate_token(next_batch)
    assert next_batch is None

    assert len(generations) == 1
    assert generations[0].generated_text.text == "a few weeks"
    assert (
        generations[0].request_id
        == default_multi_requests_seq2seq_lm_batch.requests[0].id
    )
    assert generations[0].generated_text.generated_tokens == 7
