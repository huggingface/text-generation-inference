# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import pytest
import torch

from copy import copy

from text_generation_server.pb import generate_pb2
from text_generation_server.models import get_model
from text_generation_server.models.starcoder import StarCoderCausalLMBatch
from text_generation_server.models.causal_lm import (
    PREFILL_BATCH_BUCKET_SIZE,
    PAD_SEQUENCE_TO_MULTIPLE_OF,
    MAX_TOTAL_TOKENS,
    BATCH_BUCKET_SIZE,
)
PAD_TOKEN=0


@pytest.fixture(scope="session")
def default_starcoder():
    return get_model("bigcode/starcoder", None, None, None, None)


@pytest.fixture(scope="session")
def default_tokenizer(default_starcoder):
    default_starcoder.tokenizer.pad_token_id = PAD_TOKEN
    return default_starcoder.tokenizer


@pytest.fixture
def default_pb_request(default_pb_parameters, default_pb_stop_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="Test",
        prefill_logprobs=True,
        truncate=PAD_SEQUENCE_TO_MULTIPLE_OF,
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )


@pytest.fixture
def default_pb_batch(default_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_pb_request], size=1)


@pytest.fixture
def default_starcoder_batch(default_pb_batch, default_tokenizer):
    return StarCoderCausalLMBatch.from_pb(
        default_pb_batch, default_tokenizer, torch.float32, torch.device("hpu")
    )


@pytest.fixture
def default_multi_requests_starcoder_batch(default_pb_request, default_tokenizer):
    req_0 = copy(default_pb_request)
    req_0.id = 1
    req_1 = default_pb_request
    req_1.id = 2
    req_1.stopping_parameters.max_new_tokens = 5

    batch_pb = generate_pb2.Batch(id=1, requests=[req_0, req_1], size=2)
    return StarCoderCausalLMBatch.from_pb(
        batch_pb, default_tokenizer, torch.float32, torch.device("hpu")
    )


def test_starcoder_batch_type(default_starcoder):
    assert default_starcoder.batch_type == StarCoderCausalLMBatch


def test_batch_from_pb(default_pb_batch, default_starcoder_batch):
    batch = default_starcoder_batch

    assert batch.batch_id == default_pb_batch.id
    assert len(batch.requests) == len(default_pb_batch.requests)

    for r in range(0,len(default_pb_batch.requests)):
        assert batch.requests[r].data == default_pb_batch.requests[r]

    # For Gaudi we are adding padding of multiplication of bucket size
    size_of_padded_to_bucket = ((default_pb_batch.size + PREFILL_BATCH_BUCKET_SIZE - 1) // PREFILL_BATCH_BUCKET_SIZE) * PREFILL_BATCH_BUCKET_SIZE

    assert len(batch.input_ids) == size_of_padded_to_bucket
    assert batch.input_ids.shape == torch.Size([4, 128])

    assert batch.input_ids[0][-2] == 1006
    assert batch.input_ids[1][-2] == 49
    assert batch.input_ids[2][-2] == 49
    assert batch.attention_mask[0][-2] == 1
    assert batch.attention_mask[1][-2] == 1
    assert batch.attention_mask[2][-2] == 1
    assert torch.all(batch.attention_mask[0, :-3] == 0)

    assert batch.past_key_values is None
    assert all(
        [
            torch.equal(input_ids, request.all_input_ids[:batch.input_length + 1, 0])
            for input_ids, request in zip(batch.input_ids, batch.requests)
        ]
    )

    assert len(batch) == default_pb_batch.size

    assert batch.max_input_length + 1 == default_pb_batch.requests[0].truncate


def test_starcoder_generate_token(default_starcoder, default_starcoder_batch):

    sequence_length = len(default_starcoder_batch.requests[0].all_input_ids)
    generations, next_batch, _ = default_starcoder.generate_token([default_starcoder_batch])
    padding = next_batch.requests[0].stopping_criteria.max_new_tokens

    assert isinstance(next_batch, StarCoderCausalLMBatch)
    assert len(next_batch.attention_mask[0]) == PAD_SEQUENCE_TO_MULTIPLE_OF
    assert next_batch.requests[0].all_input_ids[-padding-2] == 1006

    assert torch.all(next_batch.requests[0].all_input_ids[-padding-1:] == PAD_TOKEN)
    assert torch.all(next_batch.requests[0].all_input_ids[:-padding-3] == PAD_TOKEN)

    generations, next_batch, _ = default_starcoder.generate_token([default_starcoder_batch])
    assert torch.all(next_batch.attention_mask[0][PAD_SEQUENCE_TO_MULTIPLE_OF-2:PAD_SEQUENCE_TO_MULTIPLE_OF] == 1)
    assert torch.all(next_batch.attention_mask[0][:PAD_SEQUENCE_TO_MULTIPLE_OF-3] == 0)
    assert torch.all(next_batch.attention_mask[0][PAD_SEQUENCE_TO_MULTIPLE_OF+1:] == 0)

    assert next_batch.requests[0].all_input_ids[-padding-2] == 1006
    assert next_batch.requests[0].all_input_ids[-padding-1] == 26
    assert torch.all(next_batch.requests[0].all_input_ids[-padding:] == PAD_TOKEN)
    assert torch.all(next_batch.requests[0].all_input_ids[:-padding-3] == PAD_TOKEN)

    assert next_batch.input_length == PAD_SEQUENCE_TO_MULTIPLE_OF
    assert next_batch.max_input_length == next_batch.input_length

    assert next_batch.past_key_values is not None
    assert all(
        [p[0].shape == (MAX_TOTAL_TOKENS, 256) for p in next_batch.past_key_values]
    )
    assert all(
        [p[1].shape == (MAX_TOTAL_TOKENS, 256) for p in next_batch.past_key_values]
    )
    assert all([generation.generated_text is None for generation in generations])
    assert all([len(generation.prefill_tokens) == PAD_SEQUENCE_TO_MULTIPLE_OF-1 for generation in generations])
    assert all([generation.tokens.token_ids[0] == 26 for generation in generations])
    assert all([generation.tokens.texts[0] == "(" for generation in generations])
    assert generations[0].request_id == 0


def test_starcoder_generate_token_completion(
    default_starcoder, default_starcoder_batch
):

    next_batch = default_starcoder_batch
    generations, next_batch, _ = default_starcoder.generate_token([next_batch])

    for _ in range(default_starcoder_batch.requests[0].stopping_criteria.max_new_tokens - 1):
        generations, next_batch, _ = default_starcoder.generate_token([next_batch])
        assert len(generations) == len(next_batch)

    generations, next_batch, _ = default_starcoder.generate_token([next_batch])

    assert next_batch is None

    assert len(generations) == 1
    assert generations[0].generated_text.text == '(self):\n        """\n        Test that the test'
    assert generations[0].request_id == default_starcoder_batch.requests[0].data.id
    assert (
        generations[0].generated_text.generated_tokens
        == default_starcoder_batch.requests[0].stopping_criteria.max_new_tokens
    )


def test_starcoder_generate_token_completion_multi(
    default_starcoder, default_multi_requests_starcoder_batch
):
    next_batch = default_multi_requests_starcoder_batch
    generations, next_batch, _ = default_starcoder.generate_token([next_batch])

    for i in range(
        default_multi_requests_starcoder_batch.requests[1].stopping_criteria.max_new_tokens - 1
    ):
        generations, next_batch, _ = default_starcoder.generate_token([next_batch])
        assert len(generations) == len(next_batch)

    generations, next_batch, _ = default_starcoder.generate_token([next_batch])
    assert next_batch is not None

    assert len(generations) == 2
    assert generations[1].generated_text.text == '(self):\n        """'
    assert (
        generations[1].request_id
        == default_multi_requests_starcoder_batch.requests[1].data.id
    )
    assert (
        generations[1].generated_text.generated_tokens
        == default_multi_requests_starcoder_batch.requests[1].stopping_criteria.max_new_tokens
    )

    next_batch = next_batch.filter([next_batch.requests[0].data.id])

    for _ in range(
        default_multi_requests_starcoder_batch.requests[0].stopping_criteria.max_new_tokens - default_multi_requests_starcoder_batch.requests[1].stopping_criteria.max_new_tokens - 1
    ):
        generations, next_batch, _ = default_starcoder.generate_token([next_batch])
        assert len(generations) == len(next_batch)

    generations, next_batch, _ = default_starcoder.generate_token([next_batch])

    assert next_batch is None

    assert len(generations) == 1
    assert generations[0].generated_text.text == '(self):\n        """\n        Test that the test'
    assert (
        generations[0].request_id
        == default_multi_requests_starcoder_batch.requests[0].data.id
    )
    assert (
        generations[0].generated_text.generated_tokens
        == default_multi_requests_starcoder_batch.requests[0].stopping_criteria.max_new_tokens
    )


def test_batch_concatenate(
    default_starcoder, default_starcoder_batch, default_multi_requests_starcoder_batch
):
    next_batch_0 = default_starcoder_batch
    _, next_batch_0, _ = default_starcoder.generate_token([next_batch_0])
    _, next_batch_0, _ = default_starcoder.generate_token([next_batch_0])
    _, next_batch_0, _ = default_starcoder.generate_token([next_batch_0])

    next_batch_1 = default_multi_requests_starcoder_batch
    _, next_batch_1, _ = default_starcoder.generate_token([next_batch_1])
    _, next_batch_1, _ = default_starcoder.generate_token([next_batch_1])

    # Clone past_key_values before concatenating to compare after,
    # because they are removed from the concatenated batches
    next_batch_0_past_key_values = [x.clone() for x in next_batch_0.past_key_values]
    next_batch_1_past_key_values = [x.clone() for x in next_batch_1.past_key_values]

    next_batch = StarCoderCausalLMBatch.concatenate([next_batch_0, next_batch_1])

    assert torch.equal(next_batch.requests[0].all_input_ids, next_batch_0.requests[0].all_input_ids)
    assert torch.equal(next_batch.requests[1].all_input_ids, next_batch_1.requests[0].all_input_ids)
    assert torch.equal(next_batch.requests[2].all_input_ids, next_batch_1.requests[1].all_input_ids)


    assert torch.all(
            next_batch.attention_mask[0:2, -next_batch.right_padding - 2: -next_batch.right_padding] == 1
    )
    assert torch.all(
            next_batch.attention_mask[2, -next_batch.right_padding - 3: -next_batch.right_padding] == 1
    )
    assert torch.all(
            next_batch.attention_mask[3, -next_batch.right_padding - 2: -next_batch.right_padding] == 1
    )

    assert torch.all(
        next_batch.attention_mask[0:2, :-next_batch.right_padding-2] == 0)
    assert torch.all(
        next_batch.attention_mask[2, :-next_batch.right_padding-4] == 0)
    assert torch.all(
        next_batch.attention_mask[3, :-next_batch.right_padding-3] == 0)

    assert next_batch.batch_id == 0
    assert next_batch.input_ids[0,-next_batch.right_padding - 2] == 1006
    assert next_batch.input_ids[0,-next_batch.right_padding - 1] == 26

    assert next_batch.max_input_length == 129

    assert torch.all(next_batch.input_ids[0,-next_batch.right_padding:] == PAD_TOKEN)
    assert torch.all(next_batch.input_ids[1,-next_batch.right_padding:] == PAD_TOKEN)
    assert torch.all(next_batch.input_ids[2,-next_batch.right_padding:] == PAD_TOKEN)
    assert torch.all(next_batch.input_ids[3,-next_batch.right_padding:] == PAD_TOKEN)

    assert next_batch.input_length == PAD_SEQUENCE_TO_MULTIPLE_OF  +1
    assert next_batch.max_input_length == PAD_SEQUENCE_TO_MULTIPLE_OF + 1

    assert next_batch.requests[0] == next_batch_0.requests[0]
    assert next_batch.requests[1:] == next_batch_1.requests

    assert next_batch.requests[0].stopping_criteria == next_batch_0.requests[0].stopping_criteria
    assert next_batch.requests[1].stopping_criteria == next_batch_1.requests[0].stopping_criteria
    assert next_batch.requests[2].stopping_criteria == next_batch_1.requests[1].stopping_criteria

    assert next_batch.past_key_values is not None

    assert all([p[0].shape == (2048, 256) for p in next_batch.past_key_values])
    assert all([p[1].shape == (2048, 256) for p in next_batch.past_key_values])

    assert next_batch.past_key_values is not None

    for i, past in enumerate(next_batch.past_key_values):
        assert torch.equal(next_batch_0_past_key_values[i][0,0,0:128], past[0][1:129][0, 0:128])
        assert torch.equal(next_batch_0_past_key_values[i][0,1,0:128], past[1][1:129][0, 0:128])
        assert torch.equal(
            next_batch_1_past_key_values[i][:, :, 0:1][0][0][0], past[0][1:, :][0][0]
        )

        assert torch.equal(
            next_batch_1_past_key_values[i][1:, :, 0:1][0][0][0], past[1][1:, :][0][0]
        )

    generations, next_batch, _ = default_starcoder.generate_token([next_batch])

    for _ in range(
        default_multi_requests_starcoder_batch.requests[1].stopping_criteria.max_new_tokens - 2
    ):
        generations, next_batch, _ = default_starcoder.generate_token([next_batch])
        assert len(generations) == len(next_batch)

    generations, next_batch, _ = default_starcoder.generate_token([next_batch])
    assert next_batch is not None

    assert len(generations) == 3
    assert generations[2].generated_text.text == '(self):\n        """'

    assert (
        generations[2].request_id
        == default_multi_requests_starcoder_batch.requests[1].data.id
    )
    assert (
        generations[2].generated_text.generated_tokens
        == default_multi_requests_starcoder_batch.requests[1].stopping_criteria.max_new_tokens
    )

    next_batch = next_batch.filter(
        [next_batch.requests[0].data.id, next_batch.requests[1].data.id]
    )

    for _ in range(
        default_starcoder_batch.requests[0].stopping_criteria.max_new_tokens
        - default_multi_requests_starcoder_batch.requests[1].stopping_criteria.max_new_tokens
        - 2
    ):
        generations, next_batch, _ = default_starcoder.generate_token([next_batch])
        assert len(generations) == len(next_batch)

    generations, next_batch, _ = default_starcoder.generate_token([next_batch])
    assert next_batch is not None

    assert len(generations) == 2
    assert generations[0].generated_text.text == '(self):\n        """\n        Test that the test'
    assert generations[0].request_id == default_starcoder_batch.requests[0].data.id
    assert (
        generations[0].generated_text.generated_tokens
        == default_starcoder_batch.requests[0].stopping_criteria.max_new_tokens
    )

    next_batch = next_batch.filter([next_batch.requests[1].data.id])

    for _ in range(
        default_multi_requests_starcoder_batch.requests[0].stopping_criteria.max_new_tokens
        - default_starcoder_batch.requests[0].stopping_criteria.max_new_tokens
        - default_multi_requests_starcoder_batch.requests[1].stopping_criteria.max_new_tokens
        - 4
    ):
        generations, next_batch, _ = default_starcoder.generate_token([next_batch])
        assert len(generations) == len(next_batch)

    generations, next_batch, _ = default_starcoder.generate_token([next_batch])
    assert next_batch is None

    assert len(generations) == 1
    assert generations[0].generated_text.text == '(self):\n        """\n        Test that the test'
    assert (
        generations[0].request_id
        == default_multi_requests_starcoder_batch.requests[0].data.id
    )
    assert (
        generations[0].generated_text.generated_tokens
        == default_multi_requests_starcoder_batch.requests[0].stopping_criteria.max_new_tokens
    )
