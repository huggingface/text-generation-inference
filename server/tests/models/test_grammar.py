# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import json
import pytest
import torch

from copy import copy

from text_generation_server.pb import generate_pb2
from text_generation_server.models import get_model
from text_generation_server.models.causal_lm import (
    CausalLMBatch,
    PAD_SEQUENCE_TO_MULTIPLE_OF,
)

PAD_TOKEN=0


@pytest.fixture
def default_pb_grammar_parameters():
    grammar_schema = {
        "properties": {
            "activity": {
                "type": "string"
            },
            "animals": {
                "items": {
                    "type":"string"
                },
                "type": "array"
            }
        },
        "required": ["activity", "animals"]
    }
    return generate_pb2.NextTokenChooserParameters(
        temperature=1.0,
        repetition_penalty=1.3,
        top_k=0,
        top_p=1.0,
        typical_p=1.0,
        do_sample=False,
        grammar_type=generate_pb2.GrammarType.GRAMMAR_TYPE_JSON,
        grammar=json.dumps(grammar_schema).encode('utf-8'),
    )


@pytest.fixture(scope="session")
def default_grammar_response():
    return [
        29912, 376, 29874, 312, 2068, 1115, 29871, 13, 29908, 29890,
        638, 292, 613, 259, 376, 273, 3039, 29879, 1115,518, 1678,
        376, 26169, 3284, 4117, 3284, 336, 617, 6150, 3108, 500, 2
    ]


@pytest.fixture(scope="session")
def default_causal_lm():
    return get_model("meta-llama/Llama-2-7b-hf", None, None, None, None)


@pytest.fixture(scope="session")
def default_tokenizer(default_causal_lm):
    default_causal_lm.tokenizer.pad_token_id = PAD_TOKEN
    return default_causal_lm.tokenizer


@pytest.fixture
def default_pb_request(default_pb_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="Test",
        prefill_logprobs=True,
        truncate=PAD_SEQUENCE_TO_MULTIPLE_OF,
        parameters=default_pb_parameters,
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(stop_sequences=[], max_new_tokens=10),
    )


@pytest.fixture
def default_pb_grammar_request(default_pb_grammar_parameters):
    return generate_pb2.Request(
        id=1,
        inputs=f"Please use the following JSON schema to generate the output: I saw a puppy a cat and a raccoon during my bike ride in the park",
        prefill_logprobs=True,
        truncate=PAD_SEQUENCE_TO_MULTIPLE_OF,
        parameters=default_pb_grammar_parameters,
        stopping_parameters=generate_pb2.StoppingCriteriaParameters(stop_sequences=[], max_new_tokens=50),
    )


@pytest.fixture
def default_pb_batch(default_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_pb_request], size=1)


@pytest.fixture
def default_pb_grammar_batch(default_pb_grammar_request):
    return generate_pb2.Batch(id=1, requests=[default_pb_grammar_request], size=1)


@pytest.fixture
def default_causal_lm_batch(default_pb_batch, default_tokenizer):
    return CausalLMBatch.from_pb(
        default_pb_batch, default_tokenizer, torch.float32, torch.device("hpu")
    )


@pytest.fixture
def default_causal_lm_grammar_batch(default_pb_grammar_batch, default_tokenizer):
    return CausalLMBatch.from_pb(
        default_pb_grammar_batch, default_tokenizer, torch.float32, torch.device("hpu")
    )


@pytest.fixture
def default_two_causal_lm_grammar_batches(default_pb_grammar_request, default_tokenizer):
    req_0 = default_pb_grammar_request
    req_0.id = 0
    req_1 = copy(default_pb_grammar_request)
    req_1.id = 1

    batch_0 = generate_pb2.Batch(id=0, requests=[req_0], size=1)
    batch_1 = generate_pb2.Batch(id=1, requests=[req_1], size=1)
    return [
        CausalLMBatch.from_pb(
            b, default_tokenizer, torch.float32, torch.device("hpu")
        ) for b in [batch_0, batch_1]
    ]


def test_single_grammar_batch(
    default_causal_lm, default_causal_lm_grammar_batch, default_grammar_response
):
    counter = 0
    batch = default_causal_lm_grammar_batch

    # prefill request
    generations, next_batch, _ = default_causal_lm.generate_token([batch])

    # generate untill done
    while next_batch is not None:
        generations, next_batch, _ = default_causal_lm.generate_token([next_batch])
        assert len(generations) == 1
        assert generations[0].tokens.token_ids[0] == default_grammar_response[counter]
        counter += 1
    print(generations[0].generated_text.text)


def test_multi_grammar_batches(
    default_causal_lm, default_two_causal_lm_grammar_batches, default_grammar_response
):
    counter_0, counter_1 = 0, 0
    batch_0, batch_1 = default_two_causal_lm_grammar_batches

    # prefill first request
    generations, next_batch, _ = default_causal_lm.generate_token([batch_0])
    generations, next_batch, _ = default_causal_lm.generate_token([next_batch])
    assert len(generations) == 1
    assert generations[0].tokens.token_ids[0] == default_grammar_response[counter_0]
    counter_0 += 1

    # prefill second request
    generations, next_batch_1, _ = default_causal_lm.generate_token([batch_1])

    # concatenate and generate
    generations, next_batch, _ = default_causal_lm.generate_token([next_batch, next_batch_1])
    assert len(generations) == 2
    assert generations[0].tokens.token_ids[0] == default_grammar_response[counter_0]
    assert generations[1].tokens.token_ids[0] == default_grammar_response[counter_1]
    counter_0 += 1
    counter_1 += 1

    # generate untill first request is done
    while generations[0].generated_text is None:
        generations, next_batch, _ = default_causal_lm.generate_token([next_batch])
        assert len(generations) == 2
        assert generations[0].tokens.token_ids[0] == default_grammar_response[counter_0]
        assert generations[1].tokens.token_ids[0] == default_grammar_response[counter_1]
        counter_0 += 1
        counter_1 += 1

    # filter finished request
    response = generations[0].generated_text.text
    next_batch = next_batch.filter([next_batch.requests[1].data.id])

    # generate last tokens for second request
    while next_batch is not None:
        generations, next_batch, _ = default_causal_lm.generate_token([next_batch])
        assert len(generations) == 1
        assert generations[0].tokens.token_ids[0] == default_grammar_response[counter_1]
        counter_1 += 1

    assert response == generations[0].generated_text.text


def test_grammar_and_causal_batch(
    default_causal_lm, default_causal_lm_grammar_batch, default_causal_lm_batch, default_grammar_response
):
    counter = 0
    generations, next_batch, _ = default_causal_lm.generate_token([default_causal_lm_grammar_batch])
    generations, next_batch, _ = default_causal_lm.generate_token([next_batch])
    assert len(generations) == 1
    assert generations[0].tokens.token_ids[0] == default_grammar_response[counter]
    counter += 1

    generations, next_batch, _ = default_causal_lm.generate_token([next_batch])
    assert len(generations) == 1
    assert generations[0].tokens.token_ids[0] == default_grammar_response[counter]
    counter += 1

    # prefill second request
    generations, next_batch_1, _ = default_causal_lm.generate_token([default_causal_lm_batch])

    # concatenate and generate
    generations, next_batch, _ = default_causal_lm.generate_token([next_batch, next_batch_1])
    assert len(generations) == 2
    assert generations[0].tokens.token_ids[0] == default_grammar_response[counter]
    counter += 1

    # generate untill second request is done
    for _ in range(
        next_batch.requests[1].stopping_criteria.max_new_tokens - 1
    ):
        generations, next_batch, _ = default_causal_lm.generate_token([next_batch])
        assert len(generations) == 2
        assert generations[0].tokens.token_ids[0] == default_grammar_response[counter]
        counter += 1

    # filter finished request
    assert len(generations) == 2
    assert (
        generations[1].request_id == next_batch.requests[1].data.id
    )
    assert (
        generations[1].generated_text.generated_tokens == next_batch.requests[1].stopping_criteria.max_new_tokens
    )
    assert generations[1].generated_text.text == "ing the effect of a new method for the detection"
    next_batch = next_batch.filter([next_batch.requests[0].data.id])

    # generate untill done
    while next_batch is not None:
        generations, next_batch, _ = default_causal_lm.generate_token([next_batch])
        assert len(generations) == 1
        assert generations[0].tokens.token_ids[0] == default_grammar_response[counter]
        counter += 1
