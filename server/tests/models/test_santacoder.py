import pytest

from text_generation.pb import generate_pb2
from text_generation.models.causal_lm import CausalLMBatch
from text_generation.models.santacoder import SantaCoder


@pytest.fixture(scope="session")
def default_santacoder():
    return SantaCoder("bigcode/santacoder")


@pytest.fixture
def default_pb_request(default_pb_parameters, default_pb_stop_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="def",
        input_length=1,
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )


@pytest.fixture
def default_pb_batch(default_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_pb_request], size=1)


@pytest.fixture
def default_fim_pb_request(default_pb_parameters, default_pb_stop_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="<fim-prefix>def<fim-suffix>world<fim-middle>",
        input_length=5,
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )


@pytest.fixture
def default_fim_pb_batch(default_fim_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_fim_pb_request], size=1)


def test_santacoder_generate_token_completion(default_santacoder, default_pb_batch):
    batch = CausalLMBatch.from_pb(
        default_pb_batch, default_santacoder.tokenizer, default_santacoder.device
    )
    next_batch = batch

    for _ in range(batch.stopping_criterias[0].max_new_tokens - 1):
        generated_texts, next_batch = default_santacoder.generate_token(next_batch)
        assert generated_texts == []

    generated_texts, next_batch = default_santacoder.generate_token(next_batch)
    assert next_batch is None

    assert len(generated_texts) == 1
    assert generated_texts[0].output_text == "def test_get_all_users_with_"
    assert generated_texts[0].request == batch.requests[0]
    assert len(generated_texts[0].tokens) == len(generated_texts[0].logprobs)
    assert (
        generated_texts[0].generated_tokens
        == batch.stopping_criterias[0].max_new_tokens
    )


def test_fim_santacoder_generate_token_completion(
    default_santacoder, default_fim_pb_batch
):
    batch = CausalLMBatch.from_pb(
        default_fim_pb_batch, default_santacoder.tokenizer, default_santacoder.device
    )
    next_batch = batch

    for _ in range(batch.stopping_criterias[0].max_new_tokens - 1):
        generated_texts, next_batch = default_santacoder.generate_token(next_batch)
        assert generated_texts == []

    generated_texts, next_batch = default_santacoder.generate_token(next_batch)
    assert next_batch is None

    assert len(generated_texts) == 1
    assert (
        generated_texts[0].output_text
        == """<fim-prefix>def<fim-suffix>world<fim-middle>ineProperty(exports, "__esModule", { value"""
    )
    assert generated_texts[0].request == batch.requests[0]
    assert len(generated_texts[0].tokens) == len(generated_texts[0].logprobs)
    assert (
        generated_texts[0].generated_tokens
        == batch.stopping_criterias[0].max_new_tokens
    )
