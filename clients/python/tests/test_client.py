import pytest

from text_generation import Client, AsyncClient
from text_generation.errors import NotFoundError, ValidationError
from text_generation.types import FinishReason


def test_generate(llama_7b_url, hf_headers):
    client = Client(llama_7b_url, hf_headers)
    response = client.generate("test", max_new_tokens=1, decoder_input_details=True)

    assert response.generated_text == "_"
    assert response.details.finish_reason == FinishReason.Length
    assert response.details.generated_tokens == 1
    assert response.details.seed is None
    assert len(response.details.prefill) == 0
    # assert response.details.prefill[0] == InputToken(id=1, text="<s>", logprob=None)
    assert len(response.details.tokens) == 1
    assert response.details.tokens[0].id == 29918
    assert response.details.tokens[0].text == "_"
    assert not response.details.tokens[0].special


def test_generate_best_of(llama_7b_url, hf_headers):
    client = Client(llama_7b_url, hf_headers)
    response = client.generate(
        "test", max_new_tokens=1, best_of=2, do_sample=True, decoder_input_details=True
    )

    assert response.details.seed is not None
    assert response.details.best_of_sequences is not None
    assert len(response.details.best_of_sequences) == 1
    assert response.details.best_of_sequences[0].seed is not None


def test_generate_not_found(fake_url, hf_headers):
    client = Client(fake_url, hf_headers)
    with pytest.raises(NotFoundError):
        client.generate("test")


def test_generate_validation_error(llama_7b_url, hf_headers):
    client = Client(llama_7b_url, hf_headers)
    with pytest.raises(ValidationError):
        client.generate("test", max_new_tokens=10_000)


def test_generate_stream(llama_7b_url, hf_headers):
    client = Client(llama_7b_url, hf_headers)
    responses = [
        response for response in client.generate_stream("test", max_new_tokens=1)
    ]

    assert len(responses) == 1
    response = responses[0]

    assert response.generated_text == "_"
    assert response.details.finish_reason == FinishReason.Length
    assert response.details.generated_tokens == 1
    assert response.details.seed is None


def test_generate_stream_not_found(fake_url, hf_headers):
    client = Client(fake_url, hf_headers)
    with pytest.raises(NotFoundError):
        list(client.generate_stream("test"))


def test_generate_stream_validation_error(llama_7b_url, hf_headers):
    client = Client(llama_7b_url, hf_headers)
    with pytest.raises(ValidationError):
        list(client.generate_stream("test", max_new_tokens=10_000))


@pytest.mark.asyncio
async def test_generate_async(llama_7b_url, hf_headers):
    client = AsyncClient(llama_7b_url, hf_headers)
    response = await client.generate(
        "test", max_new_tokens=1, decoder_input_details=True
    )

    assert response.generated_text == "_"
    assert response.details.finish_reason == FinishReason.Length
    assert response.details.generated_tokens == 1
    assert response.details.seed is None
    assert len(response.details.prefill) == 0
    # assert response.details.prefill[0] == InputToken(id=1, text="<s>", logprob=None)
    # assert response.details.prefill[1] == InputToken(
    #     id=1243, text="test", logprob=-10.96875
    # )
    assert len(response.details.tokens) == 1
    assert response.details.tokens[0].id == 29918
    assert response.details.tokens[0].text == "_"
    assert not response.details.tokens[0].special


@pytest.mark.asyncio
async def test_generate_async_best_of(llama_7b_url, hf_headers):
    client = AsyncClient(llama_7b_url, hf_headers)
    response = await client.generate(
        "test", max_new_tokens=1, best_of=2, do_sample=True, decoder_input_details=True
    )

    assert response.details.seed is not None
    assert response.details.best_of_sequences is not None
    assert len(response.details.best_of_sequences) == 1
    assert response.details.best_of_sequences[0].seed is not None


@pytest.mark.asyncio
async def test_generate_async_not_found(fake_url, hf_headers):
    client = AsyncClient(fake_url, hf_headers)
    with pytest.raises(NotFoundError):
        await client.generate("test")


@pytest.mark.asyncio
async def test_generate_async_validation_error(llama_7b_url, hf_headers):
    client = AsyncClient(llama_7b_url, hf_headers)
    with pytest.raises(ValidationError):
        await client.generate("test", max_new_tokens=10_000)


@pytest.mark.asyncio
async def test_generate_stream_async(llama_7b_url, hf_headers):
    client = AsyncClient(llama_7b_url, hf_headers)
    responses = [
        response async for response in client.generate_stream("test", max_new_tokens=1)
    ]

    assert len(responses) == 1
    response = responses[0]

    assert response.generated_text == "_"
    assert response.details.finish_reason == FinishReason.Length
    assert response.details.generated_tokens == 1
    assert response.details.seed is None


@pytest.mark.asyncio
async def test_generate_stream_async_not_found(fake_url, hf_headers):
    client = AsyncClient(fake_url, hf_headers)
    with pytest.raises(NotFoundError):
        async for _ in client.generate_stream("test"):
            pass


@pytest.mark.asyncio
async def test_generate_stream_async_validation_error(llama_7b_url, hf_headers):
    client = AsyncClient(llama_7b_url, hf_headers)
    with pytest.raises(ValidationError):
        async for _ in client.generate_stream("test", max_new_tokens=10_000):
            pass
