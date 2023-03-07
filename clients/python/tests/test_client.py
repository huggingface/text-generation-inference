import pytest

from text_generation import Client, AsyncClient
from text_generation.errors import NotFoundError, ValidationError
from text_generation.types import FinishReason, PrefillToken, Token


def test_generate(bloom_url, hf_headers):
    client = Client(bloom_url, hf_headers)
    response = client.generate("test", max_new_tokens=1)

    assert response.generated_text == "."
    assert response.details.finish_reason == FinishReason.Length
    assert response.details.generated_tokens == 1
    assert response.details.seed is None
    assert len(response.details.prefill) == 1
    assert response.details.prefill[0] == PrefillToken(
        id=9234, text="test", logprob=None
    )
    assert len(response.details.tokens) == 1
    assert response.details.tokens[0] == Token(
        id=17, text=".", logprob=-1.75, special=False
    )


def test_generate_not_found(fake_url, hf_headers):
    client = Client(fake_url, hf_headers)
    with pytest.raises(NotFoundError):
        client.generate("test")


def test_generate_validation_error(bloom_url, hf_headers):
    client = Client(bloom_url, hf_headers)
    with pytest.raises(ValidationError):
        client.generate("test", max_new_tokens=10_000)


def test_generate_stream(bloom_url, hf_headers):
    client = Client(bloom_url, hf_headers)
    responses = [
        response for response in client.generate_stream("test", max_new_tokens=1)
    ]

    assert len(responses) == 1
    response = responses[0]

    assert response.generated_text == "."
    assert response.details.finish_reason == FinishReason.Length
    assert response.details.generated_tokens == 1
    assert response.details.seed is None


def test_generate_stream_not_found(fake_url, hf_headers):
    client = Client(fake_url, hf_headers)
    with pytest.raises(NotFoundError):
        list(client.generate_stream("test"))


def test_generate_stream_validation_error(bloom_url, hf_headers):
    client = Client(bloom_url, hf_headers)
    with pytest.raises(ValidationError):
        list(client.generate_stream("test", max_new_tokens=10_000))


@pytest.mark.asyncio
async def test_generate_async(bloom_url, hf_headers):
    client = AsyncClient(bloom_url, hf_headers)
    response = await client.generate("test", max_new_tokens=1)

    assert response.generated_text == "."
    assert response.details.finish_reason == FinishReason.Length
    assert response.details.generated_tokens == 1
    assert response.details.seed is None
    assert len(response.details.prefill) == 1
    assert response.details.prefill[0] == PrefillToken(
        id=9234, text="test", logprob=None
    )
    assert len(response.details.tokens) == 1
    assert response.details.tokens[0] == Token(
        id=17, text=".", logprob=-1.75, special=False
    )


@pytest.mark.asyncio
async def test_generate_async_not_found(fake_url, hf_headers):
    client = AsyncClient(fake_url, hf_headers)
    with pytest.raises(NotFoundError):
        await client.generate("test")


@pytest.mark.asyncio
async def test_generate_async_validation_error(bloom_url, hf_headers):
    client = AsyncClient(bloom_url, hf_headers)
    with pytest.raises(ValidationError):
        await client.generate("test", max_new_tokens=10_000)


@pytest.mark.asyncio
async def test_generate_stream_async(bloom_url, hf_headers):
    client = AsyncClient(bloom_url, hf_headers)
    responses = [
        response async for response in client.generate_stream("test", max_new_tokens=1)
    ]

    assert len(responses) == 1
    response = responses[0]

    assert response.generated_text == "."
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
async def test_generate_stream_async_validation_error(bloom_url, hf_headers):
    client = AsyncClient(bloom_url, hf_headers)
    with pytest.raises(ValidationError):
        async for _ in client.generate_stream("test", max_new_tokens=10_000):
            pass
