import pytest

from utils import health_check


@pytest.fixture(scope="module")
def flash_llama(launcher):
    with launcher("huggingface/llama-7b", num_shard=2) as client:
        yield client


@pytest.mark.asyncio
async def test_flash_llama(flash_llama, snapshot):
    await health_check(flash_llama, 120)

    response = await flash_llama.generate("Test request", max_new_tokens=10)

    assert response.details.generated_tokens == 10
    assert response == snapshot


@pytest.mark.asyncio
async def test_flash_llama_all_params(flash_llama, snapshot):
    await health_check(flash_llama, 120)

    response = await flash_llama.generate(
        "Test request",
        max_new_tokens=10,
        repetition_penalty=1.2,
        return_full_text=True,
        stop_sequences=["test"],
        temperature=0.5,
        top_p=0.9,
        top_k=10,
        truncate=5,
        typical_p=0.9,
        watermark=True,
        seed=0,
    )

    assert response.details.generated_tokens == 10
    assert response == snapshot


@pytest.mark.asyncio
async def test_flash_llama_load(flash_llama, generate_load, snapshot):
    await health_check(flash_llama, 120)

    responses = await generate_load(flash_llama, "Test request", max_new_tokens=10, n=4)

    assert len(responses) == 4

    assert responses == snapshot
