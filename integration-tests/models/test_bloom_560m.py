import pytest

from utils import health_check


@pytest.fixture(scope="module")
def bloom_560(launcher):
    with launcher("bigscience/bloom-560m") as client:
        yield client


@pytest.mark.asyncio
async def test_bloom_560m(bloom_560, snapshot):
    await health_check(bloom_560, 60)

    response = await bloom_560.generate(
        "Pour déguster un ortolan, il faut tout d'abord",
        max_new_tokens=10,
        top_p=0.9,
        seed=0,
    )

    assert response.details.generated_tokens == 10
    assert response == snapshot


@pytest.mark.asyncio
async def test_bloom_560m_all_params(bloom_560, snapshot):
    await health_check(bloom_560, 60)

    response = await bloom_560.generate(
        "Pour déguster un ortolan, il faut tout d'abord",
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
async def test_bloom_560m_load(bloom_560, generate_load, snapshot):
    await health_check(bloom_560, 60)

    responses = await generate_load(
        bloom_560,
        "Pour déguster un ortolan, il faut tout d'abord",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4

    assert responses == snapshot
