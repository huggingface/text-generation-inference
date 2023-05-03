import pytest

from utils import health_check


@pytest.fixture(scope="module")
def bloom_560(launcher):
    with launcher("bigscience/bloom-560m") as client:
        yield client


@pytest.mark.asyncio
async def test_bloom_560m(bloom_560, snapshot):
    await health_check(bloom_560, 60)

    response = await bloom_560.generate("Test request", max_new_tokens=10)

    assert response.details.generated_tokens == 10
    assert response == snapshot


@pytest.mark.asyncio
async def test_bloom_560m_load(bloom_560, generate_load, snapshot):
    await health_check(bloom_560, 60)

    responses = await generate_load(bloom_560, "Test request", max_new_tokens=10, n=4)

    assert len(responses) == 4

    assert responses == snapshot
