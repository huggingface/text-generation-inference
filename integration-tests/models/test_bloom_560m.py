import pytest


@pytest.fixture(scope="module")
async def bloom_560m(launcher):
    async with launcher("bigscience/bloom-560m") as client:
        yield client



@pytest.mark.asyncio
async def test_bloom_560m(bloom_560m, snapshot):
    response = await bloom_560m.generate("Test request", max_new_tokens=10)

    assert response.details.generated_tokens == 10
    assert response == snapshot


@pytest.mark.asyncio
async def test_bloom_560m_load(bloom_560m, generate_load, snapshot):
    responses = await generate_load(bloom_560m, "Test request", max_new_tokens=10, n=4)

    assert len(responses) == 4

    assert responses == snapshot
