import pytest


@pytest.fixture(scope="module")
async def bloom_560m_sharded(launcher):
    async with launcher("bigscience/bloom-560m", num_shard=2) as client:
        yield client



@pytest.mark.asyncio
async def test_bloom_560m_sharded(bloom_560m_sharded, snapshot):
    response = await bloom_560m_sharded.generate("Test request", max_new_tokens=10)

    assert response.details.generated_tokens == 10
    assert response == snapshot


@pytest.mark.asyncio
async def test_bloom_560m_sharded_load(bloom_560m_sharded, generate_load, snapshot):
    responses = await generate_load(bloom_560m_sharded, "Test request", max_new_tokens=10, n=4)

    assert len(responses) == 4

    assert responses == snapshot
