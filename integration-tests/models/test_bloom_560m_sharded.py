import pytest

from utils import health_check


@pytest.fixture(scope="module")
def bloom_560m_sharded(launcher):
    with launcher("bigscience/bloom-560m", num_shard=2) as client:
        yield client


@pytest.mark.asyncio
async def test_bloom_560m_sharded(bloom_560m_sharded, snapshot_test):
    await health_check(bloom_560m_sharded, 60)

    response = await bloom_560m_sharded.generate(
        "Pour déguster un ortolan, il faut tout d'abord",
        max_new_tokens=10,
        top_p=0.9,
        seed=0,
    )

    assert response.details.generated_tokens == 10
    assert snapshot_test(response)


@pytest.mark.asyncio
async def test_bloom_560m_sharded_load(
    bloom_560m_sharded, generate_load, snapshot_test
):
    await health_check(bloom_560m_sharded, 60)

    responses = await generate_load(
        bloom_560m_sharded,
        "Pour déguster un ortolan, il faut tout d'abord",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4

    assert snapshot_test(responses)
