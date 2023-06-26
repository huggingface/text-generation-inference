import pytest


@pytest.fixture(scope="module")
def bloom_560m_sharded_handle(launcher):
    with launcher("bigscience/bloom-560m", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def bloom_560m_sharded(bloom_560m_sharded_handle):
    await bloom_560m_sharded_handle.health(240)
    return bloom_560m_sharded_handle.client


@pytest.mark.asyncio
async def test_bloom_560m_sharded(bloom_560m_sharded, response_snapshot):
    response = await bloom_560m_sharded.generate(
        "Pour déguster un ortolan, il faut tout d'abord",
        max_new_tokens=10,
        top_p=0.9,
        decoder_input_details=True,
        seed=0,
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_bloom_560m_sharded_load(
    bloom_560m_sharded, generate_load, response_snapshot
):
    responses = await generate_load(
        bloom_560m_sharded,
        "Pour déguster un ortolan, il faut tout d'abord",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
