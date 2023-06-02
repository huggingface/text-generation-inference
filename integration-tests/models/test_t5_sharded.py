import pytest


@pytest.fixture(scope="module")
def t5_sharded_handle(launcher):
    with launcher("google/flan-t5-xxl", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def t5_sharded(t5_sharded_handle):
    await t5_sharded_handle.health(300)
    return t5_sharded_handle.client


@pytest.mark.asyncio
async def test_t5_sharded(t5_sharded, response_snapshot):
    response = await t5_sharded.generate(
        "Please answer the following question. What is the boiling point of Nitrogen?",
        max_new_tokens=10,
        decoder_input_details=True,
    )

    assert response == response_snapshot


@pytest.mark.asyncio
async def test_t5_sharded_load(t5_sharded, generate_load, response_snapshot):
    responses = await generate_load(
        t5_sharded,
        "Please answer the following question. What is the boiling point of Nitrogen?",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
