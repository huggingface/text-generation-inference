import pytest


@pytest.fixture(scope="module")
def mpt_sharded_handle(launcher):
    with launcher("mosaicml/mpt-7b", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def mpt_sharded(mpt_sharded_handle):
    await mpt_sharded_handle.health(300)
    return mpt_sharded_handle.client


@pytest.mark.release
@pytest.mark.asyncio
async def test_mpt(mpt_sharded, response_snapshot):
    response = await mpt_sharded.generate(
        "What is Deep Learning?",
        max_new_tokens=17,
        decoder_input_details=True,
    )

    assert response.details.generated_tokens == 17
    assert (
        response.generated_text
        == " - Deep Learning\nDeep Learning is a subfield of machine learning that uses artificial neural"
    )
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
async def test_mpt_load(mpt_sharded, generate_load, response_snapshot):
    responses = await generate_load(
        mpt_sharded,
        "What is Deep Learning?",
        max_new_tokens=17,
        n=4,
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])
    assert (
        responses[0].generated_text
        == " - Deep Learning\nDeep Learning is a subfield of machine learning that uses artificial neural"
    )

    assert responses == response_snapshot
