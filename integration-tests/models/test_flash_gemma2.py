import pytest


@pytest.fixture(scope="module")
def flash_gemma2_handle(launcher):
    with launcher("google/gemma-2-9b-it", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_gemma2(flash_gemma2_handle):
    await flash_gemma2_handle.health(300)
    return flash_gemma2_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_gemma2(flash_gemma2, response_snapshot):
    response = await flash_gemma2.generate(
        "<start_of_turn>user:\nWrite a poem to help me remember the first 10 elements on the periodic table, giving each element its own line.<end_of_turn>\n<start_of_turn>model:\n",
        max_new_tokens=10,
        decoder_input_details=True,
    )

    assert response.generated_text == "**Hydrogen**, light and free,\n**He"
    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_gemma2_load(flash_gemma2, generate_load, response_snapshot):
    responses = await generate_load(
        flash_gemma2,
        "<start_of_turn>user:\nWrite a poem to help me remember the first 10 elements on the periodic table, giving each element its own line.<end_of_turn>\n<start_of_turn>model:\n",
        max_new_tokens=10,
        n=4,
    )

    assert responses[0].generated_text == "**Hydrogen**, light and free,\n**He"
    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
