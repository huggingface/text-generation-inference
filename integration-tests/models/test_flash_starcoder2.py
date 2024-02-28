import pytest


@pytest.fixture(scope="module")
def flash_starcoder2_handle(launcher):
    with launcher("bigcode/starcoder2-3b", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_starcoder2(flash_starcoder2_handle):
    await flash_starcoder2_handle.health(300)
    return flash_starcoder2_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_starcoder2(flash_starcoder2, response_snapshot):
    response = await flash_starcoder2.generate(
        "def print_hello", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_starcoder2_default_params(flash_starcoder2, response_snapshot):
    response = await flash_starcoder2.generate(
        "def print_hello",
        max_new_tokens=60,
        temperature=0.2,
        top_p=0.95,
        decoder_input_details=True,
        seed=0,
    )

    assert response.details.generated_tokens == 60
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_starcoder2_load(
    flash_starcoder2, generate_load, response_snapshot
):
    responses = await generate_load(
        flash_starcoder2, "def print_hello", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
