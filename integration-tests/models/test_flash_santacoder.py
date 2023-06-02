import pytest


@pytest.fixture(scope="module")
def flash_santacoder_handle(launcher):
    with launcher("bigcode/santacoder") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_santacoder(flash_santacoder_handle):
    await flash_santacoder_handle.health(300)
    return flash_santacoder_handle.client


@pytest.mark.asyncio
async def test_flash_santacoder(flash_santacoder, response_snapshot):
    response = await flash_santacoder.generate(
        "def print_hello", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_santacoder_load(
    flash_santacoder, generate_load, response_snapshot
):
    responses = await generate_load(
        flash_santacoder, "def print_hello", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
