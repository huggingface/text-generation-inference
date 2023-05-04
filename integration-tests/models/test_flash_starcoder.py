import pytest

from utils import health_check


@pytest.fixture(scope="module")
def flash_starcoder(launcher):
    with launcher("bigcode/large-model", num_shard=2) as client:
        yield client


@pytest.mark.asyncio
async def test_flash_starcoder(flash_starcoder, snapshot):
    await health_check(flash_starcoder, 60)

    response = await flash_starcoder.generate("Test request", max_new_tokens=10)

    assert response.details.generated_tokens == 10
    assert response == snapshot


@pytest.mark.asyncio
async def test_flash_starcoder_load(flash_starcoder, generate_load, snapshot):
    await health_check(flash_starcoder, 60)

    responses = await generate_load(
        flash_starcoder, "Test request", max_new_tokens=10, n=4
    )

    assert len(responses) == 4

    assert responses == snapshot
