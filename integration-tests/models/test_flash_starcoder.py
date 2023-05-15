import pytest

from utils import health_check


@pytest.fixture(scope="module")
def flash_starcoder(launcher):
    with launcher("bigcode/starcoder", num_shard=2) as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_starcoder(flash_starcoder, snapshot_test):
    await health_check(flash_starcoder, 240)

    response = await flash_starcoder.generate("def print_hello", max_new_tokens=10)

    assert response.details.generated_tokens == 10
    assert snapshot_test(response)


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_starcoder_default_params(flash_starcoder, snapshot_test):
    await health_check(flash_starcoder, 240)

    response = await flash_starcoder.generate(
        "def print_hello", max_new_tokens=60, temperature=0.2, top_p=0.95, seed=0
    )

    assert response.details.generated_tokens == 12
    assert snapshot_test(response)


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_starcoder_load(flash_starcoder, generate_load, snapshot_test):
    await health_check(flash_starcoder, 240)

    responses = await generate_load(
        flash_starcoder, "def print_hello", max_new_tokens=10, n=4
    )

    assert len(responses) == 4

    assert snapshot_test(responses)
