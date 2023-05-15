import pytest

from utils import health_check


@pytest.fixture(scope="module")
def flash_neox(launcher):
    with launcher("OpenAssistant/oasst-sft-1-pythia-12b", num_shard=2) as client:
        yield client


@pytest.mark.asyncio
async def test_flash_neox(flash_neox, snapshot_test):
    await health_check(flash_neox, 240)

    response = await flash_neox.generate(
        "<|prompter|>What is a meme, and what's the history behind this word?<|endoftext|><|assistant|>",
        max_new_tokens=10,
    )

    assert response.details.generated_tokens == 10
    assert snapshot_test(response)


@pytest.mark.asyncio
async def test_flash_neox_load(flash_neox, generate_load, snapshot_test):
    await health_check(flash_neox, 240)

    responses = await generate_load(
        flash_neox,
        "<|prompter|>What is a meme, and what's the history behind this word?<|endoftext|><|assistant|>",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4

    assert snapshot_test(responses)
