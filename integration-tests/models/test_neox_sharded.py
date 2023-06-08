import pytest


@pytest.fixture(scope="module")
def neox_sharded_handle(launcher):
    with launcher(
        "OpenAssistant/oasst-sft-1-pythia-12b", num_shard=2, use_flash_attention=False
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def neox_sharded(neox_sharded_handle):
    await neox_sharded_handle.health(300)
    return neox_sharded_handle.client


@pytest.mark.skip
@pytest.mark.asyncio
async def test_neox(neox_sharded, response_snapshot):
    response = await neox_sharded.generate(
        "<|prompter|>What is a meme, and what's the history behind this word?<|endoftext|><|assistant|>",
        max_new_tokens=10,
        decoder_input_details=True,
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.skip
@pytest.mark.asyncio
async def test_neox_load(neox_sharded, generate_load, response_snapshot):
    responses = await generate_load(
        neox_sharded,
        "<|prompter|>What is a meme, and what's the history behind this word?<|endoftext|><|assistant|>",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
