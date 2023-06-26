import pytest


@pytest.fixture(scope="module")
def neox_handle(launcher):
    with launcher(
        "stabilityai/stablelm-tuned-alpha-3b", num_shard=1, use_flash_attention=False
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def neox(neox_handle):
    await neox_handle.health(300)
    return neox_handle.client


@pytest.mark.skip
@pytest.mark.asyncio
async def test_neox(neox, response_snapshot):
    response = await neox.generate(
        "<|USER|>What's your mood today?<|ASSISTANT|>",
        max_new_tokens=10,
        decoder_input_details=True,
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.skip
@pytest.mark.asyncio
async def test_neox_load(neox, generate_load, response_snapshot):
    responses = await generate_load(
        neox,
        "<|USER|>What's your mood today?<|ASSISTANT|>",
        max_new_tokens=10,
        n=4,
    )

    generated_texts = [r.generated_text for r in responses]

    assert len(generated_texts) == 4
    assert generated_texts, all(
        [text == generated_texts[0] for text in generated_texts]
    )

    assert responses == response_snapshot
