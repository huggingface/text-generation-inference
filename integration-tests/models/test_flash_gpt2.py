import pytest


@pytest.fixture(scope="module")
def flash_gpt2_handle(launcher):
    with launcher("openai-community/gpt2", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_gpt2(flash_gpt2_handle):
    await flash_gpt2_handle.health(300)
    return flash_gpt2_handle.client


@pytest.mark.asyncio
async def test_flash_gpt2(flash_gpt2, response_snapshot):
    response = await flash_gpt2.generate(
        "What is deep learning?",
        max_new_tokens=10,
        decoder_input_details=True,
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_gpt2_load(flash_gpt2, generate_load, response_snapshot):
    responses = await generate_load(
        flash_gpt2,
        "What is deep learning?",
        max_new_tokens=10,
        n=4,
    )

    generated_texts = [r.generated_text for r in responses]

    assert len(generated_texts) == 4
    assert all(
        [text == generated_texts[0] for text in generated_texts]
    ), generated_texts

    assert responses == response_snapshot
