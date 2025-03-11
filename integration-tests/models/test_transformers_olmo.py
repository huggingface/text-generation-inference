import pytest


@pytest.fixture(scope="module")
def flash_llama_handle(launcher):
    with launcher("allenai/OLMo-7B-0724-Instruct-hf", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama(flash_llama_handle):
    await flash_llama_handle.health(300)
    return flash_llama_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_simple(flash_llama, response_snapshot):
    response = await flash_llama.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response.generated_text == ':\n\n```json\n{\n  "'
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_load(flash_llama, generate_load, response_snapshot):
    responses = await generate_load(flash_llama, "Test request", max_new_tokens=10, n=4)

    assert len(responses) == 4
    assert responses[0].generated_text == ':\n\n```json\n{\n  "'
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
