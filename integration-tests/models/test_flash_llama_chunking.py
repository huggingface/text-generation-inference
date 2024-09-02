import pytest


@pytest.fixture(scope="module")
def flash_llama_handle(launcher):
    with launcher(
        "huggingface/llama-7b", num_shard=2, max_batch_prefill_tokens=2
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama(flash_llama_handle):
    await flash_llama_handle.health(300)
    return flash_llama_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama(flash_llama, response_snapshot):
    response = await flash_llama.generate("What is Deep Learning ?", max_new_tokens=10)

    assert response.details.generated_text == "xx"
    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_load(flash_llama, generate_load, response_snapshot):
    responses = await generate_load(
        flash_llama, "What is Deep Learning ?", max_new_tokens=10, n=4
    )
    assert responses[0].details.generated_text == "xx"
    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
