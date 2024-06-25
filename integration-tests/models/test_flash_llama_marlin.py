import pytest


@pytest.fixture(scope="module")
def flash_llama_marlin_handle(launcher):
    with launcher(
        "neuralmagic/llama-2-7b-chat-marlin", num_shard=2, quantize="marlin"
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_marlin(flash_llama_marlin_handle):
    await flash_llama_marlin_handle.health(300)
    return flash_llama_marlin_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_marlin(flash_llama_marlin, response_snapshot):
    response = await flash_llama_marlin.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_marlin_all_params(flash_llama_marlin, response_snapshot):
    response = await flash_llama_marlin.generate(
        "Test request",
        max_new_tokens=10,
        repetition_penalty=1.2,
        return_full_text=True,
        temperature=0.5,
        top_p=0.9,
        top_k=10,
        truncate=5,
        typical_p=0.9,
        watermark=True,
        decoder_input_details=True,
        seed=0,
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_marlin_load(
    flash_llama_marlin, generate_load, response_snapshot
):
    responses = await generate_load(
        flash_llama_marlin, "Test request", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
