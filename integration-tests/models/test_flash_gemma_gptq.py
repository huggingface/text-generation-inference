import pytest


@pytest.fixture(scope="module")
def flash_gemma_gptq_handle(launcher):
    with launcher("TechxGenus/gemma-2b-GPTQ", num_shard=1, quantize="gptq") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_gemma_gptq(flash_gemma_gptq_handle):
    await flash_gemma_gptq_handle.health(300)
    return flash_gemma_gptq_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_gemma_gptq(flash_gemma_gptq, ignore_logprob_response_snapshot):
    response = await flash_gemma_gptq.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response == ignore_logprob_response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_gemma_gptq_all_params(
    flash_gemma_gptq, ignore_logprob_response_snapshot
):
    response = await flash_gemma_gptq.generate(
        "Test request",
        max_new_tokens=10,
        repetition_penalty=1.2,
        return_full_text=True,
        stop_sequences=["test"],
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
    assert response == ignore_logprob_response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_gemma_gptq_load(
    flash_gemma_gptq, generate_load, ignore_logprob_response_snapshot
):
    responses = await generate_load(
        flash_gemma_gptq, "Test request", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == ignore_logprob_response_snapshot
