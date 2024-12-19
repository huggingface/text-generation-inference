import pytest


@pytest.fixture(scope="module")
def flash_llama_fp8_handle(launcher):
    with launcher("meta-llama/Meta-Llama-3-8B", num_shard=2, quantize="fp8") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_fp8(flash_llama_fp8_handle):
    await flash_llama_fp8_handle.health(300)
    return flash_llama_fp8_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_fp8(flash_llama_fp8, response_snapshot):
    response = await flash_llama_fp8.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.generated_text == " for the 2019-2020 school year"
    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_fp8_all_params(flash_llama_fp8, response_snapshot):
    response = await flash_llama_fp8.generate(
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

    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_fp8_load(flash_llama_fp8, generate_load, response_snapshot):
    responses = await generate_load(
        flash_llama_fp8, "Test request", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert responses[0].generated_text == " for the 2019-2020 school year"
    assert all(
        [r.generated_text == responses[0].generated_text for r in responses]
    ), f"Different messages : {[r.generated_text for r in responses]}"
    assert responses == response_snapshot
