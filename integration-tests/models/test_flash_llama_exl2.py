import pytest


@pytest.fixture(scope="module")
def flash_llama_exl2_handle(launcher):
    with launcher(
        "turboderp/Llama-3-8B-Instruct-exl2",
        revision="2.5bpw",
        # Set max input length to avoid OOM due to extremely large
        # scratch buffer.
        max_input_length=1024,
        num_shard=1,
        quantize="exl2",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_exl2(flash_llama_exl2_handle):
    await flash_llama_exl2_handle.health(300)
    return flash_llama_exl2_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_exl2(flash_llama_exl2, ignore_logprob_response_snapshot):
    response = await flash_llama_exl2.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response == ignore_logprob_response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_exl2_all_params(
    flash_llama_exl2, ignore_logprob_response_snapshot
):
    response = await flash_llama_exl2.generate(
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

    assert (
        response.generated_text == 'Test request. The server responds with a "200 OK"'
    )
    assert response == ignore_logprob_response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_exl2_load(
    flash_llama_exl2, generate_load, ignore_logprob_response_snapshot
):
    responses = await generate_load(
        flash_llama_exl2, "Test request", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == ignore_logprob_response_snapshot
