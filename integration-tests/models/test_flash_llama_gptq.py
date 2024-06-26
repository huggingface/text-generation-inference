import pytest

from testing_utils import is_flaky_async, SYSTEM, require_backend_async


@pytest.fixture(scope="module")
def flash_llama_gptq_handle(launcher):
    with launcher("huggingface/llama-7b-gptq", num_shard=2, quantize="gptq") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_gptq(flash_llama_gptq_handle):
    await flash_llama_gptq_handle.health(300)
    return flash_llama_gptq_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
@is_flaky_async(max_attempts=5)
async def test_flash_llama_gptq(flash_llama_gptq, response_snapshot):
    response = await flash_llama_gptq.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response.generated_text == "\nTest request\nTest request\nTest request\n"

    if SYSTEM != "rocm":
        # Logits were taken on an Nvidia GPU, and are too far off to be meaningfully compared.
        assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
@require_backend_async("cuda")
async def test_flash_llama_gptq_all_params(flash_llama_gptq, response_snapshot):
    # TODO: investigate why exllamav2 gptq kernel is this much more non-deterministic on ROCm vs on CUDA.

    response = await flash_llama_gptq.generate(
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
@require_backend_async("cuda")
async def test_flash_llama_gptq_load(
    flash_llama_gptq, generate_load, response_snapshot
):
    # TODO: investigate why exllamav2 gptq kernel is this much more non-deterministic on ROCm vs on CUDA.

    responses = await generate_load(
        flash_llama_gptq, "Test request", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
