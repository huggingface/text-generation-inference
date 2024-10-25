import pytest


@pytest.fixture(scope="module")
def flash_llama_fp8_kv_cache_handle(launcher):
    with launcher(
        "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
        num_shard=2,
        kv_cache_dtype="fp8_e4m3fn",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_fp8_kv_cache(flash_llama_fp8_kv_cache_handle):
    await flash_llama_fp8_kv_cache_handle.health(300)
    return flash_llama_fp8_kv_cache_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_fp8_kv_cache(flash_llama_fp8_kv_cache, response_snapshot):
    response = await flash_llama_fp8_kv_cache.generate(
        "What is deep learning?", max_new_tokens=10, decoder_input_details=True
    )

    assert (
        response.generated_text
        == " Deep learning is a subset of machine learning that involves"
    )
    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_fp8_kv_cache_all_params(
    flash_llama_fp8_kv_cache, response_snapshot
):
    response = await flash_llama_fp8_kv_cache.generate(
        "What is deep learning?",
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
async def test_flash_llama_fp8_kv_cache_load(
    flash_llama_fp8_kv_cache, generate_load, response_snapshot
):
    responses = await generate_load(
        flash_llama_fp8_kv_cache, "What is deep learning?", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert (
        responses[0].generated_text
        == " Deep learning is a subset of machine learning that involves"
    )
    assert all(
        [r.generated_text == responses[0].generated_text for r in responses]
    ), f"Different messages : {[r.generated_text for r in responses]}"
    assert responses == response_snapshot
