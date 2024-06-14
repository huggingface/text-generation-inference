import pytest

from testing_utils import SYSTEM, is_flaky_async, require_backend_async, require_backend


@pytest.fixture(scope="module")
@require_backend("cuda", "rocm")
def flash_llama_awq_handle_sharded(launcher):
    if SYSTEM == "rocm":
        # On ROCm, for awq checkpoints, we need to use gptq kernel that supports ROCm.
        quantize = "gptq"
    else:
        quantize = "awq"

    with launcher(
        "abhinavkulkarni/codellama-CodeLlama-7b-Python-hf-w4-g128-awq",
        num_shard=2,
        quantize=quantize,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
@require_backend_async("cuda", "rocm")
async def flash_llama_awq_sharded(flash_llama_awq_handle_sharded):
    await flash_llama_awq_handle_sharded.health(300)
    return flash_llama_awq_handle_sharded.client


@is_flaky_async(max_attempts=5)
@pytest.mark.asyncio
@require_backend_async("cuda", "rocm")
async def test_flash_llama_awq_sharded(flash_llama_awq_sharded, response_snapshot):
    response = await flash_llama_awq_sharded.generate(
        "What is Deep Learning?", max_new_tokens=10, decoder_input_details=True
    )

    # ExllamaV2 (which may be used as an AWQ backend) is highly non-deterministic, see for reference https://github.com/turboderp/exllamav2/issues/232.
    assert response.details.generated_tokens == 10

    assert (
        response.generated_text
        == "\nWhat is the difference between Deep Learning and Machine"
    )

    if SYSTEM != "rocm":
        # Logits were taken on an Nvidia GPU, and are too far off to be meaningfully compared.
        assert response == response_snapshot


@require_backend_async("cuda")
@pytest.mark.asyncio
async def test_flash_llama_awq_load_sharded(
    flash_llama_awq_sharded, generate_load, response_snapshot
):
    # TODO: This test is highly non-deterministic on ROCm.

    responses = await generate_load(
        flash_llama_awq_sharded, "What is Deep Learning?", max_new_tokens=10, n=4
    )

    assert all(
        [
            r.generated_text
            == "\nWhat is the difference between Deep Learning and Machine"
            for r in responses
        ]
    )

    # Logits were taken on an Nvidia GPU, and are too far off to be meaningfully compared.
    assert responses == response_snapshot
