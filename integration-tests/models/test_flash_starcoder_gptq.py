import pytest

from testing_utils import SYSTEM, is_flaky_async, require_backend_async


@pytest.fixture(scope="module")
def flash_starcoder_gptq_handle(launcher):
    with launcher("Narsil/starcoder-gptq", num_shard=2, quantize="gptq") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_starcoder_gptq(flash_starcoder_gptq_handle):
    await flash_starcoder_gptq_handle.health(300)
    return flash_starcoder_gptq_handle.client


@pytest.mark.asyncio
@is_flaky_async(max_attempts=10)
async def test_flash_starcoder_gptq(flash_starcoder_gptq, generous_response_snapshot):
    response = await flash_starcoder_gptq.generate(
        "def geometric_mean(L: List[float]):",
        max_new_tokens=20,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 20
    assert (
        response.generated_text
        == '\n    """\n    Calculate the geometric mean of a list of numbers.\n\n    :param L: List'
    )

    if SYSTEM != "rocm":
        assert response == generous_response_snapshot


@pytest.mark.asyncio
@is_flaky_async(max_attempts=10)
async def test_flash_starcoder_gptq_default_params(
    flash_starcoder_gptq, generous_response_snapshot
):
    response = await flash_starcoder_gptq.generate(
        "def geometric_mean(L: List[float]):",
        max_new_tokens=20,
        temperature=0.2,
        top_p=0.95,
        decoder_input_details=True,
        seed=0,
    )
    assert response.details.generated_tokens == 20
    assert (
        response.generated_text == "\n    return reduce(lambda x, y: x * y, L) ** (1.0"
    )

    if SYSTEM != "rocm":
        assert response == generous_response_snapshot


@pytest.mark.asyncio
@require_backend_async("cuda")
async def test_flash_starcoder_gptq_load(
    flash_starcoder_gptq, generate_load, generous_response_snapshot
):
    # TODO: exllamav2 gptq kernel is highly non-deterministic on ROCm.

    responses = await generate_load(
        flash_starcoder_gptq,
        "def geometric_mean(L: List[float]):",
        max_new_tokens=10,
        n=4,
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == generous_response_snapshot
