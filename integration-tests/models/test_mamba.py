import pytest


@pytest.fixture(scope="module")
def fused_kernel_mamba_handle(launcher):
    with launcher("state-spaces/mamba-130m", num_shard=1) as handle:
        yield handle


@pytest.fixture(scope="module")
async def fused_kernel_mamba(fused_kernel_mamba_handle):
    await fused_kernel_mamba_handle.health(300)
    return fused_kernel_mamba_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_mamba(fused_kernel_mamba, response_snapshot):
    response = await fused_kernel_mamba.generate(
        "What is Deep Learning?", max_new_tokens=10
    )

    assert response.details.generated_tokens == 10
    assert response.generated_text == "\n\nDeep learning is a new type of machine"
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_mamba_all_params(fused_kernel_mamba, response_snapshot):
    response = await fused_kernel_mamba.generate(
        "blue, red, yellow, ",
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
    assert (
        response.generated_text
        == "blue, red, yellow, \nand blue colors. A number of different color"
    )
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_mamba_load(
    fused_kernel_mamba, generate_load, generous_response_snapshot
):
    responses = await generate_load(
        fused_kernel_mamba, "What is Deep Learning?", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])
    assert responses[0].generated_text == "\n\nDeep learning is a new type of machine"

    assert responses == generous_response_snapshot
