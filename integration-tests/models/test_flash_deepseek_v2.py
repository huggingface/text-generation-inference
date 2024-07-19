import pytest


@pytest.fixture(scope="module")
def flash_deepseek_v2_handle(launcher):
    with launcher("deepseek-ai/DeepSeek-V2-Lite", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_deepseek_v2(flash_deepseek_v2_handle):
    await flash_deepseek_v2_handle.health(300)
    return flash_deepseek_v2_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_deepseek_v2(flash_deepseek_v2, response_snapshot):
    response = await flash_deepseek_v2.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_deepseek_v2_all_params(flash_deepseek_v2, response_snapshot):
    response = await flash_deepseek_v2.generate(
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
async def test_flash_deepseek_v2_load(
    flash_deepseek_v2, generate_load, response_snapshot
):
    responses = await generate_load(
        flash_deepseek_v2, "Test request", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
