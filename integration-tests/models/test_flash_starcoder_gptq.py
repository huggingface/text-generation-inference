import pytest


@pytest.fixture(scope="module")
def flash_santacoder_gptq_handle(launcher):
    with launcher("Narsil/starcoder-gptq", num_shard=2, quantize="gptq") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_santacoder_gptq(flash_santacoder_gptq_handle):
    await flash_santacoder_gptq_handle.health(300)
    return flash_santacoder_gptq_handle.client


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_santacoder_gptq(flash_santacoder_gptq, response_snapshot):
    response = await flash_santacoder_gptq.generate(
        'def sum(L: List[int]):\n"""Sums all elements from the list L."""', max_new_tokens=40, decoder_input_details=True
    )

    # assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_santacoder_gptq_all_params(flash_santacoder_gptq, response_snapshot):
    response = await flash_santacoder_gptq.generate(
        'def sum(L: List[int]):\n"""Sums all elements from the list L."""',
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

    #assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_santacoder_gptq_load(flash_santacoder_gptq, generate_load, response_snapshot):
    responses = await generate_load(flash_santacoder_gptq, 'def sum(L: List[int]):\n"""Sums all elements from the list L."""', max_new_tokens=10, n=4)

    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot