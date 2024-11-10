import pytest


@pytest.fixture(scope="module")
def compressed_tensors_wna16_handle(launcher):
    with launcher(
        "neuralmagic/gemma-2-2b-it-quantized.w4a16",
        num_shard=2,
        quantize="compressed-tensors",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def compressed_tensors_wna16(compressed_tensors_wna16_handle):
    await compressed_tensors_wna16_handle.health(300)
    return compressed_tensors_wna16_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_compressed_tensors_wna16(compressed_tensors_wna16, response_snapshot):
    response = await compressed_tensors_wna16.generate(
        "What is deep learning?",
        max_new_tokens=10,
        decoder_input_details=True,
    )

    assert (
        response.generated_text
        == "\n\nDeep learning is a subset of machine learning that"
    )
    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_compressed_tensors_wna16_all_params(
    compressed_tensors_wna16, response_snapshot
):
    response = await compressed_tensors_wna16.generate(
        "What is deep learning",
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
        == "What is deep learning?\n\nDeep Learning is a subset of machine learning"
    )
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_compressed_tensors_wna16_load(
    compressed_tensors_wna16, generate_load, response_snapshot
):
    responses = await generate_load(
        compressed_tensors_wna16,
        "What is deep learning?",
        max_new_tokens=10,
        n=4,
    )

    assert (
        responses[0].generated_text
        == "\n\nDeep learning is a subset of machine learning that"
    )
    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
