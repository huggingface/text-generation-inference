import pytest


@pytest.fixture(scope="module")
def compressed_tensors_w8an_handle(launcher):
    with launcher(
        "neuralmagic/Llama-3.2-1B-Instruct-FP8",
        num_shard=2,
        quantize="compressed-tensors",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def compressed_tensors_w8an(compressed_tensors_w8an_handle):
    await compressed_tensors_w8an_handle.health(300)
    return compressed_tensors_w8an_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_compressed_tensors_w8an(compressed_tensors_w8an, response_snapshot):
    response = await compressed_tensors_w8an.generate(
        "What is deep learning?",
        max_new_tokens=10,
        decoder_input_details=True,
    )

    assert (
        response.generated_text
        == " Deep learning is a type of artificial intelligence (AI"
    )
    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_compressed_tensors_w8an_all_params(
    compressed_tensors_w8an, response_snapshot
):
    response = await compressed_tensors_w8an.generate(
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
        == "What is deep learning?\nDeep learning, also known as neural network or"
    )
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_compressed_tensors_w8an_load(
    compressed_tensors_w8an, generate_load, response_snapshot
):
    responses = await generate_load(
        compressed_tensors_w8an,
        "What is deep learning?",
        max_new_tokens=10,
        n=4,
    )

    assert (
        responses[0].generated_text
        == " Deep learning is a type of artificial intelligence (AI"
    )
    assert len(responses) == 4
    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot
