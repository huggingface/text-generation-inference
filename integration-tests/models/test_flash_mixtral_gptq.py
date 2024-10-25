import pytest


@pytest.fixture(scope="module")
def flash_mixtral_gptq_handle(launcher):
    with launcher(
        "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
        revision="gptq-4bit-128g-actorder_True",
        num_shard=2,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_mixtral_gptq(flash_mixtral_gptq_handle):
    await flash_mixtral_gptq_handle.health(300)
    return flash_mixtral_gptq_handle.client


@pytest.mark.asyncio
async def test_flash_mixtral_gptq(flash_mixtral_gptq, response_snapshot):
    response = await flash_mixtral_gptq.generate(
        "What is deep learning?", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert (
        response.generated_text == "\n\nDeep learning is a subset of machine learning"
    )

    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_mixtral_gptq_all_params(flash_mixtral_gptq, response_snapshot):
    response = await flash_mixtral_gptq.generate(
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

    assert response.details.generated_tokens == 10
    assert (
        response.generated_text
        == "What is deep learning?\nDeep Learning is a subset of Machine Learning,"
    )
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_mixtral_gptq_load(
    flash_mixtral_gptq, generate_load, response_snapshot
):
    responses = await generate_load(
        flash_mixtral_gptq, "What is deep learning?", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert (
        responses[0].generated_text
        == "\n\nDeep learning is a subset of machine learning"
    )
    assert all(
        [r.generated_text == responses[0].generated_text for r in responses]
    ), f"{[r.generated_text  for r in responses]}"

    assert responses == response_snapshot
