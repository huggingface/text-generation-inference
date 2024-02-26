import pytest


@pytest.fixture(scope="module")
def flash_medusa_handle(launcher):
    with launcher(
        "FasterDecoding/medusa-vicuna-7b-v1.3", num_shard=2, revision="refs/pr/1"
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_medusa(flash_medusa_handle):
    await flash_medusa_handle.health(300)
    return flash_medusa_handle.client


@pytest.mark.asyncio
async def test_flash_medusa_simple(flash_medusa, response_snapshot):
    response = await flash_medusa.generate(
        "What is Deep Learning?", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_medusa_all_params(flash_medusa, response_snapshot):
    response = await flash_medusa.generate(
        "What is Deep Learning?",
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
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_medusa_load(flash_medusa, generate_load, response_snapshot):
    responses = await generate_load(
        flash_medusa, "What is Deep Learning?", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all(
        [r.generated_text == responses[0].generated_text for r in responses]
    ), f"{[r.generated_text for r in responses]}"
    assert (
        responses[0].generated_text == "\nDeep learning is a subset of machine learning"
    )

    assert responses == response_snapshot
