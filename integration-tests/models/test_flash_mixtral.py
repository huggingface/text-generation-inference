import pytest


@pytest.fixture(scope="module")
def flash_mixtral_handle(launcher):
    with launcher("mistralai/Mixtral-8x7B-v0.1", num_shard=8) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_mixtral(flash_mixtral_handle):
    await flash_mixtral_handle.health(300)
    return flash_mixtral_handle.client


@pytest.mark.skip(reason="requires > 4 shards")
@pytest.mark.asyncio
async def test_flash_mixtral(flash_mixtral, response_snapshot):
    response = await flash_mixtral.generate(
        "What is gradient descent?\n\n", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert (
        response.generated_text
        == "Gradient descent is an optimization algorithm used to minimize"
    )
    assert response == response_snapshot


@pytest.mark.skip(reason="requires > 4 shards")
@pytest.mark.asyncio
async def test_flash_mixtral_all_params(flash_mixtral, response_snapshot):
    response = await flash_mixtral.generate(
        "What is gradient descent?\n\n",
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
        == "What is gradient descent?\n\nIt seems to me, that if you're"
    )
    assert response == response_snapshot


@pytest.mark.skip(reason="requires > 4 shards")
@pytest.mark.asyncio
async def test_flash_mixtral_load(flash_mixtral, generate_load, response_snapshot):
    responses = await generate_load(
        flash_mixtral, "What is gradient descent?\n\n", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert responses[0].details.generated_tokens == 10
    assert (
        responses[0].generated_text
        == "Gradient descent is an optimization algorithm used to minimize"
    )
    assert all(
        [r.generated_text == responses[0].generated_text for r in responses]
    ), f"{[r.generated_text  for r in responses]}"

    assert responses == response_snapshot
