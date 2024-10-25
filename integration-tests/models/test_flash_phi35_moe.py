import pytest


@pytest.fixture(scope="module")
def flash_phi35_moe_handle(launcher):
    with launcher(
        "microsoft/Phi-3.5-MoE-instruct",
        num_shard=4,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_phi35_moe(flash_phi35_moe_handle):
    await flash_phi35_moe_handle.health(300)
    return flash_phi35_moe_handle.client


@pytest.mark.asyncio
async def test_flash_phi35_moe(flash_phi35_moe, response_snapshot):
    response = await flash_phi35_moe.generate(
        "What is gradient descent?\n\n", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert (
        response.generated_text
        == "Gradient descent is an optimization algorithm commonly used in"
    )
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_phi35_moe_all_params(flash_phi35_moe, response_snapshot):
    response = await flash_phi35_moe.generate(
        "What is gradient descent?\n",
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
        == "What is gradient descent?\nGradient Descent (GD) is an"
    )
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_phi35_moe_load(flash_phi35_moe, generate_load, response_snapshot):
    responses = await generate_load(
        flash_phi35_moe, "What is gradient descent?\n\n", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert responses[0].details.generated_tokens == 10
    assert (
        responses[0].generated_text
        == "Gradient descent is an optimization algorithm commonly used in"
    )
    assert all(
        [r.generated_text == responses[0].generated_text for r in responses]
    ), f"{[r.generated_text  for r in responses]}"

    assert responses == response_snapshot
