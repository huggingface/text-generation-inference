import pytest


@pytest.fixture(scope="module")
def flash_mistral_handle(launcher):
    with launcher("mistralai/Mistral-7B-Instruct-v0.1") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_mistral(flash_mistral_handle):
    await flash_mistral_handle.health(300)
    return flash_mistral_handle.client


@pytest.mark.asyncio
async def test_flash_mistral(flash_mistral, response_snapshot):
    response = await flash_mistral.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response.generated_text == ": Let n = 10 - 1"
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_mistral_all_params(flash_mistral, response_snapshot):
    response = await flash_mistral.generate(
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

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.asyncio
async def test_flash_mistral_load(flash_mistral, generate_load, response_snapshot):
    responses = await generate_load(
        flash_mistral, "Test request", max_new_tokens=10, n=4
    )

    assert len(responses) == 4
    assert all(
        [r.generated_text == responses[0].generated_text for r in responses]
    ), f"{[r.generated_text  for r in responses]}"
    assert responses[0].generated_text == ": Let n = 10 - 1"

    assert responses == response_snapshot
