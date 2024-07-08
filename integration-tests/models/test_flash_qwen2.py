import pytest


@pytest.fixture(scope="module")
def flash_qwen2_handle(launcher):
    with launcher("Qwen/Qwen1.5-0.5B") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_qwen2(flash_qwen2_handle):
    await flash_qwen2_handle.health(300)
    return flash_qwen2_handle.client


@pytest.mark.release
@pytest.mark.asyncio
async def test_flash_qwen2(flash_qwen2, response_snapshot):
    response = await flash_qwen2.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response.generated_text == "\n# Create a request\nrequest = requests.get"
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
async def test_flash_qwen2_all_params(flash_qwen2, response_snapshot):
    response = await flash_qwen2.generate(
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


@pytest.mark.release
@pytest.mark.asyncio
async def test_flash_qwen2_load(flash_qwen2, generate_load, response_snapshot):
    responses = await generate_load(flash_qwen2, "Test request", max_new_tokens=10, n=4)

    assert len(responses) == 4
    assert all(
        [r.generated_text == responses[0].generated_text for r in responses]
    ), f"{[r.generated_text  for r in responses]}"
    assert responses[0].generated_text == "\n# Create a request\nrequest = requests.get"

    assert responses == response_snapshot
