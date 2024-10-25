import pytest


@pytest.fixture(scope="module")
def flash_llava_next_handle(launcher):
    with launcher(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        num_shard=4,
        max_input_length=4000,
        max_total_tokens=4096,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llava_next(flash_llava_next_handle):
    await flash_llava_next_handle.health(300)
    return flash_llava_next_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llava_next_simple(flash_llava_next, response_snapshot, chicken):
    response = await flash_llava_next.generate(
        f"User:![]({chicken})Can you tell me a very short story based on the image?",
        max_new_tokens=10,
    )
    assert (
        response.generated_text == "\n\nOnce upon a time, there was a"
    ), f"{repr(response.generated_text)}"
    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llava_next_all_params(flash_llava_next, response_snapshot):
    response = await flash_llava_next.generate(
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

    assert response.details.generated_tokens == 6
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llava_next_load(
    flash_llava_next, generate_load, response_snapshot, chicken
):
    responses = await generate_load(
        flash_llava_next,
        f"User:![]({chicken})Can you tell me a very short story based on the image?",
        max_new_tokens=10,
        n=4,
    )
    generated_texts = [r.generated_text for r in responses]
    assert generated_texts[0] == "\n\nOnce upon a time, there was a"
    assert len(generated_texts) == 4
    assert all([r.generated_text == generated_texts[0] for r in responses])

    assert responses == response_snapshot
