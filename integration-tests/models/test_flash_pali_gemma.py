import pytest


@pytest.fixture(scope="module")
def flash_pali_gemma_handle(launcher):
    with launcher(
        "google/paligemma-3b-pt-224",
        num_shard=1,
        revision="float16",
        max_input_length=4000,
        max_total_tokens=4096,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_pali_gemma(flash_pali_gemma_handle):
    await flash_pali_gemma_handle.health(300)
    return flash_pali_gemma_handle.client


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_pali_gemma(flash_pali_gemma, response_snapshot, cow_beach):
    inputs = f"![]({cow_beach})Where is the cow standing?\n"
    response = await flash_pali_gemma.generate(inputs, max_new_tokens=20)

    assert response.generated_text == "beach"
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_pali_gemma_two_images(
    flash_pali_gemma, response_snapshot, chicken, cow_beach
):
    response = await flash_pali_gemma.generate(
        f"caption![]({chicken})![]({cow_beach})\n",
        max_new_tokens=20,
    )
    # Is PaliGemma not able to handle two separate images? At least we
    # get output showing that both images are used.
    assert (
        response.generated_text == "image result for chicken on the beach"
    ), f"{repr(response.generated_text)}"
    assert response == response_snapshot
