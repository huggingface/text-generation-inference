import pytest


@pytest.fixture(scope="module")
def flash_pali_gemma_handle(launcher):
    with launcher(
        "google/paligemma2-3b-pt-224",
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_pali_gemma(flash_pali_gemma_handle):
    await flash_pali_gemma_handle.health(300)
    return flash_pali_gemma_handle.client


async def test_flash_pali_gemma_image(flash_pali_gemma, response_snapshot):
    car_image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    response = await flash_pali_gemma.generate(
        f"![]({car_image})",
        max_new_tokens=20,
    )
    assert (
        response.generated_text
        == "\nBrown\nCar\nColor\nCool\nDecor\nGreen\n...\n...\n...\n..."
    )

    assert response == response_snapshot
