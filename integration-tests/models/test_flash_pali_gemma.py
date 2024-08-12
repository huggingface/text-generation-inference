import pytest
import requests
import io
import base64


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


def get_cow_beach():
    with open("integration-tests/images/cow_beach.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return f"data:image/png;base64,{encoded_string.decode('utf-8')}"


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_pali_gemma(flash_pali_gemma, response_snapshot):
    cow = get_cow_beach()
    inputs = f"![]({cow})Where is the cow standing?\n"
    response = await flash_pali_gemma.generate(inputs, max_new_tokens=20)

    assert response.generated_text == "beach"
    assert response == response_snapshot
