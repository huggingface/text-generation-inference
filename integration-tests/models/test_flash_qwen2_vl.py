import pytest


@pytest.fixture(scope="module")
def flash_qwen2_vl_handle(launcher):
    with launcher("Qwen/Qwen2-VL-7B-Instruct") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_qwen2(flash_qwen2_vl_handle):
    await flash_qwen2_vl_handle.health(300)
    return flash_qwen2_vl_handle.client


@pytest.mark.private
async def test_flash_qwen2_vl_simple(flash_qwen2, response_snapshot):
    response = await flash_qwen2.chat(
        max_tokens=100,
        seed=42,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
                        },
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ],
    )

    assert (
        response.choices[0].message.content
        == "The image depicts an anthropomorphic rabbit, wearing a futuristic spacesuit, in an extraterrestrial environment. The setting appears to be a red planet resembling Mars, with rugged terrain and rocky formations in the background. The moon is visible in the distant sky, adding to the lunar landscape."
    )

    assert response == response_snapshot


@pytest.mark.private
async def test_flash_qwen2_vl_simple_streaming(flash_qwen2, response_snapshot):
    responses = await flash_qwen2.chat(
        max_tokens=100,
        seed=42,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
                        },
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ],
        stream=True,
    )

    count = 0
    generated = ""
    last_response = None
    async for response in responses:
        count += 1
        generated += response.choices[0].delta.content
        last_response = response

    assert (
        generated
        == "The image depicts an anthropomorphic rabbit, wearing a futuristic spacesuit, in an extraterrestrial environment. The setting appears to be a red planet resembling Mars, with rugged terrain and rocky formations in the background. The moon is visible in the distant sky, adding to the lunar landscape."
    )
    assert count == 58
    assert last_response == response_snapshot
