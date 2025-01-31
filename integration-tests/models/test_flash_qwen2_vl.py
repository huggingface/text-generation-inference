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
        == "The image depicts an anthropomorphic rabbit, wearing a spacesuit, standing in a barren, rocky landscape that resembles the surface of another planet, possibly Mars. The rabbit has a red digestive system label on its chest, and the surrounding environment features red sandy terrain and a hazy, floating planet or moon in the background. The scene has a surreal, fantastical quality, blending elements of science fiction and space exploration with a whimsical character."
    )

    assert response == response_snapshot


@pytest.mark.private
async def test_flash_qwen2_vl_simple_streaming(flash_qwen2, response_snapshot):
    responses = await flash_qwen2.chat(
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
        == "The image depicts an anthropomorphic rabbit, wearing a spacesuit, standing in a barren, rocky landscape that resembles the surface of another planet, possibly Mars. The rabbit has a red digestive system label on its chest, and the surrounding environment features red sandy terrain and a hazy, floating planet or moon in the background. The scene has a surreal, fantastical quality, blending elements of science fiction and space exploration with a whimsical character."
    )
    assert count == 89
    assert last_response == response_snapshot


@pytest.mark.private
async def test_flash_qwen2_vl_bay(flash_qwen2, response_snapshot):
    response = await flash_qwen2.chat(
        seed=42,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                        },
                    },
                    {"type": "text", "text": "Describe the image"},
                ],
            },
        ],
    )
    assert response == response_snapshot


@pytest.mark.private
async def test_flash_qwen2_vl_inpaint(flash_qwen2, response_snapshot):
    response = await flash_qwen2.chat(
        seed=42,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-inpaint.png"
                        },
                    },
                    {"type": "text", "text": "Describe the image"},
                ],
            },
        ],
    )
    assert response == response_snapshot
