import pytest


@pytest.fixture(scope="module")
def flash_qwen2_5_vl_handle(launcher):
    with launcher("Qwen/Qwen2.5-VL-3B-Instruct") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_qwen2_5(flash_qwen2_5_vl_handle):
    await flash_qwen2_5_vl_handle.health(300)
    return flash_qwen2_5_vl_handle.client


@pytest.mark.private
async def test_flash_qwen2_5_vl_simple(flash_qwen2_5, response_snapshot):
    response = await flash_qwen2_5.chat(
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
                    {"type": "text", "text": "Describe the image"},
                ],
            },
        ],
    )

    assert (
        response.choices[0].message.content
        == "The image depicts an anthropomorphic rabbit character wearing an intricate space suit, which includes a helmet with a starry face pattern and multiple suitors. The rabbit's ears are significantly large and upright, and it has a hitchhiker-like star antennas on its chest. The background is a reddish-orange, rocky landscape, suggesting a Martian environment. The suit has various buttons, a red button on the chest, and a reflective or illuminated dome on the head. The overall color scheme is dominated by shades of red, orange, and gray, giving a sense of a rugged, otherworldly setting."
    )

    assert response == response_snapshot


@pytest.mark.private
async def test_flash_qwen2_5_vl_simple_streaming(flash_qwen2_5, response_snapshot):
    responses = await flash_qwen2_5.chat(
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
                    {"type": "text", "text": "Describe the image"},
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
        == "The image depicts an anthropomorphic rabbit character wearing an intricate space suit, which includes a helmet with a starry face pattern and multiple suitors. The rabbit's ears are significantly large and upright, and it has a hitchhiker-like star antennas on its chest. The background is a reddish-orange, rocky landscape, suggesting a Martian environment. The suit has various buttons, a red button on the chest, and a reflective or illuminated dome on the head. The overall color scheme is dominated by shades of red, orange, and gray, giving a sense of a rugged, otherworldly setting."
    )
    assert count == 121
    assert last_response == response_snapshot


@pytest.mark.private
async def test_flash_qwen2_5_vl_bay(flash_qwen2_5, response_snapshot):
    response = await flash_qwen2_5.chat(
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
async def test_flash_qwen2_5_vl_inpaint(flash_qwen2_5, response_snapshot):
    response = await flash_qwen2_5.chat(
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
