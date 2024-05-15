import pytest


@pytest.fixture(scope="module")
def flash_llava_next_handle(launcher):
    with launcher("llava-hf/llava-v1.6-mistral-7b-hf") as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llava_chat(flash_llava_next_handle):
    await flash_llava_next_handle.health(3000)
    return flash_llava_next_handle.inference_client


@pytest.mark.private
async def test_flash_llava_simple(flash_llava_chat, response_snapshot):
    response = await flash_llava_chat.chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Whats in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
                        },
                    },
                ],
            },
        ],
        seed=42,
        max_tokens=100,
    )

    assert (
        response.choices[0].message.content
        == " The image you've provided features an anthropomorphic rabbit in spacesuit attire. This rabbit is depicted with human-like posture and movement, standing on a rocky terrain with a vast, reddish-brown landscape in the background. The spacesuit is detailed with mission patches, circuitry, and a helmet that covers the rabbit's face and ear, with an illuminated red light on the chest area.\n\nThe artwork style is that of a"
    )
    assert response == response_snapshot
