import pytest


@pytest.fixture(scope="module")
def flash_gemma3_handle(launcher):
    with launcher("gg-hf-g/gemma-3-4b-it", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_gemma3(flash_gemma3_handle):
    await flash_gemma3_handle.health(300)
    return flash_gemma3_handle.client


async def test_flash_gemma3(flash_gemma3, response_snapshot):
    response = await flash_gemma3.generate(
        "Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
        seed=42,
        max_new_tokens=100,
    )

    assert (
        response.generated_text
        == " people died in the United States.\n\nThe generally accepted estimate is that 675,000 people died in the United States. However, some historians believe the actual number could be as high as 10 million.\n\nI am looking for more information on this discrepancy and the factors that contributed to the wide range of estimates.\n\nHere's a breakdown of the factors contributing to the wide range of estimates for the 1918 flu pandemic death toll in the United States"
    )
    assert response.details.generated_tokens == 100
    assert response == response_snapshot


async def test_flash_gemma3_image_cow_dog(flash_gemma3, response_snapshot):
    image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
    response = await flash_gemma3.chat(
        seed=42,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "What is the breed of the dog in the image?",
                    },
                ],
            },
        ],
        max_tokens=100,
    )

    assert (
        response.choices[0].message.content
        == "That's a fantastic question! However, the image doesn't show a dog. It shows a **Brown Swiss cow** standing on a beach. \n\nBrown Swiss cows are known for their reddish-brown color and distinctive white markings. \n\nIf you'd like, you can send me another image and I’ll do my best to identify it!"
    )
    assert response.usage["completion_tokens"] == 75
    assert response == response_snapshot


async def test_flash_gemma3_image_cow(flash_gemma3, response_snapshot):
    image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
    response = await flash_gemma3.chat(
        seed=42,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ],
        max_tokens=100,
    )
    assert (
        response.choices[0].message.content
        == "Here's a description of what's shown in the image:\n\nThe image depicts a brown cow standing on a sandy beach. The beach has turquoise water and a distant island visible in the background. The sky is bright blue with some white clouds. \n\nIt's a quite a humorous and unusual scene – a cow enjoying a day at the beach!"
    )
    assert response.usage["completion_tokens"] == 74
    assert response == response_snapshot


async def test_exceed_window(flash_gemma3, response_snapshot):
    response = await flash_gemma3.generate(
        "This is a nice place. " * 800 + "I really enjoy the scenery,",
        seed=42,
        max_new_tokens=20,
    )

    assert (
        response.generated_text
        == " the people, and the food.\n\nThis is a nice place.\n"
    )
    assert response.details.generated_tokens == 16
    assert response == response_snapshot
