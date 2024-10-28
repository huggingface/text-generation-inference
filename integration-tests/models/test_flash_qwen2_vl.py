import pytest


@pytest.fixture(scope="module")
def flash_qwen2_vl_handle(launcher):
    with launcher(
        "Qwen/Qwen2-VL-7B-Instruct",
        max_batch_prefill_tokens=2000,
        max_input_length=2000,
        max_total_tokens=2001,
        cuda_graphs=[0],
    ) as handle:
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
        == "The image shows a rabbit with a is on floating in outer a a in outer and seems a as an in the be an astronaut suit a a a have crew the front ag a suit the chalet"
    )

    # # TODO: return reference response
    # assert (
    #     response.choices[0].message.content
    #     == "The image depicts an astronaut with a rabbit's head standing on a rocky, reddish terrain. The astronaut is wearing a space suit with various buttons and"
    # )

    assert response == response_snapshot
