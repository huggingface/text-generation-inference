import pytest


@pytest.fixture(scope="module")
def flash_qwen2_vl_handle(launcher):
    with launcher(
        "Qwen/Qwen2-VL-2B-Instruct",
        max_input_length=40,
        max_batch_prefill_tokens=50,
        max_total_tokens=51,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_qwen2(flash_qwen2_vl_handle):
    await flash_qwen2_vl_handle.health(300)
    return flash_qwen2_vl_handle.client


@pytest.mark.private
async def test_flash_qwen2_vl_simple(flash_qwen2, response_snapshot):
    response = await flash_qwen2.chat(
        max_tokens=20,
        seed=42,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the color of the sky?"},
                ],
            },
        ],
    )

    assert response.choices[0].message.content == "The correct answer is: blue"

    assert response == response_snapshot
