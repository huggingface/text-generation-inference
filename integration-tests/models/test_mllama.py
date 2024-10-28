import pytest
import asyncio


@pytest.fixture(scope="module")
def mllama_handle(launcher):
    with launcher("meta-llama/Llama-3.2-11B-Vision-Instruct", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def mllama(mllama_handle):
    await mllama_handle.health(300)
    return mllama_handle.client


@pytest.mark.asyncio
async def test_mllama_simpl(mllama, response_snapshot):
    response = await mllama.chat(
        max_tokens=10,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Can you tell me a very short story based on the image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/huggingface/text-generation-inference/main/integration-tests/images/chicken_on_money.png"
                        },
                    },
                ],
            },
        ],
    )

    assert response.usage == {
        "completion_tokens": 10,
        "prompt_tokens": 50,
        "total_tokens": 60,
    }
    assert (
        response.choices[0].message.content
        == "In a bustling city, a chicken named Cluck"
    )
    assert response == response_snapshot


@pytest.mark.release
@pytest.mark.asyncio
async def test_mllama_load(mllama, generate_load, response_snapshot):
    futures = [
        mllama.chat(
            max_tokens=10,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Can you tell me a very short story based on the image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://raw.githubusercontent.com/huggingface/text-generation-inference/main/integration-tests/images/chicken_on_money.png"
                            },
                        },
                    ],
                },
            ],
        )
        for i in range(4)
    ]
    responses = await asyncio.gather(*futures)

    _ = [response.choices[0].message.content for response in responses]

    # XXX: TODO: Fix this test.
    # assert generated_texts[0] == "In a bustling city, a chicken named Cluck"
    # assert len(generated_texts) == 4
    # assert generated_texts, all(
    #     [text == generated_texts[0] for text in generated_texts]
    # )
    # assert responses == response_snapshot
