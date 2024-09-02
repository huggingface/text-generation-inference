import pytest


@pytest.fixture(scope="module")
def flash_llama_chat_handle(launcher):
    with launcher(
        "microsoft/Phi-3.5-MoE-instruct", 
        num_shard=4,
        cuda_graphs=[1, 2]
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_chat(flash_llama_chat_handle):
    await flash_llama_chat_handle.health(300)
    return flash_llama_chat_handle.client


@pytest.mark.private
async def test_flash_llama_simple(flash_llama_chat, response_snapshot):
    response = await flash_llama_chat.chat(
        max_tokens=40,
        seed=1337,
        messages=[
            {
                "role": "system",
                "content": "Youre a helpful assistant! Answer the users question best you can.",
            },
            {
                "role": "user",
                "content": "What is the weather like in Brooklyn, New York?",
            },
        ],
    )

    print(repr(response.choices[0].message.content))
    assert (
        response.choices[0].message.content
        == "I'keeper services don't have real-time capabilities, however, I can guide you on how to find current weather conditions in Brooklyn, New York.\n\nTo get the most accurate"
    )
    assert response == response_snapshot
