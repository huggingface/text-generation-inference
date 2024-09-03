import pytest


@pytest.fixture(scope="module")
def flash_llama_chat_handle(launcher):
    with launcher(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", num_shard=2, disable_grammar_support=False
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_chat(flash_llama_chat_handle):
    await flash_llama_chat_handle.health(300)
    return flash_llama_chat_handle.client


@pytest.mark.private
async def test_flash_llama_simple(flash_llama_chat, response_snapshot):
    response = await flash_llama_chat.chat(
        max_tokens=100,
        seed=1,
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
        == "Brooklyn, New York, is located in the northeastern part of the state of New York, and its weather is characterized by a mix of humid and temperate climates. The average temperature in Brooklyn during the winter months is around 32°F (0°C) and in the summer months is around 82°F (28°C).\n\nThe city experiences four distinct seasons, with the spring and fall being the most pleasant and"
    )
    assert response == response_snapshot
