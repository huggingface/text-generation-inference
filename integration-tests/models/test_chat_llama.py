import pytest
import json

from text_generation.types import GrammarType


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

    assert (
        response.choices[0].message.content
        == "As of today, there is a Update available for the Brooklyn, New York, area. According to the latest forecast, it's warm with high temperatures throughout the day. It's forecasted at 75°F for today and 77°F for tomorrow. However, in autumn, the weather typically changes drastically, becoming cooler and wetter. You can find the current weather forecast for the area through your local weather service. Additionally"
    )
    assert response == response_snapshot
