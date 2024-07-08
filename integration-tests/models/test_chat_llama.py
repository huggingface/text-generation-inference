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

    print(repr(response.choices[0].message.content))
    assert (
        response.choices[0].message.content
        == "As of your last question, the weather in Brooklyn, New York, is typically hot and humid throughout the year. The suburbs around New York City are jealously sheltered, and at least in the Lower Bronx, there are very few outdoor environments to explore in the middle of urban confines. In fact, typical times for humidity levels in Brooklyn include:\n\n- Early morning: 80-85% humidity, with occas"
    )
    assert response == response_snapshot
