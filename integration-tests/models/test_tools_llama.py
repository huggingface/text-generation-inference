import pytest
import json

from text_generation.types import GrammarType


@pytest.fixture(scope="module")
def flash_llama_grammar_tools_handle(launcher):
    with launcher(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", num_shard=2, disable_grammar_support=False
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_grammar_tools(flash_llama_grammar_tools_handle):
    await flash_llama_grammar_tools_handle.health(300)
    return flash_llama_grammar_tools_handle.client


# tools to be used in the following tests
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    },
                },
                "required": ["location", "format", "num_days"],
            },
        },
    },
]


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_no_tools_regex(
    flash_llama_grammar_tools, response_snapshot
):
    response = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=0,
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
        == "As for the weather in Brooklyn, New York, it can vary depending on the location within the borough. According to climatereporter.com, the average temperature in August is 73 degrees Fahrenheit (23 degrees Celsius), while the humidity is 62%. In the winter (December to February), the temperature averages between 20 and 45 degrees Fahrenheit (6 to 8 degrees Celsius), with significant"
    )
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_regex(
    flash_llama_grammar_tools, response_snapshot
):
    response = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=0,
        tools=tools,
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
    assert len(response.choices[0].message.content) == 81
    assert (
        response.choices[0].message.content
        == """{"function":{"format": "celsius", "location": "Brooklyn, NYC", "num_days": 1255}}"""
    )
    assert response == response_snapshot
