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
async def test_flash_llama_grammar_no_tools(
    flash_llama_grammar_tools, response_snapshot
):
    response = await flash_llama_grammar_tools.chat(
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


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools(flash_llama_grammar_tools, response_snapshot):
    response = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=1,
        tools=tools,
        presence_penalty=-1.1,
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
    assert response.choices[0].message.content == None
    assert response.choices[0].message.tool_calls == {
        "function": {
            "description": None,
            "name": "tools",
            "parameters": {
                "format": "celsius",
                "location": "New York, NY",
                "num_days": 14,
            },
        },
        "id": 0,
        "type": "function",
    }
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_auto(
    flash_llama_grammar_tools, response_snapshot
):
    response = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=1,
        tools=tools,
        tool_choice="auto",
        presence_penalty=-1.1,
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
    assert response.choices[0].message.content == None
    assert response.choices[0].message.tool_calls == {
        "function": {
            "description": None,
            "name": "tools",
            "parameters": {
                "format": "celsius",
                "location": "New York, NY",
                "num_days": 14,
            },
        },
        "id": 0,
        "type": "function",
    }
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_choice(
    flash_llama_grammar_tools, response_snapshot
):
    response = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=1,
        tools=tools,
        tool_choice="get_current_weather",
        presence_penalty=-1.1,
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
    assert response.choices[0].message.content == None
    assert response.choices[0].message.tool_calls == {
        "id": 0,
        "type": "function",
        "function": {
            "description": None,
            "name": "tools",
            "parameters": {"format": "celsius", "location": "New York, NY"},
        },
    }
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_stream(
    flash_llama_grammar_tools, response_snapshot
):
    responses = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=1,
        tools=tools,
        tool_choice="get_current_weather",
        presence_penalty=-1.1,
        messages=[
            {
                "role": "system",
                "content": "Youre a helpful assistant! Answer the users question best you can.",
            },
            {
                "role": "user",
                "content": "What is the weather like in Paris, France?",
            },
        ],
        stream=True,
    )

    count = 0
    async for response in responses:
        count += 1

    assert count == 20
    assert response == response_snapshot
