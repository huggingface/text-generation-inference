import pytest
import requests
import json


@pytest.fixture(scope="module")
def flash_llama_grammar_tools_handle(launcher):
    with launcher(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        num_shard=2,
        disable_grammar_support=False,
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
                "additionalProperties": False,
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
                "additionalProperties": False,
            },
        },
    },
]


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools(flash_llama_grammar_tools, response_snapshot):
    response = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=1,
        tools=tools,
        temperature=0.0,
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
    assert response.choices[0].message.content is None
    assert response.choices[0].message.tool_calls == [
        {
            "id": "0",
            "type": "function",
            "function": {
                "description": None,
                "name": "get_current_weather",
                "arguments": {"format": "celsius", "location": "Brooklyn, NY"},
            },
        }
    ]
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
        temperature=0.0,
        tool_choice="auto",
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
    assert response.choices[0].message.content is None
    assert response.choices[0].message.tool_calls == [
        {
            "id": "0",
            "type": "function",
            "function": {
                "description": None,
                "name": "get_current_weather",
                "arguments": {"format": "celsius", "location": "Brooklyn, NY"},
            },
        }
    ]

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
        temperature=0.0,
        tool_choice="get_current_weather",
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
    assert response.choices[0].message.content is None
    assert response.choices[0].message.tool_calls == [
        {
            "id": "0",
            "type": "function",
            "function": {
                "description": None,
                "name": "get_current_weather",
                "arguments": {"format": "celsius", "location": "Brooklyn, New York"},
            },
        }
    ]

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
        temperature=0.0,
        tool_choice="get_current_weather",
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
    tool_calls_generated = ""
    last_response = None
    async for response in responses:
        count += 1
        tool_calls_generated += response.choices[0].delta.tool_calls.function.arguments
        last_response = response
        assert response.choices[0].delta.content is None

    assert (
        tool_calls_generated
        == '{"function": {"_name": "get_current_weather", "format": "celsius", "location": "Paris, France"}}<|eot_id|>'
    )
    assert count == 28
    assert last_response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_insufficient_information(
    flash_llama_grammar_tools, response_snapshot
):
    responses = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=24,
        tools=tools,
        tool_choice="auto",
        messages=[
            {
                "role": "system",
                "content": "You're a helpful assistant! Answer the users question best you can.",
            },
            {
                "role": "user",
                "content": "Who are you?",
            },
        ],
        stream=False,
    )

    assert responses.choices[0].message.tool_calls is None
    assert responses.choices[0].message.content == "I am an AI assistant"

    assert responses == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_insufficient_information_stream(
    flash_llama_grammar_tools, response_snapshot
):
    responses = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=24,
        tools=tools,
        tool_choice="auto",
        messages=[
            {
                "role": "system",
                "content": "You're a helpful assistant! Answer the users question best you can.",
            },
            {
                "role": "user",
                "content": "Who are you?",
            },
        ],
        stream=True,
    )

    count = 0
    content_generated = ""
    last_response = None
    async for response in responses:
        count += 1
        content_generated += response.choices[0].delta.content
        last_response = response
        assert response.choices[0].delta.tool_calls is None

    assert count == 5
    assert content_generated == "I am an AI assistant"
    assert last_response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_sea_creatures_stream(
    flash_llama_grammar_tools, response_snapshot
):
    responses = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=24,
        tools=tools,
        tool_choice="auto",
        messages=[
            {
                "role": "system",
                "content": "You're a helpful assistant! Answer the users question best you can. If the question is not answerable by the tools, just generate a response.",
            },
            {
                "role": "user",
                "content": "Tell me a story about 3 sea creatures",
            },
        ],
        stream=True,
    )

    count = 0
    content_generated = ""
    last_response = None
    async for response in responses:
        count += 1
        content_generated += response.choices[0].delta.content
        last_response = response
        assert response.choices[0].delta.tool_calls is None

    assert count == 62
    assert (
        content_generated
        == "Once upon a time, in the ocean, there lived three sea creatures. There was a wise old octopus named Bob, a mischievous seagull named Sam, and a gentle sea turtle named Luna. They all lived together in a beautiful coral reef, surrounded by colorful fish and swaying sea fans"
    )
    assert last_response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_sea_creatures_stream_required(
    flash_llama_grammar_tools, response_snapshot
):
    responses = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=24,
        tools=tools,
        tool_choice="required",
        messages=[
            {
                "role": "system",
                "content": "You're a helpful assistant! Answer the users question best you can. If the question is not answerable by the tools, just generate a response.",
            },
            {
                "role": "user",
                "content": "Tell me a story about 3 sea creatures",
            },
        ],
        stream=True,
    )

    count = 0
    tool_calls_generated = ""
    last_response = None
    async for response in responses:
        count += 1
        assert response.choices[0].delta.content is None
        tool_calls_generated += response.choices[0].delta.tool_calls.function.arguments
        last_response = response

    assert count == 29
    assert (
        tool_calls_generated
        == '{"function": {"_name": "get_current_weather", "format": "celsius", "location": "San Francisco, CA"}}<|eot_id|>'
    )
    assert last_response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_sea_creatures_stream_none(
    flash_llama_grammar_tools, response_snapshot
):
    responses = await flash_llama_grammar_tools.chat(
        max_tokens=100,
        seed=24,
        tools=tools,
        tool_choice="none",
        messages=[
            {
                "role": "system",
                "content": "You're a helpful assistant! Answer the users question best you can. If the question is not answerable by the tools, just generate a response.",
            },
            {
                "role": "user",
                "content": "Tell me a story about 3 sea creatures",
            },
        ],
        stream=True,
    )

    count = 0
    content_generated = ""
    last_response = None
    async for response in responses:
        count += 1
        content_generated += response.choices[0].delta.content
        last_response = response
        assert response.choices[0].delta.tool_calls is None

    assert count == 100
    print(content_generated)
    assert (
        content_generated
        == "Once upon a time, in a vibrant ocean filled with coral reefs and schools of shimmering fish, lived three dear friends: Luna the sea turtle, Finley the friendly fish, and Crusty the wise crab.\n\nLuna was the oldest of the three. She had traveled the world, exploring hidden caves and shipwrecks, and collecting sparkling shells and shiny pebbles. Her shell was a beautiful mosaic of blues and greens, and her gentle eyes twinkled with the secrets of the deep"
    )
    assert last_response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_sea_creatures_stream_function_object(
    flash_llama_grammar_tools, response_snapshot
):
    # using `requests` to send the request until the client library supports tool_choice as a function object
    responses = requests.post(
        f"{flash_llama_grammar_tools.base_url}/v1/chat/completions",
        headers=flash_llama_grammar_tools.headers,
        json={
            "model": "tgi",
            "messages": [
                {
                    "role": "system",
                    "content": "You're a helpful assistant! Answer the users question best you can. If the question is not answerable by the tools, just generate a response.",
                },
                {
                    "role": "user",
                    "content": "Tell me a story about 3 sea creatures",
                },
            ],
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_n_day_weather_forecast"},
            },
            "seed": 24,
            "max_tokens": 100,
            "stream": True,
        },
        stream=True,
    )
    # iterate over the response in chunks
    count = 0
    tool_calls_generated = ""
    last_response = None
    for chunk in responses.iter_content(chunk_size=1024):
        if chunk:
            count += 1
            # remove the "data: " prefix, trailing newline, and split the chunk into individual lines
            lines = chunk.decode("utf-8").replace("data: ", "").rstrip("\n").split("\n")
            for line in lines:
                if line == "[DONE]":
                    break
                response = json.loads(line)
                tool_calls_generated += response["choices"][0]["delta"]["tool_calls"][
                    "function"
                ]["arguments"]
                last_response = response

    assert count == 39
    assert (
        tool_calls_generated
        == '{"function": {"_name": "get_n_day_weather_forecast", "format": "celsius", "location": "San Francisco, CA", "num_days":3}}<|eot_id|>'
    )
    assert last_response == response_snapshot
