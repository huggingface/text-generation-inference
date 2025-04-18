import pytest
from openai import OpenAI
from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types.chat_completion import (
    ChatCompletionOutputToolCall,
    ChatCompletionOutputFunctionDefinition,
)


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
async def test_flash_llama_grammar_tools_nostream(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    response = client.chat_completion(
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
        ChatCompletionOutputToolCall(
            id="0",
            type="function",
            function=ChatCompletionOutputFunctionDefinition(
                description=None,
                name="get_current_weather",
                arguments='{"location":"Brooklyn, NY","format":"fahrenheit"}',
            ),
        )
    ]
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_openai(
    flash_llama_grammar_tools, response_snapshot
):
    client = OpenAI(api_key="xx", base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    stream = client.chat.completions.create(
        model="tgi",
        max_tokens=100,
        seed=1,
        tools=tools,
        stream=True,
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

    chunks = []
    tool = ""
    name = ""
    for chunk in stream:
        if chunk.choices[0].delta.tool_calls[0].function.name:
            name += chunk.choices[0].delta.tool_calls[0].function.name
        tool += chunk.choices[0].delta.tool_calls[0].function.arguments
        chunks.append(chunk)

    assert name == "get_current_weather"
    assert tool == '{ "location": "Brooklyn, NY", "format": "fahrenheit"}'
    assert chunks == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_auto_nostream(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    response = client.chat_completion(
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
        ChatCompletionOutputToolCall(
            id="0",
            type="function",
            function=ChatCompletionOutputFunctionDefinition(
                description=None,
                name="get_current_weather",
                arguments='{"location":"Brooklyn, NY","format":"fahrenheit"}',
            ),
        )
    ]

    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_choice_nostream(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    response = client.chat_completion(
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
        ChatCompletionOutputToolCall(
            id="0",
            type="function",
            function=ChatCompletionOutputFunctionDefinition(
                description=None,
                name="get_current_weather",
                arguments='{"location":"Brooklyn, NY","format":"fahrenheit"}',
            ),
        )
    ]

    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_choice_stream(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    stream = client.chat_completion(
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
        stream=True,
    )

    arguments = ""
    chunks = []
    name = ""
    for chunk in stream:
        if chunk.choices[0].delta.tool_calls[0].function.name:
            name += chunk.choices[0].delta.tool_calls[0].function.name
        arguments += chunk.choices[0].delta.tool_calls[0].function.arguments
        assert chunk.choices[0].delta.content is None
        chunks.append(chunk)

    assert name == "get_current_weather"
    assert arguments == '{ "location": "Brooklyn, NY", "format": "fahrenheit"}'
    assert chunks == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_insufficient_information_nostream(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    response = client.chat_completion(
        max_tokens=20,
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

    content_generated = response.choices[0].message.content
    assert response.choices[0].message.tool_calls is None

    assert (
        content_generated
        == "I'm an artificial intelligence model known as a large language model (LLM) or conversational AI"
    )
    assert response == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_insufficient_information_stream(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    stream = client.chat_completion(
        max_tokens=20,
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

    content_generated = ""
    chunks = []
    for chunk in stream:
        content_generated += chunk.choices[0].delta.content
        chunks.append(chunk)
        assert chunk.choices[0].delta.tool_calls is None

    ######## This is exactly the same as the non streaming case
    assert (
        content_generated
        == "I'm an artificial intelligence model known as a large language model (LLM) or conversational AI"
    )
    assert chunks == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_sea_creatures_stream_auto(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    stream = client.chat_completion(
        max_tokens=20,
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

    content_generated = ""
    chunks = []
    for chunk in stream:
        content_generated += chunk.choices[0].delta.content
        chunks.append(chunk)
        assert chunk.choices[0].delta.tool_calls is None

    assert (
        content_generated
        == "Once upon a time, in a vibrant ocean filled with coral reefs and schools of shimmering fish,"
    )
    assert chunks == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_sea_creatures_stream_required(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    stream = client.chat_completion(
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

    tool_calls_generated = ""
    name = ""
    chunks = []
    for chunk in stream:
        assert chunk.choices[0].delta.content is None
        if chunk.choices[0].delta.tool_calls[0].function.name:
            name += chunk.choices[0].delta.tool_calls[0].function.name
        tool_calls_generated += chunk.choices[0].delta.tool_calls[0].function.arguments

    assert name == "get_n_day_weather_forecast"
    assert (
        tool_calls_generated
        == '{ "location": "San Francisco, CA", "format": "fahrenheit", "num_days":3}'
    )
    assert chunks == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_sea_creatures_stream_none(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    stream = client.chat_completion(
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

    content_generated = ""
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        content_generated += chunk.choices[0].delta.content
        assert chunk.choices[0].delta.tool_calls is None

    assert (
        content_generated
        == "Once upon a time, in a vibrant ocean filled with coral reefs and schools of shimmering fish, lived three dear friends: Luna the sea turtle, Finley the friendly fish, and Crusty the wise crab.\n\nLuna was the oldest of the three. She had traveled the world, exploring hidden caves and shipwrecks, and collecting sparkling shells and shiny pebbles. Her shell was a beautiful mosaic of blues and greens, and her gentle eyes twinkled with the secrets of the deep"
    )
    assert chunks == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_grammar_tools_sea_creatures_stream_function_object(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    stream = client.chat_completion(
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
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "get_n_day_weather_forecast"},
        },
        max_tokens=100,
        seed=24,
        stream=True,
    )
    chunks = []
    tool_calls_generated = ""
    name = ""
    for chunk in stream:
        assert chunk.choices[0].delta.content is None
        if chunk.choices[0].delta.tool_calls[0].function.name:
            name += chunk.choices[0].delta.tool_calls[0].function.name
        tool_calls_generated += chunk.choices[0].delta.tool_calls[0].function.arguments

    assert name == "get_n_day_weather_forecast"
    assert (
        tool_calls_generated
        == '{ "location": "San Francisco, CA", "format": "celsius", "num_days": 3}'
    )
    assert chunks == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_flash_llama_tool_reply_response(
    flash_llama_grammar_tools, response_snapshot
):
    client = InferenceClient(base_url=f"{flash_llama_grammar_tools.base_url}/v1")
    response = client.chat_completion(
        max_tokens=100,
        seed=42,
        messages=[
            {"role": "user", "content": "What's the weather like in Paris today?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "0",
                        "function": {
                            "arguments": '{"longitude": 2.2945, "latitude": 48.8567}',
                            "name": "get_weather",
                            "description": None,
                        },
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "0", "content": "6.7"},
        ],
        stream=False,
    )

    assert response.choices[0].message.tool_calls is None
    assert (
        response.choices[0].message.content
        == "I can't access real-time data, but I can provide you with current conditions and forecast for Paris, France:\n\nThe current conditions in Paris are mostly cloudy with a temperature of 6.7°C (44.1°F). \n\nPlease note that the actual weather may differ from the provided information. For up-to-date information, I suggest checking a reliable weather website or app for the latest conditions and forecast."
    )

    assert response == response_snapshot
