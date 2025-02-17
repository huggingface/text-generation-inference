from openai import OpenAI
import pytest


@pytest.fixture(scope="module")
def openai_llama_tools_handle(launcher):
    with launcher(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        num_shard=2,
        disable_grammar_support=False,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def openai_llama_tools(openai_llama_tools_handle):
    await openai_llama_tools_handle.health(300)
    return openai_llama_tools_handle.client


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
async def test_openai_llama_tools(openai_llama_tools, response_snapshot):
    client = OpenAI(
        base_url=f"{openai_llama_tools.base_url}/v1",
        api_key="_",
    )

    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
            {
                "role": "system",
                "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
            },
            {
                "role": "user",
                "content": "What's the weather like the next 3 days in San Francisco, CA?",
            },
        ],
        tools=tools,
        tool_choice="get_current_weather",
        max_tokens=500,
        stream=True,
        seed=42,
    )

    tool_call_string = ""
    for chunk in chat_completion:
        tool_call_string += chunk.choices[0].delta.tool_calls[0].function.arguments
        last_chunk = chunk.to_dict()

    assert (
        tool_call_string == '{ "location": "San Francisco, CA", "format": "fahrenheit"}'
    )
    assert last_chunk == response_snapshot
