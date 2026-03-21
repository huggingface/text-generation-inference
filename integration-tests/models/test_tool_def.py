import pytest


@pytest.fixture(scope="module")
def flash_gemma3_handle(launcher):
    with launcher("google/gemma-3-4b-it", num_shard=2) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_gemma3(flash_gemma3_handle):
    await flash_gemma3_handle.health(300)
    return flash_gemma3_handle.client


async def test_flash_gemma3_defs(flash_gemma3, response_snapshot):
    response = await flash_gemma3.chat(
        messages=[
            {
                "content": "Classify the weather: It's sunny outside with clear skies",
                "role": "user",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "classify_weather",
                    "description": "Classify weather conditions",
                    "parameters": {
                        "$defs": {
                            "WeatherType": {
                                "enum": ["sunny", "cloudy", "rainy", "snowy"],
                                "type": "string",
                            }
                        },
                        "properties": {"weather": {"$ref": "#/$defs/WeatherType"}},
                        "required": ["weather"],
                        "type": "object",
                    },
                },
            }
        ],
        tool_choice="auto",
        max_tokens=100,
        seed=42,
    )

    assert (
        response.choices[0].message.tool_calls[0]["function"]["name"]
        == "classify_weather"
    )
    assert (
        response.choices[0].message.tool_calls[0]["function"]["arguments"]
        == '{"weather":"sunny"}'
    )
    assert response == response_snapshot
