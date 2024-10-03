import pytest
from huggingface_hub import InferenceClient

# to be removed when the InferenceClient client supports latest parameters
import requests

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


# All tests are based on the following model card
# https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/

@pytest.mark.asyncio
@pytest.mark.private
async def test_basic_gen(flash_llama_grammar_tools, response_snapshot):
    client = InferenceClient(
        base_url=flash_llama_grammar_tools.base_url + "/v1",
    )

    output = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant",
            },
            {
                "role": "user",
                "content": "What is the capital of France?",
            },
        ],
        stream=True,
        seed=42,
        max_tokens=20,
    )

    final_response = []
    for chunk in output:
        final_response.append(chunk.choices[0].delta.content)
    resp = ''.join(final_response)

    assert resp == "The capital of France is Paris."

@pytest.mark.asyncio
@pytest.mark.private
async def test_code_interpreter_gen(flash_llama_grammar_tools, response_snapshot):
    client = InferenceClient(
        base_url=flash_llama_grammar_tools.base_url + "/v1",
    )
    output = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "Environment: ipython",
            },
            {
                "role": "user",
                "content": "Write code to check if number is prime, use that to see if the number 7 is prime",
            },
        ],
        stream=True,
        seed=42,
        max_tokens=20,
    )

    final_response = []
    for chunk in output:
        final_response.append(chunk.choices[0].delta.content)
    resp = ''.join(final_response)

    assert resp == "def is_prime(n):\n    if n <= 1:\n        return False\n    if n"

@pytest.mark.asyncio
@pytest.mark.private
async def test_code_builtin_tools_gen(flash_llama_grammar_tools, response_snapshot):
    url = f"{flash_llama_grammar_tools.base_url}/v1/chat/completions"

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "What is the current weather in Menlo Park, California?",
            }
        ],
        "stream": False,
        "seed": 42,
        "max_tokens": 20,
        "builtin_tools": ["brave_search", "wolfram_alpha"],
    }

    response = requests.request("POST", url, json=payload)
    response = response.json()
    resp = response.get("choices")[0].get("message").get("content")
    assert resp == "brave_search.call(query=\"current weather in Menlo Park, California\")"

@pytest.mark.asyncio
@pytest.mark.private
async def test_code_builtin_tools_explict_off_gen(flash_llama_grammar_tools, response_snapshot):
    url = f"{flash_llama_grammar_tools.base_url}/v1/chat/completions"

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "What is the current weather in Menlo Park, California?",
            }
        ],
        "stream": False,
        "seed": 42,
        "max_tokens": 20,
        # "builtin_tools": ["brave_search", "wolfram_alpha"],
    }

    response = requests.request("POST", url, json=payload)
    response = response.json()
    resp = response.get("choices")[0].get("message").get("content")
    assert resp == "I can't provide real-time weather information. However, I can encourage you to check a weather website"


@pytest.mark.asyncio
@pytest.mark.private
async def test_code_builtin_tools_two_gen(flash_llama_grammar_tools, response_snapshot):
    url = f"{flash_llama_grammar_tools.base_url}/v1/chat/completions"

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Can you help me solve this equation with wolfram_alpha: x^3 - 4x^2 + 6x - 24 = 0",
            },
        ],
        "stream": False,
        "seed": 42,
        "max_tokens": 50,
        "builtin_tools": ["brave_search", "wolfram_alpha"],
    }

    response = requests.request("POST", url, json=payload)
    response = response.json()
    resp = response.get("choices")[0].get("message").get("content")
    assert resp == "wolfram_alpha.call(query=\"solve x^3 - 4x^2 + 6x - 24 = 0\")"


@pytest.mark.asyncio
@pytest.mark.private
async def test_code_builtin_tools_function_response_gen(flash_llama_grammar_tools, response_snapshot):
    url = f"{flash_llama_grammar_tools.base_url}/v1/chat/completions"

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Can you help me solve this equation with wolfram_alpha: x^3 - 4x^2 + 6x - 24 = 0",
            },
            {
                "role": "assistant",
                "content": "wolfram_alpha.call(query=\"solve x^3 - 4x^2 + 6x - 24 = 0\")",
            },
            {
                "role": "ipython",
                "content": "{\"queryresult\": {\"success\": true, \"inputstring\": \"solve x^3 - 4x^2 + 6x - 24 = 0\", \"pods\": [{\"title\": \"Input interpretation\", \"subpods\": [{\"title\": \"\", \"plaintext\": \"solve x^3 - 4 x^2 + 6 x - 24 = 0\"}]}, {\"title\": \"Results\", \"primary\": true, \"subpods\": [{\"title\": \"\", \"plaintext\": \"x = 4\"}, {\"title\": \"\", \"plaintext\": \"x = \u00b1 (i sqrt(6))\"}]}, ... ]}}",
            },
        ],
        "stream": False,
        "seed": 42,
        "max_tokens": 50,
        "builtin_tools": ["brave_search", "wolfram_alpha"],
    }

    response = requests.request("POST", url, json=payload)
    response = response.json()
    resp = response.get("choices")[0].get("message").get("content")
    assert resp == "The solutions to the equation x^3 - 4x^2 + 6x - 24 = 0 are x = 4, x = i√6, and x = -i√6."


@pytest.mark.asyncio
@pytest.mark.private
async def test_user_supplied_json_tool_gen(flash_llama_grammar_tools, response_snapshot):
    client = InferenceClient(
        base_url=flash_llama_grammar_tools.base_url + "/v1",
    )
    output = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant with tool calling capabilities"
            },
            {
                "role": "user",
                "content": "Question: what is the weather like in San Fransisco?"
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_conditions",
                    "description": "Get the current weather conditions for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["Celsius", "Fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the user's location."
                            }
                        },
                        "required": ["location", "unit"]
                    }
                }
            }
        ],
        stream=True,
        seed=42,
        max_tokens=50,
    )

    final_response = []
    for chunk in output:
        final_response.append(chunk.choices[0].delta.content)
    resp = ''.join(final_response)

    assert resp == "{\"name\": \"get_current_conditions\", \"parameters\": {\"location\": \"San Francisco, CA\", \"unit\": \"Fahrenheit\"}}"

@pytest.mark.asyncio
@pytest.mark.private
async def test_user_supplied_json_tool_function_response_gen(flash_llama_grammar_tools, response_snapshot):
    client = InferenceClient(
        base_url=flash_llama_grammar_tools.base_url + "/v1",
    )
    output = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the orginal use question."
            },
            {
                "role": "user",
                "content": "Question: what is the weather like in San Fransisco?"
            },
            {
                "role": "assistant",
                "content": "{\"name\": \"get_current_conditions\", \"parameters\": {\"location\": \"San Francisco, CA\", \"unit\": \"Fahrenheit\"}}",
            },
            {
                "role": "ipython",
                "content": "{\"output\": \"Clouds giving way to sun Hi: 76° Tonight: Mainly clear early, then areas of low clouds forming Lo: 56°\"}",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_conditions",
                    "description": "Get the current weather conditions for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["Celsius", "Fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the user's location."
                            }
                        },
                        "required": ["location", "unit"]
                    }
                }
            }
        ],
        stream=True,
        seed=42,
        max_tokens=50,
    )

    final_response = []
    for chunk in output:
        final_response.append(chunk.choices[0].delta.content)
    resp = ''.join(final_response)
    assert resp == "The current weather conditions in San Francisco, CA are clouds giving way to sun with a high of 76°F and a low of 56°F."