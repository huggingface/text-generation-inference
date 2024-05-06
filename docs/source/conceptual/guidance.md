# Guidance

Text Generation Inference (TGI) now supports [JSON and regex grammars](#grammar-and-constraints) and [tools and functions](#tools-and-functions) to help developer guide LLM responses to fit their needs.

These feature are available starting from version `1.4.3`. They are accessible via the [text_generation](https://pypi.org/project/text-generation/) library and is compatible with OpenAI's client libraries. The following guide will walk you through the new features and how to use them!

## Quick Start

Before we jump into the deep end, ensure your system is using TGI version `1.4.3` or later to access all the features we're about to explore in this guide.

If you're not up to date, grab the latest version and let's get started!

## Table of Contents 📚

### Grammar and Constraints

- [The Grammar Parameter](#the-grammar-parameter): Shape your AI's responses with precision.
- [Constrain with Pydantic](#constrain-with-pydantic): Define a grammar using Pydantic models.
- [JSON Schema Integration](#json-schema-integration): Fine grain control over your requests via JSON schema.
- [Using the client](#using-the-client): Use TGI's client libraries to shape the AI's responses.

### Tools and Functions

- [The Tools Parameter](#the-tools-parameter): Enhance the AI's capabilities with predefined functions.
- [Via the client](#text-generation-inference-client): Use TGI's client libraries to interact with the Messages API and Tool functions.
- [OpenAI integration](#openai-integration): Use OpenAI's client libraries to interact with TGI's Messages API and Tool functions.

## Grammar and Constraints 🛣️

### The Grammar Parameter

In TGI `1.4.3`, we've introduced the grammar parameter, which allows you to specify the format of the response you want from the AI. This is a game-changer for those who need precise control over the AI's output.

Using curl, you can make a request to TGI's Messages API with the grammar parameter. This is the most primitive way to interact with the API and using [Pydantic](#constrain-with-pydantic) is recommended for ease of use and readability.

```json
curl localhost:3000/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
    "inputs": "I saw a puppy a cat and a raccoon during my bike ride in the park",
    "parameters": {
        "repetition_penalty": 1.3,
        "grammar": {
            "type": "json",
            "value": {
                "properties": {
                    "location": {
                        "type": "string"
                    },
                    "activity": {
                        "type": "string"
                    },
                    "animals_seen": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5
                    },
                    "animals": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["location", "activity", "animals_seen", "animals"]
            }
        }
    }
}'
// {"generated_text":"{ \n\n\"activity\": \"biking\",\n\"animals\": [\"puppy\",\"cat\",\"raccoon\"],\n\"animals_seen\": 3,\n\"location\": \"park\"\n}"}

```

A grammar can be defined using Pydantic models, JSON schema, or regular expressions. The AI will then generate a response that conforms to the specified grammar.

> Note: A grammar must compile to a intermediate representation to constrain the output. Grammar compilation is a computationally expensive and may take a few seconds to complete on the first request. Subsequent requests will use the cached grammar and will be much faster.

### Constrain with Pydantic

Pydantic is a powerful library for data validation and settings management. It's the perfect tool for crafting the a specific response format.

Using Pydantic models we can define a similar grammar as the previous example in a shorter and more readable way.

```python
import requests
from pydantic import BaseModel, conint
from typing import List

class Animals(BaseModel):
    location: str
    activity: str
    animals_seen: conint(ge=1, le=5)  # Constrained integer type
    animals: List[str]

prompt = "convert to JSON: I saw a puppy a cat and a raccoon during my bike ride in the park"

data = {
    "inputs": prompt,
    "parameters": {
        "repetition_penalty": 1.3,
        "grammar": {
            "type": "json",
            "value": Animals.schema()
        }
    }
}

headers = {
    "Content-Type": "application/json",
}

response = requests.post(
    'http://127.0.0.1:3000/generate',
    headers=headers,
    json=data
)
print(response.json())
# {'generated_text': '{ "activity": "bike riding", "animals": ["puppy","cat","raccoon"],"animals_seen": 3, "location":"park" }'}

```

### JSON Schema Integration

If Pydantic's not your style, go raw with direct JSON Schema integration. It's like having a conversation with the AI in its own language. This is simliar to the first example but with programmatic control.

```python
import requests

json_schema = {
    "properties": {
        "location": {
            "type": "string"
        },
        "activity": {
            "type": "string"
        },
        "animals_seen": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        },
        "animals": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": ["location", "activity", "animals_seen", "animals"]
}

data = {
    "inputs": "[INST]convert to JSON: I saw a puppy a cat and a raccoon during my bike ride in the park [/INST]",
    "parameters": {
        "max_new_tokens": 200,
        "repetition_penalty": 1.3,
        "grammar": {
            "type": "json",
            "value": json_schema
        }
    }
}

headers = {
    "Content-Type": "application/json",
}

response = requests.post(
    'http://127.0.0.1:3000/generate',
    headers=headers,
    json=data
)
print(response.json())
# {'generated_text': '{\n"activity": "biking",\n"animals": ["puppy","cat","raccoon"]\n  , "animals_seen": 3,\n   "location":"park"}'}

```

### Using the client

TGI provides a client library to that make it easy to send requests with all of the parameters we've discussed above. Here's an example of how to use the client to send a request with a grammar parameter.

```python
from text_generation import AsyncClient
from text_generation.types import GrammarType

# NOTE: tools defined above and removed for brevity

# Define an async function to encapsulate the async operation
async def main():
    client = AsyncClient(base_url="http://localhost:3000")

    # Use 'await' to wait for the async method 'chat' to complete
    response = await client.generate(
        "Whats Googles DNS",
        max_new_tokens=10,
        decoder_input_details=True,
        seed=1,
        grammar={
            "type": GrammarType.Regex,
            "value": "((25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\.){3}(25[0-5]|2[0-4]\\d|[01]?\\d\\d?)",
        },
    )

    # Once the response is received, you can process it
    print(response.generated_text)

# Ensure the main async function is run in the event loop
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# 118.8.0.84

```

## Tools and Functions 🛠️

### The Tools Parameter

In addition to the grammar parameter, we've also introduced a set of tools and functions to help you get the most out of the Messages API.

Tools are a set of user defined functions that can be used in tandem with the chat functionality to enhance the AI's capabilities. You can use these tools to perform a variety of tasks, such as data manipulation, formatting, and more.

Functions, similar to grammar are defined as JSON schema and can be passed as part of the parameters to the Messages API.

```json
curl localhost:3000/v1/chat/completions \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "tgi",
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in New York?"
        }
    ],
    "tools": [
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
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location."
                        }
                    },
                    "required": ["location", "format"]
                }
            }
        }
    ],
    "tool_choice": "get_current_weather"
}'
// {"id":"","object":"text_completion","created":1709051640,"model":"HuggingFaceH4/zephyr-7b-beta","system_fingerprint":"1.4.3-native","choices":[{"index":0,"message":{"role":"assistant","tool_calls":{"id":0,"type":"function","function":{"description":null,"name":"tools","parameters":{"format":"celsius","location":"New York"}}}},"logprobs":null,"finish_reason":"eos_token"}],"usage":{"prompt_tokens":157,"completion_tokens":19,"total_tokens":176}}
```

<details>
  <summary>Tools used in example below</summary>

  ```python
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
        }
    ]
  ```

</details>

### Text Generation Inference Client

TGI provides a client library to interact with the Messages API and Tool functions. The client library is available in both synchronous and asynchronous versions.

```python
from text_generation import AsyncClient

# NOTE: tools defined above and removed for brevity

# Define an async function to encapsulate the async operation
async def main():
    client = AsyncClient(base_url="http://localhost:3000")

    # Use 'await' to wait for the async method 'chat' to complete
    response = await client.chat(
        max_tokens=100,
        seed=1,
        tools=tools,
        presence_penalty=-1.1,
        messages=[
            {
                "role": "system",
                "content": "You're a helpful assistant! Answer the users question best you can.",
            },
            {
                "role": "user",
                "content": "What is the weather like in Brooklyn, New York?",
            },
        ],
    )

    # Once the response is received, you can process it
    print(response.choices[0].message.tool_calls)

# Ensure the main async function is run in the event loop
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

# {"id":"","object":"text_completion","created":1709051942,"model":"HuggingFaceH4/zephyr-7b-beta","system_fingerprint":"1.4.3-native","choices":[{"index":0,"message":{"role":"assistant","tool_calls":{"id":0,"type":"function","function":{"description":null,"name":"tools","parameters":{"format":"celsius","location":"New York"}}}},"logprobs":null,"finish_reason":"eos_token"}],"usage":{"prompt_tokens":157,"completion_tokens":20,"total_tokens":177}}

```

### OpenAI integration

TGI exposes an OpenAI-compatible API, which means you can use OpenAI's client libraries to interact with TGI's Messages API and Tool functions.

However there are some minor differences in the API, for example `tool_choice="auto"` will ALWAYS choose the tool for you. This is different from OpenAI's API where `tool_choice="auto"` will choose a tool if the model thinks it's necessary.

```python
from openai import OpenAI

# Initialize the client, pointing it to one of the available models
client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="_",
)

# NOTE: tools defined above and removed for brevity

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
    tool_choice="auto",  # tool selected by model
    max_tokens=500,
)


called = chat_completion.choices[0].message.tool_calls
print(called)
# {
#     "id": 0,
#     "type": "function",
#     "function": {
#         "description": None,
#         "name": "tools",
#         "parameters": {
#             "format": "celsius",
#             "location": "San Francisco, CA",
#             "num_days": 3,
#         },
#     },
# }
```
