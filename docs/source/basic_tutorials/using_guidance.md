# Guidance

Text Generation Inference (TGI) now supports [JSON and regex grammars](#grammar-and-constraints) and [tools and functions](#tools-and-functions) to help developers guide LLM responses to fit their needs.

These feature are available starting from version `1.4.3`. They are accessible via the [`huggingface_hub`](https://pypi.org/project/huggingface-hub/) library. The tool support is compatible with OpenAI's client libraries. The following guide will walk you through the new features and how to use them!

_note: guidance is supported as grammar in the `/generate` endpoint and as tools in the `v1/chat/completions` endpoint._

## How it works

TGI leverages the [outlines](https://github.com/outlines-dev/outlines) library to efficiently parse and compile the grammatical structures and tools specified by users. This integration transforms the defined grammars into an intermediate representation that acts as a framework to guide and constrain content generation, ensuring that outputs adhere to the specified grammatical rules.

If you are interested in the technical details on how outlines is used in TGI, you can check out the [conceptual guidance documentation](../conceptual/guidance).

## Table of Contents 📚

### Grammar and Constraints

- [The Grammar Parameter](#the-grammar-parameter): Shape your AI's responses with precision.
- [Constrain with Pydantic](#constrain-with-pydantic): Define a grammar using Pydantic models.
- [JSON Schema Integration](#json-schema-integration): Fine-grained control over your requests via JSON schema.
- [Using the client](#using-the-client): Use TGI's client libraries to shape the AI's responses.

### Tools and Functions

- [The Tools Parameter](#the-tools-parameter): Enhance the AI's capabilities with predefined functions.
- [Via the client](#text-generation-inference-client): Use TGI's client libraries to interact with the Messages API and Tool functions.
- [OpenAI integration](#openai-integration): Use OpenAI's client libraries to interact with TGI's Messages API and Tool functions.

## Grammar and Constraints 🛣️

### The Grammar Parameter

In TGI `1.4.3`, we've introduced the grammar parameter, which allows you to specify the format of the response you want from the LLM.

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

### Hugging Face Hub Python Library

The Hugging Face Hub Python library provides a client that makes it easy to interact with the Messages API. Here's an example of how to use the client to send a request with a grammar parameter.

```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:3000")

schema = {
    "properties": {
        "location": {"title": "Location", "type": "string"},
        "activity": {"title": "Activity", "type": "string"},
        "animals_seen": {
            "maximum": 5,
            "minimum": 1,
            "title": "Animals Seen",
            "type": "integer",
        },
        "animals": {"items": {"type": "string"}, "title": "Animals", "type": "array"},
    },
    "required": ["location", "activity", "animals_seen", "animals"],
    "title": "Animals",
    "type": "object",
}

user_input = "I saw a puppy a cat and a raccoon during my bike ride in the park"
resp = client.text_generation(
    f"convert to JSON: 'f{user_input}'. please use the following schema: {schema}",
    max_new_tokens=100,
    seed=42,
    grammar={"type": "json", "value": schema},
)

print(resp)
# { "activity": "bike ride", "animals": ["puppy", "cat", "raccoon"], "animals_seen": 3, "location": "park" }

```

A grammar can be defined using Pydantic models, JSON schemas, or regular expressions. The LLM will then generate a response that conforms to the specified grammar.

> Note: A grammar must compile to an intermediate representation to constrain the output. Grammar compilation is a computationally expensive and may take a few seconds to complete on the first request. Subsequent requests will use the cached grammar and will be much faster.

### Constrain with Pydantic

Using Pydantic models we can define a similar grammar as the previous example in a shorter and more readable way.

```python
from huggingface_hub import InferenceClient
from pydantic import BaseModel, conint
from typing import List


class Animals(BaseModel):
    location: str
    activity: str
    animals_seen: conint(ge=1, le=5)  # Constrained integer type
    animals: List[str]


client = InferenceClient("http://localhost:3000")

user_input = "I saw a puppy a cat and a raccoon during my bike ride in the park"
resp = client.text_generation(
    f"convert to JSON: 'f{user_input}'. please use the following schema: {Animals.schema()}",
    max_new_tokens=100,
    seed=42,
    grammar={"type": "json", "value": Animals.schema()},
)

print(resp)
# { "activity": "bike ride", "animals": ["puppy", "cat", "raccoon"], "animals_seen": 3, "location": "park" }


```

defining a grammar as regular expressions

```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:3000")

section_regex = "(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
regexp = f"HELLO\.{section_regex}\.WORLD\.{section_regex}"

# This is a more realistic example of an ip address regex
# regexp = f"{section_regex}\.{section_regex}\.{section_regex}\.{section_regex}"


resp = client.text_generation(
    f"Whats Googles DNS? Please use the following regex: {regexp}",
    seed=42,
    grammar={
        "type": "regex",
        "value": regexp,
    },
)


print(resp)
# HELLO.255.WORLD.255

```

## Tools and Functions 🛠️

### The Tools Parameter

In addition to the grammar parameter, we've also introduced a set of tools and functions to help you get the most out of the Messages API.

Tools are a set of user defined functions that can be used in tandem with the chat functionality to enhance the LLM's capabilities. Functions, similar to grammar are defined as JSON schema and can be passed as part of the parameters to the Messages API.

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

### Chat Completion with Tools

Grammars are supported in the `/generate` endpoint, while tools are supported in the `/chat/completions` endpoint. Here's an example of how to use the client to send a request with a tool parameter.

```python
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:3000")

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

chat = client.chat_completion(
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
    tools=tools,
    seed=42,
    max_tokens=100,
)

print(chat.choices[0].message.tool_calls)
# [ChatCompletionOutputToolCall(function=ChatCompletionOutputFunctionDefinition(arguments={'format': 'fahrenheit', 'location': 'Brooklyn, New York', 'num_days': 7}, name='get_n_day_weather_forecast', description=None), id=0, type='function')]

```

### OpenAI integration

TGI exposes an OpenAI-compatible API, which means you can use OpenAI's client libraries to interact with TGI's Messages API and Tool functions.

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

### Tool Choice Configuration

When configuring how the model interacts with tools during a chat completion, there are several options for determining if or how a tool should be called. These options are controlled by the `tool_choice` parameter, which specifies the behavior of the model in relation to tool usage. The following modes are supported:

1. **`auto`**:

   - The model decides whether to call a tool or generate a response message based on the user's input.
   - If tools are provided, this is the default mode.
   - Example usage:
     ```python
     tool_choice="auto"
     ```

2. **`none`**:

   - The model will never call any tools and will only generate a response message.
   - If no tools are provided, this is the default mode.
   - Example usage:
     ```python
     tool_choice="none"
     ```

3. **`required`**:

   - The model must call one or more tools and will not generate a response message on its own.
   - Example usage:
     ```python
     tool_choice="required"
     ```

4. **Specific Tool Call by Function Name**:
   - You can force the model to call a specific tool either by specifying the tool function directly or by using an object definition.
   - Two ways to do this:
     1. Provide the function name as a string:
        ```python
        tool_choice="get_current_weather"
        ```
     2. Use the function object format:
        ```python
        tool_choice={
          "type": "function",
          "function": {
              "name": "get_current_weather"
          }
        }
        ```

These options allow flexibility when integrating tools with the chat completions endpoint. You can configure the model to either rely on tools automatically or force it to follow a predefined behavior, based on the needs of the task at hand.

---

| **Tool Choice Option**                | **Description**                                                                                                                 | **When to Use**                                                                        |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `auto`                                | The model decides whether to call a tool or generate a message. This is the default if tools are provided.                      | Use when you want the model to decide when a tool is necessary.                        |
| `none`                                | The model generates a message without calling any tools. This is the default if no tools are provided.                          | Use when you do not want the model to call any tools.                                  |
| `required`                            | The model must call one or more tools and will not generate a message on its own.                                               | Use when a tool call is mandatory, and you do not want a regular message generated.    |
| Specific Tool Call (`name` or object) | Force the model to call a specific tool either by specifying its name (`tool_choice="get_current_weather"`) or using an object. | Use when you want to restrict the model to calling a particular tool for the response. |
