import pytest
import json
import requests


@pytest.fixture(scope="module")
def model_handle(launcher):
    """Fixture to provide the base URL for API calls."""
    with launcher(
        "google/gemma-3-4b-it",
        num_shard=2,
        disable_grammar_support=False,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def model_fixture(model_handle):
    await model_handle.health(300)
    return model_handle.client


# Sample JSON Schema for testing
person_schema = {
    "type": "object",
    "$id": "https://example.com/person.schema.json",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Person",
    "properties": {
        "firstName": {
            "type": "string",
            "description": "The person's first name.",
            "minLength": 4,
        },
        "lastName": {
            "type": "string",
            "description": "The person's last name.",
            "minLength": 4,
        },
        "hobby": {
            "description": "The person's hobby.",
            "type": "string",
            "minLength": 4,
        },
        "numCats": {
            "description": "The number of cats the person has.",
            "type": "integer",
            "minimum": 0,
        },
    },
    "required": ["firstName", "lastName", "hobby", "numCats"],
}

# More complex schema for testing nested objects and arrays
complex_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "postalCode": {"type": "string"},
            },
            "required": ["street", "city"],
        },
        "hobbies": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["name", "age", "hobbies"],
}


@pytest.mark.asyncio
@pytest.mark.private
async def test_json_schema_basic(model_fixture, response_snapshot):
    """Test basic JSON schema validation with the person schema."""
    response = requests.post(
        f"{model_fixture.base_url}/v1/chat/completions",
        json={
            "model": "tgi",
            "messages": [
                {
                    "role": "user",
                    "content": "David is a person who likes trees and nature. He enjoys studying math and science. He has 2 cats.",
                },
            ],
            "seed": 42,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "value": {"name": "person", "strict": True, "schema": person_schema},
            },
        },
    )

    result = response.json()

    # Validate response format
    content = result["choices"][0]["message"]["content"]
    parsed_content = json.loads(content)

    assert "firstName" in parsed_content
    assert "lastName" in parsed_content
    assert "hobby" in parsed_content
    assert "numCats" in parsed_content
    assert isinstance(parsed_content["numCats"], int)
    assert parsed_content["numCats"] >= 0
    assert result == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_json_schema_complex(model_fixture, response_snapshot):
    """Test complex JSON schema with nested objects and arrays."""
    response = requests.post(
        f"{model_fixture.base_url}/v1/chat/completions",
        json={
            "model": "tgi",
            "messages": [
                {
                    "role": "user",
                    "content": "John Smith is 30 years old. He lives on Maple Street in Boston. He enjoys botany, astronomy, and solving mathematical puzzles.",
                },
            ],
            "seed": 42,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "value": {
                    "name": "complex_person",
                    "strict": True,
                    "schema": complex_schema,
                },
            },
        },
    )

    result = response.json()

    # Validate response format
    content = result["choices"][0]["message"]["content"]
    parsed_content = json.loads(content)

    assert "name" in parsed_content
    assert "age" in parsed_content
    assert "hobbies" in parsed_content
    assert "address" in parsed_content
    assert "street" in parsed_content["address"]
    assert "city" in parsed_content["address"]
    assert isinstance(parsed_content["hobbies"], list)
    assert len(parsed_content["hobbies"]) >= 1
    assert result == response_snapshot


@pytest.mark.asyncio
@pytest.mark.private
async def test_json_schema_stream(model_fixture, response_snapshot):
    """Test JSON schema validation with streaming."""
    response = requests.post(
        f"{model_fixture.base_url}/v1/chat/completions",
        json={
            "model": "tgi",
            "messages": [
                {
                    "role": "user",
                    "content": "David is a person who likes to ride bicycles. He has 2 cats.",
                },
            ],
            "seed": 42,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "value": {"name": "person", "strict": True, "schema": person_schema},
            },
            "stream": True,
        },
        stream=True,
    )

    chunks = []
    content_generated = ""

    for line in response.iter_lines():
        if line:
            # Remove the "data: " prefix and handle the special case of "[DONE]"
            data = line.decode("utf-8")
            if data.startswith("data: "):
                data = data[6:]
                if data != "[DONE]":
                    chunk = json.loads(data)
                    chunks.append(chunk)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        if (
                            "delta" in chunk["choices"][0]
                            and "content" in chunk["choices"][0]["delta"]
                        ):
                            content_generated += chunk["choices"][0]["delta"]["content"]

    # Validate the final assembled JSON
    parsed_content = json.loads(content_generated)
    assert "firstName" in parsed_content
    assert "lastName" in parsed_content
    assert "hobby" in parsed_content
    assert "numCats" in parsed_content
    assert isinstance(parsed_content["numCats"], int)
    assert parsed_content["numCats"] >= 0
    assert chunks == response_snapshot
