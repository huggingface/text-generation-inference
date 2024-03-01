import pytest
import json

from text_generation.types import GrammarType


@pytest.fixture(scope="module")
def flash_llama_grammar_handle(launcher):
    with launcher(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", num_shard=2, disable_grammar_support=False
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def flash_llama_grammar(flash_llama_grammar_handle):
    await flash_llama_grammar_handle.health(300)
    return flash_llama_grammar_handle.client


@pytest.mark.asyncio
async def test_flash_llama_grammar(flash_llama_grammar, response_snapshot):
    response = await flash_llama_grammar.generate(
        "Test request", max_new_tokens=10, decoder_input_details=True
    )

    assert response.details.generated_tokens == 10
    assert response == response_snapshot


@pytest.mark.skip
@pytest.mark.asyncio
async def test_flash_llama_grammar_regex(flash_llama_grammar, response_snapshot):
    response = await flash_llama_grammar.generate(
        "Whats Googles DNS",
        max_new_tokens=10,
        decoder_input_details=True,
        seed=0,
        grammar={
            "type": GrammarType.Regex,  # "regex"
            "value": "((25[0-5]|2[0-4]\\d|[01]?\\d\\d?)\\.){3}(25[0-5]|2[0-4]\\d|[01]?\\d\\d?)",
        },
    )

    assert response.details.generated_tokens == 10
    assert response.generated_text == "42.1.1.101"
    assert response == response_snapshot


@pytest.mark.skip
@pytest.mark.asyncio
async def test_flash_llama_grammar_json(flash_llama_grammar, response_snapshot):
    response = await flash_llama_grammar.generate(
        "info: david holtz like trees and has two cats. ",
        max_new_tokens=100,
        decoder_input_details=True,
        seed=0,
        grammar={
            "type": GrammarType.Json,  # "json"
            "value": json.dumps(
                {
                    "type": "object",
                    "$id": "https://example.com/person.schema.json",
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "Person",
                    "properties": {
                        "firstName": {
                            "type": "string",
                            "description": "The person'''s first name.",
                        },
                        "lastName": {
                            "type": "string",
                            "description": "The person'''s last name.",
                        },
                        "hobby": {
                            "description": "The person'''s hobby.",
                            "type": "string",
                        },
                        "numCats": {
                            "description": "The number of cats the person has.",
                            "type": "integer",
                            "minimum": 0,
                        },
                    },
                    "required": ["firstName", "lastName", "hobby", "numCats"],
                }
            ),
        },
    )

    assert response.details.generated_tokens == 30
    assert (
        response.generated_text
        == '{"firstName":"David","hobby":"Trees","lastName":"Holtz","numCats":2}'
    )
    assert response == response_snapshot


@pytest.mark.skip
@pytest.mark.asyncio
async def test_flash_llama_grammar_load(
    flash_llama_grammar, generate_load, response_snapshot
):
    responses = await generate_load(
        flash_llama_grammar,
        "name: david. email:  ",
        max_new_tokens=10,
        n=4,
        stop_sequences=[".com"],
        seed=0,
        grammar={
            "type": GrammarType.Regex,  # "regex"
            "value": "[\\w-]+@([\\w-]+\\.)+[\\w-]+",  # email regex
        },
    )

    assert len(responses) == 4

    expected = "123456@gmail.com"

    for response in responses:
        assert response.generated_text == expected

    assert all([r.generated_text == responses[0].generated_text for r in responses])

    assert responses == response_snapshot


# this is the same as the above test, but only fires off a single request
# this is only to ensure that the parallel and single inference produce the same result
@pytest.mark.skip
@pytest.mark.asyncio
async def test_flash_llama_grammar_single_load_instance(
    flash_llama_grammar, generate_load, response_snapshot
):
    response = await flash_llama_grammar.generate(
        "name: david. email:  ",
        max_new_tokens=10,
        stop_sequences=[".com"],
        seed=0,
        grammar={
            "type": GrammarType.Regex,  # "regex"
            "value": "[\\w-]+@([\\w-]+\\.)+[\\w-]+",  # email regex
        },
    )

    # assert response.details.generated_tokens == 30
    assert response.generated_text == "123456@gmail.com"

    assert response == response_snapshot
