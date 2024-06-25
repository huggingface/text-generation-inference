import pytest
import json

from text_generation.types import GrammarType


@pytest.fixture(scope="module")
def non_flash_llama_grammar_handle(launcher):
    with launcher(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_shard=1,
        disable_grammar_support=False,
        use_flash_attention=False,
    ) as handle:
        yield handle


@pytest.fixture(scope="module")
async def non_flash_llama_grammar(non_flash_llama_grammar_handle):
    await non_flash_llama_grammar_handle.health(300)
    return non_flash_llama_grammar_handle.client


@pytest.mark.release
@pytest.mark.skip
@pytest.mark.asyncio
async def test_non_flash_llama_grammar_json(non_flash_llama_grammar, response_snapshot):
    response = await non_flash_llama_grammar.generate(
        "info: david holtz like trees and has two cats. ",
        max_new_tokens=100,
        decoder_input_details=True,
        seed=0,
        grammar={
            "type": GrammarType.Json,
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
