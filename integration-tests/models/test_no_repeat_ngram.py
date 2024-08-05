import pytest
import requests


def bloom_560_handle(launcher):
    with launcher("bigscience/bloom-560m") as handle:
        yield handle


@pytest.fixture(scope="module")
async def bloom_560(bloom_560_handle):
    await bloom_560_handle.health(240)
    return bloom_560_handle.client


@pytest.mark.asyncio
async def test_bloom_560m(bloom_560):

    base_url = bloom_560.base_url
    prompt = "The cat sat on the mat. The cat"

    repeated_2grams_control = await call_model(base_url, prompt, 0)
    assert (
        len(repeated_2grams_control) > 0
    ), "Expected to find repeated bi-grams in control case"

    repeated_2grams_test = await call_model(base_url, prompt, 2)
    assert (
        len(repeated_2grams_test) == 0
    ), f"Expected no repeated bi-grams, but found: {repeated_2grams_test}"


async def call_model(base_url, prompt, n_grams):
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 20,
            "seed": 42,
            "no_repeat_ngram_size": n_grams,
            "details": True,
        },
    }
    res = requests.post(f"{base_url}/generate", json=data)
    res = res.json()

    tokens = res["details"]["tokens"]
    token_texts = [token["text"] for token in tokens]

    # find repeated 2grams
    ngrams = [tuple(token_texts[i : i + 2]) for i in range(len(token_texts) - 2 + 1)]
    ngram_counts = {}
    for ngram in ngrams:
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
        else:
            ngram_counts[ngram] = 1

    repeated = [list(ngram) for ngram, count in ngram_counts.items() if count > 1]

    return repeated
