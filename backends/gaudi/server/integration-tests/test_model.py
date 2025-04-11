from typing import Any, Dict

from text_generation import AsyncClient
import pytest
from Levenshtein import distance as levenshtein_distance

# The "args" config is not optimized for speed but only check that the inference is working for the different models architectures
TEST_CONFIGS = {
    "meta-llama/Llama-3.1-8B-Instruct-shared": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "input": "What is Deep Learning?",
        "expected_greedy_output": " A Beginner’s Guide\nDeep learning is a subset of machine learning that involves the use",
        "expected_batch_output": " A Beginner’s Guide\nDeep learning is a subset of machine learning that involves the use",
        "args": [
            "--sharded",
            "true",
            "--num-shard",
            "8",
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "8",
            "--max-batch-prefill-tokens",
            "2048",
        ],
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "input": "What is Deep Learning?",
        "expected_greedy_output": " A Beginner’s Guide\nDeep learning is a subset of machine learning that involves the use of artificial neural networks to analyze and interpret data. It is a type of",
        "expected_batch_output": " A Beginner’s Guide\nDeep learning is a subset of machine learning that involves the use of artificial neural networks to analyze and interpret data. It is a type of",
        "env_config": {},
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
            "--max-batch-prefill-tokens",
            "2048",
        ],
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "input": "What is Deep Learning?",
        "expected_greedy_output": "\n\nDeep learning (also known as deep structured learning) is part of a broader family of machine learning techniques based on artificial neural networks\u2014specific",
        "expected_batch_output": "\n\nDeep learning (also known as deep structured learning) is part of a broader family of machine learning techniques based on artificial neural networks\u2014specific",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
            "--max-batch-prefill-tokens",
            "2048",
        ],
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "input": "What is Deep Learning?",
        "expected_greedy_output": "\n\nDeep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured",
        "expected_batch_output": "\n\nDeep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
            "--max-batch-prefill-tokens",
            "2048",
        ],
    },
    "bigcode/starcoder2-3b": {
        "model_id": "bigcode/starcoder2-3b",
        "input": "What is Deep Learning?",
        "expected_greedy_output": "\n\nDeep learning is a subset of machine learning that uses artificial neural networks to perform tasks.\n\nNeural networks are a type of machine learning algorithm that",
        "expected_batch_output": "\n\nDeep learning is a subset of machine learning that uses artificial neural networks to perform tasks.\n\nNeural networks are a type of machine learning algorithm that",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
            "--max-batch-prefill-tokens",
            "2048",
        ],
    },
    "google/gemma-7b-it": {
        "model_id": "google/gemma-7b-it",
        "input": "What is Deep Learning?",
        "expected_greedy_output": "\n\nDeep learning is a subset of machine learning that uses artificial neural networks to learn from large amounts of data. Neural networks are inspired by the structure and function of",
        "expected_batch_output": "\n\nDeep learning is a subset of machine learning that uses artificial neural networks to learn from large amounts of data. Neural networks are inspired by the structure and function of",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
            "--max-batch-prefill-tokens",
            "2048",
        ],
    },
    "Qwen/Qwen2-0.5B-Instruct": {
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "input": "What is Deep Learning?",
        "expected_greedy_output": " Deep Learning is a type of machine learning that is based on the principles of artificial neural networks. It is a type of machine learning that is used to train models",
        "expected_batch_output": " Deep Learning is a type of machine learning that is based on the principles of artificial neural networks. It is a type of machine learning that is used to train models",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
            "--max-batch-prefill-tokens",
            "2048",
        ],
    },
    "tiiuae/falcon-7b-instruct": {
        "model_id": "tiiuae/falcon-7b-instruct",
        "input": "What is Deep Learning?",
        "expected_greedy_output": "\nDeep learning is a branch of machine learning that uses artificial neural networks to learn and make decisions. It is based on the concept of hierarchical learning, where a",
        "expected_batch_output": "\nDeep learning is a branch of machine learning that uses artificial neural networks to learn and make decisions. It is based on the concept of hierarchical learning, where a",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
        ],
    },
    "microsoft/phi-1_5": {
        "model_id": "microsoft/phi-1_5",
        "input": "What is Deep Learning?",
        "expected_greedy_output": "\n\nDeep Learning is a subfield of Machine Learning that focuses on building neural networks with multiple layers of interconnected nodes. These networks are designed to learn from large",
        "expected_batch_output": "\n\nDeep Learning is a subfield of Machine Learning that focuses on building neural networks with multiple layers of interconnected nodes. These networks are designed to learn from large",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
        ],
    },
    "openai-community/gpt2": {
        "model_id": "openai-community/gpt2",
        "input": "What is Deep Learning?",
        "expected_greedy_output": "\n\nDeep learning is a new field of research that has been around for a long time. It is a new field of research that has been around for a",
        "expected_batch_output": "\n\nDeep learning is a new field of research that has been around for a long time. It is a new field of research that has been around for a",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
        ],
    },
    "facebook/opt-125m": {
        "model_id": "facebook/opt-125m",
        "input": "What is Deep Learning?",
        "expected_greedy_output": "\nAbout the Author\n\nAbout the Author\n\nAbout the Author\n\nAbout the Author\n\nAbout the Author\n\nAbout the Author\n\nAbout",
        "expected_batch_output": "\nAbout the Author\n\nAbout the Author\n\nAbout the Author\n\nAbout the Author\n\nAbout the Author\n\nAbout the Author\n\nAbout",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
        ],
    },
    "EleutherAI/gpt-j-6b": {
        "model_id": "EleutherAI/gpt-j-6b",
        "input": "What is Deep Learning?",
        "expected_greedy_output": "\n\nDeep learning is a subset of machine learning that is based on the idea of neural networks. Neural networks are a type of artificial intelligence that is inspired by",
        "expected_batch_output": "\n\nDeep learning is a subset of machine learning that is based on the idea of neural networks. Neural networks are a type of artificial intelligence that is inspired by",
        "args": [
            "--max-input-tokens",
            "512",
            "--max-total-tokens",
            "1024",
            "--max-batch-size",
            "4",
        ],
    },
}

print(f"Testing {len(TEST_CONFIGS)} models")


@pytest.fixture(scope="module", params=TEST_CONFIGS.keys())
def test_config(request) -> Dict[str, Any]:
    """Fixture that provides model configurations for testing."""
    test_config = TEST_CONFIGS[request.param]
    test_config["test_name"] = request.param
    return test_config


@pytest.fixture(scope="module")
def model_id(test_config):
    yield test_config["model_id"]


@pytest.fixture(scope="module")
def test_name(test_config):
    yield test_config["test_name"]


@pytest.fixture(scope="module")
def expected_outputs(test_config):
    return {
        "greedy": test_config["expected_greedy_output"],
        # "sampling": model_config["expected_sampling_output"],
        "batch": test_config["expected_batch_output"],
    }


@pytest.fixture(scope="module")
def input(test_config):
    return test_config["input"]


@pytest.fixture(scope="module")
def tgi_service(launcher, model_id, test_name):
    with launcher(model_id, test_name) as tgi_service:
        yield tgi_service


@pytest.fixture(scope="module")
async def tgi_client(tgi_service) -> AsyncClient:
    await tgi_service.health(1000)
    return tgi_service.client


@pytest.mark.asyncio
async def test_model_single_request(
    tgi_client: AsyncClient, expected_outputs: Dict[str, Any], input: str
):
    # Bounded greedy decoding without input
    response = await tgi_client.generate(
        input,
        max_new_tokens=32,
    )
    assert response.details.generated_tokens == 32
    assert response.generated_text == expected_outputs["greedy"]


@pytest.mark.asyncio
async def test_model_multiple_requests(
    tgi_client, generate_load, expected_outputs, input
):
    num_requests = 4
    responses = await generate_load(
        tgi_client,
        input,
        max_new_tokens=32,
        n=num_requests,
    )

    assert len(responses) == 4
    expected = expected_outputs["batch"]
    for r in responses:
        assert r.details.generated_tokens == 32
        # Compute the similarity with the expectation using the levenshtein distance
        # We should not have more than two substitutions or additions
        assert levenshtein_distance(r.generated_text, expected) < 3
