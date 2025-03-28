from typing import Any, Dict

from text_generation import AsyncClient
import pytest
from Levenshtein import distance as levenshtein_distance

TEST_CONFIGS = {
    "llama3-8b-shared-gaudi1": {
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
    "llama3-8b-gaudi1": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "input": "What is Deep Learning?",
        "expected_greedy_output": " A Beginner’s Guide\nDeep learning is a subset of machine learning that involves the use of artificial neural networks to analyze and interpret data. It is a type of",
        # "expected_sampling_output": " Part 2: Backpropagation\nIn my last post , I introduced the concept of deep learning and covered some high-level topics, including feedforward neural networks.",
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
}


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

    # TO FIX: Add sampling tests later, the gaudi backend seems to be flaky with the sampling even with a fixed seed

    # Sampling
    # response = await tgi_client.generate(
    #     "What is Deep Learning?",
    #     do_sample=True,
    #     top_k=50,
    #     top_p=0.9,
    #     repetition_penalty=1.2,
    #     max_new_tokens=100,
    #     seed=42,
    #     decoder_input_details=True,
    # )

    # assert expected_outputs["sampling"] in response.generated_text


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
