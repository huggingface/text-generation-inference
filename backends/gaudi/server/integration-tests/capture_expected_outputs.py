import json
import os
from typing import Dict, Any, Generator

import pytest
from test_model import TEST_CONFIGS

UNKNOWN_CONFIGS = {
    name: config
    for name, config in TEST_CONFIGS.items()
    if config["expected_greedy_output"] == "unknown"
    or config["expected_batch_output"] == "unknown"
}


@pytest.fixture(scope="module", params=UNKNOWN_CONFIGS.keys())
def test_config(request) -> Dict[str, Any]:
    """Fixture that provides model configurations for testing."""
    test_config = UNKNOWN_CONFIGS[request.param]
    test_config["test_name"] = request.param
    return test_config


@pytest.fixture(scope="module")
def test_name(test_config):
    yield test_config["test_name"]


@pytest.fixture(scope="module")
def tgi_service(launcher, test_config, test_name) -> Generator:
    """Fixture that provides a TGI service for testing."""
    with launcher(test_config["model_id"], test_name) as service:
        yield service


@pytest.mark.asyncio
async def test_capture_expected_outputs(tgi_service, test_config, test_name):
    """Test that captures expected outputs for models with unknown outputs."""
    print(f"Testing {test_name} with {test_config['model_id']}")

    # Wait for service to be ready
    await tgi_service.health(1000)
    client = tgi_service.client

    # Test single request (greedy)
    print("Testing single request...")
    response = await client.generate(
        test_config["input"],
        max_new_tokens=32,
    )
    greedy_output = response.generated_text

    # Test multiple requests (batch)
    print("Testing batch requests...")
    responses = []
    for _ in range(4):
        response = await client.generate(
            test_config["input"],
            max_new_tokens=32,
        )
        responses.append(response.generated_text)

    # Store results in a JSON file
    output_file = "server/integration-tests/expected_outputs.json"
    results = {}

    # Try to load existing results if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)

    # Update results for this model
    results[test_name] = {
        "model_id": test_config["model_id"],
        "input": test_config["input"],
        "greedy_output": greedy_output,
        "batch_outputs": responses,
        "args": test_config["args"],
    }

    # Save updated results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults for {test_name} saved to {output_file}")
