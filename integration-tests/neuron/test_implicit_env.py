import os

import pytest


@pytest.fixture(scope="module", params=["hub-neuron", "hub", "local-neuron"])
async def tgi_service(request, neuron_launcher, neuron_model_config):
    """Expose a TGI service corresponding to a model configuration

    For each model configuration, the service will be started using the following
    deployment options:
    - from the hub original model (export parameters chosen after hub lookup),
    - from the hub pre-exported neuron model,
    - from a local path to the neuron model.
    """
    # the tgi_env.py script will take care of setting these
    for var in [
        "MAX_BATCH_SIZE",
        "MAX_INPUT_TOKENS",
        "MAX_TOTAL_TOKENS",
        "HF_NUM_CORES",
        "HF_AUTO_CAST_TYPE",
    ]:
        if var in os.environ:
            del os.environ[var]
    if request.param == "hub":
        model_name_or_path = neuron_model_config["model_id"]
    elif request.param == "hub-neuron":
        model_name_or_path = neuron_model_config["neuron_model_id"]
    else:
        model_name_or_path = neuron_model_config["neuron_model_path"]
    service_name = neuron_model_config["name"]
    with neuron_launcher(service_name, model_name_or_path) as tgi_service:
        await tgi_service.health(600)
        yield tgi_service


@pytest.mark.asyncio
async def test_model_single_request(tgi_service):
    # Just verify that the generation works, and nothing is raised, with several set of params

    # No params
    await tgi_service.client.text_generation(
        "What is Deep Learning?",
    )

    response = await tgi_service.client.text_generation(
        "How to cook beans ?",
        max_new_tokens=17,
        details=True,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17

    # Sampling
    await tgi_service.client.text_generation(
        "What is Deep Learning?",
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=128,
        seed=42,
    )
