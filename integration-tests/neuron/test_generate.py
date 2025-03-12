import pytest


@pytest.fixture
async def tgi_service(neuron_launcher, neuron_model_config):
    model_name_or_path = neuron_model_config["neuron_model_path"]
    service_name = neuron_model_config["name"]
    with neuron_launcher(service_name, model_name_or_path) as tgi_service:
        await tgi_service.health(600)
        yield tgi_service


@pytest.mark.asyncio
async def test_model_single_request(tgi_service):
    service_name = tgi_service.client.service_name
    prompt = "What is Deep Learning?"
    # Greedy bounded without input
    response = await tgi_service.client.text_generation(
        prompt, max_new_tokens=17, details=True, decoder_input_details=True
    )
    assert response.details.generated_tokens == 17
    greedy_expectations = {
        "gpt2": "\n\nDeep learning is a new field of research that has been around for a while",
        "llama": " and How Does it Work?\nDeep learning is a subset of machine learning that uses artificial",
        "mistral": "\nWhat is Deep Learning?\nDeep Learning is a type of machine learning that",
        "qwen2": " - Part 1\n\nDeep Learning is a subset of Machine Learning that is based on",
        "granite": "\n\nDeep Learning is a subset of Machine Learning, which is a branch of Art",
    }
    assert response.generated_text == greedy_expectations[service_name]

    # Greedy bounded with input
    response = await tgi_service.client.text_generation(
        "What is Deep Learning?",
        max_new_tokens=17,
        return_full_text=True,
        details=True,
        decoder_input_details=True,
    )
    assert response.details.generated_tokens == 17
    assert response.generated_text == prompt + greedy_expectations[service_name]

    # Sampling
    response = await tgi_service.client.text_generation(
        "What is Deep Learning?",
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=128,
        seed=42,
    )
    # The response must be different
    assert not response.startswith(greedy_expectations[service_name])

    # Sampling with stop sequence (using one of the words returned from the previous test)
    stop_sequence = response.split(" ")[-5]
    response = await tgi_service.client.text_generation(
        "What is Deep Learning?",
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=128,
        seed=42,
        stop_sequences=[stop_sequence],
    )
    assert response.endswith(stop_sequence)


@pytest.mark.asyncio
async def test_model_multiple_requests(tgi_service, neuron_generate_load):
    num_requests = 4
    responses = await neuron_generate_load(
        tgi_service.client,
        "What is Deep Learning?",
        max_new_tokens=17,
        n=num_requests,
    )

    assert len(responses) == 4
    expectations = {
        "gpt2": "Deep learning is a new field of research that has been around for a while",
        "llama": "Deep learning is a subset of machine learning that uses artificial",
        "mistral": "Deep Learning is a type of machine learning that",
        "qwen2": "Deep Learning is a subset of Machine Learning that is based on",
        "granite": "Deep Learning is a subset of Machine Learning, which is a branch of Art",
    }
    expected = expectations[tgi_service.client.service_name]
    for r in responses:
        assert r.details.generated_tokens == 17
        assert expected in r.generated_text
