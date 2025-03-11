from helpers import create_request
from text_generation_server.generator import NeuronGenerator
from text_generation_server.pb.generate_pb2 import Batch


def test_prefill(neuron_model_config):
    """Verify that a prefill for a single request generates the expected output."""
    config_name = neuron_model_config["name"]
    neuron_model_path = neuron_model_config["neuron_model_path"]
    generator = NeuronGenerator.from_pretrained(neuron_model_path)
    max_batch_size = 4
    assert generator.model.batch_size >= max_batch_size
    for num_requests in [1, max_batch_size]:
        for do_sample in [True, False]:
            mode = "sample" if do_sample else "greedy"
            print(f"[{mode}]: {num_requests} requests")
            _test_prefill(config_name, generator, num_requests, do_sample)
            generator.clear()


def _test_prefill(config_name, generator, batch_size, do_sample):
    requests = []
    max_new_tokens = 20
    input_text = (
        "It was a bright cold day in April, and the clocks were striking thirteen."
    )
    for i in range(batch_size):
        requests.append(
            create_request(
                id=i,
                inputs=input_text,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
            )
        )
    # Let's be pessimistic when estimating max_tokens
    max_length = generator.model.max_length
    batch = Batch(
        id=0, requests=requests, size=batch_size, max_tokens=batch_size * max_length
    )
    generations, next_batch = generator.prefill(batch)
    assert next_batch.size == batch_size
    # Whatever was passed as max_tokens, the server will correct it
    # because of static batching
    assert next_batch.max_tokens == batch_size * max_length
    assert len(generations) == batch_size
    if do_sample:
        expectations = {
            "gpt2": [383, " The"],
            "llama": [10058, " George"],
            "mistral": [450, " The"],
            "qwen2": [362, " A"],
            "granite": [308, " ("],
        }[config_name]
    else:
        expectations = {
            "gpt2": [198, "\n"],
            "llama": [10058, " George"],
            "mistral": [13, "\n"],
            "qwen2": [358, " I"],
            "granite": [203, "\n"],
        }[config_name]
    for g in generations:
        tokens = g.tokens
        assert tokens.ids[0] == expectations[0]
        assert tokens.texts[0] == expectations[1]


def test_prefill_truncate(neuron_model_config):
    config_name = neuron_model_config["name"]
    neuron_model_path = neuron_model_config["neuron_model_path"]
    generator = NeuronGenerator.from_pretrained(neuron_model_path)
    batch_size = generator.model.batch_size
    # We apply truncation to all requests but the first one
    truncate = [
        None,
    ] + [i * 3 for i in range(1, batch_size)]
    input_text = (
        "Two gin-scented tears trickled down the sides of his nose."
        " But it was all right, everything was all right, the struggle was finished."
        " He had won the victory over himself. He loved Big Brother."
    )
    requests = []
    for i in range(batch_size):
        requests.append(create_request(id=i, inputs=input_text, truncate=truncate[i]))
    max_length = generator.model.max_length
    batch = Batch(
        id=0, requests=requests, size=batch_size, max_tokens=batch_size * max_length
    )
    generations, _ = generator.prefill(batch)
    # Even if the input text is identical for all requests, the first generated token might
    # be different because of the truncation
    expectations = {
        "gpt2": [" He", " He", "\n", " He"],
        "llama": [" —", " The", " He", " He"],
        "mistral": [" He", "\n", " He", " He"],
        "qwen2": [" He", " The", " He", " He"],
        "granite": ["\n", "\n", " I", " He"],
    }[config_name]
    for i, g in enumerate(generations):
        tokens = g.tokens
        assert tokens.texts[0] == expectations[i]
