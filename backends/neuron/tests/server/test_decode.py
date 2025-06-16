from helpers import create_request
from text_generation_server.generator import NeuronGenerator
from text_generation_server.pb.generate_pb2 import Batch


def test_decode(neuron_model_config):
    """Verify that a decoding for a single request generates the expected output."""
    config_name = neuron_model_config["name"]
    neuron_model_path = neuron_model_config["neuron_model_path"]
    generator = NeuronGenerator.from_pretrained(neuron_model_path)
    for do_sample in [True, False]:
        mode = "sample" if do_sample else "greedy"
        print(f"{config_name}[{mode}]")
        _test_decode(config_name, generator, do_sample)
        generator.clear()


def _test_decode(config_name, generator, do_sample):
    input_text = (
        "It was a bright cold day in April, and the clocks were striking thirteen."
    )
    max_new_tokens = 20
    request = create_request(
        id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=do_sample
    )
    max_length = generator.model.neuron_config.sequence_length
    batch = Batch(id=0, requests=[request], size=1, max_tokens=max_length)
    generations, next_batch = generator.prefill(batch)
    # We already generated one token: call decode max_new_tokens - 1 times
    for _ in range(max_new_tokens - 1):
        assert next_batch.size == 1
        assert next_batch.max_tokens == max_length
        assert len(generations) == 1
        assert len(generations[0].tokens.ids) == 1
        generations, next_batch = generator.decode([next_batch])
    assert next_batch is None
    assert len(generations) == 1
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert output.finish_reason == 0
    if do_sample:
        expected_text = {
            "llama": " I sat alone in the café",
            "qwen2": " The air was so still",
            "granite": "1984, George Orwell",
        }[config_name]
        assert expected_text in output.text
    else:
        print(output.text)
        expected_text = {
            "llama": " The world was holding its breath as the world's top scientists and engineers gathered at the secret underground facility",
            "qwen2": " I was sitting in my room, staring at the ceiling, when the door opened and in came a",
            "granite": "\n\nThis opening line from George Orwell's dystopian novel \"198",
        }[config_name]
        assert output.text == expected_text
