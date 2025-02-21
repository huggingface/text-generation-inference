from helpers import create_request
from text_generation_server.generator import NeuronGenerator
from text_generation_server.pb.generate_pb2 import Batch


def test_continuous_batching_two_requests(neuron_model_config):
    """Verify that two requests added to the batch at different generation steps
    generate the same outputs (continuous batching).
    """
    neuron_model_path = neuron_model_config["neuron_model_path"]
    generator = NeuronGenerator.from_pretrained(neuron_model_path)
    assert generator.model.batch_size > 1
    input_text = "Once upon a time"
    max_new_tokens = 20
    # Prefill a single request, remembering the generated token
    tokens = {0: [], 1: []}
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens)
    max_length = generator.model.max_length
    batch = Batch(id=0, requests=[request], size=1, max_tokens=max_length)
    generations, next_batch = generator.prefill(batch)
    assert next_batch.size == 1
    assert len(generations) == 1
    g = generations[0]
    tokens[g.request_id].append(g.tokens.ids[0])
    assert len(tokens[0]) == 1
    # Decode a few tokens
    gen_tokens = 4
    for _ in range(gen_tokens - 1):
        generations, next_batch = generator.decode([next_batch])
        assert len(generations) == 1
        g = generations[0]
        tokens[g.request_id].append(g.tokens.ids[0])
    assert len(tokens[0]) == gen_tokens
    assert next_batch.size == 1
    # Add a second request
    request = create_request(id=1, inputs=input_text, max_new_tokens=max_new_tokens)
    batch = Batch(id=1, requests=[request], size=1, max_tokens=max_length)
    generations, next_batch_1 = generator.prefill(batch)
    assert next_batch_1.size == 1
    # We should have generated only a single token
    assert len(generations) == 1
    g = generations[0]
    tokens[g.request_id].append(g.tokens.ids[0])
    assert len(tokens[0]) == gen_tokens
    assert len(tokens[1]) == 1
    # Decode more tokens until we reach the maximum for the first request
    batches = [next_batch, next_batch_1]
    for _ in range(max_new_tokens - gen_tokens):
        generations, next_batch = generator.decode(batches)
        for g in generations:
            tokens[g.request_id].append(g.tokens.ids[0])
        batches = [next_batch]
    # Verify we now only have one pending request
    assert next_batch.size == 1
    assert len(tokens[0]) == max_new_tokens
    assert len(tokens[1]) == max_new_tokens - gen_tokens + 1
    # Verify we have the output for the first request
    for g in generations:
        if g.request_id == 0:
            output = g.generated_text
            assert output.text != ""
            assert output.generated_tokens == max_new_tokens
            generated_text = output.text
    # Continue decoding until the end of the second request
    for _ in range(gen_tokens - 1):
        generations, next_batch = generator.decode([next_batch])
        assert len(generations) == 1
        g = generations[0]
        tokens[g.request_id].append(g.tokens.ids[0])
    assert next_batch is None
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert tokens[0] == tokens[1]
    assert output.text == generated_text
