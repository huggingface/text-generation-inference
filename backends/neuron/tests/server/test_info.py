from text_generation_server.generator import NeuronGenerator


def test_info(neuron_model_path):
    generator = NeuronGenerator.from_pretrained(neuron_model_path)
    info = generator.info
    assert info.requires_padding is True
    assert info.device_type == "xla"
    assert info.window_size == 0
    assert info.speculate == 0
