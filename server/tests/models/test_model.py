import pytest
import torch

from transformers import AutoTokenizer

from text_generation_server.models import Model


@pytest.mark.private
def test_decode_streaming():
    class TestModel(Model):
        def batch_type(self):
            raise NotImplementedError

        def generate_token(self, batch):
            raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained("huggingface/llama-7b")

    model = TestModel(
        torch.nn.Linear(1, 1), tokenizer, False, torch.float32, torch.device("cpu")
    )

    all_input_ids = [
        30672,
        232,
        193,
        139,
        233,
        135,
        162,
        235,
        179,
        165,
        30919,
        30210,
        234,
        134,
        176,
        30993,
    ]

    truth = "我很感谢你的热情"

    decoded_text = ""
    offset = None
    token_offset = None
    for i in range(len(all_input_ids)):
        text, offset, token_offset = model.decode_token(
            all_input_ids[: i + 1], offset, token_offset
        )
        decoded_text += text

    assert decoded_text == truth
