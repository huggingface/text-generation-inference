import pytest
import torch

from transformers import AutoTokenizer

from text_generation_server.pb import generate_pb2
from text_generation_server.models.flash_causal_lm import (
    FlashCausalLMBatch,
    FlashCausalLM,
)
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaForCausalLM,
)
from text_generation_server.models.globals import set_adapter_to_index
from text_generation_server.utils.import_utils import SYSTEM, empty_cache, synchronize
from unittest.mock import Mock
import base64

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

set_adapter_to_index({})


def test_flash_causal_lm_warmup():
    if SYSTEM == "cuda":
        flash_causal_lm_warmup()
    else:
        pytest.skip("Test only runs on CUDA")


def flash_causal_lm_warmup():
    revision = "main"
    quantize = None
    speculator = False
    dtype = torch.float16
    kv_cache_dtype = torch.float16
    trust_remote_code = False
    lora_adapter_ids = None
    device = torch.device("cuda:0")

    current_memory = torch.cuda.memory_allocated(device)

    default_causal_lm = FlashCausalLM(
        model_id=model_id,
        model_class=FlashLlamaForCausalLM,
        revision=revision,
        quantize=quantize,
        speculator=speculator,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        trust_remote_code=trust_remote_code,
        lora_adapter_ids=lora_adapter_ids,
    )
    model_tokenizer = AutoTokenizer.from_pretrained(model_id)

    available_memory_after_model_and_tokenizer = torch.cuda.memory_allocated(device)
    model_and_tokenizer_memory = (
        available_memory_after_model_and_tokenizer - current_memory
    )
    model_and_tokenizer_memory_mb = model_and_tokenizer_memory / 1024 / 1024
    print(f"Model and Tokenizer memory: {model_and_tokenizer_memory}")

    default_pb_parameters = generate_pb2.NextTokenChooserParameters(
        temperature=0.9,
        top_k=10,
        top_p=0.9,
        typical_p=0.9,
        repetition_penalty=1.2,
        watermark=True,
        frequency_penalty=0.1,
    )

    default_pb_stop_parameters = generate_pb2.StoppingCriteriaParameters(
        stop_sequences=[], max_new_tokens=1024, ignore_eos_token=True
    )

    # define the batches to check and the expected number of blocks and output
    batches_and_expected = [
        Mock(
            num_requests=8,
            expected_num_blocks=16376,
            expected_max_supported_total_tokens=22449,
        ),
        Mock(
            num_requests=4,
            expected_num_blocks=8188,
            expected_max_supported_total_tokens=30768,
        ),
        Mock(
            num_requests=2,
            expected_num_blocks=4094,
            expected_max_supported_total_tokens=40445,
        ),
    ]
    for batch_of_size_n in batches_and_expected:
        empty_cache()
        synchronize()

        # build the inputs (similar to prefill used in warmup)
        inputs_text = "_test " * 1024
        b64_encoded_image = "iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAABg2lDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV/TSotUROxQxCFDdbKLijjWKhShQqgVWnUwufQLmrQkKS6OgmvBwY/FqoOLs64OroIg+AHi7OCk6CIl/i8ptIjx4Lgf7+497t4BQqvKNDOQADTdMjKppJjLr4rBVwQQwhAERGVm1uckKQ3P8XUPH1/v4jzL+9yfY0AtmAzwicQJVjcs4g3imU2rznmfOMLKskp8Tjxh0AWJH7muuPzGueSwwDMjRjYzTxwhFks9rPQwKxsa8TRxTNV0yhdyLquctzhr1Qbr3JO/MFzQV5a5TnMUKSxiCRJEKGiggiosxGnVSTGRof2kh3/E8UvkUshVASPHAmrQIDt+8D/43a1ZnJp0k8JJoO/Ftj/GgOAu0G7a9vexbbdPAP8zcKV3/bUWMPtJerOrxY6AwW3g4rqrKXvA5Q4QfarLhuxIfppCsQi8n9E35YHhW6B/ze2ts4/TByBLXaVvgINDYLxE2ese7w719vbvmU5/PycecohsjayNAAAACXBIWXMAAC4jAAAuIwF4pT92AAAAB3RJTUUH6AQIEQMnlTSSjwAAABl0RVh0Q29tbWVudABDcmVhdGVkIHdpdGggR0lNUFeBDhcAAAASSURBVDjLY2AYBaNgFIyCoQsABMQAAeRw1DoAAAAASUVORK5CYII="
        inputs_image = f"![](data:image/jpeg;base64,{b64_encoded_image})"
        inputs = inputs_text + inputs_image
        # inputs are also added as chunks to define the type
        input_chunks = [
            generate_pb2.InputChunk(text=inputs_text),
            generate_pb2.InputChunk(
                image=generate_pb2.Image(
                    # convert the base64 encoded image to bytes by decoding it
                    data=base64.b64decode(b64_encoded_image),
                    mimetype="image/jpeg;base64",
                )
            ),
        ]

        # build a batch of size n requests
        default_pb_requests = []
        for i in range(batch_of_size_n.num_requests):
            req = generate_pb2.Request(
                id=i,
                inputs=inputs,
                input_chunks=generate_pb2.Input(chunks=input_chunks),
                prefill_logprobs=True,
                truncate=1024,
                parameters=default_pb_parameters,
                stopping_parameters=default_pb_stop_parameters,
            )
            default_pb_requests.append(req)

        # convert the list of requests to a FlashCausalLMBatch this will calculate the number of blocks
        default_pb_batch = generate_pb2.Batch(
            id=0, requests=default_pb_requests, size=batch_of_size_n.num_requests
        )
        default_flash_causal_lm_batch = FlashCausalLMBatch.from_pb(
            default_pb_batch, model_tokenizer, torch.float16, torch.device("cuda:0")
        )
        print("number of blocks", default_flash_causal_lm_batch.num_blocks)
        assert (
            default_flash_causal_lm_batch.num_blocks
            == batch_of_size_n.expected_num_blocks
        )
        max_supported_total_tokens = default_causal_lm.warmup(
            default_flash_causal_lm_batch
        )
        print("output", max_supported_total_tokens)
        assert (
            max_supported_total_tokens
            == batch_of_size_n.expected_max_supported_total_tokens
        )
        warmup_response = generate_pb2.WarmupResponse(
            max_supported_total_tokens=max_supported_total_tokens
        )


if __name__ == "__main__":
    test_flash_causal_lm_warmup()
