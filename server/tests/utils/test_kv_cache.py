import torch
import pytest
from text_generation_server.models.globals import ATTENTION, BLOCK_SIZE
from text_generation_server.layers.attention import KVCache
from text_generation_server.utils.import_utils import SYSTEM

def test_kvcache_memory():
    if SYSTEM == "cuda":
        kvcache_memory()
    else:
        pytest.skip("Test only runs on CUDA")

def kvcache_memory():
    num_blocks = 8188
    num_kv_heads = 8
    head_size = 128
    kv_cache_dtype = torch.float16
    device = torch.device("cuda:0")
    num_layers = 32

    current_memory = torch.cuda.memory_allocated(device)

    kv_cache = [
        KVCache(
            num_blocks=num_blocks,
            num_heads=num_kv_heads,
            head_size=head_size,
            dtype=kv_cache_dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]

    available_memory_after_kv_cache = torch.cuda.memory_allocated(device)
    kv_cache_memory = available_memory_after_kv_cache - current_memory
    kv_cache_memory_mb = kv_cache_memory / 1024 / 1024
    
    print(f"KV Cache memory: {kv_cache_memory}")
    assert kv_cache_memory_mb > 1023
    assert kv_cache_memory_mb < 1025


if __name__ == "__main__":
    test_kvcache_memory()
