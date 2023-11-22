# encoding:utf-8
# -------------------------------------------#
# Filename: optims -- test_llama_fa.py
#
# Description:
# Version:       1.0
# Created:       2023/9/18-20:50
# Last modified by:
# Author:        'zhaohuayang@myhexin.com'
# Company:       同花顺网络信息股份有限公司
# -------------------------------------------#
import math
import time
from pathlib import Path
from typing import List, Optional

# Flash attention imports
import numpy as np
import torch
import torch.distributed
import transformers

from flash_llama_modeling import FlashLlamaForCausalLM
from utils import initialize_torch_distributed
from weights import Weights

BLOCK_SIZE = 16


class CacheManager:
    def __init__(
            self,
            num_blocks: int,
            num_layers: int,
            num_heads: int,
            head_size: int,
            dtype: torch.dtype,
            device: torch.device,
    ):
        self.block_size = BLOCK_SIZE
        self.num_blocks = num_blocks
        self.device = device

        element_size = torch.tensor([], dtype=dtype).element_size()
        x = self.block_size // element_size

        self.kv_cache = [
            (
                torch.empty(
                    (num_blocks, num_heads, head_size // x, self.block_size, x),
                    dtype=dtype,
                    device=device,
                ),
                torch.empty(
                    (num_blocks, num_heads, head_size, self.block_size),
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in range(num_layers)
        ]
        self.free_block_mask = torch.ones(num_blocks, dtype=torch.int32, device="cpu")
        self.slots = torch.arange(
            0, num_blocks * self.block_size, dtype=torch.int32
        ).view(num_blocks, self.block_size)

    def allocate(self, blocks, max_blocks, needed_blocks_slots):
        """
        blocks: 总共需要的blocks数量
        max_blocks: 最大的blocks数量大小
        needed_blocks_slots: 每个序列所需的blocks及其对应的序列长度
        """
        # Get free blocks indices by finding values in mask that are not set to 0
        free_block_indices = self.free_block_mask.nonzero()
        assert (
                len(free_block_indices) >= blocks
        ), f"Out of available cache blocks: asked {blocks}, only {len(free_block_indices)} free blocks"

        # Slice by the number of required blocks
        block_indices = free_block_indices[: blocks]
        block_indices = block_indices.flatten()

        # Padded block tables
        block_tables_tensor = torch.zeros(
            (len(needed_blocks_slots), max_blocks), dtype=torch.int32
        )

        # Allocate paged attention blocks
        cumulative_blocks = 0
        slots = []
        block_tables = []
        for i, (needed_blocks, needed_slots) in enumerate(needed_blocks_slots):
            # Get allocated blocks for this sequence
            allocated_blocks = block_indices[
                               cumulative_blocks: cumulative_blocks + needed_blocks
                               ]
            # Get slots for the allocated blocks
            allocated_slots = self.slots[allocated_blocks].flatten()[:needed_slots]

            slots.append(allocated_slots)
            block_tables.append(allocated_blocks.tolist())
            block_tables_tensor[i, :needed_blocks] = allocated_blocks
            cumulative_blocks += needed_blocks

        # Allocate the required number of blocks by setting the mask to 0
        self.free_block_mask[block_indices] = 0

        return block_tables, block_tables_tensor.to(self.device), torch.concat(slots).to(self.device)

    def free(self, block_indices: Optional[List[int]]):
        if block_indices is not None and block_indices:
            # Reset mask
            self.free_block_mask[block_indices] = 1


def generate(tokenizer, model, config, device, prompt, max_new_tokens=10):
    input_ids = tokenizer(prompt).input_ids

    def warmup():
        print("start warmup...")
        global CACHE_MANAGER
        blocks = 260
        CACHE_MANAGER = CacheManager(blocks,
                                     len(model.model.layers),
                                     model.model.num_key_value_heads,
                                     model.model.head_size,
                                     torch.float16,
                                     device)
        input_length = 1024
        bs = 4
        warmup_inputs = {
            'input_ids': torch.arange(1, input_length + 1, dtype=torch.int64, device=device).repeat(bs),
            'position_ids': torch.arange(0, input_length, dtype=torch.int32, device=device).repeat(bs),
            'cu_seqlen_prefill': torch.tensor([i * input_length for i in range(bs + 1)], dtype=torch.int32,
                                              device=device),
            'block_tables': torch.arange(0, blocks, dtype=torch.int32, device=device).split(blocks // bs),
            'slots': torch.arange(0, 4144, dtype=torch.int32, device=device),
            'input_lengths': torch.tensor([input_length] * 4, dtype=torch.int32, device=device),
            'max_s': 1024,
            'lm_head_indices': None
        }
        model.forward(**warmup_inputs, kv_cache=CACHE_MANAGER.kv_cache)

        del CACHE_MANAGER
        torch.cuda.empty_cache()

    # 预热
    warmup()

    print("start speed test running")
    # 申请缓存空间
    global CACHE_MANAGER
    CACHE_MANAGER = CacheManager(100,
                                 len(model.model.layers),
                                 model.model.num_key_value_heads,
                                 model.model.head_size,
                                 torch.float16,
                                 device)
    total_tokens = len(input_ids) + max_new_tokens - 1
    needed_blocks = math.ceil(total_tokens / BLOCK_SIZE)
    needed_blocks_slots = [(needed_blocks, total_tokens)]
    _, block_tables_tensor, slots = CACHE_MANAGER.allocate(needed_blocks, needed_blocks, needed_blocks_slots)
    # forward循环
    loops = 10
    tpss = []
    for loop in range(loops):
        print(f"loop {loop}...")
        times = []
        new_tokens = []
        for step in range(max_new_tokens):
            if step == 0:
                # prefill step
                slot_indices = torch.arange(0, 0 + len(input_ids), dtype=torch.int64)
                inputs = {
                    'input_ids': torch.tensor(input_ids, dtype=torch.int64, device=device),
                    'position_ids': torch.arange(0, len(input_ids), dtype=torch.int32, device=device),
                    'cu_seqlen_prefill': torch.tensor([0, len(input_ids)], dtype=torch.int32, device=device),
                    'block_tables': block_tables_tensor,
                    'slots': slots[slot_indices],
                    'input_lengths': torch.tensor([len(input_ids)], dtype=torch.int32, device=device),
                    'max_s': len(input_ids),
                    'lm_head_indices': torch.tensor([0 + len(input_ids) - 1], dtype=torch.int32, device=device)
                }
            else:
                # incremental step
                current_length = len(input_ids) + step
                inputs = {
                    'input_ids': new_tokens[-1],
                    'position_ids': torch.tensor([current_length - 1], dtype=torch.int32, device=device),
                    'cu_seqlen_prefill': None,
                    'block_tables': block_tables_tensor,
                    'slots': torch.tensor([current_length - 1], dtype=torch.int32, device=device),
                    'input_lengths': torch.tensor([current_length], dtype=torch.int32, device=device),
                    'max_s': current_length,
                    'lm_head_indices': None
                }
            torch.cuda.synchronize()
            s_time = time.time()
            logits = model.forward(**inputs, kv_cache=CACHE_MANAGER.kv_cache)
            torch.cuda.synchronize()
            cost_time = time.time() - s_time
            next_token_id = logits.argmax(dim=-1)
            new_tokens.append(next_token_id)
            times.append(round(cost_time, 6))

        if loop == 0:
            new_tokens = torch.concat(new_tokens)
            print(tokenizer.decode(new_tokens, skip_special_tokens=True))

        elapsed_time = np.mean(times)
        tps = 1 / elapsed_time
        tpss.append(tps)
        print(times)
        print(f"total new tokens: {max_new_tokens}, cost time: {sum(times):.6f} s\n"
              f"time_per_token: {elapsed_time * 1000:.3f} ms, tps: {tps:.2f} tokens/s")
    print(f'mean tps: {np.mean(tpss):.2f} tokens/s')


def main(model_path):
    # init env
    process_group, rank, world_size = initialize_torch_distributed()
    # step 0: 定义路径与属性
    model_path = Path(model_path)
    config = transformers.AutoConfig.from_pretrained(model_path)
    config.quantize = None
    model_files = list(model_path.glob('*.safetensors'))

    # step 1: 定义tokenizer与权重
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="left", truncation_side="left")
    device = torch.device(f"cuda:{rank}")
    weights = Weights(model_files, device, torch.float16, process_group=process_group)

    # step2: 定义模型
    torch.distributed.barrier(group=process_group)
    model = FlashLlamaForCausalLM(config, weights).eval()
    torch.distributed.barrier(group=process_group)
    print(model)

    # step3: 推理
    with torch.no_grad():
        prompt = "who are you?"
        generate(tokenizer, model, config, device, prompt, max_new_tokens=100)


if __name__ == '__main__':
    CACHE_MANAGER: Optional[CacheManager] = None
    main('/code/models/llama-7b-hf')
