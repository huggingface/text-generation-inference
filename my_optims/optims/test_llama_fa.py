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
from typing import List, Dict, Optional
from typing import Tuple

# Flash attention imports
import dropout_layer_norm
import flash_attn_2_cuda
import numpy as np
import rotary_emb
import torch
import torch.distributed
import transformers
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from vllm import attention_ops, cache_ops

# vllm imports

BLOCK_SIZE = 16


class FastLinear(nn.Module):
    def __init__(
            self,
            weight,
            bias,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class PositionRotaryEmbedding(nn.Module):
    def __init__(self, inv_freq, scaling_factor):
        super().__init__()
        self.inv_freq = inv_freq
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.scaling_factor = scaling_factor
        self.dynamic_args = None

    @staticmethod
    def _create_inv_freq(dim, base, device):
        inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        return inv_freq

    @classmethod
    def static(cls, config, dim, base, device):
        inv_freq = cls._create_inv_freq(dim, base, device)
        scaling_factor = None
        return cls(inv_freq, scaling_factor)

    def _update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            if self.scaling_factor is not None:
                t /= self.scaling_factor
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def get_cos_sin(
            self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype
    ):
        """
        Return cos and sin for the asked position ids
        """

        self._update_cos_sin_cache(dtype, position_ids.device, max_s)

        cos = torch.index_select(self._cos_cached, 0, position_ids)
        sin = torch.index_select(self._sin_cached, 0, position_ids)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        rotary_dim = cos.shape[-1]
        x1 = x[..., :rotary_dim]
        x2 = x[..., rotary_dim: 2 * rotary_dim]

        rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)
        return x


class Weights:
    def __init__(
            self,
            filenames: List[Path],
            device,
            dtype,
            process_group,
            aliases: Optional[Dict[str, List[str]]] = None,
    ):
        routing = {}
        for filename in filenames:
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        raise RuntimeError(
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                        )
                    routing[k] = filename
        if aliases is None:
            aliases = {}
        self.aliases = aliases
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self.process_group = process_group
        self._handles = {}

    def _get_handle(self, filename):
        if filename not in self._handles:
            f = safe_open(filename, framework="pytorch")
            self._handles[filename] = f

        return self._handles[filename]

    def get_filename(self, tensor_name: str) -> (str, str):
        filename = self.routing.get(tensor_name, None)
        if filename is None:
            aliases = self.aliases.get(tensor_name, [])
            for alias in aliases:
                filename = self.routing.get(alias, None)
                if filename is not None:
                    return str(filename), alias
            raise RuntimeError(f"weight {tensor_name} does not exist")
        return str(filename), tensor_name

    def get_tensor(self, tensor_name: str, to_device=True):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype not in [torch.int32, torch.int64]:
            tensor = tensor.to(dtype=self.dtype)
        if to_device:
            tensor = tensor.to(device=self.device)
        return tensor

    def load_multi_linear(self, config, prefixes):
        weight = torch.cat([self.get_tensor(f"{p}.weight") for p in prefixes], dim=0)
        return FastLinear(weight, bias=None)


class LlamaRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        # faster post attention rms norm
        normed_hidden_states, res, *rest = dropout_layer_norm.dropout_add_ln_fwd(
            hidden_states,
            residual,
            self.weight,
            None,
            None,
            None,
            None,
            None,
            0.0,
            self.variance_epsilon,
            1.0,
            0,
            None,
            False,
            True,  # Activate RMSNorm
        )
        if res is None:
            res = hidden_states

        return normed_hidden_states, res


class FlashLlamaAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        self.softmax_scale = self.head_size ** -0.5
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        # q,k,v,o and rotary
        self.query_key_value = weights.load_multi_linear(config,
                                                         [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"])
        self.o_proj = FastLinear.load(config, f"{prefix}.o_proj", weights, bias=False)
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config, dim=self.head_size, base=config.rope_theta, device=weights.device
        )

    def forward(
            self,
            hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        qkv = self.query_key_value(hidden_states)
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, cos, sin)
        self.rotary_emb(torch.select(kv, dim=1, index=0), cos, sin)

        cache_ops.reshape_and_cache(
            kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
        )

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn_2_cuda.varlen_fwd(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                attn_output,
                cu_seqlen_prefill,
                cu_seqlen_prefill,
                max_s,
                max_s,
                0.0,
                self.softmax_scale,
                False,
                True,
                -1,
                0,
                False,
                None,
            )
        # Decode
        else:
            # kv_cache[1] => [num_blocks, num_heads, head_size, block_size]
            block_size = kv_cache[1].shape[3]
            attention_ops.paged_attention_v1(
                attn_output,
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                block_size,
                max_s,
                None
            )

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))


class LlamaMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        self.intermediate_size = config.intermediate_size
        self.act = ACT2FN[act]
        # Fuse gate and up proj
        self.gate_up_proj = weights.load_multi_linear(config, [f"{prefix}.gate_proj", f"{prefix}.up_proj"])
        self.down_proj = FastLinear.load(config, f"{prefix}.down_proj", weights, bias=False)

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class FlashLlamaLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"

        self.input_layernorm = LlamaRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.self_attn = FlashLlamaAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.mlp = LlamaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

    def forward(
            self,
            hidden_states,
            residual,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashLlamaModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        embeddings = weights.get_tensor(f"model.embed_tokens.weight")
        self.embed_tokens = nn.Embedding.from_pretrained(F.pad(embeddings, (0, 0, 0, 1)),
                                                         padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [
                FlashLlamaLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
                # for layer_id in range(1)
            ]
        )
        self.norm = LlamaRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            cu_seqlen_prefill: Optional[torch.Tensor],
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_s: int,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashLlamaForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        self.model = FlashLlamaModel(config, weights)
        self.lm_head = FastLinear.load(config, prefix="lm_head", weights=weights, bias=False)

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            cu_seqlen_prefill: Optional[torch.Tensor],
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_s: int,
            lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)
        return logits


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


def generate(tokenizer, model, prompt, max_new_tokens=10):
    input_ids = tokenizer(prompt).input_ids

    def warmup():
        print("start warmup...")
        global CACHE_MANAGER
        blocks = 260
        CACHE_MANAGER = CacheManager(blocks,
                                     model.config.num_hidden_layers,
                                     model.config.num_key_value_heads,
                                     model.config.hidden_size // model.config.num_attention_heads,
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
                                 model.config.num_hidden_layers,
                                 model.config.num_key_value_heads,
                                 model.config.hidden_size // model.config.num_attention_heads,
                                 torch.float16,
                                 device)
    total_tokens = len(input_ids) + max_new_tokens - 1
    needed_blocks = math.ceil(total_tokens / BLOCK_SIZE)
    needed_blocks_slots = [(needed_blocks, total_tokens)]
    _, block_tables_tensor, slots = CACHE_MANAGER.allocate(needed_blocks, needed_blocks, needed_blocks_slots)
    # forward循环
    loops = 10
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
        print(f"total new tokens: {max_new_tokens}, cost time: {sum(times):.6f} s\n"
              f"time_per_token: {elapsed_time * 1000:.3f} ms, tps: {1 / elapsed_time:.2f} tokens/s")


def main(model_path):
    # step 0: 定义路径与属性
    model_path = Path(model_path)
    config = transformers.AutoConfig.from_pretrained(model_path)
    model_files = list(model_path.glob('*.safetensors'))

    # step 1: 定义tokenizer与权重
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="left", truncation_side="left")
    weights = Weights(model_files, device, torch.float16, process_group=None)

    # step2: 定义模型
    model = FlashLlamaForCausalLM(config, weights).eval()
    print(model)

    # step3: 推理
    with torch.no_grad():
        prompt = "who are you?"
        generate(tokenizer, model, prompt, max_new_tokens=100)


if __name__ == '__main__':
    CACHE_MANAGER: Optional[CacheManager] = None
    device = torch.device("cuda")
    main('/code/models/llama-7b-hf')
