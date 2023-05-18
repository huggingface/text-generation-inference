from typing import List, Union

import torch

from transformers import GPTBigCodeConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt_bigcode.configuration_gpt_bigcode import (
    InferenceRunnerType,
)
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeBlock, softmax_function


def _align_tensor(x):
    return x + -x % 128


class GPTBigCodeInferenceRunner:
    def __init__(self, config: GPTBigCodeConfig, model):
        self.batch_size = None
        self.model = model
        self.n_layer = len(self.model.h)

        self.inference_runner_type = InferenceRunnerType(config.inference_runner)
        assert self.inference_runner_type != InferenceRunnerType.NO_RUNNER
        assert config.pre_allocate_kv_cache
        self.validate_input = config.validate_runner_input
        self.pad_key_length = 8 if config.pad_key_length else 1
        self.fused_softmax = True if config.fused_softmax is None and config.pad_key_length else config.fused_softmax

        # TODO: Support other attention types?
        assert model.multi_query

        self.max_sequence_length = config.max_sequence_length or config.n_positions

    def _allocate(self, batch_size, device, dtype):
        block: GPTBigCodeBlock = self.model.h[0]
        attn = block.attn
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.softmax_dtype = torch.float32 if attn.attention_softmax_in_fp32 else self.dtype
        self.upcast = self.softmax_dtype != self.dtype

        do_unscale = attn.scale_attention_softmax_in_fp32 and self.upcast
        self.unscale = [i + 1.0 if do_unscale else 1.0 for i in range(self.n_layer)]
        scale = attn.head_dim**-0.5 if attn.scale_attn_weights else 1
        self.scale = [scale / unscale for unscale in self.unscale]

        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        hidden_end = self.batch_size * attn.embed_dim
        # Query: (bs, embed_dim), also used for attn outputs (no overlap with value).
        query_begin = _align_tensor(hidden_end)
        query_end = query_begin + self.batch_size * attn.embed_dim
        # KV: (bs, 2 * kv_dim), combines with query into c_attn.
        kv_end = query_end + 2 * self.batch_size * attn.kv_dim
        # Attn weights: (batch_size, num_heads, key_length), no overlap with value
        attn_weights_begin = _align_tensor(kv_end)
        attn_weights_end = attn_weights_begin + self.batch_size * attn.num_heads * self.max_sequence_length
        # Projection: (batch_size, embed_dim), no overlap with attn outputs ~ query.
        # Also used for MLP projection
        c_proj_begin = _align_tensor(query_end)
        c_proj_end = c_proj_begin + self.batch_size * attn.embed_dim
        c_fc_begin = query_begin
        c_fc_end = c_fc_begin + self.batch_size * block.inner_dim
        pool_size = max(attn_weights_end, c_proj_end, c_fc_end)

        print(
            f"Allocating inference buffers (batch size = {self.batch_size}, max sequence length ="
            f" {self.max_sequence_length})..."
        )

        kv_caches = []
        for block in self.model.h:
            block.attn.freeze_kv_cache()
            kv_cache = block.attn.get_kv_cache(self.batch_size, self.max_sequence_length, self.device, self.dtype)
            if attn.multi_query:
                kv_cache = kv_cache.unsqueeze(1)
            kv_caches.append(kv_cache)

        kv_cache_size = sum(kv_cache.numel() for kv_cache in kv_caches)

        print(f"  Activation pool size: {pool_size:,}")
        print(f"  KV cache size: {kv_cache_size:,}")
        buffer_memory = (pool_size + kv_cache_size) * torch.finfo(
            self.dtype
        ).bits / 8 + self.batch_size * self.max_sequence_length
        print(f"  Memory usage: {buffer_memory/2**20:.0f} MiB")

        activation_pool = torch.empty(pool_size, **factory_kwargs)
        self.mask_value = torch.full(
            [], torch.finfo(self.softmax_dtype).min, dtype=self.softmax_dtype, device=self.device
        )
        # We ensure mask tensors are contiguous to enable more efficient kernels.
        attn_mask = torch.empty(self.batch_size * self.max_sequence_length, dtype=torch.bool, device=self.device)

        if self.device.type == "cuda":
            print(f"  Memory allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB")
            # Max stats give some insight on the prefill memory usage.
            print(f"  Max memory allocated {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
            print(f"  Max memory reserved {torch.cuda.max_memory_reserved()/2**20:.0f} MiB")

        key_lengths = range(self.max_sequence_length + 1)
        padded_key_lengths = [key_length + -key_length % self.pad_key_length for key_length in key_lengths]

        self.padded_attn_masks = [
            attn_mask[: self.batch_size * key_length].view(self.batch_size, 1, key_length)
            for key_length in padded_key_lengths
        ]
        self.attn_masks = [
            padded_attn_mask[:, :, :key_length].squeeze(1)
            for key_length, padded_attn_mask in enumerate(self.padded_attn_masks)
        ]
        self.attn_mask_pads = [
            padded_attn_mask[:, :, key_length:].squeeze(1)
            for key_length, padded_attn_mask in enumerate(self.padded_attn_masks)
        ]

        # Hidden: (batch_size, 1, embed_dim), no overlap allowed.
        self.hidden_states_squeezed = activation_pool[:hidden_end].view(self.batch_size, -1)
        self.hidden_states = self.hidden_states_squeezed.unsqueeze(1)
        # QKV: (bs, embed_dim + 2 * kv_dim).
        self.c_attn = activation_pool[query_begin:kv_end].view(self.batch_size, -1)
        self.query = self.c_attn[:, : attn.embed_dim].view(self.batch_size, attn.num_heads, attn.head_dim)
        self.kv_attn = self.c_attn[:, attn.embed_dim :]

        keys, values = zip(*(kv_cache.split((attn.head_dim, attn.head_dim), dim=-1) for kv_cache in kv_caches))
        head_slice = 0 if attn.multi_query else slice(None)

        self.padded_keys = [
            [key[:, head_slice, :key_length, :].transpose(-1, -2) for key in keys] for key_length in padded_key_lengths
        ]
        self.padded_values = [
            [value[:, head_slice, :key_length, :] for value in values] for key_length in padded_key_lengths
        ]

        # This is nonsense for key_length == 0, but we never need the value.
        self.current_key_values = [
            [kv_cache[:, head_slice, key_length - 1, :] for kv_cache in kv_caches] for key_length in key_lengths
        ]
        self.past_key_values = [
            [kv_cache[:, head_slice, : key_length - 1, :] for kv_cache in kv_caches] for key_length in key_lengths
        ]

        # Attn weights: (batch_size, num_heads, key_length), no overlap with value.
        attn_weights = activation_pool[attn_weights_begin:attn_weights_end].view(
            self.batch_size, attn.num_heads, self.max_sequence_length
        )
        self.padded_attn_weights = [attn_weights[:, :, :key_length] for key_length in padded_key_lengths]

        # Attn outputs: (batch_size, embed_dim), no overlap with value.
        self.attn_output = activation_pool[query_begin:query_end].view(self.batch_size, -1)
        self.attn_output_expanded = self.attn_output.view(self.batch_size, attn.num_heads, attn.head_dim)
        # Attn projection: (batch_size, embed_dim), no overlap with attn outputs.
        self.c_proj = activation_pool[c_proj_begin:c_proj_end].view(self.batch_size, -1)

        # MLP first layer: (batch_size, embed_dim)
        self.mlp_c_fc = activation_pool[c_fc_begin:c_fc_end].view(self.batch_size, -1)
        # MLP projection: (batch_size, inner_dim)
        self.mlp_c_proj = activation_pool[query_begin:query_end].view(self.batch_size, -1)

        if self.inference_runner_type != InferenceRunnerType.BASE_RUNNER:
            print("Generating cuda graphs")
            self.memory_pool = None
            # This prevents some issue with cublas initialization.
            # https://github.com/pytorch/pytorch/issues/99397
            dummy_matrix = self.mask_value.view([1, 1])
            torch.matmul(dummy_matrix, dummy_matrix)
            if self.inference_runner_type == InferenceRunnerType.FULL_GRAPH:
                self.cuda_graphs = {}
                # The output may not always be at the same memory location.
                self.output_hidden_states = {}
                # Generate the largest one first to warm up the memory pool.
                # The other ones are generated lazily.
                self._generate_full_cuda_graph(self.max_sequence_length)
            else:
                self._generate_cuda_graphs()

    def _generate_cuda_graphs(self):
        self.cuda_graphs = {}
        for layer_idx in range(self.n_layer + 1):
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph, pool=self.memory_pool):
                if layer_idx > 0:
                    self._forward_post_attn(self.model.h[layer_idx - 1])
                if layer_idx < self.n_layer:
                    self._forward_qkv(self.model.h[layer_idx])
                else:
                    self.output_hidden_states = self._forward_end()
            if self.memory_pool is None:
                self.memory_pool = graph.pool()
            self.cuda_graphs[layer_idx] = graph

    def _generate_full_cuda_graph(self, key_length):
        # We need to warmup the jit function before creating the graph, otherwise it will crash.
        # https://github.com/pytorch/pytorch/issues/99397
        # Warmup needs to be done for every input shape (key length), and for both scale == 1 and scale != 1
        if self.fused_softmax or (self.fused_softmax is None and key_length % 8 == 0):
            for scale in (1.0, 2.0):
                softmax_function(
                    self.padded_attn_weights[key_length],
                    self.padded_attn_masks[key_length],
                    self.mask_value,
                    scale,
                    self.softmax_dtype,
                    self.upcast,
                    self.fused_softmax,
                )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.memory_pool):
            self.output_hidden_states[key_length] = self._forward(key_length)
        if self.memory_pool is None:
            self.memory_pool = graph.pool()
        self.cuda_graphs[key_length] = graph

    def _forward_embed(self, input_ids, position_ids):
        # Embedding doesn't support out argument.
        inputs_embeds = self.model.wte(input_ids)
        position_embeds = self.model.wpe(position_ids)
        torch.add(inputs_embeds, position_embeds, out=self.hidden_states)

    def _forward_qkv(self, block):
        # LN doesn't support out argument.
        hidden_states = block.ln_1(self.hidden_states_squeezed)
        torch.nn.functional.linear(
            hidden_states,
            block.attn.c_attn.weight,
            block.attn.c_attn.bias,
            out=self.c_attn,
        )

    def _forward_attn(self, block, key_length):
        layer_idx = block.attn.layer_idx
        self.current_key_values[key_length][layer_idx].copy_(self.kv_attn)
        attn_weights = self.padded_attn_weights[key_length]

        torch.baddbmm(
            attn_weights,
            self.query,
            self.padded_keys[key_length][layer_idx],
            beta=0,
            alpha=self.scale[layer_idx],
            out=attn_weights,
        )
        # Jit doesn't allow inplace kernel.
        attn_weights = softmax_function(
            attn_weights,
            self.padded_attn_masks[key_length],
            self.mask_value,
            self.unscale[layer_idx],
            self.softmax_dtype,
            self.upcast,
            self.fused_softmax,
        )

        torch.bmm(attn_weights, self.padded_values[key_length][layer_idx], out=self.attn_output_expanded)

    def _forward_post_attn(self, block):
        torch.nn.functional.linear(
            self.attn_output,
            block.attn.c_proj.weight,
            block.attn.c_proj.bias,
            out=self.c_proj,
        )
        self.hidden_states_squeezed.add_(self.c_proj)
        # LN doesn't support out argument.
        hidden_states = block.ln_2(self.hidden_states_squeezed)
        torch.nn.functional.linear(hidden_states, block.mlp.c_fc.weight, block.mlp.c_fc.bias, out=self.mlp_c_fc)
        # Most activations don't support out argument.
        feed_forward_hidden_states = block.mlp.act(self.mlp_c_fc)
        torch.nn.functional.linear(
            feed_forward_hidden_states, block.mlp.c_proj.weight, block.mlp.c_proj.bias, out=self.mlp_c_proj
        )
        self.hidden_states_squeezed.add_(self.mlp_c_proj)

    def _forward_end(self):
        # LN doesn't support out argument.
        return self.model.ln_f(self.hidden_states)

    def _forward(self, key_length):
        for block in self.model.h:
            self._forward_qkv(block)
            self._forward_attn(block, key_length)
            self._forward_post_attn(block)
        return self._forward_end()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Union[List[torch.Tensor], int],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        batch_size, query_length = input_ids.shape
        assert query_length == 1
        if self.batch_size is None:
            self._allocate(batch_size, device=input_ids.device, dtype=self.model.dtype)
        elif self.validate_input:
            assert batch_size == self.batch_size
            assert self.dtype == self.model.dtype
            assert self.device == input_ids.device

        if self.validate_input:
            assert attention_mask.dim() == 2
            assert attention_mask.shape[0] == batch_size
            key_length = attention_mask.shape[1]
            assert key_length <= self.max_sequence_length
            if isinstance(past_key_values, int):
                assert key_length == past_key_values + 1
        else:
            key_length = attention_mask.shape[1]

        self._forward_embed(input_ids, position_ids)

        self.attn_masks[key_length].copy_(attention_mask)

        attn_mask_pad = self.attn_mask_pads[key_length]
        if attn_mask_pad.size(1) > 0:
            attn_mask_pad.fill_(False)

        if self.inference_runner_type == InferenceRunnerType.FULL_GRAPH:
            if key_length not in self.cuda_graphs:
                self._generate_full_cuda_graph(key_length)
            self.cuda_graphs[key_length].replay()
            hidden_states = self.output_hidden_states[key_length]
        elif self.inference_runner_type == InferenceRunnerType.PARTIAL_GRAPH:
            for i, block in enumerate(self.model.h):
                self.cuda_graphs[i].replay()
                self._forward_attn(block, key_length)
            self.cuda_graphs[self.n_layer].replay()
            hidden_states = self.output_hidden_states
        else:
            hidden_states = self._forward(key_length)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=key_length,
        )
