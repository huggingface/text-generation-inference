import math
import os
from typing import TYPE_CHECKING, Optional, Tuple, List

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch import nn
from torch.nn import functional as F
from torch.distributed import ProcessGroup

from text_generation_server.utils.sgmv import (
    add_lora_a_bgmv,
    add_lora_b_bgmv,
    has_sgmv,
    lora_a_sgmv_cutlass,
    lora_b_sgmv_cutlass,
    orient_for_rank,
)

if TYPE_CHECKING:
    from text_generation_server.adapters import AdapterBatchData
    from text_generation_server.adapters.lora import BatchLoraWeights


class LoraLinear(nn.Module):
    def __init__(
        self, base_layer: nn.Module, layer_id: int, process_group: ProcessGroup
    ):
        super().__init__()
        self.base_layer = base_layer
        self.layer_id = layer_id
        self.process_group = process_group

    def forward_layer_type(
        self,
        result: torch.Tensor,
        input: torch.Tensor,
        adapter_data: "AdapterBatchData",
        layer_type: str,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        if adapter_data is None:
            return result
        data = adapter_data.data.get(layer_type)
        data: Optional["BatchLoraWeights"] = (
            data.get("lora") if data is not None else None
        )

        if has_sgmv() and data is not None and data.can_vectorize(self.process_group):
            # In tensor-parallel configurations, each GPU processes a specific segment of the output.
            # The 'result' tensor represents the full output, which can vary in size based on
            # the layer type (e.g., attention vs. feed-forward layers). We define the current
            # segment using start_idx and end_idx. If the segment size doesn't match this GPU's
            # slice of 'result', we create a zero tensor of the correct size for LoRA computation.
            # This approach ensures accurate LoRA application across various layer sizes and
            # configurations, adapting to different model architectures and parallelization strategies.
            #
            # Example scenarios where this is necessary:
            # 1. The adapter's size doesn't evenly divide across GPUs.
            # 2. We're processing the last segment which might be smaller.
            # 3. Different projection layers (q, k, v) have different sizes.
            if end_idx - start_idx != result.shape[1]:
                proj = torch.zeros_like(result[:, start_idx:end_idx])
            else:
                proj = result

            for r, rank_segments in data.rank_data.items():
                lora_a_ptr = rank_segments.lora_a_ptr
                lora_b_ptr = rank_segments.lora_b_ptr

                if lora_a_ptr is None or lora_b_ptr is None:
                    raise ValueError("LoRA data is missing")

                if data.use_sgmv:
                    # Use SGMV for prefill
                    v = lora_a_sgmv_cutlass(
                        input,
                        rank_segments.tmp_shrink,
                        lora_a_ptr,
                        rank_segments.segment_starts,
                        rank_segments.segment_ends,
                        self.layer_id,
                        r,
                    )

                    if self.process_group.size() > 1:
                        v = self.collect_lora_a(v)

                    lora_b_sgmv_cutlass(
                        proj,
                        v,
                        rank_segments.tmp_expand,
                        lora_b_ptr,
                        rank_segments.segment_starts,
                        rank_segments.segment_ends,
                        self.layer_id,
                    )
                else:
                    # Use BGMV for decode
                    v = torch.zeros(
                        (input.size(0), r), dtype=input.dtype, device=input.device
                    )
                    # TODO: error with [-1, 0], but not [0, -1]
                    add_lora_a_bgmv(
                        v,
                        input,
                        lora_a_ptr,
                        rank_segments.indices,
                        self.layer_id,
                    )

                    if self.process_group.size() > 1:
                        v = self.collect_lora_a(v)

                    add_lora_b_bgmv(
                        proj,
                        v,
                        lora_b_ptr,
                        rank_segments.indices,
                        self.layer_id,
                    )

            if end_idx - start_idx != result.shape[1]:
                result[:, start_idx:end_idx] += proj
        else:
            for adapter_index in adapter_data.meta.adapter_set:
                if data is not None and data.has_adapter(adapter_index):
                    adapter_mask = (
                        (adapter_data.meta.adapter_indices == adapter_index)
                        .to(input.dtype)
                        .view(-1, 1)
                    )
                    layer_result = self.forward_lora(
                        input, data, adapter_index, adapter_mask
                    )
                    result[:, start_idx:end_idx] += layer_result

        return result

    def forward_lora(
        self,
        input: torch.Tensor,
        data: "BatchLoraWeights",
        adapter_index: int,
        adapter_mask: torch.Tensor,
    ) -> torch.Tensor:
        lora_a = data.lora_a[adapter_index][self.layer_id, :, :]
        lora_b = data.lora_b[adapter_index][self.layer_id, :, :]

        lora_a = orient_for_rank(lora_a, lora_b.size(0))

        a_out = input @ lora_a
        if self.process_group.size() > 1:
            a_out = self.collect_lora_a(a_out)

        result = (a_out @ lora_b) * adapter_mask
        return result

    def collect_lora_a(self, a_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implemented in subclasses")


class TensorParallelMultiAdapterLinear(LoraLinear):
    def __init__(
        self,
        base_layer: nn.Module,
        layer_id: int,
        layer_names: List[str],
        sizes: List[int],
        process_group: ProcessGroup,
    ):
        super().__init__(base_layer, layer_id, process_group)
        self.layer_names = layer_names
        self.sizes = sizes

    @classmethod
    def load(
        cls,
        base_layer: nn.Module,
        layer_id: int,
        layer_names: List[str],
        sizes: List[int],
        process_group: ProcessGroup,
    ):
        return TensorParallelMultiAdapterLinear(
            base_layer, layer_id, layer_names, sizes, process_group
        )

    def forward(
        self, input: torch.Tensor, adapter_data: "AdapterBatchData"
    ) -> torch.Tensor:
        result = self.base_layer(input)

        # noop if no layer names are provided (e.g. for models without adapters)
        if self.layer_names is None:
            return result

        # handle models like Bloom that have inputs of shape
        # (batch_size, sequence_length, hidden_size)
        # we need to reshape them to (batch_size * sequence_length, hidden_size)
        # for the LoRA computation, then reshape back
        prev_shape = result.shape
        is_3d = len(input.shape) >= 3
        if is_3d:
            input = input.reshape(-1, input.shape[-1])
            result = result.reshape(-1, result.shape[-1])

        offset = 0
        for i, layer_name in enumerate(self.layer_names):
            start_idx = offset // self.process_group.size()
            # The 'sizes' parameter is essential in tensor-parallel setups for handling multiple
            # projection layers (q_proj, k_proj, v_proj) by defining their output dimensions. It
            # ensures correct slicing of the result tensor, accommodating variations like grouped-query
            # attention where k_proj and v_proj differ from q_proj. This allows precise application of
            # LoRA adapters to each sub-component of the multi-head attention mechanism, managing the
            # different projection sizes across layers and model architectures.
            if self.sizes is not None:
                offset += self.sizes[i]
                end_idx = offset // self.process_group.size()
            else:
                end_idx = result.shape[1]

            result = self.forward_layer_type(
                result, input, adapter_data, layer_name, start_idx, end_idx
            )

        if is_3d:
            result = result.reshape(prev_shape)

        return result

    def collect_lora_a(self, a_out: torch.Tensor) -> torch.Tensor:
        # Tensor parallel implementation of X @ A@B, where A and B are sharded column-wise.
        # We use an all-gather between X@A and (X@A)@B to ensure alignment across ranks.
        #
        # TODO(travis): this is not very efficient as we do an all-gather for every adapter,
        #   instead we could pre-allocate a (B, a, r) tensor for all adapters with the same
        #   rank, compute `a_out` on each, and then slice them into the buffer as shown here:
        #   https://discuss.pytorch.org/t/concatenate-tensors-without-memory-copying/34609
        gathered_tensors = [
            torch.empty_like(a_out) for _ in range(self.process_group.size())
        ]
        torch.distributed.all_gather(gathered_tensors, a_out)
        return torch.cat(gathered_tensors, dim=1)


class TensorParallelAdapterRowLinear(LoraLinear):
    def __init__(self, base_layer, layer_id, layer_name, process_group):
        super().__init__(base_layer, layer_id, process_group)
        self.layer_name = layer_name

    @classmethod
    def load(cls, base_layer, layer_id, layer_name, process_group):
        return cls(base_layer, layer_id, layer_name, process_group)

    def forward(
        self, input: torch.Tensor, adapter_data: "AdapterBatchData"
    ) -> torch.Tensor:
        result = self.base_layer(input)

        if self.layer_name is None:
            return result

        # Fused all-gather + all-reduce from S-LoRA paper: https://arxiv.org/abs/2311.03285
        stride = result.shape[-1] // self.process_group.size()
        start_idx = self.process_group.rank() * stride
        end_idx = (self.process_group.rank() + 1) * stride

        self.forward_layer_type(
            result, input, adapter_data, self.layer_name, start_idx, end_idx
        )

        return result

    def collect_lora_a(self, a_out: torch.Tensor) -> torch.Tensor:
        # Tensor parallel implementation of X @ A@B, where A and B are sharded row-wise.
        # We use an all-reduce between X@A and (X@A)@B to ensure alignment across ranks.
        #
        # TODO(travis): this is not very efficient as we do an all-reduce for every adapter,
        #   instead we could pre-allocate a (B, a, r) tensor for all adapters with the same
        #   rank, compute `a_out` on each, and then slice them into the buffer as shown here:
        #   https://discuss.pytorch.org/t/concatenate-tensors-without-memory-copying/34609
        torch.distributed.all_reduce(a_out, group=self.process_group)
        return a_out
