from typing import Optional

import torch
import torch.distributed
from torch.distributed import ProcessGroup

from text_generation_server.adapters import AdapterBatchData
from text_generation_server.adapters.lora import BatchLoraWeights
from text_generation_server.utils.sgmv import (
    add_lora_a_bgmv,
    add_lora_b_bgmv,
    has_sgmv,
    lora_a_sgmv_cutlass,
    lora_b_sgmv_cutlass,
    orient_for_rank,
)


def gather_lora_weights(
    process_group: ProcessGroup,
    weights: torch.Tensor,
    use_all_gather: bool = False,
) -> BatchLoraWeights:
    if use_all_gather:
        # Tensor parallel implementation of X @ A@B, where A and B are sharded column-wise.
        # We use an all-gather between X@A and (X@A)@B to ensure alignment across ranks.
        #
        # TODO: this is not very efficient as we do an all-gather for every adapter,
        # instead we could pre-allocate a (B, a, r) tensor for all adapters with the same
        # rank, compute `a_out` on each, and then slice them into the buffer as shown here:
        # https://discuss.pytorch.org/t/concatenate-tensors-without-memory-copying/34609
        gathered_tensors = [
            torch.empty_like(weights) for _ in range(process_group.size())
        ]
        torch.distributed.all_gather(gathered_tensors, weights, group=process_group)
        return torch.cat(gathered_tensors, dim=1)
    else:
        torch.distributed.all_reduce(weights, group=process_group)
        return weights


def forward_layer_type(
    process_group: ProcessGroup,
    layer_id: int,
    result: torch.Tensor,
    input: torch.Tensor,
    adapter_data: "AdapterBatchData",
    layer_type: str,
    start_idx: int,
    end_idx: int,
    use_all_gather: bool = False,
) -> torch.Tensor:
    if adapter_data is None:
        return result
    data = adapter_data.data.get(layer_type)
    data: Optional["BatchLoraWeights"] = data.get("lora") if data is not None else None

    if has_sgmv() and data is not None and data.can_vectorize(process_group):
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
                    layer_id,
                    r,
                )

                if process_group.size() > 1:
                    v = gather_lora_weights(process_group, v, use_all_gather)

                lora_b_sgmv_cutlass(
                    proj,
                    v,
                    rank_segments.tmp_expand,
                    lora_b_ptr,
                    rank_segments.segment_starts,
                    rank_segments.segment_ends,
                    layer_id,
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
                    layer_id,
                )

                if process_group.size() > 1:
                    v = gather_lora_weights(process_group, v, use_all_gather)

                add_lora_b_bgmv(
                    proj,
                    v,
                    lora_b_ptr,
                    rank_segments.indices,
                    layer_id,
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
                layer_result = forward_lora(
                    process_group=process_group,
                    layer_id=layer_id,
                    input=input,
                    data=data,
                    adapter_index=adapter_index,
                    adapter_mask=adapter_mask,
                    use_all_gather=use_all_gather,
                )
                result[:, start_idx:end_idx] += layer_result
    return result


def forward_lora(
    process_group: ProcessGroup,
    layer_id,
    input: torch.Tensor,
    data: "BatchLoraWeights",
    adapter_index: int,
    adapter_mask: torch.Tensor,
    use_all_gather: bool = False,
) -> torch.Tensor:
    lora_a = data.lora_a[adapter_index][layer_id, :, :]
    lora_b = data.lora_b[adapter_index][layer_id, :, :]

    lora_a = orient_for_rank(lora_a, lora_b.size(0))

    a_out = input @ lora_a
    if process_group.size() > 1:
        a_out = gather_lora_weights(process_group, a_out, use_all_gather)

    result = (a_out @ lora_b) * adapter_mask
    return result
