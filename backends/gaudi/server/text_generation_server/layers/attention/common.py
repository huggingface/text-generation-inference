from dataclasses import dataclass
import torch
from typing import Optional, List, Dict
import collections

_TYPE_CACHE = {}


@dataclass
class HPUPagedAttentionMetadata:
    """Metadata for PagedAttention."""

    block_list: Optional[torch.Tensor]
    block_mapping: Optional[torch.Tensor]
    block_usage: Optional[torch.Tensor]
    block_scales: Optional[torch.Tensor]
    block_groups: Optional[torch.Tensor]
    attn_bias: Optional[torch.Tensor]


def subtuple(
    obj: object,
    typename: str,
    to_copy: List[str],
    to_override: Optional[Dict[str, object]] = None,
):
    if obj is None:
        return None
    if to_override is None:
        to_override = {}
    fields = set(to_copy) | set(to_override.keys())
    if isinstance(obj, dict):
        values = {key: obj[key] for key in fields if key in obj}
    else:
        values = {f: to_override.get(f, getattr(obj, f)) for f in fields}
    if typename not in _TYPE_CACHE:
        _TYPE_CACHE[typename] = collections.namedtuple(typename, " ".join(fields))
    return _TYPE_CACHE[typename](**values)


def trim_attn_metadata(metadata: HPUPagedAttentionMetadata) -> object:
    # NOTE(kzawora): To anyone working on this in the future:
    # Trimming metadata is required when using HPUGraphs.
    # Attention metadata is going to be hashed by PT bridge, and
    # appropriate HPUGraphs will be matched based on all inputs' hash.

    # Before you put more keys in here, make sure you know their
    # value type and make sure you know how it's going to be hashed.
    # You can find that information in input_hash function
    # in habana_frameworks/torch/hpu/graphs.py. You can also hash
    # it manually with torch.hpu.graphs.input_hash(attention_metadata)

    # If you use primitive types here - they will get hashed based
    # on their value. You *will* get lots of excessive graph captures
    # (and an OOM eventually) if you decide to put something like
    # seq_len int here.
    # If you absolutely need a scalar, put it in a tensor. Tensors
    # get hashed using their metadata, not their values:
    # input_hash(torch.tensor(123)) == input_hash(torch.tensor(321))
    # input_hash(123) != input_hash(321)
    # input_hash("abc") != input_hash("cba")
    attention_metadata = subtuple(
        metadata,
        "TrimmedAttentionMetadata",
        [
            "block_list",
            "block_mapping",
            "block_usage",
            "block_scales",
            "block_groups",
            "attn_bias",
        ],
    )
    return attention_metadata


@dataclass
class Seqlen:
    input_lengths: torch.Tensor
    cache_lengths: torch.Tensor
    cu_seqlen_q: Optional[torch.Tensor]
    cu_seqlen_k: Optional[torch.Tensor]

    def __init__(
        self,
        input_lengths,
        cache_lengths,
        cu_seqlen_q=None,
    ):
        self.input_lengths = input_lengths
        self.cache_lengths = cache_lengths
        device = self.input_lengths.device
        shape = self.input_lengths.shape
        if cu_seqlen_q is None:
            cu_seqlen_q = torch.arange(
                shape[0] + 1,
                device=device,
                dtype=torch.int32,
            )
        cu_seqlen_k = torch.zeros(shape[-1] + 1, device=device, dtype=torch.int32)

        # cuda graphs don't like this and this is necessary to clamp within mistral
        # Although FA2 might not want the clamping
        # cu_seqlen_k[0] = 0
        total = self.input_lengths + self.cache_lengths
        torch.cumsum(total, -1, out=cu_seqlen_k[1:])

        self.cu_seqlen_q = cu_seqlen_q
        self.cu_seqlen_k = cu_seqlen_k

    def clamp(self, max):
        # Flash decoding doesn't need to clamp
        return self


def trim_seqlen_metadata(metadata: Seqlen) -> object:
    # NOTE(kzawora): To anyone working on this in the future:
    # Trimming metadata is required when using HPUGraphs.
    # Attention metadata is going to be hashed by PT bridge, and
    # appropriate HPUGraphs will be matched based on all inputs' hash.

    # Before you put more keys in here, make sure you know their
    # value type and make sure you know how it's going to be hashed.
    # You can find that information in input_hash function
    # in habana_frameworks/torch/hpu/graphs.py. You can also hash
    # it manually with torch.hpu.graphs.input_hash(attention_metadata)

    # If you use primitive types here - they will get hashed based
    # on their value. You *will* get lots of excessive graph captures
    # (and an OOM eventually) if you decide to put something like
    # seq_len int here.
    # If you absolutely need a scalar, put it in a tensor. Tensors
    # get hashed using their metadata, not their values:
    # input_hash(torch.tensor(123)) == input_hash(torch.tensor(321))
    # input_hash(123) != input_hash(321)
    # input_hash("abc") != input_hash("cba")
    attention_metadata = subtuple(
        metadata,
        "TrimmedSeqlen",
        [
            "input_lengths",
            "cache_lengths",
            "cu_seqlen_q",
            "cu_seqlen_k",
        ],
    )
    return attention_metadata
