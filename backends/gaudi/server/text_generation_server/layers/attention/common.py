from dataclasses import dataclass
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.models.globals import ATTENTION
import torch
from typing import Optional


if ATTENTION in {"flashinfer", "flashdecoding"}:

    @dataclass
    class Seqlen:
        input_lengths: torch.Tensor
        prefix_lengths: torch.Tensor
        cu_seqlen_q: Optional[torch.Tensor]
        cu_seqlen_k: Optional[torch.Tensor]
        max_q: int
        max_k: int

        def __init__(
            self,
            input_lengths,
            prefix_lengths,
            cu_seqlen_q=None,
            max_q=None,
            max_k=None,
        ):
            self.input_lengths = input_lengths
            self.prefix_lengths = prefix_lengths
            device = self.input_lengths.device
            shape = self.input_lengths.shape
            if cu_seqlen_q is None:
                cu_seqlen_q = torch.arange(
                    shape[0] + 1,
                    device=device,
                    dtype=torch.int32,
                )
                max_q = 1
            else:
                assert max_q is not None
            assert max_k is not None
            cu_seqlen_k = torch.zeros(shape[-1] + 1, device=device, dtype=torch.int32)

            # cuda graphs don't like this and this is necessary to clamp within mistral
            # Although FA2 might not want the clamping
            # cu_seqlen_k[0] = 0
            total = self.input_lengths + self.prefix_lengths
            torch.cumsum(total, -1, out=cu_seqlen_k[1:])

            self.cu_seqlen_q = cu_seqlen_q
            self.cu_seqlen_k = cu_seqlen_k
            self.max_q = max_q
            self.max_k = max_k

        def clamp(self, max):
            # Flash decoding doesn't need to clamp
            return self

else:

    @dataclass
    class Seqlen:
        input_lengths: torch.Tensor
        prefix_lengths: torch.Tensor
        cu_seqlen_q: torch.Tensor
        max_q: int
        max_k: int

        def clamp(self, max):
            if SYSTEM == "rocm":
                return self
            raise NotImplementedError("Not implemented seqlen for paged")
            return Seqlen(torch.clamp(self.input_lengths, max=max))
