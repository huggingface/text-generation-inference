from dataclasses import dataclass
from text_generation_server.models.globals import ATTENTION
import torch
from typing import Optional


if ATTENTION in {"flashinfer", "flashdecoding"}:

    @dataclass
    class Seqlen:
        input_lengths: torch.Tensor
        cu_seqlen_q: Optional[torch.Tensor]
        cu_seqlen_k: Optional[torch.Tensor]

        def __init__(self, input_lengths):
            self.input_lengths = input_lengths
            device = self.input_lengths.device
            shape = self.input_lengths.shape
            cu_seqlen_q = torch.arange(
                shape[0] + 1,
                device=device,
                dtype=torch.int32,
            )
            cu_seqlen_k = torch.zeros(shape[-1] + 1, device=device, dtype=torch.int32)
            # cuda graphs don't like this and this is necessary to clamp within mistral
            # Although FA2 might not want the clamping
            # cu_seqlen_k[0] = 0
            torch.cumsum(self.input_lengths, -1, out=cu_seqlen_k[1:])

            self.cu_seqlen_q = cu_seqlen_q
            self.cu_seqlen_k = cu_seqlen_k

        def clamp(self, max):
            # Flash decoding doesn't need to clamp
            return self

else:

    @dataclass
    class Seqlen:
        input_lengths: torch.Tensor

        def clamp(self, max):
            return Seqlen(torch.clamp(self.input_lengths, max=max))
