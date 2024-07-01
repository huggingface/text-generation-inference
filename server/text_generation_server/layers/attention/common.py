from dataclasses import dataclass
from text_generation_server.models.globals import FLASH_DECODING
import torch
from typing import Optional


@dataclass
class Seqlen:
    input_lengths: torch.Tensor
    cu_seqlen_q: Optional[torch.Tensor]
    cu_seqlen_k: Optional[torch.Tensor]

    def __init__(self, input_lengths):
        self.input_lengths = input_lengths
        if FLASH_DECODING:
            device = self.input_lengths.device
            shape = self.input_lengths.shape
            cu_seqlen_q = torch.arange(
                shape[0] + 1,
                device=device,
                dtype=torch.int32,
            )
            cu_seqlen_k = torch.empty(shape[-1] + 1, device=device, dtype=torch.int32)
            cu_seqlen_k[0] = 0
            torch.cumsum(self.input_lengths, -1, out=cu_seqlen_k[1:])

            self.cu_seqlen_q = cu_seqlen_q
            self.cu_seqlen_k = cu_seqlen_k
        else:
            self.cu_seqlen_q = None
            self.cu_seqlen_k = None
