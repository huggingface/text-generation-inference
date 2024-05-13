import torch
from typing import Tuple, Optional
from text_generation_server.layers.medusa import MedusaHeadV1, MedusaHeadV2
from text_generation_server.layers.tensor_parallel import TensorParallelHead


class SpeculativeHead(torch.nn.Module):
    def __init__(self, lm_head, medusa):
        super().__init__()
        self.head = lm_head
        self.medusa = medusa

    @staticmethod
    def load(config, prefix: str, weights):
        use_medusa = config.use_medusa
        if use_medusa:
            lm_head = None
            try:
                medusa = MedusaHeadV1.load(config, prefix, weights)
            except:
                medusa = MedusaHeadV2(config, prefix, weights)
        else:
            lm_head = TensorParallelHead.load(config, prefix, weights)
            medusa = None
        return SpeculativeHead(lm_head, medusa)

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.medusa is not None:
            return self.medusa(input)

        assert self.head is not None
        logits = self.head(input)
        return logits, None
