import torch
import json
from typing import Tuple, Optional
from text_generation_server.layers.tensor_parallel import TensorParallelHead
from text_generation_server.layers.medusa import MedusaHeadV1, MedusaHeadV2
from text_generation_server.layers.mlp import MLPSpeculatorHead


class SpeculativeHead(torch.nn.Module):
    def __init__(self, lm_head, speculator):
        super().__init__()
        self.head = lm_head
        self.speculator = speculator

    @staticmethod
    def load(config, prefix: str, weights):
        speculator = config.speculator
        if speculator:
            speculator_path = config.speculator["path"]
            speculator_config = str(speculator_path / "config.json")

            with open(speculator_config, "r") as f:
                speculator_config = json.load(f)

            config.speculator_config = speculator_config
            try:
                architecture = speculator_config["architectures"][0]

                if architecture == "MLPSpeculatorPreTrainedModel":
                    speculator = MLPSpeculatorHead.load(config, prefix, weights)
                else:
                    speculator = None
            except KeyError:
                try:
                    speculator = MedusaHeadV1.load(config, prefix, weights)
                except:
                    speculator = MedusaHeadV2(config, prefix, weights)
            lm_head = None
        else:
            lm_head = TensorParallelHead.load(config, prefix, weights)
            speculator = None
        return SpeculativeHead(lm_head, speculator)

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.speculator is not None:
            return self.speculator(input)

        assert self.head is not None
        logits = self.head(input)
        return logits, None
