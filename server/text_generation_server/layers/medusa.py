import torch
from torch import nn
from typing import Tuple, Optional
from text_generation_server.utils.speculate import get_speculate
from text_generation_server.layers.linear import FastLinear
from text_generation_server.layers.tensor_parallel import (
    TensorParallelHead,
    TensorParallelColumnLinear,
)


class ResBlock(torch.nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.linear = FastLinear.load(
            config, prefix=f"{prefix}.linear", weights=weights, bias=True
        )
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaModel(torch.nn.Module):
    def __init__(self, config, medusa_config, weights):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                MedusaHead(config, medusa_config, prefix=f"{i}", weights=weights)
                for i in range(get_speculate())
            ]
        )

    def forward(self, x):
        if not self.heads:
            return None
        speculative_logits = torch.stack([head(x) for head in self.heads], dim=1)
        return speculative_logits


class MedusaHead(torch.nn.Module):
    def __init__(self, config, medusa_config, prefix, weights):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                ResBlock(config, prefix=f"{prefix}.{i}", weights=weights)
                for i in range(medusa_config["medusa_num_layers"])
            ]
        )
        n = len(self.blocks)
        self.out = FastLinear.load(
            config, prefix=f"{prefix}.{n}", weights=weights, bias=False
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x


class MedusaHeadV1(nn.Module):
    def __init__(self, lm_head, medusa):
        super().__init__()
        self.lm_head = lm_head
        self.medusa = medusa

    @staticmethod
    def load(config, prefix: str, weights):
        from pathlib import Path
        from safetensors import safe_open
        import json

        speculator = config.speculator

        path = speculator["path"]
        medusa_config = str(Path(path) / "config.json")

        for fname in speculator["model_paths"]:
            filename = str(Path(path) / fname)

            with open(medusa_config, "r") as f:
                medusa_config = json.load(f)
            routing = weights.routing
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing and routing[k] != filename:
                        raise RuntimeError(
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                        )
                    routing[k] = filename

        medusa = MedusaModel(config, medusa_config, weights)
        lm_head = TensorParallelHead.load(config, prefix, weights)
        return MedusaHeadV1(lm_head, medusa)

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.lm_head(input)
        # If we have too many tokens, we skip speculative logits
        if input.shape[0] > 128:
            return logits, None

        speculative_logits = self.medusa(input)
        return logits, speculative_logits


class MedusaHeadV2(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        from pathlib import Path
        from safetensors import safe_open
        import json

        speculator_path = config.speculator["path"]

        medusa_config = str(Path(speculator_path) / "config.json")
        filename = str(Path(speculator_path) / "medusa_lm_head.safetensors")

        with open(medusa_config, "r") as f:
            medusa_config = json.load(f)
        routing = weights.routing
        with safe_open(filename, framework="pytorch") as f:
            for k in f.keys():
                if k in routing and routing[k] != filename:
                    raise RuntimeError(
                        f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                    )
                routing[k] = filename

        self.n_medusa_heads = get_speculate()

        assert medusa_config["medusa_num_layers"] == 1
        self.linear = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{i}.0.linear" for i in range(self.n_medusa_heads)],
            dim=0,
            weights=weights,
            bias=True,
        )
        self.process_group = weights.process_group
        self.world_size = self.process_group.size()
        self.rank = self.process_group.rank()

        self.act = torch.nn.SiLU()

        self.lm_head = TensorParallelHead.load(config, prefix, weights)

    def forward(self, x):
        # If we have too many tokens, we skip speculative logits
        if x.shape[0] > 128:
            logits = self.lm_head(x)
            return logits, None

        size = x.shape[-1]
        block_size = (size + self.world_size - 1) // self.world_size
        start = self.rank * block_size
        stop = (self.rank + 1) * block_size

        x_block = x[:, start:stop]

        # Compute all medusa heads at the same time, then reshape and move the n_medusa_heads dim to dim 1
        medusa_res = self.act(self.linear(x)).reshape(
            *x_block.shape[:-1], self.n_medusa_heads, x_block.shape[-1]
        )

        # Apply all residual medusa heads
        output = x[:, start:stop].unsqueeze(-2) + medusa_res

        # Gather medusa heads
        world_output = [
            torch.empty_like(output) for _ in range(self.process_group.size())
        ]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)

        # Stack x and medusa residual x
        stacked_x = torch.cat([x.unsqueeze(-2), world_output], dim=-2)

        # Compute lm head on x + medusa residual x
        logits = self.lm_head(stacked_x)

        # Finally, split logits from speculative logits
        logits, speculative_logits = torch.split(
            logits, [1, self.n_medusa_heads], dim=-2
        )
        # Squeeze added dimension
        logits = logits.squeeze(-2)

        return logits, speculative_logits
