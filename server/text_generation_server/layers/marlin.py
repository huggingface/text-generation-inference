from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    import marlin
except ImportError:
    marlin = None

try:
    major, _minor = torch.cuda.get_device_capability()
    has_sm_8_0 = major >= 8
except Exception:
    has_sm_8_0 = False

MARLIN_TILE_SIZE = 16


@dataclass
class MarlinWeight:
    """
    Marlin weights.

    Attributes:
        B (torch.Tensor): int4-quantized weights packed into int32.
        s (torch.Tensor): float16 scales.
    """

    B: torch.Tensor
    s: torch.Tensor


class MarlinLinear(nn.Module):
    def __init__(
        self, *, B: torch.Tensor, s: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        super().__init__()

        if not has_sm_8_0:
            raise NotImplementedError(
                "Using quantized marlin models requires CUDA capability 8.0 or later"
            )

        if marlin is None:
            raise NotImplementedError(
                "You do not seem to have marlin installed, either install it (cd server &&  make install-marlin)"
            )

        assert B.dtype == torch.int32
        assert s.dtype == torch.float16

        in_features = B.shape[0] * MARLIN_TILE_SIZE
        out_features = s.shape[1]
        assert (
            in_features % 128 == 0
        ), f"Number of input features ({in_features}) not divisable by 128"
        assert (
            out_features % 256 == 0
        ), f"Number of output features ({out_features}) not divisable by 256"

        group_size = -1 if s.shape[0] == 1 else in_features // s.shape[0]
        assert group_size in {
            -1,
            128,
        }, f"Group size must be -1 or 128, was {group_size}"

        self.register_buffer("B", B)
        self.register_buffer("s", s)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.workspace = torch.zeros(
            out_features // 128 * 16, dtype=torch.int, device=B.device
        )

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        assert marlin is not None
        C = torch.empty(
            A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device
        )
        marlin.mul(
            A.view((-1, A.shape[-1])),
            self.B,
            C.view((-1, C.shape[-1])),
            self.s,
            self.workspace,
        )

        if self.bias is not None:
            C += self.bias

        return C
