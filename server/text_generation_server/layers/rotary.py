import os
import torch
from torch import nn

from text_generation_server.utils.import_utils import SYSTEM

if SYSTEM == "cuda":
    from flash_attn.layers.rotary import RotaryEmbedding
    import rotary_emb
elif SYSTEM == "rocm":
    from vllm._C import ops
elif SYSTEM == "ipex":
    import intel_extension_for_pytorch as ipex


def _create_inv_freq(dim, base, device):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    return inv_freq


def _get_rope_config(config):
    if os.getenv("ROPE_SCALING", None) is not None:
        rope_scaling = {
            "type": os.environ["ROPE_SCALING"],
            "factor": float(os.environ["ROPE_FACTOR"]),
        }
        return rope_scaling
    return getattr(config, "rope_scaling", None)


class PositionRotaryEmbedding(nn.Module):
    def __init__(self, inv_freq, scaling_factor):
        super().__init__()
        self.inv_freq = inv_freq
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.scaling_factor = scaling_factor
        self.dynamic_args = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        # Such controlflows may add some overhead.
        if SYSTEM == "cuda":
            rotary_dim = cos.shape[-1]
            q1 = query[..., :rotary_dim]
            q2 = query[..., rotary_dim : 2 * rotary_dim]

            rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)

            k1 = key[..., :rotary_dim]
            k2 = key[..., rotary_dim : 2 * rotary_dim]

            rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)
        elif SYSTEM == "rocm":
            # NOTE: On RoCm systems, we use a ROPE implementatation adapted from VLLM which launches a single kernel for both query/key, contrary to flash-attn implementation used on NVIDIA systems.
            # Compiling flash-attn rotary on RoCm, it appears hipcc is unable to unroll loops, resulting in an even slower inference compared to eager: https://github.com/pytorch/pytorch/issues/113773

            head_size = query.shape[-1]

            # Inplace operation, updating query and key.
            ops.rotary_embedding(query, key, head_size, cos, sin, True)
        elif SYSTEM == "ipex":
            ipex.llm.functional.rotary_embedding(
                query, key, sin, cos, query.size(-1), True
            )
        else:
            raise ValueError(
                "Your system seem to be not supported. Please check your install or open an issue at https://github.com/huggingface/text-generation-inference/issues with a clear reproduction."
            )

    @classmethod
    def static(cls, config, dim, base, device):
        inv_freq = _create_inv_freq(dim, base, device)
        scaling_factor = None
        rope_scaling = _get_rope_config(config)
        if rope_scaling is not None:
            if rope_scaling["type"] == "linear":
                pass
            elif rope_scaling["type"] == "dynamic":
                scaling_factor = rope_scaling["factor"]
                return DynamicPositionRotaryEmbedding(
                    dim=dim,
                    max_position_embeddings=config.max_position_embeddings,
                    base=base,
                    device=inv_freq.device,
                    scaling_factor=scaling_factor,
                )
            elif rope_scaling["type"] == "yarn":
                scaling_factor = rope_scaling["factor"]
                return YarnPositionRotaryEmbedding(
                    dim=2 * inv_freq.shape[0],
                    max_position_embeddings=rope_scaling[
                        "original_max_position_embeddings"
                    ],
                    base=base,
                    device=inv_freq.device,
                    scaling_factor=scaling_factor,
                    extrapolation_factor=1,
                    attn_factor=1,
                    beta_fast=32,
                    beta_slow=1,
                )
            elif rope_scaling["type"] in ["su", "longrope"]:
                short_factor = torch.tensor(
                    rope_scaling["short_factor"], dtype=torch.float32, device=device
                )
                short_inv_freq = 1.0 / (
                    short_factor
                    * base
                    ** (
                        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                        / dim
                    )
                )
                long_factor = torch.tensor(
                    rope_scaling["long_factor"], dtype=torch.float32, device=device
                )
                long_inv_freq = 1.0 / (
                    long_factor
                    * base
                    ** (
                        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                        / dim
                    )
                )

                original_max_position_embeddings = (
                    config.original_max_position_embeddings
                )
                max_position_embeddings = config.max_position_embeddings
                if max_position_embeddings <= original_max_position_embeddings:
                    scaling_factor = 1.0
                else:
                    scale = max_position_embeddings / original_max_position_embeddings
                    scaling_factor = math.sqrt(
                        1 + math.log(scale) / math.log(original_max_position_embeddings)
                    )

                return SuRotaryEmbedding(
                    short_inv_freq=short_inv_freq,
                    long_inv_freq=long_inv_freq,
                    scaling_factor=scaling_factor,
                    original_max_position_embeddings=original_max_position_embeddings,
                )
            else:
                raise NotImplementedError(
                    f"rope scaling type {rope_scaling['type']} is not implemented or invalid"
                )
        return cls(inv_freq, scaling_factor)

    @classmethod
    def load(cls, config, prefix, weights):
        # XXX: Always load this in float32 !
        dtype = weights.dtype
        weights.dtype = torch.float32
        inv_freq = weights.get_tensor(f"{prefix}.inv_freq")
        weights.dtype = dtype

        scaling_factor = None
        rope_scaling = _get_rope_config(config)
        if rope_scaling is not None:
            scaling_factor = rope_scaling["factor"]
            if rope_scaling["type"] == "linear":
                pass
            elif rope_scaling["type"] == "dynamic":
                return DynamicPositionRotaryEmbedding(
                    dim=2 * inv_freq.shape[0],
                    max_position_embeddings=config.max_position_embeddings,
                    base=10000.0,
                    device=inv_freq.device,
                    scaling_factor=scaling_factor,
                )
            elif rope_scaling["type"] == "yarn":
                return YarnPositionRotaryEmbedding(
                    dim=2 * inv_freq.shape[0],
                    max_position_embeddings=rope_scaling[
                        "original_max_position_embeddings"
                    ],
                    base=10000.0,
                    device=inv_freq.device,
                    scaling_factor=scaling_factor,
                    extrapolation_factor=1,
                    attn_factor=1,
                    beta_fast=32,
                    beta_slow=1,
                )
            else:
                raise NotImplementedError(
                    f"rope scaling type {rope_scaling['type']} is not implemented or invalid"
                )
        return cls(inv_freq, scaling_factor)

    def _update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            if self.scaling_factor is not None:
                t /= self.scaling_factor
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def get_cos_sin(self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype):
        """
        Return cos and sin for the asked position ids
        """
        if SYSTEM == "rocm":
            # For RoCm, we always use float cos/sin to avoid a cast.
            # For NVIDIA, for some reason, the flash-attn rotary kernel requires cos/sin and query/key to be of same dtype: https://github.com/Dao-AILab/flash-attention/blob/017716451d446e464dde9aca3a3c1ed2209caaa9/csrc/rotary/rotary.cpp#L26
            # But later on goes and cast cos/sin to float anyway: https://github.com/Dao-AILab/flash-attention/blob/017716451d446e464dde9aca3a3c1ed2209caaa9/csrc/rotary/rotary_cuda.cu#L29, which looks suboptimal.
            dtype = torch.float32

        self._update_cos_sin_cache(dtype, position_ids.device, max_s)

        cos = torch.index_select(self._cos_cached, 0, position_ids)
        sin = torch.index_select(self._sin_cached, 0, position_ids)

        # Note: this unsqueeze is not necessary on RoCm + VLLM ROPE implementation, but we leave it as is to avoid yet an other controlflow.
        return cos.unsqueeze(1), sin.unsqueeze(1)


class SuRotaryEmbedding(PositionRotaryEmbedding):
    def __init__(
        self,
        short_inv_freq,
        long_inv_freq,
        scaling_factor,
        original_max_position_embeddings,
    ):
        super(PositionRotaryEmbedding, self).__init__()
        self.short_inv_freq = short_inv_freq
        self.long_inv_freq = long_inv_freq
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.dynamic_args = None

    def _update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen

            t = torch.arange(seqlen, device=device, dtype=self.short_inv_freq.dtype)
            short_freqs = torch.outer(
                t[: self.original_max_position_embeddings],
                self.short_inv_freq.to(device=t.device),
            )
            long_freqs = torch.outer(
                t[self.original_max_position_embeddings :],
                self.long_inv_freq.to(device=t.device),
            )

            freqs = torch.cat([short_freqs, long_freqs])

            self._cos_cached = (torch.cos(freqs) * self.scaling_factor).to(dtype)
            self._sin_cached = (torch.sin(freqs) * self.scaling_factor).to(dtype)


class DynamicPositionRotaryEmbedding(PositionRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings, base, device, scaling_factor):
        inv_freq = _create_inv_freq(dim, base, device)
        super().__init__(inv_freq, scaling_factor)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

    def _update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            if seqlen > self.max_position_embeddings:
                newbase = self.base * (
                    (self.scaling_factor * seqlen / self.max_position_embeddings)
                    - (self.scaling_factor - 1)
                ) ** (self.dim / (self.dim - 2))
                self.inv_freq = _create_inv_freq(
                    self.dim, newbase, self.inv_freq.device
                )
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)


# Inverse dim formula to find dim based on number of rotations
import math


def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YarnPositionRotaryEmbedding(PositionRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings,
        base,
        device,
        scaling_factor,
        *,
        extrapolation_factor,
        attn_factor,
        beta_fast,
        beta_slow,
    ):
        inv_freq = _create_inv_freq(dim, base, device)
        super().__init__(inv_freq, scaling_factor)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = float(
            get_mscale(self.scaling_factor) * self.attn_factor
        )  # Get n-d magnitude scaling corrected for interpolation

    def _update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            if seqlen > self.max_position_embeddings:
                inv_freq_extrapolation = _create_inv_freq(
                    self.dim, self.base, self.inv_freq.device
                )
                freqs = 1.0 / inv_freq_extrapolation
                inv_freq_interpolation = 1.0 / (self.scaling_factor * freqs)
                low, high = find_correction_range(
                    self.beta_fast,
                    self.beta_slow,
                    self.dim,
                    self.base,
                    self.max_position_embeddings,
                )
                inv_freq_mask = (
                    1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)
                ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
                inv_freq = (
                    inv_freq_interpolation * (1 - inv_freq_mask)
                    + inv_freq_extrapolation * inv_freq_mask
                )

                self.inv_freq = inv_freq
                self.mscale = float(
                    get_mscale(self.scaling_factor) * self.attn_factor
                )  # Get n-d magnitude scaling corrected for interpolation

            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = (torch.cos(freqs) * self.mscale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * self.mscale).to(dtype)
