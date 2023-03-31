import torch
import rotary_emb

from flash_attn.layers.rotary import RotaryEmbedding


class PositionRotaryEmbedding(RotaryEmbedding):
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
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def get_cos_sin(self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype):
        """
        Return cos and sin for the asked position ids
        """

        self._update_cos_sin_cache(dtype, position_ids.device, max_s)

        cos = torch.index_select(self._cos_cached, 0, position_ids)
        sin = torch.index_select(self._sin_cached, 0, position_ids)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def forward(self, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        rotary_dim = cos.shape[-1]
        q1 = qkv[:, 0, :, :rotary_dim]
        q2 = qkv[:, 0, :, rotary_dim : 2 * rotary_dim]
        k1 = qkv[:, 1, :, :rotary_dim]
        k2 = qkv[:, 1, :, rotary_dim : 2 * rotary_dim]

        rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)
        rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)
        return qkv
