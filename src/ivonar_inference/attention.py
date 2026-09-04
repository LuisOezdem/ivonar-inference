from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quantization import TernaryLinear


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x_float = x.float()
    x_even, x_odd = x_float[..., 0::2], x_float[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(dtype=x.dtype)


class Rotary(nn.Module):

    def __init__(
        self,
        head_dim: int,
        base_context: int,
        active_context: int,
        rope_base: float = 10000.0,
        scaling_strategy: str = "linear",
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary position encoding")
        self.head_dim = int(head_dim)
        self.base_context = int(base_context)
        self.active_context = int(active_context)
        self.rope_base = float(rope_base)
        self.scaling_strategy = scaling_strategy.strip().lower()
        if self.scaling_strategy not in {"linear", "yarn_like", "none"}:
            raise ValueError("Rotary scaling_strategy must be linear, yarn_like, or none")
        values = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        self.register_buffer("inv_freq", 1.0 / (self.rope_base ** (values / self.head_dim)))

    def rotate(self, x: Tensor, offset: int = 0) -> Tensor:
        seq_len = x.shape[1]
        positions = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.float32)
        inv_freq = self._scaled_inv_freq(positions.device)
        positions = self._scaled_positions(positions)
        freqs = torch.einsum("s,d->sd", positions, inv_freq)
        cos = freqs.cos().view(1, seq_len, 1, -1)
        sin = freqs.sin().view(1, seq_len, 1, -1)
        return _apply_rope(x, cos, sin)

    def _interpolation_ratio(self) -> float:
        if self.active_context <= self.base_context:
            return 1.0
        return self.active_context / self.base_context

    def _scaled_positions(self, positions: Tensor) -> Tensor:
        ratio = self._interpolation_ratio()
        if self.scaling_strategy == "none" or ratio <= 1.0:
            return positions
        if self.scaling_strategy == "linear":
            return positions / ratio
        return positions

    def _scaled_inv_freq(self, device: torch.device) -> Tensor:
        inv_freq = self.inv_freq.to(device)
        ratio = self._interpolation_ratio()
        if self.scaling_strategy != "yarn_like" or ratio <= 1.0:
            return inv_freq
        dim = inv_freq.numel()
        ramp = torch.linspace(0.0, 1.0, steps=dim, device=device)
        correction = 1.0 + (ratio - 1.0) * ramp.square()
        return inv_freq / correction


class LatentAttention(nn.Module):

    def __init__(
        self,
        q_proj: TernaryLinear,
        kv_down_proj: TernaryLinear,
        k_nope_up_proj: TernaryLinear,
        k_rope_proj: TernaryLinear,
        v_up_proj: TernaryLinear,
        out_proj: TernaryLinear,
        num_heads: int,
        base_context: int,
        active_context: int,
        rope_base: float = 10000.0,
        scaling_strategy: str = "linear",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(q_proj.in_features)
        self.num_heads = int(num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = self.hidden_dim // self.num_heads
        if self.head_dim < 4:
            raise ValueError("head_dim must be at least 4 for MLA decoupled RoPE")
        self.rope_dim = max(2, (self.head_dim // 2) // 2 * 2)
        self.nope_dim = self.head_dim - self.rope_dim
        self.latent_dim = int(kv_down_proj.out_features)
        expected = {
            "q_proj": (self.hidden_dim, self.hidden_dim),
            "kv_down_proj": (self.hidden_dim, self.latent_dim),
            "k_nope_up_proj": (self.latent_dim, self.num_heads * self.nope_dim),
            "k_rope_proj": (self.hidden_dim, self.num_heads * self.rope_dim),
            "v_up_proj": (self.latent_dim, self.hidden_dim),
            "out_proj": (self.hidden_dim, self.hidden_dim),
        }
        projections = {
            "q_proj": q_proj,
            "kv_down_proj": kv_down_proj,
            "k_nope_up_proj": k_nope_up_proj,
            "k_rope_proj": k_rope_proj,
            "v_up_proj": v_up_proj,
            "out_proj": out_proj,
        }
        for name, projection in projections.items():
            shape = (projection.in_features, projection.out_features)
            if shape != expected[name]:
                raise ValueError(f"{name} has shape {shape}, expected {expected[name]}")
        self.q_proj = q_proj
        self.kv_down_proj = kv_down_proj
        self.k_nope_up_proj = k_nope_up_proj
        self.k_rope_proj = k_rope_proj
        self.v_up_proj = v_up_proj
        self.out_proj = out_proj
        self.rope = Rotary(
            self.rope_dim,
            base_context=base_context,
            active_context=active_context,
            rope_base=rope_base,
            scaling_strategy=scaling_strategy,
        )

    def forward(self, x: Tensor, state: dict[str, Tensor] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if state is not None:
            raise ValueError("attention runs a prompt from the beginning; the decoder continues it")
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        latent = self.kv_down_proj(x)
        k_rope = self.k_rope_proj(x).view(batch_size, seq_len, self.num_heads, self.rope_dim)
        k_nope = self.k_nope_up_proj(latent).view(batch_size, seq_len, self.num_heads, self.nope_dim)
        v = self.v_up_proj(latent).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q_rotated = torch.cat((q[..., : self.nope_dim], self.rope.rotate(q[..., self.nope_dim :], offset=0)), dim=-1)
        k_rotated = torch.cat((k_nope, self.rope.rotate(k_rope, offset=0)), dim=-1)
        y = F.scaled_dot_product_attention(
            q_rotated.transpose(1, 2), k_rotated.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        out = self.out_proj(y.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim))
        return out, {"latent": latent, "k_rope": k_rope}
