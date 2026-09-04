from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quantization import TernaryLinear


class RMSNorm(nn.Module):

    def __init__(self, weight: Tensor, eps: float = 1e-6, clip: float = 8.0) -> None:
        super().__init__()
        if weight.ndim != 1:
            raise ValueError("RMSNorm weight must be one-dimensional")
        self.hidden_dim = int(weight.shape[0])
        self.eps = float(eps)
        self.clip = float(clip)
        self.register_buffer("weight", weight.detach().clone().float())

    def forward(self, x: Tensor) -> Tensor:
        x_float = x.float()
        inverse_rms = x_float.square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        clipped = (x_float * inverse_rms).clamp(-self.clip, self.clip)
        return (clipped * self.weight).to(dtype=x.dtype)


class TokenHead(nn.Module):

    def __init__(self, projection: TernaryLinear) -> None:
        super().__init__()
        self.projection = projection
        self.vocab_size = int(projection.out_features)
        self.hidden_dim = int(projection.in_features)

    def embed(self, input_ids: Tensor) -> Tensor:
        if input_ids.dtype != torch.int64:
            raise TypeError("input_ids must use int64 token indices")
        if input_ids.numel():
            minimum, maximum = torch.aminmax(input_ids)
            if int(minimum.item()) < 0 or int(maximum.item()) >= self.vocab_size:
                raise IndexError(f"token id is out of range for vocab_size={self.vocab_size}")
        weight = self.projection.runtime_weight(input_ids.device)
        return F.embedding(input_ids, weight).to(dtype=torch.float32)

    def project(self, hidden: Tensor) -> Tensor:
        return self.projection(hidden)


class GatedFeedForward(nn.Module):
    def __init__(self, gate_proj: TernaryLinear, up_proj: TernaryLinear, down_proj: TernaryLinear) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
