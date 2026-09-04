from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

TERNARY_GROUP_SIZE = 128


def unpack_ternary_codes(packed: Tensor, in_features: int) -> Tensor:
    values = torch.stack(
        (packed & 0x03, (packed >> 2) & 0x03, (packed >> 4) & 0x03, (packed >> 6) & 0x03),
        dim=-1,
    ).reshape(*packed.shape[:-1], -1)
    return values[..., :in_features].to(torch.int8) - 1


def dequantize_ternary_codes(codes: Tensor, scales: Tensor, group_size: int, dtype: torch.dtype) -> Tensor:
    in_features = int(codes.shape[-1])
    groups = math.ceil(in_features / group_size)
    pad = groups * group_size - in_features
    values = codes.float()
    if pad:
        values = F.pad(values, (0, pad))
    values = values.reshape(*codes.shape[:-1], groups, group_size) * scales.float().unsqueeze(-1)
    values = values.reshape(*codes.shape[:-1], -1)
    if pad:
        values = values[..., :in_features]
    return values.to(dtype)


def quantize_activations(x: Tensor, eps: float = 1e-5) -> Tensor:
    scale = x.abs().amax(dim=-1, keepdim=True).float().div(127.0).clamp_min(eps)
    quantized = (torch.round(x.float() / scale).clamp(-127, 127) * scale).to(x.dtype)
    return x + (quantized - x)


_ZERO_BY_DEVICE: dict[torch.device, Tensor] = {}


def _zero(device: torch.device) -> Tensor:
    zero = _ZERO_BY_DEVICE.get(device)
    if zero is None:
        zero = torch.zeros(1, dtype=torch.float32, device=device)
        _ZERO_BY_DEVICE[device] = zero
    return zero


def quantized_linear(x: Tensor, weight: Tensor, bias: Tensor | None, eps: float = 1e-5) -> Tensor:
    if weight.dtype == x.dtype:
        return F.linear(quantize_activations(x, eps), weight, bias)
    amax = torch.linalg.vector_norm(x.detach(), ord=float("inf"), dim=-1, keepdim=True)
    bound = amax.clamp_min_(127.0 * eps)
    zero = _zero(x.device)
    codes = torch.empty_like(x, dtype=weight.dtype)
    torch.addcdiv(zero, x.detach(), bound, value=127.0, out=codes)
    out = F.linear(codes.round_(), weight)
    return torch.addcmul(zero if bias is None else bias, out, bound, value=1.0 / 127.0).to(dtype=x.dtype)


def _same_device(first: torch.device, second: torch.device) -> bool:
    if first.type != second.type:
        return False
    if first.type != "cuda" or first.index is None or second.index is None:
        return True
    return first.index == second.index


class TernaryLinear(nn.Module):

    def __init__(
        self,
        packed_weight: Tensor,
        weight_scales: Tensor,
        in_features: int,
        bias: Tensor | None = None,
        eps: float = 1e-5,
        group_size: int = TERNARY_GROUP_SIZE,
    ) -> None:
        super().__init__()
        if packed_weight.ndim != 2 or packed_weight.dtype != torch.uint8:
            raise ValueError("packed_weight must be a two-dimensional uint8 tensor")
        self.in_features = int(in_features)
        self.out_features = int(packed_weight.shape[0])
        self.group_size = int(group_size)
        self.eps = float(eps)
        expected_packed = (self.out_features, math.ceil(self.in_features / 4))
        expected_scales = (self.out_features, math.ceil(self.in_features / self.group_size))
        if tuple(packed_weight.shape) != expected_packed:
            raise ValueError(f"invalid packed shape: expected {expected_packed}, got {tuple(packed_weight.shape)}")
        if tuple(weight_scales.shape) != expected_scales:
            raise ValueError(f"invalid scale shape: expected {expected_scales}, got {tuple(weight_scales.shape)}")
        if bias is not None and tuple(bias.shape) != (self.out_features,):
            raise ValueError(f"invalid bias shape: expected ({self.out_features},), got {tuple(bias.shape)}")
        self.register_buffer("packed_weight", packed_weight.contiguous())
        self.register_buffer("weight_scales", weight_scales.contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().clone().float())
        self._runtime_weight: Tensor | None = None

    def unpacked_weight(self, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> Tensor:
        target = torch.device(device) if device is not None else self.packed_weight.device
        codes = unpack_ternary_codes(self.packed_weight.to(target), self.in_features)
        return dequantize_ternary_codes(codes, self.weight_scales.to(target), self.group_size, dtype)

    def materialize(self, dtype: torch.dtype = torch.float32) -> None:
        self._runtime_weight = self.unpacked_weight(dtype=dtype).detach()

    def cached_runtime_weight(self, device: torch.device | str | None = None) -> Tensor | None:
        target = self.packed_weight.device if device is None else torch.device(device)
        cached = self._runtime_weight
        if cached is None or not _same_device(cached.device, target):
            return None
        return cached

    def runtime_weight(self, device: torch.device | str | None = None) -> Tensor:
        cached = self.cached_runtime_weight(device)
        if cached is not None:
            return cached
        return self.unpacked_weight(device=device)

    def _apply(self, fn, recurse: bool = True):
        module = super()._apply(fn, recurse)
        if self._runtime_weight is not None:
            self._runtime_weight = fn(self._runtime_weight)
        return module

    def forward(self, x: Tensor) -> Tensor:
        return quantized_linear(x, self.runtime_weight(x.device), self.bias, eps=self.eps)
