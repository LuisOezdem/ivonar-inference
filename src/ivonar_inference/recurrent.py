from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .quantization import TernaryLinear

_SSD_CHUNK_LENGTHS = (256, 128, 64)
_SSD_MAX_LOG_RANGE = 30.0


def _scan_chunk_states(
    gate_real: Tensor,
    gate_imag: Tensor,
    increment_real: Tensor,
    increment_imag: Tensor,
    initial_real: Tensor,
    initial_imag: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    chunks = gate_real.shape[1]
    opening_real: list[Tensor] = []
    opening_imag: list[Tensor] = []
    current_real = initial_real
    current_imag = initial_imag
    for index in range(chunks):
        opening_real.append(current_real)
        opening_imag.append(current_imag)
        step_gate_real = gate_real[:, index]
        step_gate_imag = gate_imag[:, index]
        next_real = step_gate_real * current_real - step_gate_imag * current_imag + increment_real[:, index]
        next_imag = step_gate_real * current_imag + step_gate_imag * current_real + increment_imag[:, index]
        current_real, current_imag = next_real, next_imag
    return (
        torch.stack(opening_real, dim=1),
        torch.stack(opening_imag, dim=1),
        torch.stack((current_real, current_imag), dim=-1),
    )


class RecurrentMixer(nn.Module):

    def __init__(
        self,
        in_proj: TernaryLinear,
        bc_dt_proj: TernaryLinear,
        out_proj: TernaryLinear,
        log_decay: Tensor,
        theta: Tensor,
        skip: Tensor,
        conv_weight: Tensor,
        conv_bias: Tensor,
        num_heads: int,
        state_dim: int,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(in_proj.in_features)
        self.num_heads = int(num_heads)
        self.state_dim = int(state_dim)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.head_dim = self.hidden_dim // self.num_heads
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        if tuple(log_decay.shape) != (self.num_heads, self.state_dim) or tuple(theta.shape) != (
            self.num_heads,
            self.state_dim,
        ):
            raise ValueError("log_decay and theta must have shape [num_heads, state_dim]")
        if tuple(skip.shape) != (self.num_heads, self.head_dim):
            raise ValueError("skip must have shape [num_heads, head_dim]")
        if conv_weight.ndim != 3 or conv_weight.shape[0] != self.hidden_dim or conv_weight.shape[1] != 1:
            raise ValueError("conv_weight must have shape [hidden_dim, 1, conv_kernel_size]")
        self.conv_kernel_size = int(conv_weight.shape[2])
        self.in_proj = in_proj
        self.bc_dt_proj = bc_dt_proj
        self.out_proj = out_proj
        self.register_buffer("log_decay", log_decay.detach().clone().float())
        self.register_buffer("theta", theta.detach().clone().float())
        self.register_buffer("skip", skip.detach().clone().float())
        self.register_buffer("conv_weight", conv_weight.detach().clone().float())
        self.register_buffer("conv_bias", conv_bias.detach().clone().float())
        self.chunk_length = self._ssd_chunk_length()

    def _ssd_chunk_length(self) -> int:
        peak_decay = float(F.softplus(self.log_decay.float()).max().item())
        span = max(peak_decay * self.dt_max, torch.finfo(torch.float32).tiny)
        for length in _SSD_CHUNK_LENGTHS:
            if length * span <= _SSD_MAX_LOG_RANGE:
                return length
        raise RuntimeError("the trained decay leaves the range the chunked scan can absorb")

    def initial_state(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        return {
            "ssm_state": torch.zeros(
                batch_size, self.num_heads, self.state_dim, self.head_dim, 2, dtype=torch.float32, device=device
            ),
            "prev_force": torch.zeros(
                batch_size, self.num_heads, self.state_dim, self.head_dim, dtype=torch.float32, device=device
            ),
            "conv_state": torch.zeros(
                batch_size, self.hidden_dim, self.conv_kernel_size - 1, dtype=torch.float32, device=device
            ),
        }

    def _causal_conv(self, value: Tensor, conv_state: Tensor) -> tuple[Tensor, Tensor]:
        signal = value.transpose(1, 2)
        padded = torch.cat([conv_state.to(dtype=signal.dtype), signal], dim=2)
        convolved = F.conv1d(
            padded,
            self.conv_weight.to(dtype=signal.dtype),
            self.conv_bias.to(dtype=signal.dtype),
            groups=self.hidden_dim,
        )
        next_state = padded[:, :, padded.shape[2] - (self.conv_kernel_size - 1) :]
        return F.silu(convolved.transpose(1, 2)), next_state

    def forward(self, x: Tensor, state: dict[str, Tensor] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 3 or x.shape[-1] != self.hidden_dim:
            raise ValueError(f"x must have shape [batch_size, seq_len, {self.hidden_dim}]")
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self.initial_state(batch_size, x.device)
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value, conv_state = self._causal_conv(value, state["conv_state"])
        value_float = value.view(batch_size, seq_len, self.num_heads, self.head_dim).float()
        gate = torch.sigmoid(gate).view(batch_size, seq_len, self.num_heads, self.head_dim)
        bc_dt = self.bc_dt_proj(x).view(batch_size, seq_len, self.num_heads, 3, self.state_dim)
        b_coeff = torch.tanh(bc_dt[:, :, :, 0, :]).float()
        c_coeff = torch.tanh(bc_dt[:, :, :, 1, :]).float()
        dt = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(bc_dt[:, :, :, 2, :].float())
        y, ssm_state, prev_force = self._forward_ssd(
            value_float, b_coeff, c_coeff, dt, state["ssm_state"], state["prev_force"]
        )
        y = y + self.skip.view(1, 1, self.num_heads, self.head_dim) * value_float
        y = (y.to(gate.dtype) * gate).reshape(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(y), {"ssm_state": ssm_state, "prev_force": prev_force, "conv_state": conv_state}

    def _forward_ssd(
        self,
        drive: Tensor,
        b_coeff: Tensor,
        c_coeff: Tensor,
        dt: Tensor,
        ssm_state: Tensor,
        prev_force: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        seq_len = drive.shape[1]
        chunk_length = self.chunk_length
        decay = -F.softplus(self.log_decay)
        theta = self.theta
        state = ssm_state
        force = prev_force
        outputs: list[Tensor] = []
        aligned_len = (seq_len // chunk_length) * chunk_length
        if aligned_len:
            block_y, state, force = self._ssd_aligned_blocks(
                drive[:, :aligned_len],
                b_coeff[:, :aligned_len],
                c_coeff[:, :aligned_len],
                dt[:, :aligned_len],
                state,
                force,
                decay,
                theta,
                chunk_length,
            )
            outputs.append(block_y)
        for start in range(aligned_len, seq_len, chunk_length):
            end = min(start + chunk_length, seq_len)
            chunk_y, state, force = self._ssd_chunk(
                drive[:, start:end],
                b_coeff[:, start:end],
                c_coeff[:, start:end],
                dt[:, start:end],
                state,
                force,
                decay,
                theta,
            )
            outputs.append(chunk_y)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1), state, force

    def _ssd_aligned_blocks(
        self,
        drive: Tensor,
        b_coeff: Tensor,
        c_coeff: Tensor,
        dt: Tensor,
        state: Tensor,
        force: Tensor,
        decay: Tensor,
        theta: Tensor,
        chunk_length: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, num_heads, head_dim = drive.shape
        chunk_len = chunk_length
        chunks = seq_len // chunk_len
        state_dim = self.state_dim
        coefficient_shape = (batch_size, chunks, chunk_len, num_heads, state_dim)
        drive_blocks = drive.reshape(batch_size, chunks, chunk_len, num_heads, head_dim)
        b_blocks = b_coeff.reshape(coefficient_shape)
        c_blocks = c_coeff.reshape(coefficient_shape)
        dt_blocks = dt.reshape(coefficient_shape)

        decay_view = decay.view(1, 1, 1, num_heads, state_dim)
        theta_view = theta.view(1, 1, 1, num_heads, state_dim)
        step_real = (dt_blocks * decay_view).clamp(min=-60.0, max=0.0)
        log_real = torch.cumsum(step_real, dim=2)
        log_imag = torch.cumsum(dt_blocks * theta_view, dim=2)
        cos_imag = torch.cos(log_imag)
        sin_imag = torch.sin(log_imag)
        forward_magnitude = torch.exp(log_real)
        inverse_magnitude = torch.exp(-log_real)
        a_real = forward_magnitude * cos_imag
        a_imag = forward_magnitude * sin_imag
        inverse_real = inverse_magnitude * cos_imag
        inverse_imag = -inverse_magnitude * sin_imag

        if chunk_len > 1:
            weights = torch.cat(
                [0.5 * (dt_blocks[:, :, :-1] + dt_blocks[:, :, 1:]), torch.zeros_like(dt_blocks[:, :, -1:])],
                dim=2,
            )
        else:
            weights = torch.zeros_like(dt_blocks)
        scaled_b = b_blocks * weights

        c_real = (c_blocks * a_real).permute(0, 1, 3, 2, 4)
        c_imag = (c_blocks * a_imag).permute(0, 1, 3, 2, 4)
        b_real = (scaled_b * inverse_real).permute(0, 1, 3, 2, 4)
        b_imag = (scaled_b * inverse_imag).permute(0, 1, 3, 2, 4)
        drive_heads = drive_blocks.permute(0, 1, 3, 2, 4)

        transfer = torch.matmul(c_real, b_real.transpose(-1, -2)) - torch.matmul(c_imag, b_imag.transpose(-1, -2))
        history = torch.matmul(transfer.tril(-1), drive_heads)
        diagonal = (c_blocks * b_blocks * dt_blocks * 0.5).sum(dim=-1).permute(0, 1, 3, 2).unsqueeze(-1)

        chunk_force = b_blocks[:, :, -1].unsqueeze(-1) * drive_blocks[:, :, -1].unsqueeze(-2)
        incoming_force = torch.cat([force.unsqueeze(1), chunk_force[:, :-1]], dim=1)
        first_half_dt = (0.5 * dt_blocks[:, :, 0]).unsqueeze(-1)
        last_half_dt = (0.5 * dt_blocks[:, :, -1]).unsqueeze(-1)

        local_real = torch.matmul(b_real.transpose(-1, -2), drive_heads)
        local_imag = torch.matmul(b_imag.transpose(-1, -2), drive_heads)
        seed_real = incoming_force * first_half_dt + local_real
        seed_imag = local_imag
        gate_real = a_real[:, :, -1].unsqueeze(-1)
        gate_imag = a_imag[:, :, -1].unsqueeze(-1)
        increment_real = gate_real * seed_real - gate_imag * seed_imag + last_half_dt * chunk_force
        increment_imag = gate_real * seed_imag + gate_imag * seed_real

        opening_real, opening_imag, state = _scan_chunk_states(
            gate_real, gate_imag, increment_real, increment_imag, state[..., 0], state[..., 1]
        )

        carry_real = opening_real + incoming_force * first_half_dt
        carry_imag = opening_imag
        carried = torch.matmul(c_real, carry_real) - torch.matmul(c_imag, carry_imag)
        block_y = (carried + history + diagonal * drive_heads).permute(0, 1, 3, 2, 4)
        return block_y.reshape(batch_size, seq_len, num_heads, head_dim), state, chunk_force[:, -1]

    def _ssd_chunk(
        self,
        drive: Tensor,
        b_coeff: Tensor,
        c_coeff: Tensor,
        dt: Tensor,
        state: Tensor,
        force: Tensor,
        decay: Tensor,
        theta: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        chunk_len = drive.shape[1]
        decay_view = decay.view(1, 1, self.num_heads, self.state_dim)
        theta_view = theta.view(1, 1, self.num_heads, self.state_dim)
        step_real = (dt * decay_view).clamp(min=-60.0, max=0.0)
        log_real = torch.cumsum(step_real, dim=1)
        log_imag = torch.cumsum(dt * theta_view, dim=1)
        cos_imag = torch.cos(log_imag)
        sin_imag = torch.sin(log_imag)
        forward_magnitude = torch.exp(log_real)
        inverse_magnitude = torch.exp(-log_real)
        a_real = forward_magnitude * cos_imag
        a_imag = forward_magnitude * sin_imag
        inverse_real = inverse_magnitude * cos_imag
        inverse_imag = -inverse_magnitude * sin_imag

        carry_re = state[..., 0] + force * (0.5 * dt[:, 0]).unsqueeze(-1)
        carry_im = state[..., 1]

        if chunk_len > 1:
            weights = torch.cat([0.5 * (dt[:, :-1] + dt[:, 1:]), torch.zeros_like(dt[:, -1:])], dim=1)
        else:
            weights = torch.zeros_like(dt)
        scaled_b = b_coeff * weights

        c_real = (c_coeff * a_real).permute(0, 2, 1, 3)
        c_imag = (c_coeff * a_imag).permute(0, 2, 1, 3)
        b_real = (scaled_b * inverse_real).permute(0, 2, 1, 3)
        b_imag = (scaled_b * inverse_imag).permute(0, 2, 1, 3)
        drive_heads = drive.permute(0, 2, 1, 3)

        transfer = torch.matmul(c_real, b_real.transpose(-1, -2)) - torch.matmul(c_imag, b_imag.transpose(-1, -2))
        history = torch.matmul(transfer.tril(-1), drive_heads)
        carried = torch.matmul(c_real, carry_re) - torch.matmul(c_imag, carry_im)
        diagonal = (c_coeff * b_coeff * dt * 0.5).sum(dim=-1).permute(0, 2, 1).unsqueeze(-1)
        y = (carried + history + diagonal * drive_heads).permute(0, 2, 1, 3)

        residual_real = carry_re + torch.matmul(b_real.transpose(-1, -2), drive_heads)
        residual_imag = carry_im + torch.matmul(b_imag.transpose(-1, -2), drive_heads)
        last_real = a_real[:, -1].unsqueeze(-1)
        last_imag = a_imag[:, -1].unsqueeze(-1)
        next_force = b_coeff[:, -1].unsqueeze(-1) * drive[:, -1].unsqueeze(-2)
        next_state = torch.stack(
            (
                last_real * residual_real - last_imag * residual_imag + (0.5 * dt[:, -1]).unsqueeze(-1) * next_force,
                last_real * residual_imag + last_imag * residual_real,
            ),
            dim=-1,
        )
        return y, next_state, next_force
