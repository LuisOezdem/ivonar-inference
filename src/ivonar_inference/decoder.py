from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import LatentAttention
from .layers import RMSNorm
from .recurrent import RecurrentMixer
from .model import IvonarModel
from .quantization import TernaryLinear, quantized_linear


def _default_attention_lengths(max_len: int) -> tuple[int, ...]:
    lengths = []
    length = 512
    while length < max_len:
        lengths.append(length)
        length *= 2
    return (*lengths, max_len)


@dataclass(frozen=True)
class _FusedProjection:
    weight: Tensor
    bias: Tensor | None
    splits: tuple[int, ...]
    eps: float

    def __call__(self, x: Tensor) -> tuple[Tensor, ...]:
        return quantized_linear(x, self.weight, self.bias, eps=self.eps).split(self.splits, dim=-1)


def _fuse_projections(projections: tuple[TernaryLinear, ...], device: torch.device) -> _FusedProjection | None:
    weights = []
    for projection in projections:
        weight = projection.cached_runtime_weight(device)
        if weight is None:
            return None
        weights.append(weight)
    if len({weight.dtype for weight in weights}) != 1 or len({float(p.eps) for p in projections}) != 1:
        return None
    biases = [projection.bias for projection in projections]
    if any(bias is None for bias in biases) != all(bias is None for bias in biases):
        return None
    bias = None if biases[0] is None else torch.cat([b.to(device) for b in biases], dim=0)
    return _FusedProjection(
        torch.cat(weights, dim=0), bias, tuple(int(weight.shape[0]) for weight in weights), float(projections[0].eps)
    )


class StaticDecoder:

    def __init__(
        self,
        model: IvonarModel,
        max_len: int | None = None,
        batch_size: int = 1,
        device: str | torch.device | None = None,
        sampling: bool = False,
        max_top_k: int = 256,
        attention_lengths: Iterable[int] | None = None,
        fuse_projections: bool | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_top_k <= 0:
            raise ValueError("max_top_k must be positive")
        self.model = model
        self.device = model.device if device is None else torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_len = int(model.config.seq_len if max_len is None else max_len)
        if self.max_len <= 0 or self.max_len > int(model.config.seq_len):
            raise ValueError("max_len must lie within the model context")
        self.batch_size = int(batch_size)
        lengths = {int(length) for length in (attention_lengths or _default_attention_lengths(self.max_len))}
        lengths.add(self.max_len)
        if any(length <= 0 or length > self.max_len for length in lengths):
            raise ValueError("attention_lengths must lie within max_len")
        self.attention_lengths = tuple(sorted(lengths))
        self.host_position = 0
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.cache_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.fuse_projections = self.device.type == "cuda" if fuse_projections is None else bool(fuse_projections)
        self.token = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.position = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.logits = torch.zeros(
            self.batch_size, int(model.config.vocab_size), dtype=torch.float32, device=self.device
        )
        self.slots = torch.arange(self.max_len, device=self.device)
        self.recurrent_states: dict[int, dict[str, Tensor]] = {}
        self.recurrent_complex: dict[int, Tensor] = {}
        self.recurrent_constants: dict[int, dict[str, Tensor]] = {}
        self.attention_caches: dict[int, tuple[Tensor, Tensor]] = {}
        self.attention_scales: dict[int, float] = {}
        self.rope_tables: dict[int, Tensor] = {}
        self.recurrent_projections: dict[int, _FusedProjection] = {}
        self.attention_in_projections: dict[int, _FusedProjection] = {}
        self.attention_up_projections: dict[int, _FusedProjection] = {}
        self.ffn_projections: dict[int, _FusedProjection] = {}
        self.norm_bounds: dict[int, tuple[Tensor, Tensor]] = {}
        self.sampling = bool(sampling)
        self.max_top_k = min(int(max_top_k), int(model.config.vocab_size))
        self.temperature = torch.ones(1, dtype=torch.float32, device=self.device)
        self.penalty = torch.ones(1, dtype=torch.float32, device=self.device)
        self.inverse_penalty = torch.ones(1, dtype=torch.float32, device=self.device)
        self.unit = torch.ones(1, dtype=torch.float32, device=self.device)
        self.top_k = torch.full((1,), self.max_top_k, dtype=torch.int64, device=self.device)
        self.ranks = torch.arange(self.max_top_k, device=self.device)
        self.seen = torch.zeros(self.batch_size, int(model.config.vocab_size), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            for index, block in enumerate(model.blocks):
                mixer = block.mixer
                if isinstance(mixer, RecurrentMixer):
                    self._prepare_recurrent(index, mixer)
                elif isinstance(mixer, LatentAttention):
                    self._prepare_attention(index, mixer)
                else:
                    raise TypeError(f"unsupported mixer for static decoding: {type(mixer).__name__}")
                self._prepare_ffn(index, block.ffn)
                self._prepare_norm(block.mixer_norm)
                self._prepare_norm(block.ffn_norm)
            self._prepare_norm(model.final_norm)

    def _fuse(self, projections: tuple[TernaryLinear, ...]) -> _FusedProjection | None:
        if not self.fuse_projections:
            return None
        return _fuse_projections(projections, self.device)

    def _prepare_norm(self, norm: RMSNorm) -> None:
        magnitude = norm.clip * norm.weight.abs().to(self.device)
        self.norm_bounds[id(norm)] = (-magnitude, magnitude)

    def _norm(self, norm: RMSNorm, x: Tensor) -> Tensor:
        low, high = self.norm_bounds[id(norm)]
        normalized = F.rms_norm(x.float(), (norm.hidden_dim,), norm.weight, norm.eps)
        return torch.clamp(normalized, min=low, max=high).to(dtype=x.dtype)

    def _prepare_recurrent(self, index: int, mixer: RecurrentMixer) -> None:
        state = mixer.initial_state(self.batch_size, self.device)
        self.recurrent_states[index] = state
        self.recurrent_complex[index] = torch.view_as_complex(state["ssm_state"])
        fused = self._fuse((mixer.in_proj, mixer.bc_dt_proj))
        if fused is not None:
            self.recurrent_projections[index] = fused
        decay = -F.softplus(mixer.log_decay).to(self.device)
        theta = mixer.theta.to(self.device)
        self.recurrent_constants[index] = {
            "double_cdecay": 2.0 * torch.complex(decay, theta).view(1, mixer.num_heads, mixer.state_dim, 1),
            "skip": mixer.skip.to(self.device).view(1, mixer.num_heads, mixer.head_dim),
            "half_dt_min": torch.tensor([0.5 * mixer.dt_min], dtype=torch.float32, device=self.device),
            "half_dt_range": torch.tensor(
                [0.5 * (mixer.dt_max - mixer.dt_min)], dtype=torch.float32, device=self.device
            ),
        }

    def _prepare_attention(self, index: int, mixer: LatentAttention) -> None:
        shape = (self.batch_size, mixer.num_heads, self.max_len, mixer.head_dim)
        self.attention_caches[index] = (
            torch.zeros(shape, dtype=self.cache_dtype, device=self.device),
            torch.zeros(shape, dtype=self.cache_dtype, device=self.device),
        )
        self.attention_scales[index] = 1.0 / math.sqrt(mixer.head_dim)
        unit = torch.zeros(1, self.max_len, 1, mixer.rope_dim, dtype=torch.float32, device=self.device)
        unit[..., 0::2] = 1.0
        rotated = mixer.rope.rotate(unit, offset=0).reshape(self.max_len, mixer.rope_dim // 2, 2).contiguous()
        identity = torch.ones(self.max_len, mixer.nope_dim // 2, dtype=torch.complex64, device=self.device)
        self.rope_tables[index] = torch.cat((identity, torch.view_as_complex(rotated)), dim=-1)
        fused_in = self._fuse((mixer.q_proj, mixer.kv_down_proj, mixer.k_rope_proj))
        if fused_in is not None:
            self.attention_in_projections[index] = fused_in
        fused_up = self._fuse((mixer.k_nope_up_proj, mixer.v_up_proj))
        if fused_up is not None:
            self.attention_up_projections[index] = fused_up

    def _prepare_ffn(self, index: int, ffn: nn.Module) -> None:
        fused = self._fuse((ffn.gate_proj, ffn.up_proj))
        if fused is not None:
            self.ffn_projections[index] = fused

    @torch.no_grad()
    def prefill(self, input_ids: Tensor) -> Tensor:
        """Run the prompt through the parallel path and load the buffers from its states."""

        if input_ids.ndim != 2 or input_ids.shape[0] != self.batch_size:
            raise ValueError(f"input_ids must have shape [{self.batch_size}, seq_len]")
        length = int(input_ids.shape[1])
        if length == 0 or length > self.max_len:
            raise ValueError("prompt length must lie within the decoder capacity")
        logits, states = self.model.prefill(input_ids.to(self.device))
        for index, block in enumerate(self.model.blocks):
            state = states[index]
            mixer = block.mixer
            if index in self.recurrent_states:
                for key, buffer in self.recurrent_states[index].items():
                    buffer.copy_(state[key].to(dtype=buffer.dtype))
            else:
                latent = state["latent"].to(dtype=torch.float32)
                k_rope = state["k_rope"].to(dtype=torch.float32)
                k_nope = mixer.k_nope_up_proj(latent).reshape(self.batch_size, length, mixer.num_heads, mixer.nope_dim)
                v = mixer.v_up_proj(latent).reshape(self.batch_size, length, mixer.num_heads, mixer.head_dim)
                k = torch.cat((k_nope, mixer.rope.rotate(k_rope, offset=0)), dim=-1)
                k_cache, v_cache = self.attention_caches[index]
                k_cache[:, :, :length].copy_(k.permute(0, 2, 1, 3))
                v_cache[:, :, :length].copy_(v.permute(0, 2, 1, 3))
        self.position.fill_(length)
        self.host_position = length
        self.logits.copy_(logits.to(dtype=torch.float32))
        return self.logits

    def attention_length(self, position: int) -> int:
        """The cache bucket a step at ``position`` attends over."""

        for length in self.attention_lengths:
            if position < length:
                return length
        raise ValueError("the decoder has no free position left")

    def _embed(self, token: Tensor) -> Tensor:
        return F.embedding(token, self.model.token_io.projection.runtime_weight(token.device)).to(dtype=torch.float32)

    def _recurrent_step(self, index: int, mixer: RecurrentMixer, h: Tensor) -> Tensor:
        state = self.recurrent_states[index]
        constants = self.recurrent_constants[index]
        batch_size, heads, state_dim, head_dim = self.batch_size, mixer.num_heads, mixer.state_dim, mixer.head_dim
        fused = self.recurrent_projections.get(index)
        if fused is not None:
            value_gate, bc_dt = fused(h)
        else:
            value_gate, bc_dt = mixer.in_proj(h), mixer.bc_dt_proj(h)
        value, gate = value_gate.chunk(2, dim=-1)
        window = torch.cat((state["conv_state"], value.unsqueeze(-1)), dim=-1)
        convolved = F.conv1d(window, mixer.conv_weight, mixer.conv_bias, groups=mixer.hidden_dim)
        state["conv_state"].copy_(window[:, :, 1:])
        value = F.silu(convolved).view(batch_size, heads, head_dim)
        gate = torch.sigmoid(gate).reshape(batch_size, heads, head_dim)
        bc_dt = bc_dt.reshape(batch_size, heads, 3, state_dim)
        bc = torch.tanh(bc_dt[:, :, :2])
        b_t = bc[:, :, 0].unsqueeze(-1)
        c_t = bc[:, :, 1].unsqueeze(-1)
        half_dt = torch.addcmul(
            constants["half_dt_min"], torch.sigmoid(bc_dt[:, :, 2]), constants["half_dt_range"]
        ).unsqueeze(-1)
        alpha = torch.exp(half_dt * constants["double_cdecay"])
        force = b_t * value.unsqueeze(2)
        drive = torch.addcmul(force, alpha, state["prev_force"]) * half_dt
        self.recurrent_complex[index].mul_(alpha).add_(drive)
        state["prev_force"].copy_(force)
        y = (c_t * state["ssm_state"][..., 0]).sum(dim=2)
        y = torch.addcmul(y, constants["skip"], value)
        return mixer.out_proj((y * gate).reshape(batch_size, mixer.hidden_dim))

    @staticmethod
    def _rotate(x: Tensor, rotation: Tensor) -> Tensor:
        pairs = torch.view_as_complex(x.contiguous().view(*x.shape[:-1], -1, 2))
        return torch.view_as_real(pairs * rotation).flatten(-2)

    def _attention_step(self, index: int, mixer: LatentAttention, h: Tensor, invalid: Tensor, length: int) -> Tensor:
        batch_size, heads = self.batch_size, mixer.num_heads
        fused_in = self.attention_in_projections.get(index)
        if fused_in is not None:
            q_full, latent, k_rope_full = fused_in(h)
        else:
            q_full, latent, k_rope_full = mixer.q_proj(h), mixer.kv_down_proj(h), mixer.k_rope_proj(h)
        q = q_full.reshape(batch_size, 1, heads, mixer.head_dim)
        k_rope = k_rope_full.reshape(batch_size, 1, heads, mixer.rope_dim)
        fused_up = self.attention_up_projections.get(index)
        if fused_up is not None:
            k_nope_full, v_full = fused_up(latent)
        else:
            k_nope_full, v_full = mixer.k_nope_up_proj(latent), mixer.v_up_proj(latent)
        k_nope = k_nope_full.reshape(batch_size, 1, heads, mixer.nope_dim)
        v = v_full.reshape(batch_size, 1, heads, mixer.head_dim)
        rotation = self.rope_tables[index].index_select(0, self.position).view(1, 1, 1, -1)
        k_new = self._rotate(torch.cat((k_nope, k_rope), dim=-1), rotation)
        q_new = self._rotate(q, rotation)
        k_cache, v_cache = self.attention_caches[index]
        k_cache.index_copy_(2, self.position, k_new.permute(0, 2, 1, 3).to(dtype=k_cache.dtype))
        v_cache.index_copy_(2, self.position, v.permute(0, 2, 1, 3).to(dtype=v_cache.dtype))
        q_heads = (q_new.permute(0, 2, 1, 3) * self.attention_scales[index]).to(dtype=k_cache.dtype)
        scores = torch.matmul(q_heads, k_cache[:, :, :length].transpose(-1, -2))
        probs = torch.softmax(scores.masked_fill(invalid, float("-inf")), dim=-1, dtype=torch.float32)
        y = torch.matmul(probs.to(dtype=v_cache.dtype), v_cache[:, :, :length]).to(dtype=q.dtype)
        return mixer.out_proj(y.permute(0, 2, 1, 3).reshape(batch_size, mixer.hidden_dim))

    def _ffn(self, index: int, block: nn.Module, x: Tensor) -> Tensor:
        h = self._norm(block.ffn_norm, x)
        fused = self.ffn_projections.get(index)
        if fused is None:
            return block.ffn(h.unsqueeze(1))[:, 0]
        gate, up = fused(h)
        return block.ffn.down_proj(F.silu(gate) * up)

    def _step(self, length: int) -> None:
        model = self.model
        invalid = (self.slots[:length] > self.position).view(1, 1, 1, length)
        x = self._embed(self.token)
        for index, block in enumerate(model.blocks):
            h = self._norm(block.mixer_norm, x)
            if index in self.recurrent_states:
                out = self._recurrent_step(index, block.mixer, h)
            else:
                out = self._attention_step(index, block.mixer, h, invalid, length)
            x = x + out
            x = x + self._ffn(index, block, x)
        self.logits.copy_(model.token_io.project(self._norm(model.final_norm, x)).to(dtype=torch.float32))
        self.position.add_(1)
        if self.sampling:
            self._sample()

    def _sample(self) -> None:
        scores = self.logits / self.temperature
        multiplier = torch.where(scores > 0, self.inverse_penalty, self.penalty)
        scores = scores * torch.where(self.seen, multiplier, self.unit)
        values, indices = torch.topk(scores, self.max_top_k, dim=-1)
        values = values.masked_fill(self.ranks >= self.top_k, float("-inf"))
        choice = torch.multinomial(torch.softmax(values, dim=-1), 1)
        sampled = indices.gather(-1, choice)
        self.seen.scatter_(1, sampled, True)
        self.token.copy_(sampled.squeeze(-1))

    def configure_sampling(
        self,
        temperature: float,
        top_k: int,
        repetition_penalty: float = 1.0,
        penalized_ids: Iterable[int] = (),
    ) -> None:
        """Set the sampling rule for the answer that starts now.

        ``penalized_ids`` start out penalized, which is how the caller keeps an
        answer from repeating the one before it word for word.
        """

        if not self.sampling:
            raise RuntimeError("this decoder was created without sampling")
        if temperature <= 0 or repetition_penalty <= 0:
            raise ValueError("temperature and repetition_penalty must be positive")
        effective = int(top_k) if 0 < int(top_k) <= self.max_top_k else self.max_top_k
        self.temperature.fill_(float(temperature))
        self.penalty.fill_(float(repetition_penalty))
        self.inverse_penalty.fill_(1.0 / float(repetition_penalty))
        self.top_k.fill_(effective)
        self.seen.zero_()
        seeds = [int(token) for token in penalized_ids if 0 <= int(token) < self.seen.shape[1]]
        if seeds:
            index = torch.tensor(sorted(set(seeds)), dtype=torch.int64, device=self.device)
            self.seen.index_fill_(1, index, True)

    @torch.no_grad()
    def sample(self) -> int:
        """Sample from the logits the last prefill or step produced."""

        if not self.sampling:
            raise RuntimeError("this decoder was created without sampling")
        self._sample()
        return int(self.token[0])

    @torch.no_grad()
    def advance(self) -> int:
        """Feed the sampled token, run one step and return the token sampled from it."""

        if not self.sampling:
            raise RuntimeError("this decoder was created without sampling")
        self._run_step()
        return int(self.token[0])

    def _run_step(self) -> None:
        length = self.attention_length(self.host_position)
        graph = self.graphs.get(length)
        if graph is not None:
            graph.replay()
        else:
            self._step(length)
        self.host_position += 1

    @property
    def graph(self) -> torch.cuda.CUDAGraph | None:
        """The graph for the full capacity, or None while the decoder runs eagerly."""

        return self.graphs.get(self.max_len)

    @torch.no_grad()
    def capture(self, warmup_steps: int = 2) -> bool:
        """Record the decode step as one CUDA graph per attention bucket; False where capture is unavailable."""

        if self.device.type != "cuda" or not torch.cuda.is_available():
            return False
        saved_position = int(self.position)
        saved_token = self.token.clone()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        graphs: dict[int, torch.cuda.CUDAGraph] = {}
        try:
            for length in self.attention_lengths:
                with torch.cuda.stream(stream):
                    for _ in range(max(1, warmup_steps)):
                        self.position.fill_(0)
                        self._step(length)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                self.position.fill_(0)
                with torch.cuda.graph(graph, stream=stream):
                    self._step(length)
                torch.cuda.synchronize(self.device)
                graphs[length] = graph
        except Exception:
            self.graphs = {}
            return False
        self.graphs = graphs
        self.position.fill_(saved_position)
        self.token.copy_(saved_token)
        return True

    @torch.no_grad()
    def step(self, token: Tensor) -> Tensor:
        """Feed one token per batch row and return the next logits."""

        if token.ndim == 2:
            token = token[:, 0]
        if token.shape != (self.batch_size,):
            raise ValueError(f"token must have shape [{self.batch_size}]")
        self.token.copy_(token.to(device=self.device, dtype=torch.int64))
        self._run_step()
        return self.logits

    def reset(self) -> None:
        self.position.zero_()
        self.host_position = 0
        self.seen.zero_()
