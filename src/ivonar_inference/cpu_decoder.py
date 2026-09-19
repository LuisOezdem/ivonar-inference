from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .attention import LatentAttention
from .decoder import StaticDecoder
from .model import IvonarModel
from .paths import ivonar_home
from .quantization import TERNARY_GROUP_SIZE, TernaryLinear
from .recurrent import RecurrentMixer

os.environ.setdefault("NUMBA_CACHE_DIR", str(ivonar_home() / "numba"))

import numba

GROUP = TERNARY_GROUP_SIZE
PLANE = GROUP // 4
_lock = threading.Lock()
_f32 = np.float32
FAST = {"reassoc", "contract", "nsz"}


@numba.njit(cache=True)
def quantize_into(values, q, gsum):
    n = values.shape[0]
    amax = _f32(0.0)
    for i in range(n):
        magnitude = abs(values[i])
        if magnitude > amax:
            amax = magnitude
    scale = amax / _f32(127.0)
    if scale < _f32(1e-5):
        scale = _f32(1e-5)
    gsum[:] = 0
    for i in range(n):
        level = np.rint(values[i] / scale)
        if level > _f32(127.0):
            level = _f32(127.0)
        elif level < _f32(-127.0):
            level = _f32(-127.0)
        code = np.int32(level)
        q[i] = code
        gsum[i // 128] += code
    q[n:] = 0
    return scale


@numba.njit(cache=True)
def norm_quantize(x, weight, eps, clip, work, q, gsum):
    n = x.shape[0]
    squares = _f32(0.0)
    for i in range(n):
        squares += x[i] * x[i]
    inverse = _f32(1.0) / np.sqrt(squares / _f32(n) + eps)
    for i in range(n):
        value = x[i] * inverse * weight[i]
        bound = clip * abs(weight[i])
        if value > bound:
            value = bound
        elif value < -bound:
            value = -bound
        work[i] = value
    return quantize_into(work[:n], q, gsum)


@numba.njit(inline="always")
def row_dot(planes, scales, r, q, gsum):
    groups = planes.shape[1] // PLANE
    total = _f32(0.0)
    for g in range(groups):
        acc = np.int32(0)
        base = g * GROUP
        row = planes[r, g * PLANE : g * PLANE + PLANE]
        for b in range(PLANE):
            byte = np.int32(row[b])
            acc += (byte & 3) * q[base + b] + ((byte >> 2) & 3) * q[base + PLANE + b]
            acc += ((byte >> 4) & 3) * q[base + 2 * PLANE + b] + ((byte >> 6) & 3) * q[base + 3 * PLANE + b]
        total += scales[r, g] * _f32(acc - gsum[g])
    return total


@numba.njit(parallel=True, fastmath=FAST, cache=True)
def gemv(values, planes, scales, bias, q, gsum, out, accumulate):
    act_scale = quantize_into(values, q, gsum)
    with_bias = bias.shape[0] > 0
    for r in numba.prange(planes.shape[0]):
        result = row_dot(planes, scales, r, q, gsum) * act_scale
        if with_bias:
            result += bias[r]
        if accumulate:
            out[r] += result
        else:
            out[r] = result


@numba.njit(parallel=True, fastmath=FAST, cache=True)
def normed_gemv(x, weight, eps, clip, work, planes, scales, bias, q, gsum, out):
    act_scale = norm_quantize(x, weight, eps, clip, work, q, gsum)
    with_bias = bias.shape[0] > 0
    for r in numba.prange(planes.shape[0]):
        result = row_dot(planes, scales, r, q, gsum) * act_scale
        if with_bias:
            result += bias[r]
        out[r] = result


@numba.njit(parallel=True, fastmath=FAST, cache=True)
def normed_gemv_silu(x, weight, eps, clip, work, planes, scales, bias, q, gsum, out):
    act_scale = norm_quantize(x, weight, eps, clip, work, q, gsum)
    with_bias = bias.shape[0] > 0
    dense = out.shape[0]
    for r in numba.prange(dense):
        gate = row_dot(planes, scales, r, q, gsum) * act_scale
        up = row_dot(planes, scales, dense + r, q, gsum) * act_scale
        if with_bias:
            gate += bias[r]
            up += bias[dense + r]
        out[r] = gate / (_f32(1.0) + np.exp(-gate)) * up


@numba.njit(cache=True)
def embed_row(planes, scales, token, out):
    n = out.shape[0]
    for i in range(n):
        byte = np.int32(planes[token, (i >> 7) * PLANE + (i & 31)])
        code = (byte >> (2 * ((i >> 5) & 3))) & 3
        out[i] = _f32(code - 1) * scales[token, i >> 7]


@numba.njit(parallel=True, cache=True)
def recurrent_step(
    mixer_in, hidden, heads, state_dim, head_dim, conv_taps, conv_bias, conv_state, decay, theta, skip,
    ssm_state, prev_force, half_dt_min, half_dt_range, values, out,
):
    taps = conv_taps.shape[1]
    bc_base = 2 * hidden
    for h in numba.prange(heads):
        first = h * head_dim
        for c in range(first, first + head_dim):
            convolved = conv_bias[c]
            for t in range(taps - 1):
                convolved += conv_state[c, t] * conv_taps[c, t]
            fresh = mixer_in[c]
            convolved += fresh * conv_taps[c, taps - 1]
            for t in range(taps - 2):
                conv_state[c, t] = conv_state[c, t + 1]
            if taps > 1:
                conv_state[c, taps - 2] = fresh
            values[c] = convolved / (_f32(1.0) + np.exp(-convolved))
            out[c] = _f32(0.0)
        row = bc_base + h * 3 * state_dim
        for s in range(state_dim):
            b_t = np.tanh(mixer_in[row + s])
            c_t = np.tanh(mixer_in[row + state_dim + s])
            gate_dt = mixer_in[row + 2 * state_dim + s]
            half_dt = half_dt_min + half_dt_range / (_f32(1.0) + np.exp(-gate_dt))
            magnitude = np.exp(_f32(2.0) * half_dt * decay[h, s])
            angle = _f32(2.0) * half_dt * theta[h, s]
            alpha_re = magnitude * np.cos(angle)
            alpha_im = magnitude * np.sin(angle)
            for d in range(head_dim):
                force = b_t * values[first + d]
                previous = prev_force[h, s, d]
                drive_re = (force + alpha_re * previous) * half_dt
                drive_im = alpha_im * previous * half_dt
                state_re = ssm_state[h, s, d, 0]
                state_im = ssm_state[h, s, d, 1]
                new_re = state_re * alpha_re - state_im * alpha_im + drive_re
                ssm_state[h, s, d, 0] = new_re
                ssm_state[h, s, d, 1] = state_re * alpha_im + state_im * alpha_re + drive_im
                prev_force[h, s, d] = force
                out[first + d] += c_t * new_re
        for d in range(head_dim):
            index = first + d
            gate = mixer_in[hidden + index]
            out[index] = (out[index] + skip[h, d] * values[index]) / (_f32(1.0) + np.exp(-gate))


@numba.njit(parallel=True, cache=True)
def attention_step(
    mixer_in, kv, rope, position, k_cache, v_cache, out, heads, head_dim, nope_dim, rope_dim, latent_dim, scale,
):
    hidden = heads * head_dim
    half = head_dim // 2
    for h in numba.prange(heads):
        rotated_q = np.empty(head_dim, dtype=np.float32)
        key = np.empty(head_dim, dtype=np.float32)
        for d in range(nope_dim):
            key[d] = kv[h * nope_dim + d]
        for d in range(rope_dim):
            key[nope_dim + d] = mixer_in[hidden + latent_dim + h * rope_dim + d]
        for pair in range(half):
            cos = rope[position, pair, 0]
            sin = rope[position, pair, 1]
            q_re = mixer_in[h * head_dim + 2 * pair]
            q_im = mixer_in[h * head_dim + 2 * pair + 1]
            rotated_q[2 * pair] = (q_re * cos - q_im * sin) * scale
            rotated_q[2 * pair + 1] = (q_re * sin + q_im * cos) * scale
            k_re = key[2 * pair]
            k_im = key[2 * pair + 1]
            k_cache[h, position, 2 * pair] = k_re * cos - k_im * sin
            k_cache[h, position, 2 * pair + 1] = k_re * sin + k_im * cos
        for d in range(head_dim):
            v_cache[h, position, d] = kv[heads * nope_dim + h * head_dim + d]
        scores = np.empty(position + 1, dtype=np.float32)
        peak = _f32(-np.inf)
        for t in range(position + 1):
            total = _f32(0.0)
            for d in range(head_dim):
                total += rotated_q[d] * k_cache[h, t, d]
            scores[t] = total
            if total > peak:
                peak = total
        norm = _f32(0.0)
        for t in range(position + 1):
            weight = np.exp(scores[t] - peak)
            scores[t] = weight
            norm += weight
        for d in range(head_dim):
            out[h * head_dim + d] = _f32(0.0)
        for t in range(position + 1):
            weight = scores[t] / norm
            for d in range(head_dim):
                out[h * head_dim + d] += weight * v_cache[h, t, d]


@numba.njit(cache=True)
def sample_token(logits, seen, temperature, penalty, inverse_penalty, top_k, heap_values, heap_indices):
    count = min(top_k, heap_values.shape[0])
    size = 0
    for i in range(logits.shape[0]):
        score = logits[i] / temperature
        if seen[i]:
            score = score * inverse_penalty if score > _f32(0.0) else score * penalty
        if size < count:
            j = size
            size += 1
            while j > 0:
                parent = (j - 1) >> 1
                if heap_values[parent] <= score:
                    break
                heap_values[j] = heap_values[parent]
                heap_indices[j] = heap_indices[parent]
                j = parent
            heap_values[j] = score
            heap_indices[j] = i
        elif score > heap_values[0]:
            j = 0
            while True:
                child = 2 * j + 1
                if child >= size:
                    break
                if child + 1 < size and heap_values[child + 1] < heap_values[child]:
                    child += 1
                if heap_values[child] >= score:
                    break
                heap_values[j] = heap_values[child]
                heap_indices[j] = heap_indices[child]
                j = child
            heap_values[j] = score
            heap_indices[j] = i
    peak = heap_values[0]
    for j in range(size):
        if heap_values[j] > peak:
            peak = heap_values[j]
    total = _f32(0.0)
    for j in range(size):
        total += np.exp(heap_values[j] - peak)
    target = _f32(np.random.random()) * total
    chosen = heap_indices[size - 1]
    running = _f32(0.0)
    for j in range(size):
        running += np.exp(heap_values[j] - peak)
        if running > target:
            chosen = heap_indices[j]
            break
    seen[chosen] = True
    return chosen


def planes_of(projections: list[TernaryLinear]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack projections and repack their 2-bit codes so each byte holds four inputs 32 apart."""

    in_features = int(projections[0].in_features)
    groups = (in_features + GROUP - 1) // GROUP
    codes = []
    scales = []
    biases = []
    for projection in projections:
        packed = projection.packed_weight.cpu()
        unpacked = torch.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)], dim=-1).reshape(packed.shape[0], -1)
        padded = torch.ones(packed.shape[0], groups * GROUP, dtype=torch.uint8)
        padded[:, :in_features] = unpacked[:, :in_features]
        codes.append(padded)
        scales.append(projection.weight_scales.cpu().float())
        if projection.bias is not None:
            biases.append(projection.bias.cpu().float())
    stacked = torch.cat(codes, dim=0).reshape(-1, groups, 4, PLANE)
    planes = stacked[:, :, 0] | stacked[:, :, 1] << 2 | stacked[:, :, 2] << 4 | stacked[:, :, 3] << 6
    bias = torch.cat(biases).numpy() if len(biases) == len(projections) else np.zeros(0, dtype=np.float32)
    return (
        np.ascontiguousarray(planes.reshape(-1, groups * PLANE).numpy()),
        np.ascontiguousarray(torch.cat(scales, dim=0).numpy()),
        np.ascontiguousarray(bias, dtype=np.float32),
    )


@dataclass
class _Matrix:
    planes: np.ndarray
    scales: np.ndarray
    bias: np.ndarray

    @classmethod
    def of(cls, *projections: TernaryLinear) -> _Matrix:
        return cls(*planes_of(list(projections)))

    @property
    def rows(self) -> int:
        return int(self.planes.shape[0])


def _array(tensor: Tensor) -> np.ndarray:
    return tensor.detach().numpy()


class CpuDecoder(StaticDecoder):
    """The torch decoder with its per-token step replaced by numba kernels over the packed 2-bit weights.

    Prompts still run through the torch prefill; each generated token then
    reads 93 MB of ternary weights instead of 700 MB of float16 ones.
    """

    def __init__(
        self,
        model: IvonarModel,
        max_len: int | None = None,
        sampling: bool = False,
        max_top_k: int = 256,
    ) -> None:
        super().__init__(model, max_len=max_len, device="cpu", sampling=sampling, max_top_k=max_top_k)
        if self.batch_size != 1:
            raise ValueError("the numba decoder runs one sequence")
        config = model.config
        self.hidden = int(config.hidden_dim)
        widths = [self.hidden, int(config.dense_dim)]
        self.layers: list[dict[str, object]] = []
        mixer_width = self.hidden
        kv_width = 1
        for index, block in enumerate(model.blocks):
            mixer = block.mixer
            layer: dict[str, object] = {
                "mixer_norm": _array(block.mixer_norm.weight.float()),
                "mixer_eps": np.float32(block.mixer_norm.eps),
                "mixer_clip": np.float32(block.mixer_norm.clip),
                "ffn_norm": _array(block.ffn_norm.weight.float()),
                "ffn_eps": np.float32(block.ffn_norm.eps),
                "ffn_clip": np.float32(block.ffn_norm.clip),
                "gate_up": _Matrix.of(block.ffn.gate_proj, block.ffn.up_proj),
                "down": _Matrix.of(block.ffn.down_proj),
            }
            if isinstance(mixer, RecurrentMixer):
                state = self.recurrent_states[index]
                layer.update(
                    kind="mamba",
                    entry=_Matrix.of(mixer.in_proj, mixer.bc_dt_proj),
                    out=_Matrix.of(mixer.out_proj),
                    heads=int(mixer.num_heads),
                    state_dim=int(mixer.state_dim),
                    head_dim=int(mixer.head_dim),
                    conv_taps=np.ascontiguousarray(_array(mixer.conv_weight.float()[:, 0, :])),
                    conv_bias=_array(mixer.conv_bias.float()),
                    conv_state=_array(state["conv_state"][0]),
                    decay=np.ascontiguousarray(_array(-torch.nn.functional.softplus(mixer.log_decay.float()))),
                    theta=np.ascontiguousarray(_array(mixer.theta.float())),
                    skip=np.ascontiguousarray(_array(mixer.skip.float()).reshape(mixer.num_heads, mixer.head_dim)),
                    ssm_state=_array(state["ssm_state"][0]),
                    prev_force=_array(state["prev_force"][0]),
                    half_dt_min=np.float32(0.5 * mixer.dt_min),
                    half_dt_range=np.float32(0.5 * (mixer.dt_max - mixer.dt_min)),
                )
                mixer_width = max(mixer_width, 2 * self.hidden + 3 * mixer.num_heads * mixer.state_dim)
            elif isinstance(mixer, LatentAttention):
                k_cache, v_cache = self.attention_caches[index]
                if k_cache.dtype != torch.float32:
                    raise ValueError("the numba decoder needs a float32 attention cache")
                rope = torch.view_as_real(self.rope_tables[index]).contiguous()
                layer.update(
                    kind="attention",
                    entry=_Matrix.of(mixer.q_proj, mixer.kv_down_proj, mixer.k_rope_proj),
                    up=_Matrix.of(mixer.k_nope_up_proj, mixer.v_up_proj),
                    out=_Matrix.of(mixer.out_proj),
                    heads=int(mixer.num_heads),
                    head_dim=int(mixer.head_dim),
                    nope_dim=int(mixer.nope_dim),
                    rope_dim=int(mixer.rope_dim),
                    latent_dim=int(mixer.latent_dim),
                    rope=_array(rope),
                    rope_tensor=rope,
                    k_cache=_array(k_cache[0]),
                    v_cache=_array(v_cache[0]),
                    scale=np.float32(self.attention_scales[index]),
                )
                mixer_width = max(mixer_width, self.hidden + mixer.latent_dim + mixer.num_heads * mixer.rope_dim)
                kv_width = max(kv_width, mixer.num_heads * mixer.nope_dim + self.hidden)
                widths.append(int(mixer.latent_dim))
            else:
                raise TypeError(f"unsupported mixer for the numba decoder: {type(mixer).__name__}")
            self.layers.append(layer)
        self.head = _Matrix.of(model.token_io.projection)
        self.final_norm = _array(model.final_norm.weight.float())
        self.final_eps = np.float32(model.final_norm.eps)
        self.final_clip = np.float32(model.final_norm.clip)
        padded = (max(widths) + GROUP - 1) // GROUP * GROUP
        self.q = np.zeros(padded, dtype=np.int32)
        self.gsum = np.zeros(padded // GROUP, dtype=np.int32)
        self.work = np.zeros(padded, dtype=np.float32)
        self.residual = np.zeros(self.hidden, dtype=np.float32)
        self.mixer_in = np.zeros(mixer_width, dtype=np.float32)
        self.mixer_out = np.zeros(self.hidden, dtype=np.float32)
        self.values = np.zeros(self.hidden, dtype=np.float32)
        self.kv = np.zeros(kv_width, dtype=np.float32)
        self.ffn_mid = np.zeros(int(config.dense_dim), dtype=np.float32)
        self.logits_np = _array(self.logits[0])
        self.token_np = _array(self.token)
        self.position_np = _array(self.position)
        self.seen_np = _array(self.seen[0])
        self.temperature_np = _array(self.temperature)
        self.penalty_np = _array(self.penalty)
        self.inverse_penalty_np = _array(self.inverse_penalty)
        self.top_k_np = _array(self.top_k)
        self.heap_values = np.zeros(self.max_top_k, dtype=np.float32)
        self.heap_indices = np.zeros(self.max_top_k, dtype=np.int64)
        set_threads(torch.get_num_threads())

    def _step(self, length: int) -> None:
        with _lock:
            self._numba_step()
            self.position_np[0] += 1
        if self.sampling:
            self._sample()

    def _sample(self) -> None:
        with _lock:
            self.token_np[0] = sample_token(
                self.logits_np, self.seen_np, self.temperature_np[0], self.penalty_np[0],
                self.inverse_penalty_np[0], int(self.top_k_np[0]), self.heap_values, self.heap_indices,
            )

    def _numba_step(self) -> None:
        position = int(self.position_np[0])
        residual, q, gsum, work = self.residual, self.q, self.gsum, self.work
        embed_row(self.head.planes, self.head.scales, int(self.token_np[0]), residual)
        for layer in self.layers:
            entry = layer["entry"]
            mixer_in = self.mixer_in[: entry.rows]
            normed_gemv(
                residual, layer["mixer_norm"], layer["mixer_eps"], layer["mixer_clip"], work,
                entry.planes, entry.scales, entry.bias, q, gsum, mixer_in,
            )
            if layer["kind"] == "mamba":
                recurrent_step(
                    mixer_in, self.hidden, layer["heads"], layer["state_dim"], layer["head_dim"],
                    layer["conv_taps"], layer["conv_bias"], layer["conv_state"], layer["decay"], layer["theta"],
                    layer["skip"], layer["ssm_state"], layer["prev_force"], layer["half_dt_min"],
                    layer["half_dt_range"], self.values, self.mixer_out,
                )
            else:
                up = layer["up"]
                latent = mixer_in[self.hidden : self.hidden + layer["latent_dim"]]
                gemv(latent, up.planes, up.scales, up.bias, q, gsum, self.kv[: up.rows], False)
                attention_step(
                    mixer_in, self.kv, layer["rope"], position, layer["k_cache"], layer["v_cache"], self.mixer_out,
                    layer["heads"], layer["head_dim"], layer["nope_dim"], layer["rope_dim"], layer["latent_dim"],
                    layer["scale"],
                )
            out = layer["out"]
            gemv(self.mixer_out, out.planes, out.scales, out.bias, q, gsum, residual, True)
            gate_up = layer["gate_up"]
            normed_gemv_silu(
                residual, layer["ffn_norm"], layer["ffn_eps"], layer["ffn_clip"], work,
                gate_up.planes, gate_up.scales, gate_up.bias, q, gsum, self.ffn_mid,
            )
            down = layer["down"]
            gemv(self.ffn_mid, down.planes, down.scales, down.bias, q, gsum, residual, True)
        head = self.head
        normed_gemv(
            residual, self.final_norm, self.final_eps, self.final_clip, work,
            head.planes, head.scales, head.bias, q, gsum, self.logits_np,
        )

    @torch.no_grad()
    def self_check(self, steps: int = 3) -> None:
        """Run the numba step and the torch step on one prompt; raises unless numba agrees and is faster.

        Where numba lacks a fast thread pool its step can lose to the torch
        one, so both are timed on this machine before the numba one is kept.
        """

        vocab = int(self.model.config.vocab_size)
        ids = torch.randint(2, vocab, (1, 6 + steps), generator=torch.Generator().manual_seed(0))
        torch_step = lambda length: StaticDecoder._step(self, length)
        results: dict[str, tuple[Tensor, float]] = {}
        sampling, self.sampling = self.sampling, False
        try:
            for name, step in (("numba", self._step), ("torch", torch_step)):
                self.prefill(ids[:, :5])
                self.token.copy_(ids[:, 5])
                step(self.max_len)
                logits = self.logits.clone()
                started = time.perf_counter()
                for index in range(steps):
                    self.token.copy_(ids[:, 6 + index])
                    step(self.max_len)
                results[name] = (logits, time.perf_counter() - started)
        finally:
            self.sampling = sampling
            self.reset()
        actual, numba_seconds = results["numba"]
        expected, torch_seconds = results["torch"]
        scale = float(expected.norm())
        if not bool(torch.isfinite(actual).all()) or scale == 0.0:
            raise RuntimeError("the numba decoder produced invalid numbers")
        difference = float((actual - expected).norm()) / scale
        if difference > 0.25:
            raise RuntimeError(f"the numba decoder disagrees with the torch decoder ({difference:.1e})")
        if numba_seconds >= torch_seconds:
            raise RuntimeError("the numba decoder is not faster than the torch decoder on this CPU")


def set_threads(count: int) -> None:
    numba.set_num_threads(max(1, min(int(count), numba.config.NUMBA_NUM_THREADS)))
