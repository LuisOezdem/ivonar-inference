from __future__ import annotations

from collections.abc import Sequence
from importlib import resources

import torch
import torch.nn.functional as F
from torch import Tensor

from .attention import LatentAttention
from .decoder import StaticDecoder
from .kernels import runtime
from .kernels.runtime import KernelError, Launch, float32, int32, pointer
from .model import IvonarModel
from .quantization import TERNARY_GROUP_SIZE, TernaryLinear
from .recurrent import RecurrentMixer

CHUNK = 64
THREADS = 256
NARROW_WIDTH = 4 * THREADS
ATTENTION_THREADS = 64
RECURRENT_MAX_THREADS = 1024
RECURRENT_MAX_STATE = 512
SAMPLE_THREADS = 1024
ROW_ALIGNMENT = 16
DEFAULT_SPLITS = 4
TARGET_BLOCKS = 48
TILE = 32
TILE_SPLITS = 1
TILE_STATE = 16
TILE_HEAD = 128
TILE_TAPS = 8


def kernel_source() -> str:
    return resources.files("ivonar_inference").joinpath("kernels/ternary.cu").read_text(encoding="utf-8")


def _lanes_for(chunks: int, rows: int, max_lanes: int) -> int:
    lanes = 4
    while lanes < max_lanes and (chunks > 4 * lanes or rows * lanes < TARGET_BLOCKS * THREADS):
        lanes *= 2
    return lanes


def _stack(tensors: Sequence[Tensor], interleave: bool) -> Tensor:
    if len(tensors) == 1:
        return tensors[0]
    if interleave:
        return torch.stack(tensors, dim=1).reshape(-1, *tensors[0].shape[1:])
    return torch.cat(tensors, dim=0)


class Arena:
    __slots__ = ("buffer", "used")

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.buffer = torch.empty(max(int(capacity), 0), dtype=torch.uint8, device=device)
        self.used = 0

    @property
    def base(self) -> int:
        return self.buffer.data_ptr()

    def place(self, tensor: Tensor) -> Tensor:
        source = tensor.contiguous()
        nbytes = source.numel() * source.element_size()
        start = (self.used + 255) // 256 * 256
        if start + nbytes > self.buffer.numel():
            return source
        self.used = start + nbytes
        slot = self.buffer[start : start + nbytes].view(source.dtype).view(source.shape)
        slot.copy_(source)
        return slot


class Matrix:
    __slots__ = ("packed", "scales", "bias", "rows", "in_features", "row_bytes", "groups", "chunks", "lanes")

    def __init__(
        self,
        projections: Sequence[TernaryLinear],
        device: torch.device,
        interleave: bool = False,
        max_lanes: int = 32,
        arena: Arena | None = None,
    ) -> None:
        first = projections[0]
        in_features = int(first.in_features)
        rows = int(first.out_features)
        for projection in projections:
            if int(projection.in_features) != in_features:
                raise KernelError("fused projections must share their input width")
            if interleave and int(projection.out_features) != rows:
                raise KernelError("interleaved projections must share their output width")
            if int(projection.group_size) != TERNARY_GROUP_SIZE:
                raise KernelError(f"ternary kernels need group size {TERNARY_GROUP_SIZE}")
        packed = _stack([projection.packed_weight for projection in projections], interleave)
        row_bytes = int(packed.shape[1])
        padded = (row_bytes + ROW_ALIGNMENT - 1) // ROW_ALIGNMENT * ROW_ALIGNMENT
        if padded != row_bytes:
            packed = F.pad(packed, (0, padded - row_bytes))
        self.packed = packed.contiguous().to(device)
        scales = _stack([projection.weight_scales for projection in projections], interleave)
        if scales.dtype != torch.float16:
            raise KernelError("ternary kernels need float16 group scales")
        self.scales = scales.contiguous().to(device)
        biases = [projection.bias for projection in projections]
        if all(bias is None for bias in biases):
            self.bias: Tensor | None = None
        elif any(bias is None for bias in biases):
            raise KernelError("fused projections must all carry a bias or none")
        else:
            self.bias = _stack([bias.float() for bias in biases], interleave).contiguous().to(device)
        if arena is not None:
            self.packed = arena.place(self.packed)
            self.scales = arena.place(self.scales)
            if self.bias is not None:
                self.bias = arena.place(self.bias)
        self.rows = int(self.packed.shape[0])
        self.in_features = in_features
        self.row_bytes = padded
        self.groups = int(self.scales.shape[1])
        self.chunks = (in_features + CHUNK - 1) // CHUNK
        self.lanes = _lanes_for(self.chunks, self.rows, max_lanes)

    @property
    def grid(self) -> int:
        rows_per_block = THREADS // self.lanes
        return (self.rows + rows_per_block - 1) // rows_per_block

    @property
    def wide(self) -> bool:
        return self.in_features > NARROW_WIDTH


class KernelDecoder(StaticDecoder):

    def __init__(
        self,
        model: IvonarModel,
        max_len: int | None = None,
        device: str | torch.device | None = None,
        sampling: bool = False,
        max_top_k: int = 256,
        splits: int = DEFAULT_SPLITS,
    ) -> None:
        target = model.device if device is None else torch.device(device)
        if target.type != "cuda":
            raise KernelError("ternary kernels need a CUDA device")
        capacity = int(model.config.seq_len if max_len is None else max_len)
        super().__init__(
            model,
            max_len=capacity,
            batch_size=1,
            device=target,
            sampling=sampling,
            max_top_k=max_top_k,
            attention_lengths=(capacity,),
            fuse_projections=False,
        )
        if splits <= 0:
            raise ValueError("splits must be positive")
        self.splits = int(splits)
        self._launches: list[Launch] = []
        self._rope_tables: dict[int, Tensor] = {}
        self._build()

    def _build(self) -> None:
        model = self.model
        config = model.config
        device = self.device
        hidden = int(config.hidden_dim)
        heads = int(config.num_heads)
        head_dim = hidden // heads
        if head_dim % 2 != 0:
            raise KernelError("ternary kernels need an even head_dim")
        mixer_width = hidden
        kv_width = 0
        widths = [hidden, int(config.dense_dim)]
        for block in model.blocks:
            mixer = block.mixer
            if isinstance(mixer, RecurrentMixer):
                if mixer.state_dim * mixer.head_dim > RECURRENT_MAX_THREADS or mixer.state_dim > RECURRENT_MAX_STATE:
                    raise KernelError("ternary kernels need state_dim times head_dim of at most 1024")
                mixer_width = max(mixer_width, 2 * hidden + 3 * mixer.num_heads * mixer.state_dim)
            elif isinstance(mixer, LatentAttention):
                mixer_width = max(mixer_width, hidden + mixer.latent_dim + heads * mixer.rope_dim)
                kv_width = max(kv_width, heads * mixer.nope_dim + hidden)
                widths.append(int(mixer.latent_dim))
            else:
                raise KernelError(f"unsupported mixer for ternary kernels: {type(mixer).__name__}")
        max_chunks = (max(widths) + CHUNK - 1) // CHUNK
        self.residual = torch.zeros(hidden, dtype=torch.float32, device=device)
        self.mixer_in = torch.zeros(mixer_width, dtype=torch.float32, device=device)
        self.mixer_out = torch.zeros(hidden, dtype=torch.float32, device=device)
        self.kv = torch.zeros(max(kv_width, 1), dtype=torch.float32, device=device)
        self.ffn_mid = torch.zeros(int(config.dense_dim), dtype=torch.float32, device=device)
        self.partials = torch.full((heads, self.splits, head_dim + 2), float("-inf"), dtype=torch.float32, device=device)
        self.q8 = torch.zeros(max_chunks * CHUNK, dtype=torch.int8, device=device)
        self.q8sum = torch.zeros(max_chunks, dtype=torch.int32, device=device)
        self.q8scale = torch.zeros(1, dtype=torch.float32, device=device)
        self.uniform = torch.zeros(1, dtype=torch.float32, device=device)
        max_top_k = min(self.max_top_k, SAMPLE_THREADS)
        defines = {"HEAD_DIM": head_dim, "MAX_CHUNKS": max_chunks, "MAX_TOPK": max_top_k, "TILE": TILE}
        self._module = runtime.compile_module(kernel_source(), defines, device)
        self._sample_launch: Launch | None = None
        if self.sampling and self.max_top_k <= SAMPLE_THREADS:
            self._sample_launch = self._sampler(int(config.vocab_size))
        capacity = runtime.driver().persisting_capacity(int(device.index))
        self.arena = Arena(capacity, device) if capacity > 0 else None
        self._persisted_stream: int | None = None
        arena = self.arena
        self._token_matrix = Matrix([model.token_io.projection], device, arena=arena)
        self._layer_matrices: list[dict[str, Matrix]] = []
        self._recurrent_tensors: dict[int, tuple[Tensor, ...]] = {}
        self._launches.append(self._embed(self._token_matrix, hidden))
        for index, block in enumerate(model.blocks):
            mixer = block.mixer
            matrices: dict[str, Matrix] = {}
            if isinstance(mixer, RecurrentMixer):
                matrices["entry"] = Matrix([mixer.in_proj, mixer.bc_dt_proj], device, arena=arena)
                self._launches.append(
                    self._gemv("gemv_norm_store", matrices["entry"], self.residual, block.mixer_norm.weight, self.mixer_in, eps=block.mixer_norm.eps)
                )
                self._launches.append(self._recurrent(index, mixer, hidden))
                matrices["out"] = Matrix([mixer.out_proj], device, arena=arena)
                self._launches.append(self._gemv("gemv_plain_residual", matrices["out"], self.mixer_out, None, self.residual))
            else:
                matrices["entry"] = Matrix([mixer.q_proj, mixer.kv_down_proj, mixer.k_rope_proj], device, arena=arena)
                self._launches.append(
                    self._gemv("gemv_norm_store", matrices["entry"], self.residual, block.mixer_norm.weight, self.mixer_in, eps=block.mixer_norm.eps)
                )
                latent = self.mixer_in.narrow(0, hidden, mixer.latent_dim)
                matrices["up"] = Matrix([mixer.k_nope_up_proj, mixer.v_up_proj], device, arena=arena)
                self._launches.append(self._gemv("gemv_plain_store", matrices["up"], latent, None, self.kv))
                self._launches.append(self._attention(index, mixer, hidden, heads))
                matrices["out"] = Matrix([mixer.out_proj], device, arena=arena)
                self._launches.append(
                    self._gemv("gemv_attn_residual", matrices["out"], self.partials, None, self.residual, splits=self.splits)
                )
            matrices["gate_up"] = Matrix([block.ffn.gate_proj, block.ffn.up_proj], device, interleave=True, max_lanes=16, arena=arena)
            self._launches.append(
                self._gemv("gemv_norm_silu_pair", matrices["gate_up"], self.residual, block.ffn_norm.weight, self.ffn_mid, eps=block.ffn_norm.eps)
            )
            matrices["down"] = Matrix([block.ffn.down_proj], device, arena=arena)
            self._launches.append(self._gemv("gemv_plain_residual", matrices["down"], self.ffn_mid, None, self.residual))
            self._layer_matrices.append(matrices)
        self._head_launches = [
            self._quantize("quantize_norm", self.residual, model.final_norm.weight, hidden, eps=model.final_norm.eps),
            self._gemv("gemv_prequant_store", self._token_matrix, None, None, self.logits, prequant=True),
        ]
        self._launches.extend(self._head_launches)
        if arena is not None and arena.used > 0:
            runtime.driver().reserve_persisting(int(device.index), arena.used)
        self._mixer_width = mixer_width
        self._kv_width = max(kv_width, 1)
        self._max_chunks = max_chunks
        self._tile_launches: list[Launch] | None = None
        self._tile_graph: torch.cuda.CUDAGraph | None = None
        if self._tile_supported():
            self._build_tile(hidden, heads, head_dim)

    @property
    def persisted_bytes(self) -> int:
        return 0 if self.arena is None else self.arena.used

    @property
    def tile_prefill(self) -> bool:
        return self._tile_launches is not None

    def _tile_supported(self) -> bool:
        for block in self.model.blocks:
            mixer = block.mixer
            if isinstance(mixer, RecurrentMixer):
                if mixer.state_dim > TILE_STATE or mixer.head_dim > TILE_HEAD or mixer.conv_kernel_size > TILE_TAPS:
                    return False
            elif mixer.head_dim > TILE_HEAD:
                return False
        return True

    def _build_tile(self, hidden: int, heads: int, head_dim: int) -> None:
        model = self.model
        device = self.device
        config = model.config
        dense = int(config.dense_dim)
        self.tile_tokens = torch.zeros(TILE, dtype=torch.int64, device=device)
        self.tile_valid = torch.zeros(1, dtype=torch.int64, device=device)
        self.residual_tile = torch.zeros(TILE, hidden, dtype=torch.float32, device=device)
        self.mixer_in_tile = torch.zeros(TILE, self._mixer_width, dtype=torch.float32, device=device)
        self.mixer_out_tile = torch.zeros(TILE, hidden, dtype=torch.float32, device=device)
        self.kv_tile = torch.zeros(TILE, self._kv_width, dtype=torch.float32, device=device)
        self.ffn_tile = torch.zeros(TILE, dense, dtype=torch.float32, device=device)
        partial_stride = heads * TILE_SPLITS * (head_dim + 2)
        self.partials_tile = torch.full((TILE, partial_stride), float("-inf"), dtype=torch.float32, device=device)
        self.q8_tile = torch.zeros(TILE, self._max_chunks * CHUNK, dtype=torch.int8, device=device)
        self.xsum_tile = torch.zeros(TILE, self._max_chunks, dtype=torch.int32, device=device)
        self.scale_tile = torch.zeros(TILE, dtype=torch.float32, device=device)
        launches: list[Launch] = []
        token = self._token_matrix
        launches.append(
            Launch(
                self._module.kernel("embed_tile"),
                ((hidden + THREADS - 1) // THREADS, TILE, 1),
                (THREADS, 1, 1),
                [
                    pointer(self.tile_tokens), pointer(token.packed), pointer(token.scales), pointer(self.residual_tile),
                    int32(hidden), int32(token.row_bytes), int32(token.groups), int32(TILE),
                ],
            )
        )
        for index, block in enumerate(model.blocks):
            mixer = block.mixer
            matrices = self._layer_matrices[index]
            launches.append(
                self._quantize_tile("quantize_tile_norm", self.residual_tile, hidden, hidden, block.mixer_norm.weight, eps=block.mixer_norm.eps)
            )
            launches.append(self._gemm_tile("gemm_tile_store", matrices["entry"], self.mixer_in_tile, self._mixer_width))
            if isinstance(mixer, RecurrentMixer):
                launches.append(self._recurrent_tile(index, mixer, hidden))
                launches.append(self._quantize_tile("quantize_tile_plain", self.mixer_out_tile, hidden, hidden))
                launches.append(self._gemm_tile("gemm_tile_residual", matrices["out"], self.residual_tile, hidden))
            else:
                latent = self.mixer_in_tile.narrow(1, hidden, mixer.latent_dim)
                launches.append(self._quantize_tile("quantize_tile_plain", latent, mixer.latent_dim, self._mixer_width))
                launches.append(self._gemm_tile("gemm_tile_store", matrices["up"], self.kv_tile, self._kv_width))
                launches.extend(self._attention_tile(index, mixer, hidden, heads, partial_stride))
                launches.append(
                    self._quantize_tile("quantize_tile_attn", self.partials_tile, hidden, partial_stride, splits=TILE_SPLITS)
                )
                launches.append(self._gemm_tile("gemm_tile_residual", matrices["out"], self.residual_tile, hidden))
            launches.append(
                self._quantize_tile("quantize_tile_norm", self.residual_tile, hidden, hidden, block.ffn_norm.weight, eps=block.ffn_norm.eps)
            )
            launches.append(self._gemm_tile("gemm_tile_silu_pair", matrices["gate_up"], self.ffn_tile, dense))
            launches.append(self._quantize_tile("quantize_tile_plain", self.ffn_tile, dense, dense))
            launches.append(self._gemm_tile("gemm_tile_residual", matrices["down"], self.residual_tile, hidden))
        launches.append(
            Launch(
                self._module.kernel("gather_token"),
                ((hidden + THREADS - 1) // THREADS, 1, 1),
                (THREADS, 1, 1),
                [pointer(self.residual_tile), pointer(self.residual), int32(hidden), pointer(self.tile_valid)],
            )
        )
        launches.extend(self._head_launches)
        self._tile_launches = launches

    def _quantize_tile(
        self,
        name: str,
        source: Tensor,
        width: int,
        in_stride: int,
        aux: Tensor | None = None,
        eps: float = 1e-6,
        splits: int = 1,
    ) -> Launch:
        args = [
            pointer(source), int32(in_stride), pointer(aux), int32(width), float32(eps), int32(splits),
            pointer(self.q8_tile), pointer(self.xsum_tile), pointer(self.scale_tile),
        ]
        if width > NARROW_WIDTH:
            name += "_wide"
        return Launch(self._module.kernel(name), (TILE, 1, 1), (THREADS, 1, 1), args)

    def _gemm_tile(self, name: str, matrix: Matrix, output: Tensor, out_stride: int) -> Launch:
        args = [
            pointer(self.q8_tile), pointer(self.xsum_tile), pointer(self.scale_tile),
            pointer(matrix.packed), pointer(matrix.scales), pointer(matrix.bias), pointer(output),
            int32(matrix.rows), int32(matrix.in_features), int32(matrix.row_bytes), int32(matrix.groups),
            int32(matrix.lanes), int32(out_stride), int32(TILE),
        ]
        if matrix.wide:
            name += "_wide"
        return Launch(self._module.kernel(name), (matrix.grid, 1, 1), (THREADS, 1, 1), args)

    def _recurrent_tile(self, index: int, mixer: RecurrentMixer, hidden: int) -> Launch:
        state = self.recurrent_states[index]
        decay, theta, skip, conv_weight, conv_bias = self._recurrent_tensors[index]
        args = [
            pointer(self.mixer_in_tile), int32(self._mixer_width), pointer(conv_weight), pointer(conv_bias),
            pointer(state["conv_state"]), pointer(decay), pointer(theta), pointer(skip),
            pointer(state["ssm_state"]), pointer(state["prev_force"]), pointer(self.mixer_out_tile),
            int32(hidden), int32(mixer.state_dim), int32(mixer.head_dim), int32(mixer.conv_kernel_size),
            float32(0.5 * mixer.dt_min), float32(0.5 * (mixer.dt_max - mixer.dt_min)),
            int32(TILE), pointer(self.tile_valid),
        ]
        return Launch(
            self._module.kernel("recurrent_tile"),
            (mixer.num_heads, 1, 1),
            (mixer.state_dim * mixer.head_dim, 1, 1),
            args,
        )

    def _attention_tile(self, index: int, mixer: LatentAttention, hidden: int, heads: int, partial_stride: int) -> list[Launch]:
        k_cache, v_cache = self.attention_caches[index]
        rope = self._rope_tables[index]
        write = Launch(
            self._module.kernel("kv_write_tile"),
            (heads, TILE, 1),
            (ATTENTION_THREADS, 1, 1),
            [
                pointer(self.mixer_in_tile), int32(self._mixer_width), pointer(self.kv_tile), int32(self._kv_width),
                pointer(rope), pointer(self.position), pointer(k_cache), pointer(v_cache),
                int32(hidden + mixer.latent_dim), int32(mixer.nope_dim), int32(mixer.rope_dim), int32(heads * mixer.nope_dim),
                int32(self.max_len), int32(TILE), pointer(self.tile_valid),
            ],
        )
        attend = Launch(
            self._module.kernel("attention_tile"),
            (heads, TILE_SPLITS, TILE),
            (ATTENTION_THREADS, 1, 1),
            [
                pointer(self.mixer_in_tile), int32(self._mixer_width), pointer(rope), pointer(self.position),
                pointer(k_cache), pointer(v_cache), pointer(self.partials_tile), int32(partial_stride),
                int32(self.max_len), int32(TILE_SPLITS), float32(self.attention_scales[index]),
                int32(TILE), pointer(self.tile_valid),
            ],
        )
        return [write, attend]

    def _run_tile(self) -> None:
        if self._tile_graph is not None:
            self._tile_graph.replay()
            return
        runtime.driver().bind(int(self.device.index))
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if not torch.cuda.is_current_stream_capturing():
            self._persist(stream)
        for launch in self._tile_launches:
            launch(stream)
        self.position.add_(self.tile_valid)

    @torch.no_grad()
    def prefill(self, input_ids: Tensor) -> Tensor:
        if self._tile_launches is None:
            return super().prefill(input_ids)
        if input_ids.ndim != 2 or input_ids.shape[0] != self.batch_size:
            raise ValueError(f"input_ids must have shape [{self.batch_size}, seq_len]")
        length = int(input_ids.shape[1])
        if length == 0 or length > self.max_len:
            raise ValueError("prompt length must lie within the decoder capacity")
        ids = input_ids[0].to(device=self.device, dtype=torch.int64)
        for state in self.recurrent_states.values():
            for buffer in state.values():
                buffer.zero_()
        self.position.zero_()
        self.host_position = 0
        for start in range(0, length, TILE):
            chunk = ids[start : start + TILE]
            count = int(chunk.numel())
            self.tile_tokens.zero_()
            self.tile_tokens[:count].copy_(chunk)
            self.tile_valid.fill_(count)
            self._run_tile()
            self.host_position += count
        return self.logits

    @torch.no_grad()
    def capture(self, warmup_steps: int = 2) -> bool:
        captured = super().capture(warmup_steps)
        if not captured or self._tile_launches is None:
            return captured
        saved_position = int(self.position)
        saved_valid = int(self.tile_valid)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        self._prepare_stream(stream)
        try:
            self.tile_valid.fill_(TILE)
            with torch.cuda.stream(stream):
                for _ in range(max(1, warmup_steps)):
                    self.position.fill_(0)
                    self._run_tile()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            self.position.fill_(0)
            with torch.cuda.graph(graph, stream=stream):
                self._run_tile()
            torch.cuda.synchronize(self.device)
            self._tile_graph = graph
        except Exception:
            self._tile_graph = None
        self.position.fill_(saved_position)
        self.tile_valid.fill_(saved_valid)
        return captured

    def _persist(self, stream: int) -> None:
        if self.arena is None or self.arena.used == 0 or self._persisted_stream == stream:
            return
        runtime.driver().persist_on_stream(stream, self.arena.base, self.arena.used)
        self._persisted_stream = stream

    def _prepare_stream(self, stream: torch.cuda.Stream) -> None:
        runtime.driver().bind(int(self.device.index))
        self._persist(stream.cuda_stream)

    def _embed(self, matrix: Matrix, hidden: int) -> Launch:
        args = [
            pointer(self.token), pointer(matrix.packed), pointer(matrix.scales), pointer(self.residual),
            int32(hidden), int32(matrix.row_bytes), int32(matrix.groups),
        ]
        return Launch(self._module.kernel("embed_token"), ((hidden + THREADS - 1) // THREADS, 1, 1), (THREADS, 1, 1), args)

    def _gemv(
        self,
        name: str,
        matrix: Matrix,
        source: Tensor | None,
        aux: Tensor | None,
        output: Tensor,
        eps: float = 1e-6,
        splits: int = 1,
        prequant: bool = False,
    ) -> Launch:
        args = [
            pointer(source), pointer(aux),
            pointer(self.q8 if prequant else None), pointer(self.q8sum if prequant else None), pointer(self.q8scale if prequant else None),
            pointer(matrix.packed), pointer(matrix.scales), pointer(matrix.bias), pointer(output),
            int32(matrix.rows), int32(matrix.in_features), int32(matrix.row_bytes), int32(matrix.groups),
            int32(matrix.lanes), int32(splits), float32(eps),
        ]
        if matrix.wide and not prequant:
            name += "_wide"
        return Launch(self._module.kernel(name), (matrix.grid, 1, 1), (THREADS, 1, 1), args)

    def _quantize(self, name: str, source: Tensor, aux: Tensor | None, width: int, eps: float = 1e-6, splits: int = 1) -> Launch:
        args = [
            pointer(source), pointer(aux), int32(width), float32(eps), int32(splits),
            pointer(self.q8), pointer(self.q8sum), pointer(self.q8scale),
        ]
        if width > NARROW_WIDTH:
            name += "_wide"
        return Launch(self._module.kernel(name), (1, 1, 1), (THREADS, 1, 1), args)

    def _sampler(self, vocab: int) -> Launch:
        args = [
            pointer(self.logits), pointer(self.temperature), pointer(self.penalty), pointer(self.inverse_penalty),
            pointer(self.seen), pointer(self.top_k), pointer(self.uniform), pointer(self.token), int32(vocab),
        ]
        return Launch(self._module.kernel("sample_token"), (1, 1, 1), (SAMPLE_THREADS, 1, 1), args)

    def _recurrent(self, index: int, mixer: RecurrentMixer, hidden: int) -> Launch:
        state = self.recurrent_states[index]
        device = self.device
        decay = (-F.softplus(mixer.log_decay.float())).contiguous().to(device)
        theta = mixer.theta.float().contiguous().to(device)
        skip = mixer.skip.float().contiguous().to(device)
        conv_weight = mixer.conv_weight.float().reshape(mixer.hidden_dim, mixer.conv_kernel_size).contiguous().to(device)
        conv_bias = mixer.conv_bias.float().contiguous().to(device)
        self._recurrent_tensors[index] = (decay, theta, skip, conv_weight, conv_bias)
        args = [
            pointer(self.mixer_in), pointer(conv_weight), pointer(conv_bias), pointer(state["conv_state"]),
            pointer(decay), pointer(theta), pointer(skip), pointer(state["ssm_state"]), pointer(state["prev_force"]),
            pointer(self.mixer_out), int32(hidden), int32(mixer.state_dim), int32(mixer.head_dim),
            int32(mixer.conv_kernel_size), float32(0.5 * mixer.dt_min), float32(0.5 * (mixer.dt_max - mixer.dt_min)),
        ]
        return Launch(
            self._module.kernel("recurrent_step"),
            (mixer.num_heads, 1, 1),
            (mixer.state_dim * mixer.head_dim, 1, 1),
            args,
        )

    def _attention(self, index: int, mixer: LatentAttention, hidden: int, heads: int) -> Launch:
        k_cache, v_cache = self.attention_caches[index]
        rope = torch.view_as_real(self.rope_tables[index]).contiguous()
        self._rope_tables[index] = rope
        args = [
            pointer(self.mixer_in), pointer(self.kv), pointer(rope), pointer(self.position),
            pointer(k_cache), pointer(v_cache), pointer(self.partials),
            int32(hidden + mixer.latent_dim), int32(mixer.nope_dim), int32(mixer.rope_dim), int32(heads * mixer.nope_dim),
            int32(self.max_len), int32(self.splits), float32(self.attention_scales[index]),
        ]
        return Launch(self._module.kernel("attention_step"), (heads, self.splits, 1), (ATTENTION_THREADS, 1, 1), args)

    def _sample(self) -> None:
        if self._sample_launch is None:
            super()._sample()
            return
        runtime.driver().bind(int(self.device.index))
        self.uniform.uniform_()
        self._sample_launch(torch.cuda.current_stream(self.device).cuda_stream)

    def _step(self, length: int) -> None:
        runtime.driver().bind(int(self.device.index))
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if not torch.cuda.is_current_stream_capturing():
            self._persist(stream)
        for launch in self._launches:
            launch(stream)
        self.position.add_(1)
        if self.sampling:
            self._sample()
