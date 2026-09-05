from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .attention import LatentAttention
from .config import ModelConfig
from .layers import GatedFeedForward, TokenHead, RMSNorm
from .recurrent import RecurrentMixer
from .model import IvonarBlock, IvonarModel
from .quantization import TernaryLinear

PACKED_FORMAT = "ivonar_packed_ternary_inference"
PACKED_SCHEMA_VERSION = 1


def resolve_device(requested: str = "auto") -> str:
    """Turn ``auto`` into the fastest device this machine can actually run."""

    if requested.strip().lower() != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class LoadedModel:
    model: IvonarModel
    config: ModelConfig
    path: Path
    lineup: str | None
    stage: str | None
    phase: str | None
    step: int
    tokenizer_sha256: str | None


def _validate_state(state: dict[str, Any]) -> None:
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"model_state entry is not a tensor: {key}")
        if key.endswith(".packed_weight"):
            if value.dtype != torch.uint8:
                raise RuntimeError(f"packed ternary buffer must be uint8: {key}")
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), 8 * 1024 * 1024):
                chunk = flat[start : start + 8 * 1024 * 1024]
                if bool((chunk & (chunk >> 1) & 0x55).any().item()):
                    raise RuntimeError(f"packed ternary buffer contains the reserved code 3: {key}")
        elif key.endswith(".weight_scales"):
            if value.dtype != torch.float16:
                raise RuntimeError(f"g128 scale buffer must be float16: {key}")
            if value.numel() and not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"g128 scale buffer must contain only finite values: {key}")
            if value.numel() and not bool((value > 0).all().item()):
                raise RuntimeError(f"g128 scale buffer must contain only positive values: {key}")


class _StateReader:
    """Hands out state tensors by key and reports what was never asked for."""

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self.state = state
        self.used: set[str] = set()

    def take(self, key: str) -> torch.Tensor:
        if key not in self.state:
            raise RuntimeError(f"model_state is missing {key}")
        self.used.add(key)
        return self.state[key]

    def take_optional(self, key: str) -> torch.Tensor | None:
        if key not in self.state:
            return None
        self.used.add(key)
        return self.state[key]

    def linear(self, prefix: str, in_features: int) -> TernaryLinear:
        return TernaryLinear(
            self.take(f"{prefix}.packed_weight"),
            self.take(f"{prefix}.weight_scales"),
            in_features,
            bias=self.take_optional(f"{prefix}.bias"),
        )

    def unused(self) -> list[str]:
        return sorted(set(self.state) - self.used)


def build_model(config: ModelConfig, state: dict[str, torch.Tensor]) -> IvonarModel:
    """Assemble the model on the CPU from a packed state dictionary."""

    _validate_state(state)
    reader = _StateReader(state)
    hidden = config.hidden_dim
    token_io = TokenHead(reader.linear("token_io.projection", hidden))
    if token_io.vocab_size != config.vocab_size:
        raise RuntimeError(f"token weight has {token_io.vocab_size} rows, config says {config.vocab_size}")
    blocks: list[IvonarBlock] = []
    for index, kind in enumerate(config.layer_types):
        prefix = f"blocks.{index}"
        if kind == "mamba":
            mixer: RecurrentMixer | LatentAttention = RecurrentMixer(
                reader.linear(f"{prefix}.mixer.in_proj", hidden),
                reader.linear(f"{prefix}.mixer.bc_dt_proj", hidden),
                reader.linear(f"{prefix}.mixer.out_proj", hidden),
                reader.take(f"{prefix}.mixer.log_decay"),
                reader.take(f"{prefix}.mixer.theta"),
                reader.take(f"{prefix}.mixer.skip"),
                reader.take(f"{prefix}.mixer.conv_weight"),
                reader.take(f"{prefix}.mixer.conv_bias"),
                num_heads=config.num_heads,
                state_dim=config.state_dim,
            )
        else:
            mixer = LatentAttention(
                reader.linear(f"{prefix}.mixer.q_proj", hidden),
                reader.linear(f"{prefix}.mixer.kv_down_proj", hidden),
                reader.linear(f"{prefix}.mixer.k_nope_up_proj", config.latent_dim),
                reader.linear(f"{prefix}.mixer.k_rope_proj", hidden),
                reader.linear(f"{prefix}.mixer.v_up_proj", config.latent_dim),
                reader.linear(f"{prefix}.mixer.out_proj", hidden),
                num_heads=config.num_heads,
                base_context=config.base_context,
                active_context=config.seq_len,
                rope_base=config.rope_base,
                scaling_strategy=config.longrope_scaling,
            )
        ffn = GatedFeedForward(
            reader.linear(f"{prefix}.ffn.gate_proj", hidden),
            reader.linear(f"{prefix}.ffn.up_proj", hidden),
            reader.linear(f"{prefix}.ffn.down_proj", config.dense_dim),
        )
        blocks.append(
            IvonarBlock(
                kind,
                RMSNorm(reader.take(f"{prefix}.mixer_norm.weight")),
                mixer,
                RMSNorm(reader.take(f"{prefix}.ffn_norm.weight")),
                ffn,
            )
        )
    final_norm = RMSNorm(reader.take("final_norm.weight"))
    unused = reader.unused()
    if unused:
        raise RuntimeError(f"model_state has entries the model does not use: {', '.join(unused[:5])}")
    return IvonarModel(config, token_io, blocks, final_norm)


def materialize_weights(model: IvonarModel, dtype: torch.dtype) -> None:
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            module.materialize(dtype)


def load_model(
    path: str | Path,
    device: str | torch.device = "cpu",
    expected_tokenizer_sha256: str | None = None,
    materialize: bool = True,
) -> LoadedModel:
    """Load an Ivonar model file and prepare it for decoding on ``device``.

    ``materialize`` unpacks the ternary weights into full-precision matrices
    for the torch code path; the ternary kernels read the packed weights and
    skip it.
    """

    model_file = Path(path)
    payload = torch.load(model_file, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid Ivonar checkpoint: {model_file}")
    if payload.get("format") != PACKED_FORMAT or payload.get("schema_version") != PACKED_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported checkpoint format; expected {PACKED_FORMAT} schema {PACKED_SCHEMA_VERSION}")
    config = ModelConfig.from_payload(payload.get("config"))
    tokenizer_sha256 = str(payload["tokenizer_sha256"]) if payload.get("tokenizer_sha256") else None
    if expected_tokenizer_sha256 is not None:
        if tokenizer_sha256 is None:
            raise RuntimeError("checkpoint is missing tokenizer_sha256")
        if tokenizer_sha256 != str(expected_tokenizer_sha256):
            raise RuntimeError(
                f"tokenizer SHA-256 mismatch: expected {expected_tokenizer_sha256}, found {tokenizer_sha256}"
            )
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint model_state is invalid")
    model = build_model(config, state)
    target = torch.device(device)
    model.to(target)
    model.eval()
    if materialize:
        materialize_weights(model, torch.float16 if target.type == "cuda" else torch.float32)
    return LoadedModel(
        model=model,
        config=config,
        path=model_file,
        lineup=str(payload["lineup"]) if payload.get("lineup") is not None else None,
        stage=str(payload["stage"]) if payload.get("stage") is not None else None,
        phase=str(payload["phase"]) if payload.get("phase") is not None else None,
        step=int(payload.get("step", 0) or 0),
        tokenizer_sha256=tokenizer_sha256,
    )
