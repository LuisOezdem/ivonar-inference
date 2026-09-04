from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_REQUIRED_CONTRACT = {
    "architecture": "ivonar_native_ternary",
    "architecture_version": 1,
    "ternary_group_size": 128,
    "ternary_storage": "q2_g128",
    "activation_quantization": "int8_per_token",
    "output_head_type": "tied_ternary_embedding_head",
    "mla_rope_layout": "decoupled_direct_rope",
    "mamba_implementation": "ivonar_mamba3_conv_et_mimo",
    "mamba_variant": "MIMO",
    "tie_embeddings": True,
}


@dataclass(frozen=True)
class ModelConfig:
    """The part of a checkpoint's configuration that running the model needs."""

    name: str
    vocab_size: int
    seq_len: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    state_dim: int
    layer_types: tuple[str, ...]
    dense_dim: int
    latent_dim: int
    base_context: int
    final_context: int
    rope_base: float
    longrope_scaling: str
    pad_token_id: int
    endoftext_token_id: int

    @classmethod
    def from_payload(cls, config: Any) -> "ModelConfig":
        if not isinstance(config, dict):
            raise ValueError("checkpoint config is invalid")
        mismatches = [
            f"{key}={config.get(key)!r}" for key, expected in _REQUIRED_CONTRACT.items() if config.get(key) != expected
        ]
        if mismatches:
            raise ValueError("incompatible Ivonar checkpoint: " + ", ".join(mismatches))
        if int(config.get("num_experts", 0)) > 0:
            raise ValueError("mixture-of-experts checkpoints are not supported")
        hidden_dim = int(config["hidden_dim"])
        num_layers = int(config["num_layers"])
        layer_types = tuple(str(kind) for kind in config.get("layer_types") or ())
        if not layer_types:
            layer_types = tuple("mamba" for _ in range(num_layers))
        if len(layer_types) != num_layers or any(kind not in {"mamba", "mla"} for kind in layer_types):
            raise ValueError("layer_types must name mamba or mla for every layer")
        return cls(
            name=str(config.get("name", "")),
            vocab_size=int(config["vocab_size"]),
            seq_len=int(config["seq_len"]),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=int(config["num_heads"]),
            state_dim=int(config["state_dim"]),
            layer_types=layer_types,
            dense_dim=int(
                config.get("dense_intermediate_dim", 0) or max(int(config.get("intermediate_dim", 0)), 4 * hidden_dim)
            ),
            latent_dim=int(config.get("mla_latent_dim", 0) or max(128, hidden_dim // 4)),
            base_context=int(config.get("base_context", 8192)),
            final_context=int(config.get("final_context", 8192)),
            rope_base=float(config.get("rope_base", 10000.0)),
            longrope_scaling=str(config.get("longrope_scaling", "linear")),
            pad_token_id=int(config.get("pad_token_id", 0)),
            endoftext_token_id=int(config.get("endoftext_token_id", 1)),
        )
