from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from ivonar_inference.config import ModelConfig
from ivonar_inference.engine import Engine, GenerationSettings, ModelInfo
from ivonar_inference.loader import build_model, materialize_weights
from ivonar_inference.model import IvonarModel
from ivonar_inference.tokenizer import TernaryTokenizer

ANSWER = "Paris is the capital of France and it is lovely in spring."

CONTRACT = {
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


def tiny_config(vocab_size: int = 300) -> ModelConfig:
    return ModelConfig.from_payload(
        {
            **CONTRACT,
            "name": "TINY",
            "vocab_size": vocab_size,
            "seq_len": 32,
            "hidden_dim": 16,
            "num_layers": 3,
            "num_heads": 4,
            "state_dim": 3,
            "layer_types": ["mamba", "mla", "mamba"],
            "dense_intermediate_dim": 32,
            "mla_latent_dim": 8,
            "base_context": 32,
            "final_context": 32,
            "num_experts": 0,
        }
    )


def _packed(generator: torch.Generator, out_features: int, in_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = math.ceil(in_features / 4)
    codes = torch.randint(0, 3, (out_features, width, 4), generator=generator, dtype=torch.uint8)
    packed = codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)
    groups = math.ceil(in_features / 128)
    scales = (torch.rand(out_features, groups, generator=generator) * 0.05 + 0.01).half()
    return packed.contiguous(), scales


def tiny_state(config: ModelConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    """A random but well-formed packed state for ``config``."""

    generator = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {}

    def linear(prefix: str, out_features: int, in_features: int, bias: bool = True) -> None:
        packed, scales = _packed(generator, out_features, in_features)
        state[f"{prefix}.packed_weight"] = packed
        state[f"{prefix}.weight_scales"] = scales
        if bias:
            state[f"{prefix}.bias"] = torch.randn(out_features, generator=generator) * 0.02

    hidden, heads, state_dim = config.hidden_dim, config.num_heads, config.state_dim
    head_dim = hidden // heads
    linear("token_io.projection", config.vocab_size, hidden, bias=False)
    for index, kind in enumerate(config.layer_types):
        prefix = f"blocks.{index}"
        state[f"{prefix}.mixer_norm.weight"] = 1.0 + 0.1 * torch.randn(hidden, generator=generator)
        state[f"{prefix}.ffn_norm.weight"] = 1.0 + 0.1 * torch.randn(hidden, generator=generator)
        if kind == "mamba":
            linear(f"{prefix}.mixer.in_proj", 2 * hidden, hidden)
            linear(f"{prefix}.mixer.bc_dt_proj", 3 * heads * state_dim, hidden)
            linear(f"{prefix}.mixer.out_proj", hidden, hidden)
            state[f"{prefix}.mixer.log_decay"] = torch.rand(heads, state_dim, generator=generator) * 0.5 - 1.0
            state[f"{prefix}.mixer.theta"] = torch.rand(heads, state_dim, generator=generator) * math.pi
            state[f"{prefix}.mixer.skip"] = 1.0 + 0.1 * torch.randn(heads, head_dim, generator=generator)
            conv = 0.1 * torch.randn(hidden, 1, 4, generator=generator)
            conv[:, :, -1] += 1.0
            state[f"{prefix}.mixer.conv_weight"] = conv
            state[f"{prefix}.mixer.conv_bias"] = 0.01 * torch.randn(hidden, generator=generator)
        else:
            rope_dim = max(2, (head_dim // 2) // 2 * 2)
            nope_dim = head_dim - rope_dim
            linear(f"{prefix}.mixer.q_proj", hidden, hidden)
            linear(f"{prefix}.mixer.kv_down_proj", config.latent_dim, hidden)
            linear(f"{prefix}.mixer.k_nope_up_proj", heads * nope_dim, config.latent_dim)
            linear(f"{prefix}.mixer.k_rope_proj", heads * rope_dim, hidden)
            linear(f"{prefix}.mixer.v_up_proj", hidden, config.latent_dim)
            linear(f"{prefix}.mixer.out_proj", hidden, hidden)
        linear(f"{prefix}.ffn.gate_proj", config.dense_dim, hidden)
        linear(f"{prefix}.ffn.up_proj", config.dense_dim, hidden)
        linear(f"{prefix}.ffn.down_proj", hidden, config.dense_dim)
    state["final_norm.weight"] = 1.0 + 0.1 * torch.randn(hidden, generator=generator)
    return state


def tiny_model(config: ModelConfig | None = None, seed: int = 0, device: str = "cpu") -> IvonarModel:
    config = config or tiny_config()
    model = build_model(config, tiny_state(config, seed)).to(device).eval()
    materialize_weights(model, torch.float16 if torch.device(device).type == "cuda" else torch.float32)
    return model


def tiny_tokenizer_file(path: Path) -> Path:
    """A byte-level tokenizer without merges, written in the python backend format."""

    tokenizer = TernaryTokenizer()
    payload = {
        "format": "ivonar_tokenizer_v1",
        "backend": "python_byte_bpe",
        "special_tokens": tokenizer.special_tokens,
        "vocab": {str(idx): list(value) for idx, value in sorted(tokenizer.vocab.items())},
        "merges": [],
        "next_id": tokenizer.next_id,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FakeTokenizer:
    special_tokens = {"<|im_end|>": 3}
    endoftext_id = 1

    def __len__(self) -> int:
        return 300

    def encode(self, text: str) -> list[int]:
        return [7] * len(text.split())

    def decode(self, ids: list[int]) -> str:
        return " ".join("x" for _ in ids)


def fake_stream(**kwargs) -> Iterator[str]:
    fake_stream.calls.append(kwargs)
    words = ANSWER.split()
    for index, word in enumerate(words[: kwargs["max_new_tokens"]]):
        yield word if index == 0 else " " + word


fake_stream.calls = []


@pytest.fixture
def engine() -> Engine:
    fake_stream.calls.clear()
    info = ModelInfo(
        model_id="ivonar-nano",
        path=Path("model.pt"),
        context_tokens=64,
        device="cpu",
        lineup="nano",
        stage="sft",
        phase="base",
        step=1,
    )
    return Engine(object(), FakeTokenizer(), info, GenerationSettings(max_tokens=32), stream_fn=fake_stream)
