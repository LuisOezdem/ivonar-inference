from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ivonar_inference.config import ModelConfig
from ivonar_inference.engine import Engine, GenerationSettings
from ivonar_inference.loader import PACKED_FORMAT, PACKED_SCHEMA_VERSION, build_model, load_model
from ivonar_inference.tokenizer import TernaryTokenizer, format_chat_messages, tokenizer_file_sha256
from conftest import CONTRACT, tiny_config, tiny_state, tiny_tokenizer_file


def test_config_rejects_other_architectures_and_experts() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        ModelConfig.from_payload({**CONTRACT, "architecture_version": 2, "vocab_size": 10, "seq_len": 4, "hidden_dim": 4, "num_layers": 1, "num_heads": 1, "state_dim": 1})
    with pytest.raises(ValueError, match="mixture-of-experts"):
        ModelConfig.from_payload({**CONTRACT, "num_experts": 4, "vocab_size": 10, "seq_len": 4, "hidden_dim": 4, "num_layers": 1, "num_heads": 1, "state_dim": 1})
    config = tiny_config()
    assert config.layer_types == ("mamba", "mla", "mamba")
    assert config.latent_dim == 8 and config.dense_dim == 32


def test_build_model_uses_every_state_entry() -> None:
    config = tiny_config()
    state = tiny_state(config)
    model = build_model(config, state)
    assert len(model.blocks) == 3
    extra = dict(state)
    extra["blocks.0.mixer.unknown"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="does not use"):
        build_model(config, extra)
    missing = dict(state)
    del missing["final_norm.weight"]
    with pytest.raises(RuntimeError, match="missing final_norm.weight"):
        build_model(config, missing)
    broken = dict(state)
    broken["token_io.projection.packed_weight"] = torch.full_like(state["token_io.projection.packed_weight"], 0xFF)
    with pytest.raises(RuntimeError, match="reserved code"):
        build_model(config, broken)


@torch.no_grad()
def test_forward_and_prefill_agree() -> None:
    torch.manual_seed(1)
    config = tiny_config()
    model = build_model(config, tiny_state(config, seed=1)).eval()
    ids = torch.randint(2, 300, (1, 10))
    logits = model(ids)
    assert logits.shape == (1, 10, 300)
    last, states = model.prefill(ids)
    torch.testing.assert_close(last, logits[:, -1])
    assert set(states[0]) == {"ssm_state", "prev_force", "conv_state"}
    assert set(states[1]) == {"latent", "k_rope"}
    with pytest.raises(ValueError, match="context"):
        model(torch.randint(2, 300, (1, 33)))


def _write_checkpoint(path: Path, config: ModelConfig, tokenizer_sha256: str | None, **extra) -> Path:
    payload = {
        "format": PACKED_FORMAT,
        "schema_version": PACKED_SCHEMA_VERSION,
        "config": {**CONTRACT, "name": config.name, "vocab_size": config.vocab_size, "seq_len": config.seq_len, "hidden_dim": config.hidden_dim, "num_layers": config.num_layers, "num_heads": config.num_heads, "state_dim": config.state_dim, "layer_types": list(config.layer_types), "dense_intermediate_dim": config.dense_dim, "mla_latent_dim": config.latent_dim, "base_context": config.base_context, "final_context": config.final_context},
        "model_state": tiny_state(config, seed=2),
        "lineup": "nano",
        "stage": "sft",
        "phase": "base",
        "step": 7,
        "tokenizer_sha256": tokenizer_sha256,
        **extra,
    }
    torch.save(payload, path)
    return path


def test_load_model_checks_format_and_tokenizer(tmp_path: Path) -> None:
    tokenizer_file = tiny_tokenizer_file(tmp_path / "tokenizer.json")
    sha = tokenizer_file_sha256(tokenizer_file)
    config = tiny_config(vocab_size=266)
    model_file = _write_checkpoint(tmp_path / "packed_inference_checkpoint.pt", config, sha)
    loaded = load_model(model_file, device="cpu", expected_tokenizer_sha256=sha)
    assert loaded.lineup == "nano" and loaded.stage == "sft" and loaded.step == 7
    assert loaded.model.token_io.projection.cached_runtime_weight("cpu").dtype == torch.float32
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_model(model_file, device="cpu", expected_tokenizer_sha256="0" * 64)
    _write_checkpoint(tmp_path / "other.pt", config, sha, schema_version=2)
    with pytest.raises(RuntimeError, match="format"):
        load_model(tmp_path / "other.pt")

    engine = Engine.load(model_file, device="cpu")
    assert engine.decoder is not None and engine.decoder.sampling
    assert engine.info.context_tokens == 32 and engine.info.graph is False
    result = engine.complete([{"role": "user", "content": "hi"}], GenerationSettings(max_tokens=8))
    assert isinstance(result.text, str) and result.finish_reason in {"stop", "length"}
    assert engine.decoder.host_position <= engine.count_tokens(engine.build_prompt([{"role": "user", "content": "hi"}], GenerationSettings(max_tokens=8))[0]) + 8


def test_tokenizer_python_backend_round_trips(tmp_path: Path) -> None:
    tokenizer = TernaryTokenizer.load(tiny_tokenizer_file(tmp_path / "tokenizer.json"))
    assert len(tokenizer) == 266
    text = "Hello, world! <|im_start|>x<|im_end|> ümlaut"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert ids[ids.index(tokenizer.special_tokens["<|im_start|>"])] == 2
    assert tokenizer.endoftext_id == 1 and tokenizer.pad_id == 0


def test_format_chat_messages_matches_the_training_layout() -> None:
    prompt = format_chat_messages(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi <|assistant|> there"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Bye"},
        ]
    )
    assert prompt == (
        "<|im_start|><|system|>Be brief.<|im_end|>"
        "<|im_start|><|user|>Hi [assistant] there<|im_end|>"
        "<|im_start|><|assistant|>Hello.<|im_end|>"
        "<|im_start|><|user|>Bye<|im_end|>"
        "<|im_start|><|assistant|>"
    )
    with pytest.raises(ValueError, match="last chat message"):
        format_chat_messages([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}])
    with pytest.raises(ValueError, match="system message must come first"):
        format_chat_messages([{"role": "user", "content": "Hi"}, {"role": "system", "content": "x"}])
