from __future__ import annotations

import sys

import pytest
import torch

from ivonar_inference.decoder import StaticDecoder, build_decoder
from conftest import tiny_config, tiny_model

CpuDecoder = pytest.importorskip("ivonar_inference.cpu_decoder").CpuDecoder


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).norm() / expected.norm())


@torch.no_grad()
def test_the_numba_step_matches_the_torch_step() -> None:
    model = tiny_model(seed=7)
    decoder = CpuDecoder(model, max_len=32)
    reference = StaticDecoder(model, max_len=32, device="cpu")
    ids = torch.randint(2, 300, (1, 20), generator=torch.Generator().manual_seed(7))
    decoder.prefill(ids[:, :6])
    reference.prefill(ids[:, :6])
    agreements = 0
    for index in range(6, 20):
        actual = decoder.step(ids[:, index]).clone()
        expected = reference.step(ids[:, index]).clone()
        assert _relative(actual, expected) < 0.05
        agreements += int(actual.argmax() == expected.argmax())
        assert int(decoder.position) == int(reference.position) == index + 1
    assert agreements >= 12


@torch.no_grad()
def test_the_numba_sampler_follows_the_sampling_rule() -> None:
    model = tiny_model(seed=8)
    decoder = CpuDecoder(model, max_len=32, sampling=True)
    reference = StaticDecoder(model, max_len=32, device="cpu", sampling=True)
    logits = torch.randn(1, 300, generator=torch.Generator().manual_seed(8))
    for penalized in ((), (int(logits.argmax()),)):
        for current in (decoder, reference):
            current.configure_sampling(temperature=0.7, top_k=1, repetition_penalty=3.0, penalized_ids=penalized)
            current.logits.copy_(logits)
        assert decoder.sample() == reference.sample()
    decoder.configure_sampling(temperature=1.0, top_k=5, repetition_penalty=1.0)
    allowed = set(torch.topk(logits[0], 5).indices.tolist())
    for _ in range(50):
        decoder.logits.copy_(logits)
        assert decoder.sample() in allowed
    assert bool(decoder.seen[0, list(allowed)].any())


@torch.no_grad()
def test_generation_on_the_numba_decoder_is_reproducible_for_greedy_sampling() -> None:
    model = tiny_model(seed=9)
    tokens = []
    for kind in (CpuDecoder, StaticDecoder):
        decoder = kind(model, max_len=32, sampling=True) if kind is CpuDecoder else kind(model, max_len=32, device="cpu", sampling=True)
        decoder.prefill(torch.tensor([[3, 4, 5]]))
        decoder.configure_sampling(temperature=1.0, top_k=1, repetition_penalty=4.0)
        tokens.append([decoder.sample(), *decoder.advance_many(6)])
    matches = sum(a == b for a, b in zip(*tokens))
    assert matches >= 5


def test_build_decoder_keeps_the_numba_decoder_when_it_checks_out(monkeypatch) -> None:
    monkeypatch.setattr(CpuDecoder, "self_check", lambda self, steps=3: None)
    decoder, backend, note = build_decoder(tiny_model(seed=1), "cpu", sampling=True, kernels=True)
    assert type(decoder) is CpuDecoder
    assert backend == "ternary"
    assert note is None


def test_build_decoder_uses_the_torch_decoder_when_numba_loses(monkeypatch) -> None:
    def slower(self, steps=3):
        raise RuntimeError("the numba decoder is not faster than the torch decoder on this CPU")

    monkeypatch.setattr(CpuDecoder, "self_check", slower)
    decoder, backend, note = build_decoder(tiny_model(seed=1), "cpu", sampling=True, kernels=True)
    assert type(decoder) is StaticDecoder
    assert backend == "torch"
    assert "not faster" in note


def test_build_decoder_works_without_numba(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.delitem(sys.modules, "ivonar_inference.cpu_decoder")
    decoder, backend, note = build_decoder(tiny_model(seed=1), "cpu", sampling=True, kernels=True)
    assert type(decoder) is StaticDecoder
    assert backend == "torch"
    assert "numba" in note


def test_self_check_passes_on_a_tiny_model() -> None:
    decoder = CpuDecoder(tiny_model(seed=10, config=tiny_config()), max_len=32)
    try:
        decoder.self_check()
    except RuntimeError as exc:
        assert "not faster" in str(exc)
    assert int(decoder.position) == 0
