from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ivonar_inference.decoder import StaticDecoder
from ivonar_inference.generation import stream_text
from ivonar_inference.tokenizer import TernaryTokenizer
from ivonar_inference.verify import check_decoder
from conftest import tiny_config, tiny_model, tiny_tokenizer_file


def _teacher_forced(model, decoder: StaticDecoder, ids: list[int], split: int) -> tuple[torch.Tensor, torch.Tensor]:
    reference = model(torch.tensor([ids], device=model.device))[0].float().cpu()[split - 1 : -1]
    predictions = [decoder.prefill(torch.tensor([ids[:split]], device=decoder.device))[0].float().cpu().clone()]
    for token_id in ids[split:-1]:
        predictions.append(decoder.step(torch.tensor([token_id], device=decoder.device))[0].float().cpu().clone())
    return torch.stack(predictions), reference


@torch.no_grad()
def test_fused_decoder_matches_the_one_pass_forward_on_cpu() -> None:
    torch.manual_seed(3)
    model = tiny_model(seed=3)
    assert [block.mixer_type for block in model.blocks] == ["mamba", "mla", "mamba"]
    ids = torch.randint(2, 300, (14,)).tolist()
    decoder = StaticDecoder(model, max_len=32, device="cpu", attention_lengths=(8, 16), fuse_projections=True)
    assert decoder.graph is None
    assert decoder.attention_length(0) == 8 and decoder.attention_length(8) == 16 and decoder.attention_length(31) == 32
    assert set(decoder.ffn_projections) == {0, 1, 2}
    assert set(decoder.recurrent_projections) == {0, 2}
    assert set(decoder.attention_in_projections) == {1}
    assert set(decoder.attention_up_projections) == {1}
    predicted, reference = _teacher_forced(model, decoder, ids, split=7)
    torch.testing.assert_close(predicted, reference, rtol=1e-3, atol=1e-3)
    assert int(decoder.position) == 14 - 1
    assert decoder.host_position == 14 - 1
    with pytest.raises(ValueError, match="no free position"):
        decoder.attention_length(32)

    decoder.reset()
    again, _ = _teacher_forced(model, decoder, ids, split=7)
    torch.testing.assert_close(again, predicted, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_cpu_decoder_keeps_the_module_projections() -> None:
    torch.manual_seed(4)
    model = tiny_model(seed=4)
    ids = torch.randint(2, 300, (9,)).tolist()
    decoder = StaticDecoder(model, max_len=32, device="cpu")
    assert decoder.ffn_projections == {} and decoder.recurrent_projections == {}
    predicted, reference = _teacher_forced(model, decoder, ids, split=5)
    torch.testing.assert_close(predicted, reference, rtol=1e-3, atol=1e-3)


@torch.no_grad()
def test_decoder_refuses_prompts_beyond_its_capacity() -> None:
    model = tiny_model(seed=5)
    decoder = StaticDecoder(model, max_len=8, device="cpu")
    with pytest.raises(ValueError, match="capacity"):
        decoder.prefill(torch.randint(2, 300, (1, 9)))
    with pytest.raises(ValueError, match="max_len"):
        StaticDecoder(model, max_len=64, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")
@torch.no_grad()
def test_graph_replay_matches_the_one_pass_forward() -> None:
    torch.manual_seed(6)
    model = tiny_model(seed=6, device="cuda")
    ids = torch.randint(2, 300, (11,)).tolist()
    decoder = StaticDecoder(model, max_len=32, device="cuda", attention_lengths=(8, 16))
    assert decoder.attention_lengths == (8, 16, 32)
    assert decoder.capture() is True
    assert set(decoder.graphs) == {8, 16, 32}
    assert decoder.graph is not None
    check = check_decoder(tiny_model(seed=6), decoder, ids, split=5)
    assert decoder.host_position == 11 - 1
    assert check.top1_agreement >= 0.9
    assert check.max_abs_diff < 0.1


@torch.no_grad()
def test_device_sampler_applies_top_k_and_the_repetition_penalty() -> None:
    torch.manual_seed(8)
    decoder = StaticDecoder(tiny_model(seed=8), max_len=32, device="cpu", sampling=True, max_top_k=8)
    decoder.configure_sampling(temperature=1.0, top_k=1, repetition_penalty=1.5)
    decoder.logits.zero_()
    decoder.logits[0, 5] = 6.0
    decoder.logits[0, 9] = 5.0
    assert decoder.sample() == 5
    assert bool(decoder.seen[0, 5]) and not bool(decoder.seen[0, 9])
    assert decoder.sample() == 9
    decoder.configure_sampling(temperature=1.0, top_k=1, repetition_penalty=1.0)
    assert decoder.sample() == 5
    with pytest.raises(RuntimeError, match="without sampling"):
        StaticDecoder(tiny_model(seed=8), max_len=32, device="cpu").sample()


@torch.no_grad()
def test_device_sampler_top_k_zero_covers_the_candidate_set() -> None:
    torch.manual_seed(9)
    decoder = StaticDecoder(tiny_model(seed=9), max_len=32, device="cpu", sampling=True, max_top_k=4)
    decoder.configure_sampling(temperature=0.5, top_k=0)
    assert int(decoder.top_k) == 4
    decoder.logits.fill_(-50.0)
    decoder.logits[0, :4] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    draws = {decoder.sample() for _ in range(64)}
    assert draws <= {0, 1, 2, 3}
    assert len(draws) > 1


def test_stream_text_decodes_through_the_device_sampler(tmp_path: Path) -> None:
    torch.manual_seed(10)
    tokenizer = TernaryTokenizer.load(tiny_tokenizer_file(tmp_path / "tokenizer.json"))
    model = tiny_model(tiny_config(len(tokenizer)), seed=10)
    decoder = StaticDecoder(model, max_len=32, device="cpu", sampling=True)
    text = "".join(stream_text(tokenizer, decoder, "hello", max_new_tokens=5, temperature=1.0, top_k=4))
    assert isinstance(text, str)
    produced = decoder.host_position - len(tokenizer.encode("hello"))
    assert 0 <= produced <= 4
    with pytest.raises(ValueError, match="max_new_tokens"):
        list(stream_text(tokenizer, decoder, "hello", max_new_tokens=0, temperature=1.0, top_k=4))


def test_stream_text_stops_at_a_stop_token(tmp_path: Path, monkeypatch) -> None:
    tokenizer = TernaryTokenizer.load(tiny_tokenizer_file(tmp_path / "tokenizer.json"))
    model = tiny_model(tiny_config(len(tokenizer)), seed=11)
    decoder = StaticDecoder(model, max_len=32, device="cpu", sampling=True)
    word_id = tokenizer.encode("w")[0]
    draws = iter([word_id, word_id, 3, word_id])
    monkeypatch.setattr(decoder, "sample", lambda: next(draws))
    monkeypatch.setattr(decoder, "advance", lambda: next(draws))
    text = "".join(
        stream_text(tokenizer, decoder, "hello", max_new_tokens=8, temperature=1.0, top_k=4, stop_token_ids=(3,))
    )
    assert text == "ww"


@torch.no_grad()
def test_configure_sampling_can_penalize_earlier_words() -> None:
    torch.manual_seed(12)
    decoder = StaticDecoder(tiny_model(seed=12), max_len=32, device="cpu", sampling=True, max_top_k=8)
    decoder.logits.zero_()
    decoder.logits[0, 5] = 6.0
    decoder.logits[0, 9] = 5.0
    decoder.configure_sampling(temperature=1.0, top_k=1, repetition_penalty=1.5)
    assert decoder.sample() == 5
    decoder.configure_sampling(temperature=1.0, top_k=1, repetition_penalty=1.5, penalized_ids=[5, -1, 99999])
    assert decoder.sample() == 9


def test_stream_text_penalizes_the_previous_answer(tmp_path: Path, monkeypatch) -> None:
    tokenizer = TernaryTokenizer.load(tiny_tokenizer_file(tmp_path / "tokenizer.json"))
    model = tiny_model(tiny_config(len(tokenizer)), seed=13)
    decoder = StaticDecoder(model, max_len=32, device="cpu", sampling=True)
    seen: dict[str, object] = {}
    original = decoder.configure_sampling
    monkeypatch.setattr(
        decoder,
        "configure_sampling",
        lambda *args, **kwargs: (seen.update(kwargs), original(*args, **kwargs))[1],
    )
    list(stream_text(tokenizer, decoder, "hi", max_new_tokens=2, temperature=1.0, top_k=4, penalized_text="abc"))
    assert list(seen["penalized_ids"]) == tokenizer.encode("abc")
