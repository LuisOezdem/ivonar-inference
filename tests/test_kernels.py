from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ivonar_inference.bench import run_benchmark
from ivonar_inference.config import ModelConfig
from ivonar_inference.decoder import StaticDecoder, build_decoder
from ivonar_inference.engine import Engine, GenerationSettings
from ivonar_inference.generation import stream_text
from ivonar_inference.loader import build_model, materialize_weights
from ivonar_inference.tokenizer import TernaryTokenizer, tokenizer_file_sha256
from ivonar_inference.verify import check_decoder
from conftest import CONTRACT, tiny_config, tiny_model, tiny_state, tiny_tokenizer_file
from test_model import _write_checkpoint

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="ternary kernels need a GPU")


def _config(
    hidden: int, heads: int, state_dim: int, latent: int, dense: int, vocab_size: int = 300, seq_len: int = 32
) -> ModelConfig:
    return ModelConfig.from_payload(
        {
            **CONTRACT,
            "name": "TINY",
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "hidden_dim": hidden,
            "num_layers": 3,
            "num_heads": heads,
            "state_dim": state_dim,
            "layer_types": ["mamba", "mla", "mamba"],
            "dense_intermediate_dim": dense,
            "mla_latent_dim": latent,
            "base_context": seq_len,
            "final_context": seq_len,
            "num_experts": 0,
        }
    )


def _reference_and_served(config: ModelConfig, seed: int):
    state = tiny_state(config, seed=seed)
    reference = build_model(config, state).eval()
    materialize_weights(reference, torch.float32)
    served = build_model(config, state).to("cuda").eval()
    materialize_weights(served, torch.float16)
    return reference, served


@cuda
@pytest.mark.parametrize(
    "config", [_config(16, 4, 3, 8, 32), _config(64, 4, 5, 24, 96)], ids=["head_dim=4", "head_dim=16"]
)
def test_kernel_decoder_matches_the_one_pass_forward(config: ModelConfig) -> None:
    from ivonar_inference.kernel_decoder import KernelDecoder

    torch.manual_seed(3)
    reference, served = _reference_and_served(config, seed=3)
    ids = torch.randint(2, 300, (20,)).tolist()
    decoder = KernelDecoder(served, max_len=32, device="cuda")
    assert decoder.attention_lengths == (32,)
    assert decoder.graph is None
    eager = check_decoder(reference, decoder, ids, split=8)
    assert eager.max_abs_diff < 0.05
    assert abs(eager.decoder_log_prob - eager.reference_log_prob) < 0.05
    torch_check = check_decoder(reference, StaticDecoder(served, max_len=32, device="cuda"), ids, split=8)
    assert eager.mean_abs_diff <= 2 * torch_check.mean_abs_diff + 1e-3
    assert decoder.capture() is True
    assert set(decoder.graphs) == {32}
    replayed = check_decoder(reference, decoder, ids, split=8)
    assert replayed.decoder_log_prob == eager.decoder_log_prob
    decoder.reset()
    again = check_decoder(reference, decoder, ids, split=8)
    assert again.decoder_log_prob == replayed.decoder_log_prob


@cuda
@pytest.mark.parametrize("length", [1, 5, 32, 33, 70])
def test_tile_prefill_matches_the_one_pass_forward(length: int) -> None:
    from ivonar_inference.kernel_decoder import TILE, KernelDecoder

    torch.manual_seed(length)
    reference, served = _reference_and_served(_config(64, 4, 5, 24, 96, seq_len=96), seed=7)
    ids = torch.randint(2, 300, (length + 6,)).tolist()
    decoder = KernelDecoder(served, max_len=96, device="cuda")
    assert decoder.tile_prefill
    assert served.token_io.projection.cached_runtime_weight("cuda") is not None
    kernel = check_decoder(reference, decoder, ids, split=length)
    assert decoder.host_position == len(ids) - 1
    assert int(decoder.position) == len(ids) - 1
    torch_check = check_decoder(reference, StaticDecoder(served, max_len=96, device="cuda"), ids, split=length)
    assert kernel.max_abs_diff < 0.05
    assert kernel.mean_abs_diff <= 3 * torch_check.mean_abs_diff + 2e-3
    assert decoder.capture() is True
    assert decoder._tile_graph is not None
    replayed = check_decoder(reference, decoder, ids, split=length)
    assert replayed.decoder_log_prob == kernel.decoder_log_prob
    assert (length + TILE - 1) // TILE >= 1


@cuda
def test_kernel_decoder_refuses_prompts_beyond_its_capacity() -> None:
    from ivonar_inference.kernel_decoder import KernelDecoder

    _, served = _reference_and_served(_config(16, 4, 3, 8, 32), seed=5)
    decoder = KernelDecoder(served, max_len=8, device="cuda")
    with pytest.raises(ValueError, match="capacity"):
        decoder.prefill(torch.randint(2, 300, (1, 9), device="cuda"))


@cuda
def test_stream_text_runs_on_the_kernel_decoder(tmp_path: Path) -> None:
    from ivonar_inference.kernel_decoder import KernelDecoder

    torch.manual_seed(10)
    tokenizer = TernaryTokenizer.load(tiny_tokenizer_file(tmp_path / "tokenizer.json"))
    _, served = _reference_and_served(_config(16, 4, 3, 8, 32, vocab_size=len(tokenizer)), seed=10)
    decoder = KernelDecoder(served, max_len=32, device="cuda", sampling=True)
    assert decoder.capture() is True
    text = "".join(stream_text(tokenizer, decoder, "hello", max_new_tokens=5, temperature=1.0, top_k=4))
    assert isinstance(text, str)
    produced = decoder.host_position - len(tokenizer.encode("hello"))
    assert 0 <= produced <= 4


@cuda
def test_sample_token_follows_the_top_k_distribution() -> None:
    from ivonar_inference.kernel_decoder import KernelDecoder

    torch.manual_seed(21)
    _, served = _reference_and_served(_config(16, 4, 3, 8, 32), seed=21)
    decoder = KernelDecoder(served, max_len=32, device="cuda", sampling=True, max_top_k=16)
    assert decoder._sample_launch is not None
    logits = torch.randn(300, device="cuda") * 3
    decoder.logits.copy_(logits.unsqueeze(0))
    decoder.configure_sampling(temperature=0.7, top_k=8, repetition_penalty=1.0)
    draws = 4000
    counts = torch.zeros(300)
    for _ in range(draws):
        decoder.logits.copy_(logits.unsqueeze(0))
        decoder.seen.zero_()
        counts[decoder.sample()] += 1
    values, indices = torch.topk(logits / 0.7, 8)
    expected = torch.softmax(values, dim=0).cpu()
    assert counts.sum() == draws
    assert counts[[i for i in range(300) if i not in indices.tolist()]].sum() == 0
    frequencies = counts[indices.cpu()] / draws
    tolerance = 5 * (expected * (1 - expected) / draws).sqrt() + 0.005
    assert bool(((frequencies - expected).abs() <= tolerance).all())
    decoder.configure_sampling(temperature=0.7, top_k=8, repetition_penalty=2.0, penalized_ids=[int(indices[0])])
    penalized = 0
    for _ in range(draws):
        decoder.logits.copy_(logits.unsqueeze(0))
        decoder.seen.zero_()
        decoder.seen[0, int(indices[0])] = True
        penalized += int(decoder.sample() == int(indices[0]))
    assert penalized / draws < float(expected[0]) * 0.8
    decoder.configure_sampling(temperature=1.0, top_k=1, repetition_penalty=1.0)
    decoder.logits.copy_(logits.unsqueeze(0))
    assert decoder.sample() == int(indices[0])


@cuda
def test_engine_load_serves_the_ternary_kernels(tmp_path: Path) -> None:
    tokenizer_file = tiny_tokenizer_file(tmp_path / "tokenizer.json")
    config = _config(16, 4, 3, 8, 32, vocab_size=266)
    model_file = _write_checkpoint(tmp_path / "packed_inference_checkpoint.pt", config, tokenizer_file_sha256(tokenizer_file))
    engine = Engine.load(model_file, device="cuda")
    assert engine.info.backend == "ternary"
    assert engine.info.graph is True
    assert engine.info.notes == ()
    assert engine.decoder.tile_prefill
    assert engine.model.token_io.projection.cached_runtime_weight("cuda") is None
    result = engine.complete([{"role": "user", "content": "hi"}], GenerationSettings(max_tokens=8))
    assert isinstance(result.text, str)
    assert result.finish_reason in {"stop", "length"}
    fallback = Engine.load(model_file, device="cuda", kernels=False)
    assert fallback.info.backend == "torch"
    assert type(fallback.decoder) is StaticDecoder


def test_build_decoder_without_a_gpu_uses_the_torch_decoder() -> None:
    decoder, backend, note = build_decoder(tiny_model(seed=1), "cpu", sampling=True, kernels=True)
    assert type(decoder) is StaticDecoder
    assert backend == "torch"
    assert note is None
    assert decoder.sampling


@cuda
def test_compile_module_caches_the_binary(tmp_path: Path, monkeypatch) -> None:
    from ivonar_inference.kernels import runtime

    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path / "kernels")
    source = 'extern "C" __global__ void probe(float* out) { out[threadIdx.x] = (float)threadIdx.x * SCALE; }'
    device = torch.device("cuda", torch.cuda.current_device())
    first = runtime.compile_module(source, {"SCALE": 2.0}, device, name="probe")
    assert list((tmp_path / "kernels").glob("*.cubin"))
    assert runtime.compile_module(source, {"SCALE": 2.0}, device, name="probe") is first
    assert runtime.compile_module(source, {"SCALE": 3.0}, device, name="probe") is not first
    out = torch.zeros(32, device=device)
    runtime.Launch(first.kernel("probe"), (1, 1, 1), (32, 1, 1), [runtime.pointer(out)])(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(out, torch.arange(32, device=device, dtype=torch.float32) * 2)


def test_benchmark_reports_prefill_and_decode(tmp_path: Path) -> None:
    tokenizer_file = tiny_tokenizer_file(tmp_path / "tokenizer.json")
    config = tiny_config(vocab_size=266)
    model_file = _write_checkpoint(tmp_path / "packed_inference_checkpoint.pt", config, tokenizer_file_sha256(tokenizer_file))
    engine = Engine.load(model_file, device="cpu")
    result = run_benchmark(engine, prompt="hi", tokens=4)
    assert result.prompt_tokens > 0
    assert 1 <= result.step_tokens <= 4
    assert result.step_tokens_per_second > 0
    assert len(result.lines()) == 3
