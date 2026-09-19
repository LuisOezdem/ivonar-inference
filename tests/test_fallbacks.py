from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from ivonar_inference import cli, loader
from ivonar_inference.download import PullJob
from ivonar_inference.engine import Engine, GenerationSettings, ModelInfo
from ivonar_inference.paths import MODEL_FILE, TOKENIZER_FILE
from ivonar_inference.registry import ModelRegistry
from ivonar_inference.server import create_app
from ivonar_inference.store import ChatStore

from conftest import FakeTokenizer, fake_stream, tiny_model

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _install(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MODEL_FILE).write_bytes(b"weights")
    (folder / TOKENIZER_FILE).write_text("{}", encoding="utf-8")
    return folder / MODEL_FILE


def _engine(model_path: Path, device: str = "cpu", model_id: str = "model", stream_fn=fake_stream, **options) -> Engine:
    info = ModelInfo(model_id=model_id, path=Path(model_path), context_tokens=64, device=device)
    return Engine(object(), FakeTokenizer(), info, GenerationSettings(max_tokens=16), stream_fn=stream_fn)


def test_auto_uses_the_cpu_without_a_gpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert loader.pick_device("auto") == ("cpu", None)
    assert loader.pick_device("cuda") == ("cuda", None)


def test_auto_skips_a_gpu_this_torch_build_cannot_run(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80", "sm_86", "sm_90"])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Future GPU")
    device, note = loader.pick_device("auto")
    assert device == "cpu"
    assert "Future GPU" in note and "not supported" in note and "CPU" in note


def test_auto_skips_a_gpu_that_does_not_respond(monkeypatch) -> None:
    def broken(device=None):
        raise RuntimeError("CUDA driver version is insufficient")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(torch.cuda, "get_device_capability", broken)
    device, note = loader.pick_device("auto")
    assert device == "cpu"
    assert "did not respond" in note and "insufficient" in note


@pytest.mark.parametrize(
    ("arches", "capability", "expected"),
    [
        (["sm_80", "sm_86"], (8, 9), True),
        (["sm_90"], (8, 9), False),
        (["sm_75", "compute_80"], (8, 9), True),
        (["sm_75", "compute_90"], (8, 9), False),
        (["sm_90a"], (9, 0), True),
        (["sm_100", "sm_120"], (12, 0), True),
        ([], (7, 5), True),
    ],
)
def test_the_arch_list_decides_which_gpus_this_build_runs(monkeypatch, arches, capability, expected) -> None:
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arches)
    assert loader._built_for(*capability) is expected


def test_a_model_that_fails_on_the_gpu_runs_on_the_cpu(tmp_path: Path) -> None:
    _install(tmp_path / "models" / "nano")

    def gpu_fails(model_path: Path, device: str = "cpu", **options) -> Engine:
        if device != "cpu":
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return _engine(model_path, device=device, **options)

    registry = ModelRegistry(root=tmp_path, device="cuda", loader=gpu_fails)
    engine = registry.load_first()
    assert engine.info.device == "cpu"
    assert registry.device == "cpu"
    assert "no kernel image" in registry.notes[0] and "CPU" in registry.notes[0]


def test_a_model_that_fails_everywhere_reports_the_real_error(tmp_path: Path) -> None:
    _install(tmp_path / "models" / "nano")

    def always_fails(model_path: Path, device: str = "cpu", **options) -> Engine:
        raise RuntimeError(f"checkpoint is damaged ({device})")

    registry = ModelRegistry(root=tmp_path, device="cuda", loader=always_fails)
    with pytest.raises(RuntimeError, match="checkpoint is damaged"):
        registry.load_first()
    assert registry.notes == []
    assert not registry.ready


def test_a_failed_switch_keeps_the_previous_model(tmp_path: Path) -> None:
    _install(tmp_path / "models" / "good")
    _install(tmp_path / "models" / "broken")

    def picky(model_path: Path, device: str = "cpu", **options) -> Engine:
        if Path(model_path).parent.name == "broken":
            raise RuntimeError("checkpoint is damaged")
        return _engine(model_path, device=device, **options)

    registry = ModelRegistry(root=tmp_path, loader=picky)
    registry.switch("good")
    with pytest.raises(RuntimeError, match="damaged"):
        registry.switch("broken")
    assert registry.current == "good"
    assert registry.ready

    client = TestClient(create_app(registry=registry))
    refused = client.post("/api/model", json={"name": "broken"})
    assert refused.status_code == 400
    assert "broken could not be loaded" in refused.json()["detail"]
    assert client.get("/api/status").json()["model"] == "good"


def _breaking_stream(**kwargs):
    yield "Paris"
    raise RuntimeError("CUDA error: an illegal memory access was encountered")


def test_errors_while_answering_reach_the_page_and_the_api(tmp_path: Path) -> None:
    engine = _engine(Path("model.pt"), stream_fn=_breaking_stream)
    client = TestClient(create_app(engine, store=ChatStore(tmp_path / "chats")))
    chat_id = client.post("/api/chats").json()["id"]
    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"content": "Where is Paris?"}) as response:
        events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
    assert events[0] == {"delta": "Paris"}
    assert "illegal memory access" in events[-1]["error"]
    assert [m["role"] for m in client.get(f"/api/chats/{chat_id}").json()["messages"]] == ["user", "assistant"]

    body = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert "illegal memory access" in json.loads(lines[-2][6:])["error"]["message"]
    assert lines[-1] == "data: [DONE]"

    failed = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert failed.status_code == 500
    assert "illegal memory access" in failed.json()["detail"]


def test_a_model_that_cannot_be_loaded_at_start_is_explained(tmp_path: Path, isolated_home: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _install(isolated_home / "models" / "nano")

    def broken(cls, model_path, **options):
        raise RuntimeError("checkpoint is damaged")

    monkeypatch.setattr(Engine, "load", classmethod(broken))
    args = argparse.Namespace(
        model=None, tokenizer=None, device="cpu", no_graph=False, no_kernels=False, system="", max_tokens=16,
        temperature=0.5, top_k=40, repetition_penalty=1.15,
    )
    registry, failure = cli._registry(args)
    assert not registry.ready
    assert "checkpoint is damaged" in failure

    job = PullJob(installed=lambda: True, load=lambda folder: None)
    job.fail(failure)
    assert job.snapshot()["phase"] == "error"
    assert job.snapshot()["failed"] == "loading"

    monkeypatch.setattr("builtins.input", lambda prompt="": "/exit")
    assert cli.main(["chat", "--device", "cpu"]) == 0
    assert "could not be loaded: checkpoint is damaged" in capsys.readouterr().out

    args.model = "nano"
    with pytest.raises(RuntimeError, match="damaged"):
        cli._registry(args)


def test_ctrl_c_ends_a_command_quietly(monkeypatch, capsys) -> None:
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_models", interrupted)
    assert cli.main(["models"]) == 130
    assert "Traceback" not in capsys.readouterr().err


def test_ctrl_c_stops_an_answer_but_not_the_chat(tmp_path: Path, isolated_home: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _install(isolated_home / "models" / "nano")

    def interrupted_stream(**kwargs):
        yield "Par"
        raise KeyboardInterrupt

    monkeypatch.setattr(
        Engine, "load", classmethod(lambda cls, model_path, **options: _engine(model_path, stream_fn=interrupted_stream))
    )
    feed = iter(["Where is Paris?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(feed))
    assert cli.main(["chat", "--device", "cpu", "--max-tokens", "8"]) == 0
    assert "Answer stopped" in capsys.readouterr().out


@cuda
def test_kernels_that_fail_their_self_check_fall_back_to_the_torch_decoder(monkeypatch) -> None:
    from ivonar_inference.decoder import StaticDecoder, build_decoder
    from ivonar_inference.kernel_decoder import KernelDecoder
    from ivonar_inference.kernels.runtime import KernelError

    def broken(self) -> None:
        raise KernelError("the ternary kernels disagree with themselves on this GPU")

    monkeypatch.setattr(KernelDecoder, "self_check", broken)
    model = tiny_model(seed=3, device="cuda")
    decoder, backend, note = build_decoder(model, "cuda", sampling=True)
    assert type(decoder) is StaticDecoder
    assert backend == "torch"
    assert "disagree" in note


@cuda
def test_the_kernels_run_without_a_persisting_l2_cache(monkeypatch) -> None:
    from ivonar_inference.kernel_decoder import KernelDecoder
    from ivonar_inference.kernels import runtime
    from ivonar_inference.kernels.runtime import KernelError

    def unsupported(*args, **kwargs):
        raise KernelError("CUDA driver error 801: operation not supported")

    monkeypatch.setattr(runtime._Driver, "reserve_persisting", unsupported)
    monkeypatch.setattr(runtime._Driver, "persist_on_stream", unsupported)
    decoder = KernelDecoder(tiny_model(seed=4, device="cuda"), device="cuda", sampling=True)
    decoder.self_check()
    assert decoder.persisted_bytes == 0
    assert decoder.capture()
    decoder.prefill(torch.tensor([[5, 6, 7]]))
    decoder.configure_sampling(temperature=0.5, top_k=5)
    assert 0 <= decoder.advance() < int(decoder.model.config.vocab_size)


def test_an_unreadable_tokenizer_is_named_in_the_error(tmp_path: Path) -> None:
    from ivonar_inference.tokenizer import TernaryTokenizer

    path = tmp_path / "tokenizer.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="is not an Ivonar tokenizer"):
        TernaryTokenizer.load(path)


def test_any_load_failure_while_switching_is_a_clear_refusal(tmp_path: Path) -> None:
    _install(tmp_path / "models" / "good")
    _install(tmp_path / "models" / "odd")

    def odd(model_path: Path, device: str = "cpu", **options) -> Engine:
        if Path(model_path).parent.name == "odd":
            raise KeyError("special_tokens")
        return _engine(model_path, device=device, **options)

    registry = ModelRegistry(root=tmp_path, loader=odd)
    registry.switch("good")
    client = TestClient(create_app(registry=registry))
    refused = client.post("/api/model", json={"name": "odd"})
    assert refused.status_code == 400
    assert "odd could not be loaded" in refused.json()["detail"]
    assert client.get("/api/status").json()["model"] == "good"
