from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ivonar_inference import cli
from ivonar_inference.download import PullJob
from ivonar_inference.engine import Engine, GenerationSettings, ModelInfo
from ivonar_inference.paths import MODEL_FILE, TOKENIZER_FILE
from ivonar_inference.registry import ModelRegistry
from ivonar_inference.server import create_app
from ivonar_inference.store import ChatStore
from conftest import FakeTokenizer, fake_stream


def _fake_loader(model_path: Path, device: str = "cpu", model_id: str = "model", **options) -> Engine:
    info = ModelInfo(model_id=model_id, path=Path(model_path), context_tokens=64, device=device)
    return Engine(object(), FakeTokenizer(), info, options.get("defaults") or GenerationSettings(), stream_fn=fake_stream)


def _install(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MODEL_FILE).write_bytes(b"weights")
    (folder / TOKENIZER_FILE).write_text("{}", encoding="utf-8")
    return folder


def _setup(tmp_path: Path, isolated_home: Path):
    registry = ModelRegistry(root=tmp_path, loader=_fake_loader, defaults=GenerationSettings(max_tokens=32))

    def fetch(name: str, repo: str | None = None, progress=None) -> Path:
        progress(MODEL_FILE, 3, 7)
        progress(TOKENIZER_FILE, 7, 7)
        return _install(isolated_home / "models" / name)

    pull = PullJob(installed=registry.installed, load=registry.load_first, fetch=fetch)
    app = create_app(registry=registry, store=ChatStore(tmp_path / "chats"), pull=pull)
    return registry, pull, TestClient(app)


def _wait_for(client: TestClient, phase: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get("/api/download").json()
        if state["phase"] == phase:
            return state
        time.sleep(0.02)
    raise AssertionError(f"download never reached {phase}: {state}")


def test_server_starts_without_a_model_and_explains_what_to_do(tmp_path: Path, isolated_home: Path) -> None:
    _, _, client = _setup(tmp_path, isolated_home)
    status = client.get("/api/status").json()
    assert status["ready"] is False and status["model"] is None
    assert status["models"] == []
    assert status["download"]["phase"] == "idle"
    assert status["catalog"]["name"] == "ivonar-nano"
    assert status["catalog"]["folder"] == str(isolated_home / "models")
    assert client.get("/v1/models").json()["data"] == []
    refused = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert refused.status_code == 503 and "Download" in refused.json()["detail"]
    chat = client.post("/api/chats").json()["id"]
    assert client.post(f"/api/chats/{chat}/messages", json={"content": "hi"}).status_code == 503
    assert client.post(f"/api/chats/{chat}/regenerate").status_code == 503
    assert client.get("/api/chats").json()["chats"][0]["id"] == chat
    assert client.post("/api/settings", json={"temperature": 0.9}).json()["defaults"]["temperature"] == 0.9


def test_download_button_fetches_loads_and_then_refuses_a_second_download(tmp_path: Path, isolated_home: Path) -> None:
    registry, _, client = _setup(tmp_path, isolated_home)
    started = client.post("/api/download")
    assert started.status_code == 202 and started.json()["phase"] in {"downloading", "loading", "ready"}
    state = _wait_for(client, "ready")
    assert state["done"] == state["total"] == 7
    status = client.get("/api/status").json()
    assert status["ready"] is True and status["model"] == "ivonar-nano"
    assert [entry["name"] for entry in status["models"]] == ["ivonar-nano"]
    assert registry.defaults.max_tokens == 32
    reply = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert reply.status_code == 200
    assert client.post("/api/download").status_code == 409


def test_a_failed_download_can_be_retried(tmp_path: Path, isolated_home: Path) -> None:
    registry = ModelRegistry(root=tmp_path, loader=_fake_loader)
    attempts = []

    def fetch(name: str, repo: str | None = None, progress=None) -> Path:
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("connection reset")
        return _install(isolated_home / "models" / name)

    pull = PullJob(installed=registry.installed, load=registry.load_first, fetch=fetch)
    client = TestClient(create_app(registry=registry, pull=pull))
    client.post("/api/download")
    failed = _wait_for(client, "error")
    assert "connection reset" in failed["error"]
    assert client.get("/api/status").json()["ready"] is False
    client.post("/api/download")
    _wait_for(client, "ready")
    assert client.get("/api/status").json()["ready"] is True


def test_models_are_found_in_the_user_folder_from_any_directory(tmp_path: Path, isolated_home: Path, monkeypatch) -> None:
    _install(isolated_home / "models" / "ivonar-nano")
    elsewhere = tmp_path / "somewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    registry = ModelRegistry(loader=_fake_loader)
    assert registry.installed()
    engine = registry.load_first()
    assert engine.info.model_id == "ivonar-nano"


def _run_chat(monkeypatch, lines: list[str], capsys) -> str:
    feed = iter(lines)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(feed)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.main(["chat", "--device", "cpu"]) == 0
    return capsys.readouterr().out


def test_terminal_chat_offers_the_download_only_without_a_model(tmp_path: Path, isolated_home: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Engine, "load", classmethod(lambda cls, model_path, **options: _fake_loader(model_path, **options)))

    def fake_pull(name: str, repo: str | None = None, progress=None) -> Path:
        progress(MODEL_FILE, 5, 5)
        return _install(isolated_home / "models" / name)

    monkeypatch.setattr(cli, "pull_model", fake_pull)
    output = _run_chat(monkeypatch, ["hello", "/download", "hello", "/download", "/exit"], capsys)
    assert "No model is installed yet. Type /download" in output
    assert "There is no model yet; type /download, or copy a model folder into" in output
    assert f"Saved to {isolated_home / 'models' / 'ivonar-nano'}" in output
    assert "assistant>" in output
    assert "A model is already installed" in output

    output = _run_chat(monkeypatch, ["/download", "/exit"], capsys)
    assert "No model is installed yet" not in output
    assert "A model is already installed" in output


def test_terminal_chat_picks_up_a_model_copied_in_while_it_waits(
    tmp_path: Path, isolated_home: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Engine, "load", classmethod(lambda cls, model_path, **options: _fake_loader(model_path, **options)))
    monkeypatch.setattr(cli, "pull_model", lambda *args, **kwargs: pytest.fail("nothing should be downloaded"))
    feed = iter(["hello", "copy", "/exit"])

    def fake_input(prompt: str = "") -> str:
        line = next(feed, None)
        if line is None:
            raise EOFError
        if line == "copy":
            _install(isolated_home / "models" / "my-model")
            return "Where is Paris?"
        return line

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.main(["chat", "--device", "cpu", "--max-tokens", "16"]) == 0
    output = capsys.readouterr().out
    assert "There is no model yet" in output
    assert "Preparing the model" in output
    assert "my-model" in output
    assert output.count("assistant>") == 1
    assert fake_stream.calls[-1]["prompt"].rstrip().endswith("Where is Paris?<|im_end|><|im_start|><|assistant|>")


def test_the_page_loads_a_model_copied_in_without_downloading(tmp_path: Path, isolated_home: Path) -> None:
    registry = ModelRegistry(root=tmp_path, loader=_fake_loader, defaults=GenerationSettings(max_tokens=32))

    def fetch(*args, **kwargs) -> Path:
        pytest.fail("a copied model must not be downloaded again")

    pull = PullJob(installed=registry.installed, load=registry.load_first, fetch=fetch)
    client = TestClient(create_app(registry=registry, pull=pull))
    assert client.get("/api/status").json()["models"] == []
    _install(isolated_home / "models" / "my-model")
    assert [entry["name"] for entry in client.get("/api/status").json()["models"]] == ["my-model"]
    assert client.post("/api/download").json()["phase"] == "loading"
    _wait_for(client, "ready")
    assert client.get("/api/status").json()["model"] == "my-model"


def test_a_model_that_fails_to_load_is_reported_as_such(tmp_path: Path, isolated_home: Path) -> None:
    def broken_loader(model_path: Path, **options) -> Engine:
        raise RuntimeError("checkpoint is damaged")

    registry = ModelRegistry(root=tmp_path, loader=broken_loader)
    pull = PullJob(installed=registry.installed, load=registry.load_first)
    client = TestClient(create_app(registry=registry, pull=pull))
    _install(isolated_home / "models" / "broken")
    client.post("/api/download")
    failed = _wait_for(client, "error")
    assert failed["failed"] == "loading"
    assert "checkpoint is damaged" in failed["error"]
    assert client.get("/api/status").json()["ready"] is False


def test_pull_says_when_everything_is_already_there(tmp_path: Path, isolated_home: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "pull_model", lambda name, repo=None, progress=None: _install(isolated_home / "models" / name))
    assert cli.main(["pull"]) == 0
    output = capsys.readouterr().out
    assert "Already up to date" in output
    assert "Saved to" not in output


def test_switching_to_the_model_in_use_keeps_the_conversation(tmp_path: Path, isolated_home: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    _install(isolated_home / "models" / "ivonar-nano")
    monkeypatch.setattr(Engine, "load", classmethod(lambda cls, model_path, **options: _fake_loader(model_path, **options)))
    output = _run_chat(monkeypatch, ["/model ivonar-nano", "/exit"], capsys)
    assert "Already using ivonar-nano." in output
    assert "New conversation" not in output


def test_terminal_chat_can_quit_before_downloading(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "pull_model", lambda *args, **kwargs: pytest.fail("nothing should be downloaded"))
    output = _run_chat(monkeypatch, ["/help", "/exit"], capsys)
    assert "/download" in output


def test_commands_that_need_a_model_say_how_to_get_one(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["models"]) == 1
    assert "ivonar pull" in capsys.readouterr().out
    assert cli.main(["bench", "--device", "cpu"]) == 1
    assert "no model is installed yet" in capsys.readouterr().err


def test_serve_picks_the_next_free_port(monkeypatch) -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]
        assert cli._free_port("127.0.0.1", taken) != taken
