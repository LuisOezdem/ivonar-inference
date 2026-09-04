from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ivonar_inference.engine import Engine, GenerationSettings, ModelInfo
from ivonar_inference.registry import DEFAULT_SYSTEM, ModelRegistry
from ivonar_inference.server import create_app
from ivonar_inference.store import ChatStore
from conftest import FakeTokenizer, fake_stream


def _engine(name: str, context: int = 64) -> Engine:
    info = ModelInfo(model_id=name, path=Path(f"{name}.pt"), context_tokens=context, device="cpu", stage="sft", step=1)
    return Engine(object(), FakeTokenizer(), info, GenerationSettings(max_tokens=32), stream_fn=fake_stream)


class FakeRegistry(ModelRegistry):
    """A registry whose switch swaps in a prepared engine instead of loading a file."""

    def __init__(self, root: Path, engines: dict[str, Engine], current: str) -> None:
        for name in engines:
            folder = root / "models" / name
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "packed_inference_checkpoint.pt").write_bytes(b"x")
        super().__init__(
            engines[current],
            root / "models" / current / "packed_inference_checkpoint.pt",
            root=root,
            defaults=engines[current].defaults,
        )
        self.engines = engines
        self.loads = 0

    def switch(self, name: str) -> Engine:
        if name not in self.engines:
            raise FileNotFoundError(f"model not found: {name}")
        self.loads += 1
        self._engine = self.engines[name]
        self._path = self.root / "models" / name / "packed_inference_checkpoint.pt"
        return self._engine


@pytest.fixture
def registry(tmp_path: Path) -> FakeRegistry:
    fake_stream.calls.clear()
    return FakeRegistry(tmp_path, {"nano": _engine("nano"), "mini": _engine("mini", context=48)}, current="nano")


@pytest.fixture
def client(registry: FakeRegistry, tmp_path: Path) -> TestClient:
    app = create_app(
        registry.engine, default_system=DEFAULT_SYSTEM, store=ChatStore(tmp_path / "chats"), registry=registry
    )
    return TestClient(app)


def test_registry_lists_every_model_and_marks_the_loaded_one(registry: FakeRegistry) -> None:
    names = [entry.name for entry in registry.entries()]
    assert names == ["mini", "nano"]
    assert [entry.loaded for entry in registry.entries()] == [False, True]
    assert registry.current == "nano"
    registry.switch("mini")
    assert registry.current == "mini"
    assert [entry.loaded for entry in registry.entries()] == [True, False]


def test_status_reports_models_and_settings(client: TestClient) -> None:
    status = client.get("/api/status").json()
    assert status["model"] == "nano"
    assert status["system"] == DEFAULT_SYSTEM
    assert [item["name"] for item in status["models"]] == ["mini", "nano"]
    assert status["defaults"]["max_tokens"] == 32
    assert client.get("/v1/models").json()["data"][0]["id"] == "mini"


def test_switching_models_through_the_api(client: TestClient, registry: FakeRegistry) -> None:
    status = client.post("/api/model", json={"name": "mini"}).json()
    assert status["model"] == "mini"
    assert status["context_tokens"] == 48
    assert registry.loads == 1
    missing = client.post("/api/model", json={"name": "nope"})
    assert missing.status_code == 404


def test_settings_change_the_system_message_and_defaults(client: TestClient, registry: FakeRegistry) -> None:
    status = client.post("/api/settings", json={"system": "Answer in one line.", "temperature": 0.9, "top_k": 5}).json()
    assert status["system"] == "Answer in one line."
    assert status["defaults"]["temperature"] == 0.9 and status["defaults"]["top_k"] == 5
    assert registry.engine.defaults.temperature == 0.9
    chat = client.post("/api/chats").json()
    with client.stream("POST", f"/api/chats/{chat['id']}/messages", json={"content": "hi"}) as response:
        list(response.iter_lines())
    assert fake_stream.calls[-1]["temperature"] == 0.9
    assert client.post("/api/settings", json={"temperature": 0}).status_code == 422


def test_a_request_may_switch_the_model(client: TestClient, registry: FakeRegistry) -> None:
    payload = {"model": "mini", "messages": [{"role": "user", "content": "hi"}]}
    body = client.post("/v1/chat/completions", json=payload).json()
    assert body["model"] == "mini"
    assert registry.current == "mini"
    assert client.post("/v1/chat/completions", json={**payload, "model": "ghost"}).status_code == 404


def _send(client: TestClient, chat_id: str, content: str) -> dict:
    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"content": content}) as response:
        events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
    return events[-1]


def test_the_stream_reports_context_use_and_drops_old_turns(client: TestClient) -> None:
    chat = client.post("/api/chats").json()["id"]
    first = _send(client, chat, "one two three")
    assert first["done"] is True and first["context_tokens"] == 64
    assert first["dropped_messages"] == 0 and first["prompt_tokens"] > 0
    dropped = 0
    for _ in range(6):
        dropped += _send(client, chat, "another question about the topic")["dropped_messages"]
    assert dropped > 0
    assert client.post(f"/api/chats/{chat}/messages", json={"content": "word " * 60}).status_code == 400
