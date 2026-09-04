from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ivonar_inference.engine import Engine
from ivonar_inference.server import create_app
from ivonar_inference.store import ChatStore

from conftest import ANSWER, fake_stream


@pytest.fixture
def client(engine: Engine, tmp_path: Path) -> TestClient:
    return TestClient(create_app(engine, default_system="Be brief.", store=ChatStore(tmp_path / "chats")))


def _events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def test_chat_lifecycle_through_the_api(client: TestClient) -> None:
    created = client.post("/api/chats", json={"title": "Trip"})
    assert created.status_code == 201
    chat_id = created.json()["id"]
    assert client.get("/api/chats").json()["chats"][0]["title"] == "Trip"

    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"content": "Where is Paris?"}) as response:
        assert response.status_code == 200
        events = _events(response)
    text = "".join(event.get("delta", "") for event in events)
    assert text == ANSWER
    assert events[-1]["done"] is True
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["completion_tokens"] == len(ANSWER.split())
    assert fake_stream.calls[-1]["prompt"].startswith("<|im_start|><|system|>Be brief.<|im_end|>")

    chat = client.get(f"/api/chats/{chat_id}").json()
    assert [message["role"] for message in chat["messages"]] == ["user", "assistant"]
    assert chat["messages"][1]["content"] == ANSWER

    with client.stream(
        "POST", f"/api/chats/{chat_id}/messages", json={"content": "And Berlin?", "system": "Pirate."}
    ) as response:
        _events(response)
    prompt = fake_stream.calls[-1]["prompt"]
    assert prompt.startswith("<|im_start|><|system|>Pirate.<|im_end|>")
    assert "<|user|>Where is Paris?<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|><|user|>And Berlin?<|im_end|><|im_start|><|assistant|>")

    assert client.patch(f"/api/chats/{chat_id}", json={"title": "Cities"}).json()["title"] == "Cities"
    assert client.delete(f"/api/chats/{chat_id}").status_code == 204
    assert client.get(f"/api/chats/{chat_id}").status_code == 404
    assert client.get("/api/chats").json()["chats"] == []


def test_new_chat_is_titled_from_the_first_message(client: TestClient) -> None:
    chat_id = client.post("/api/chats").json()["id"]
    with client.stream("POST", f"/api/chats/{chat_id}/messages", json={"content": "Plan my week please"}) as response:
        _events(response)
    assert client.get(f"/api/chats/{chat_id}").json()["title"] == "Plan my week please"


def test_bad_turns_are_rejected_and_nothing_is_stored(client: TestClient) -> None:
    chat_id = client.post("/api/chats").json()["id"]
    assert client.post(f"/api/chats/{chat_id}/messages", json={"content": ""}).status_code == 422
    assert client.post(f"/api/chats/{chat_id}/messages", json={"content": "x", "temperature": 0}).status_code == 400
    assert client.post("/api/chats/missing/messages", json={"content": "x"}).status_code == 404
    assert client.get(f"/api/chats/{chat_id}").json()["messages"] == []


def test_app_without_store_has_no_chat_routes(engine: Engine) -> None:
    client = TestClient(create_app(engine))
    assert client.get("/api/chats").status_code == 404
    assert client.get("/health").json()["chats"] is False
