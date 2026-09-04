from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ivonar_inference.engine import Engine
from ivonar_inference.server import create_app

from conftest import ANSWER, fake_stream


@pytest.fixture
def client(engine: Engine) -> TestClient:
    return TestClient(create_app(engine, default_system="Be brief."))


def test_health_and_models(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model"] == "ivonar-nano"
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "ivonar-nano"


def test_index_serves_the_chat_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Ivonar</title>" in response.text


def test_chat_completion_without_streaming(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "ivonar-nano", "messages": [{"role": "user", "content": "Where is Paris?"}]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == ANSWER
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["completion_tokens"] == len(ANSWER.split())
    assert fake_stream.calls[-1]["prompt"].startswith("<|im_start|><|system|>Be brief.<|im_end|>")


def test_client_system_message_is_kept(client: TestClient) -> None:
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "Pirate."}, {"role": "user", "content": "Hi"}]},
    )
    prompt = fake_stream.calls[-1]["prompt"]
    assert prompt.startswith("<|im_start|><|system|>Pirate.<|im_end|>")
    assert "Be brief." not in prompt


def test_chat_completion_streams_server_sent_events(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Where is Paris?"}], "stream": True, "max_tokens": 3},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    assert text == "Paris is the"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_unknown_model_and_bad_requests_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert response.status_code == 404
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hi"}], "temperature": 0},
    )
    assert response.status_code == 400
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "word " * 80}]},
    )
    assert response.status_code == 400


def test_the_logo_and_favicon_are_served(client: TestClient) -> None:
    for path in ("/assets/ivonar-logo.png", "/favicon.ico"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert 'src="/assets/ivonar-logo.png"' in client.get("/").text
