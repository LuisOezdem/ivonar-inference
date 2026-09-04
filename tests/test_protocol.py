from __future__ import annotations

import pytest
from pydantic import ValidationError

from ivonar_inference.engine import GenerationResult, GenerationSettings
from ivonar_inference.protocol import (
    ChatCompletionRequest,
    chunk_payload,
    completion_id,
    completion_payload,
    model_card,
)


def test_request_maps_onto_settings_with_defaults() -> None:
    request = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7, "stop": "END"}
    )
    settings = request.settings(GenerationSettings(max_tokens=100, top_k=10))
    assert settings == GenerationSettings(max_tokens=100, temperature=0.7, top_k=10, stop=("END",))
    request = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}], "stop": ["a", "b"], "max_tokens": 8}
    )
    assert request.settings(GenerationSettings()).stop == ("a", "b")
    assert request.settings(GenerationSettings()).max_tokens == 8


def test_request_rejects_bad_roles_empty_lists_and_too_many_stops() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({"messages": []})
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate({"messages": [{"role": "tool", "content": "x"}]})
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(
            {"messages": [{"role": "user", "content": "x"}], "stop": ["1", "2", "3", "4", "5"]}
        )


def test_payloads_follow_the_chat_completion_shapes() -> None:
    request_id = completion_id()
    assert request_id.startswith("chatcmpl-")
    result = GenerationResult(prompt_tokens=4, completion_tokens=6, finish_reason="stop", text="hello")
    payload = completion_payload(request_id, "ivonar-nano", result)
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert payload["usage"] == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
    chunk = chunk_payload(request_id, "ivonar-nano", 1, {"content": "he"})
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"][0]["delta"] == {"content": "he"}
    assert chunk["choices"][0]["finish_reason"] is None
    assert model_card("ivonar-nano")["id"] == "ivonar-nano"
