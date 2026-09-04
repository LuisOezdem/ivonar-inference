from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .engine import GenerationResult, GenerationSettings


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    repetition_penalty: float | None = None

    @field_validator("stop")
    @classmethod
    def _stop_as_list(cls, value: str | list[str] | None) -> list[str] | None:
        if value is None:
            return None
        items = [value] if isinstance(value, str) else list(value)
        if len(items) > 4:
            raise ValueError("at most four stop strings are supported")
        return items

    def settings(self, defaults: GenerationSettings) -> GenerationSettings:
        return GenerationSettings(
            max_tokens=self.max_tokens if self.max_tokens is not None else defaults.max_tokens,
            temperature=self.temperature if self.temperature is not None else defaults.temperature,
            top_k=self.top_k if self.top_k is not None else defaults.top_k,
            repetition_penalty=(
                self.repetition_penalty if self.repetition_penalty is not None else defaults.repetition_penalty
            ),
            stop=tuple(self.stop) if self.stop else defaults.stop,
        )


def completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def model_card(model_id: str) -> dict[str, object]:
    return {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "ivonar"}


def completion_payload(request_id: str, model_id: str, result: GenerationResult) -> dict[str, object]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def chunk_payload(
    request_id: str,
    model_id: str,
    created: int,
    delta: dict[str, str],
    finish_reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
