from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from importlib import resources

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .engine import Engine, GenerationSettings
from .protocol import (
    ChatCompletionRequest,
    chunk_payload,
    completion_id,
    completion_payload,
    model_card,
)
from .registry import DEFAULT_SYSTEM, ModelRegistry
from .store import ChatStore


LOGO_PATH = "/assets/ivonar-logo.png"


def _sse(payload: object) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class ChatCreate(BaseModel):
    title: str = "New chat"


class ChatRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ModelSwitch(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class SettingsUpdate(BaseModel):
    system: str | None = Field(default=None, max_length=4000)
    temperature: float | None = Field(default=None, gt=0, le=5)
    top_k: int | None = Field(default=None, ge=0, le=100000)
    max_tokens: int | None = Field(default=None, gt=0)
    repetition_penalty: float | None = Field(default=None, gt=0, le=5)


class ChatTurn(BaseModel):
    content: str = Field(min_length=1)
    system: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None

    def settings(self, defaults: GenerationSettings) -> GenerationSettings:
        return GenerationSettings(
            max_tokens=self.max_tokens if self.max_tokens is not None else defaults.max_tokens,
            temperature=self.temperature if self.temperature is not None else defaults.temperature,
            top_k=self.top_k if self.top_k is not None else defaults.top_k,
            repetition_penalty=(
                self.repetition_penalty if self.repetition_penalty is not None else defaults.repetition_penalty
            ),
            stop=defaults.stop,
        )


def create_app(
    engine: Engine,
    default_system: str | None = None,
    store: ChatStore | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    from . import __version__

    app = FastAPI(title="Ivonar inference", version=__version__)
    state = {"system": DEFAULT_SYSTEM if default_system is None else default_system}

    def active() -> Engine:
        return registry.engine if registry is not None else engine

    def defaults() -> GenerationSettings:
        return active().defaults

    def prepared(messages: Sequence[dict[str, str]], settings: GenerationSettings) -> GenerationSettings:
        current = active()
        try:
            checked = settings.validated(current.info.context_tokens)
            current.fit_messages(messages, checked)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return checked

    def with_system(messages: list[dict[str, str]], system: str | None) -> list[dict[str, str]]:
        text = (system if system is not None else state["system"]) or ""
        if text.strip() and (not messages or messages[0]["role"] != "system"):
            return [{"role": "system", "content": text.strip()}, *messages]
        return messages

    def status() -> dict[str, object]:
        current = active()
        info = current.info
        return {
            "status": "ok",
            "model": info.model_id,
            "device": info.device,
            "graph": info.graph,
            "context_tokens": info.context_tokens,
            "system": state["system"],
            "defaults": {
                "temperature": current.defaults.temperature,
                "top_k": current.defaults.top_k,
                "max_tokens": current.defaults.max_tokens,
                "repetition_penalty": current.defaults.repetition_penalty,
            },
            "models": [entry.as_dict() for entry in registry.entries()] if registry is not None else [],
            "chats": store is not None,
        }

    @app.get("/health")
    def health() -> dict[str, object]:
        return status()

    @app.get("/api/status")
    def api_status() -> dict[str, object]:
        return status()

    @app.post("/api/settings")
    def update_settings(body: SettingsUpdate) -> dict[str, object]:
        current = active()
        if body.system is not None:
            state["system"] = body.system.strip()
        merged = GenerationSettings(
            max_tokens=body.max_tokens if body.max_tokens is not None else current.defaults.max_tokens,
            temperature=body.temperature if body.temperature is not None else current.defaults.temperature,
            top_k=body.top_k if body.top_k is not None else current.defaults.top_k,
            repetition_penalty=(
                body.repetition_penalty if body.repetition_penalty is not None else current.defaults.repetition_penalty
            ),
            stop=current.defaults.stop,
        )
        try:
            current.defaults = merged.validated(current.info.context_tokens)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if registry is not None:
            registry.defaults = current.defaults
            registry.system = state["system"]
        return status()

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        if registry is None:
            return {"object": "list", "data": [model_card(active().info.model_id)]}
        return {"object": "list", "data": [model_card(entry.name) for entry in registry.entries()]}

    @app.post("/api/model")
    def switch_model(body: ModelSwitch) -> dict[str, object]:
        if registry is None:
            raise HTTPException(status_code=404, detail="this server serves a single model")
        try:
            registry.switch(body.name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return status()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return resources.files("ivonar_inference").joinpath("web/index.html").read_text(encoding="utf-8")

    @app.get(LOGO_PATH, response_class=Response)
    def logo() -> Response:
        data = resources.files("ivonar_inference").joinpath("web/assets/ivonar-logo.png").read_bytes()
        return Response(data, media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @app.get("/favicon.ico", response_class=Response)
    def favicon() -> Response:
        return logo()

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        current = active()
        known = ({entry.name for entry in registry.entries()} if registry is not None else {current.info.model_id}) | {
            current.info.model_id
        }
        if request.model is not None and request.model not in known:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
        if registry is not None and request.model is not None and request.model != registry.current:
            current = registry.switch(request.model)
        messages = with_system([message.model_dump() for message in request.messages], None)
        settings = prepared(messages, request.settings(current.defaults))
        request_id = completion_id()
        model_id = current.info.model_id
        if request.stream:
            return StreamingResponse(
                _stream_chunks(current, request_id, model_id, messages, settings),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        result = current.complete(messages, settings)
        return completion_payload(request_id, model_id, result)

    if store is None:
        return app

    def chat_or_404(chat_id: str):
        try:
            return store.get(chat_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown chat") from exc

    @app.get("/api/chats")
    def list_chats() -> dict[str, object]:
        return {"chats": store.list()}

    @app.post("/api/chats", status_code=201)
    def create_chat(body: ChatCreate | None = None) -> dict[str, object]:
        return store.create((body or ChatCreate()).title).summary()

    @app.get("/api/chats/{chat_id}")
    def get_chat(chat_id: str) -> dict[str, object]:
        chat = chat_or_404(chat_id)
        return {**chat.summary(), "messages": chat.messages}

    @app.patch("/api/chats/{chat_id}")
    def rename_chat(chat_id: str, body: ChatRename) -> dict[str, object]:
        chat_or_404(chat_id)
        return store.rename(chat_id, body.title).summary()

    @app.delete("/api/chats/{chat_id}", status_code=204, response_class=Response)
    def delete_chat(chat_id: str) -> Response:
        chat_or_404(chat_id)
        store.delete(chat_id)
        return Response(status_code=204)

    @app.post("/api/chats/{chat_id}/messages")
    def send_message(chat_id: str, turn: ChatTurn):
        chat = chat_or_404(chat_id)
        current = active()
        history = with_system([*chat.messages, {"role": "user", "content": turn.content}], turn.system)
        settings = prepared(history, turn.settings(current.defaults))
        store.append(chat_id, "user", turn.content)
        return StreamingResponse(
            _stream_turn(current, store, chat_id, history, settings),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _stream_chunks(engine: Engine, request_id: str, model_id: str, messages, settings) -> Iterator[str]:
    created = int(time.time())
    yield _sse(chunk_payload(request_id, model_id, created, {"role": "assistant", "content": ""}))
    for delta in engine.stream(messages, settings):
        yield _sse(chunk_payload(request_id, model_id, created, {"content": delta}))
    yield _sse(chunk_payload(request_id, model_id, created, {}, engine.last_generation.finish_reason))
    yield "data: [DONE]\n\n"


def _stream_turn(engine: Engine, store: ChatStore, chat_id: str, messages, settings) -> Iterator[str]:
    pieces: list[str] = []
    try:
        for delta in engine.stream(messages, settings):
            pieces.append(delta)
            yield _sse({"delta": delta})
    finally:
        answer = "".join(pieces)
        if answer.strip():
            store.append(chat_id, "assistant", answer)
    result = engine.last_generation
    yield _sse(
        {
            "done": True,
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "context_tokens": engine.info.context_tokens,
            "dropped_messages": result.dropped_messages,
            "seconds": round(result.seconds, 2),
            "tokens_per_second": round(result.tokens_per_second, 1),
        }
    )
