from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from importlib import resources

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .download import DEFAULT_MODEL, DEFAULT_SIZE_MB, DEFAULT_TITLE, PullJob
from .engine import Engine, GenerationSettings
from .paths import user_models_dir
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
NO_MODEL = "No model is installed yet. Open the chat page and click Download, or run 'ivonar pull'."
UNBOUNDED_CONTEXT = 1 << 20


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


class TurnSettings(BaseModel):
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


class ChatTurn(TurnSettings):
    content: str = Field(min_length=1)


def create_app(
    engine: Engine | None = None,
    default_system: str | None = None,
    store: ChatStore | None = None,
    registry: ModelRegistry | None = None,
    pull: PullJob | None = None,
) -> FastAPI:
    """The chat page and the API over one engine, or over a registry that may still be waiting for a model.

    With a ``pull`` job the page offers to download the default release while
    no model is installed, and the server loads it once the download is done.
    """

    from . import __version__

    app = FastAPI(title="Ivonar inference", version=__version__)
    state = {"system": DEFAULT_SYSTEM if default_system is None else default_system}

    def active() -> Engine | None:
        return registry.engine if registry is not None else engine

    def require() -> Engine:
        current = active()
        if current is None:
            raise HTTPException(status_code=503, detail=NO_MODEL)
        return current

    def base_settings() -> GenerationSettings:
        current = active()
        if current is not None:
            return current.defaults
        return registry.defaults if registry is not None else GenerationSettings()

    def prepared(messages: Sequence[dict[str, str]], settings: GenerationSettings) -> GenerationSettings:
        current = require()
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
        settings = base_settings()
        payload: dict[str, object] = {
            "status": "ok",
            "ready": current is not None,
            "model": None,
            "device": registry.device if registry is not None else None,
            "backend": None,
            "graph": False,
            "context_tokens": 0,
            "system": state["system"],
            "defaults": {
                "temperature": settings.temperature,
                "top_k": settings.top_k,
                "max_tokens": settings.max_tokens,
                "repetition_penalty": settings.repetition_penalty,
            },
            "models": [entry.as_dict() for entry in registry.entries()] if registry is not None else [],
            "chats": store is not None,
            "download": pull.snapshot() if pull is not None else None,
            "catalog": {
                "name": pull.name if pull is not None else DEFAULT_MODEL,
                "title": DEFAULT_TITLE,
                "size_mb": DEFAULT_SIZE_MB,
                "folder": str(user_models_dir()),
            },
        }
        if current is not None:
            info = current.info
            payload.update(
                model=info.model_id,
                device=info.device,
                backend=info.backend,
                graph=info.graph,
                context_tokens=info.context_tokens,
            )
        return payload

    @app.get("/health")
    def health() -> dict[str, object]:
        return status()

    @app.get("/api/status")
    def api_status() -> dict[str, object]:
        return status()

    @app.post("/api/settings")
    def update_settings(body: SettingsUpdate) -> dict[str, object]:
        current = active()
        base = base_settings()
        if body.system is not None:
            state["system"] = body.system.strip()
        merged = GenerationSettings(
            max_tokens=body.max_tokens if body.max_tokens is not None else base.max_tokens,
            temperature=body.temperature if body.temperature is not None else base.temperature,
            top_k=body.top_k if body.top_k is not None else base.top_k,
            repetition_penalty=body.repetition_penalty if body.repetition_penalty is not None else base.repetition_penalty,
            stop=base.stop,
        )
        try:
            checked = merged.validated(current.info.context_tokens if current is not None else UNBOUNDED_CONTEXT)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if current is not None:
            current.defaults = checked
        if registry is not None:
            registry.defaults = checked
            registry.system = state["system"]
        return status()

    @app.get("/api/download")
    def download_state() -> dict[str, object]:
        if pull is None:
            raise HTTPException(status_code=404, detail="this server does not download models")
        return pull.snapshot()

    @app.post("/api/download", status_code=202)
    def start_download() -> dict[str, object]:
        if pull is None:
            raise HTTPException(status_code=404, detail="this server does not download models")
        if active() is not None:
            raise HTTPException(status_code=409, detail="a model is already installed")
        return pull.start()

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        if registry is None:
            current = active()
            return {"object": "list", "data": [model_card(current.info.model_id)] if current is not None else []}
        return {"object": "list", "data": [model_card(entry.name) for entry in registry.entries()]}

    def switched(name: str) -> Engine:
        try:
            return registry.switch(name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"{name} could not be loaded: {_failure(exc)}") from exc

    @app.post("/api/model")
    def switch_model(body: ModelSwitch) -> dict[str, object]:
        if registry is None:
            raise HTTPException(status_code=404, detail="this server serves a single model")
        switched(body.name)
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
        current = require()
        known = ({entry.name for entry in registry.entries()} if registry is not None else set()) | {
            current.info.model_id
        }
        if request.model is not None and request.model not in known:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
        if registry is not None and request.model is not None and request.model != registry.current:
            current = switched(request.model)
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
        try:
            result = current.complete(messages, settings)
        except (RuntimeError, MemoryError) as exc:
            raise HTTPException(status_code=500, detail=_failure(exc)) from exc
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

    def stream_answer(current: Engine, chat_id: str, history: list[dict[str, str]], settings) -> StreamingResponse:
        return StreamingResponse(
            _stream_turn(current, store, chat_id, history, settings),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/chats/{chat_id}/messages")
    def send_message(chat_id: str, turn: ChatTurn):
        chat = chat_or_404(chat_id)
        current = require()
        history = with_system([*chat.messages, {"role": "user", "content": turn.content}], turn.system)
        settings = prepared(history, turn.settings(current.defaults))
        store.append(chat_id, "user", turn.content)
        return stream_answer(current, chat_id, history, settings)

    @app.post("/api/chats/{chat_id}/regenerate")
    def regenerate(chat_id: str, turn: TurnSettings | None = None):
        chat = chat_or_404(chat_id)
        current = require()
        messages = list(chat.messages)
        while messages and messages[-1]["role"] == "assistant":
            messages.pop()
        if not messages:
            raise HTTPException(status_code=409, detail="there is no message to answer again")
        turn = turn or TurnSettings()
        history = with_system(messages, turn.system)
        settings = prepared(history, turn.settings(current.defaults))
        store.drop_answer(chat_id)
        return stream_answer(current, chat_id, history, settings)

    return app


def _failure(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _stream_chunks(engine: Engine, request_id: str, model_id: str, messages, settings) -> Iterator[str]:
    created = int(time.time())
    yield _sse(chunk_payload(request_id, model_id, created, {"role": "assistant", "content": ""}))
    try:
        for delta in engine.stream(messages, settings):
            yield _sse(chunk_payload(request_id, model_id, created, {"content": delta}))
    except Exception as exc:
        yield _sse({"error": {"message": _failure(exc), "type": "server_error"}})
        yield "data: [DONE]\n\n"
        return
    yield _sse(chunk_payload(request_id, model_id, created, {}, engine.last_generation.finish_reason))
    yield "data: [DONE]\n\n"


def _stream_turn(engine: Engine, store: ChatStore, chat_id: str, messages, settings) -> Iterator[str]:
    pieces: list[str] = []
    failure = None
    try:
        for delta in engine.stream(messages, settings):
            pieces.append(delta)
            yield _sse({"delta": delta})
    except Exception as exc:
        failure = _failure(exc)
    finally:
        answer = "".join(pieces)
        if answer.strip():
            store.append(chat_id, "assistant", answer)
    if failure is not None:
        yield _sse({"error": failure})
        return
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
            "first_token_seconds": round(result.first_token_seconds, 3),
            "decode_tokens_per_second": round(result.decode_tokens_per_second, 1),
        }
    )
