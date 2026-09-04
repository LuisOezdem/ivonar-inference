from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .decoder import StaticDecoder
from .generation import stream_text
from .loader import load_model
from .tokenizer import IM_END, TernaryTokenizer, format_chat_messages, tokenizer_file_sha256

Message = Mapping[str, str]
StreamFn = Callable[..., Iterator[str]]


@dataclass(frozen=True)
class GenerationSettings:
    max_tokens: int = 512
    temperature: float = 0.5
    top_k: int = 40
    repetition_penalty: float = 1.15
    stop: tuple[str, ...] = ()

    def validated(self, context_tokens: int) -> GenerationSettings:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if any(not item for item in self.stop):
            raise ValueError("stop strings cannot be empty")
        ceiling = max(1, context_tokens - 1)
        if self.max_tokens > ceiling:
            return replace(self, max_tokens=ceiling)
        return self


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    path: Path
    context_tokens: int
    device: str
    lineup: str | None = None
    stage: str | None = None
    phase: str | None = None
    step: int = 0
    graph: bool = False


@dataclass
class GenerationResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    seconds: float = 0.0
    text: str = ""
    dropped_messages: int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds > 0 else 0.0


class Engine:
    """Serialised text generation over one loaded Ivonar model."""

    def __init__(
        self,
        model: object,
        tokenizer: TernaryTokenizer,
        info: ModelInfo,
        defaults: GenerationSettings | None = None,
        stream_fn: StreamFn = stream_text,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.defaults = defaults or GenerationSettings()
        self._stream_fn = stream_fn
        self._lock = threading.Lock()
        self.last_generation = GenerationResult()
        self.decoder: StaticDecoder | None = None

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        tokenizer_path: str | Path | None = None,
        device: str = "cpu",
        model_id: str = "ivonar-nano",
        defaults: GenerationSettings | None = None,
        graph: bool = True,
    ) -> Engine:
        """Load a model file and its tokenizer.

        ``tokenizer_path`` defaults to the ``tokenizer.json`` next to the model.
        Decoding always runs through the static decoder; ``graph`` records its
        step as CUDA graphs on a GPU, which removes the per-token launch
        overhead.
        """

        model_file = Path(model_path)
        tokenizer_file = Path(tokenizer_path) if tokenizer_path is not None else model_file.parent / "tokenizer.json"
        if not model_file.is_file():
            raise FileNotFoundError(f"model file does not exist: {model_file}")
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"tokenizer file does not exist: {tokenizer_file}")
        tokenizer = TernaryTokenizer.load(tokenizer_file)
        loaded = load_model(model_file, device=device, expected_tokenizer_sha256=tokenizer_file_sha256(tokenizer_file))
        if len(tokenizer) != loaded.config.vocab_size:
            raise RuntimeError(
                f"tokenizer vocabulary {len(tokenizer)} does not match the model vocabulary {loaded.config.vocab_size}"
            )
        info = ModelInfo(
            model_id=model_id,
            path=model_file,
            context_tokens=int(loaded.config.seq_len),
            device=device,
            lineup=loaded.lineup,
            stage=loaded.stage,
            phase=loaded.phase,
            step=loaded.step,
        )
        engine = cls(loaded.model, tokenizer, info, defaults)
        engine.decoder = StaticDecoder(loaded.model, device=device, sampling=True)
        if graph and torch.device(device).type == "cuda" and engine.decoder.capture():
            engine.info = replace(info, graph=True)
        return engine

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text)) if text else 0

    def fit_messages(
        self, messages: Sequence[Message], settings: GenerationSettings
    ) -> tuple[list[dict[str, str]], str, int, int]:
        """Drop the oldest turns until the conversation leaves room for the answer.

        Returns the kept messages, the rendered prompt, its token count and how
        many messages were dropped. A system message and the newest question
        always stay.
        """

        kept = [dict(message) for message in messages]
        dropped = 0
        while True:
            prompt = format_chat_messages(kept)
            prompt_tokens = self.count_tokens(prompt)
            if prompt_tokens + settings.max_tokens <= self.info.context_tokens:
                return kept, prompt, prompt_tokens, dropped
            droppable = [index for index, message in enumerate(kept[:-1]) if message.get("role") != "system"]
            if not droppable:
                raise ValueError(
                    f"the last message alone needs {prompt_tokens} tokens, which leaves no room "
                    f"for {settings.max_tokens} answer tokens in a {self.info.context_tokens}-token context"
                )
            first = droppable[0]
            del kept[first]
            dropped += 1
            if first < len(kept) - 1 and kept[first].get("role") == "assistant":
                del kept[first]
                dropped += 1

    def build_prompt(self, messages: Sequence[Message], settings: GenerationSettings) -> tuple[str, int]:
        """Render the conversation so it fits the context; see ``fit_messages``."""

        _, prompt, prompt_tokens, _ = self.fit_messages(messages, settings)
        return prompt, prompt_tokens

    def stream(self, messages: Sequence[Message], settings: GenerationSettings | None = None) -> Iterator[str]:
        """Yield answer text as it is generated; one generation runs at a time."""

        active = (settings or self.defaults).validated(self.info.context_tokens)
        kept, prompt, prompt_tokens, dropped = self.fit_messages(messages, active)
        stop_ids = (int(self.tokenizer.special_tokens[IM_END]),)
        previous = next((m["content"] for m in reversed(kept) if m.get("role") == "assistant"), "")
        with self._lock:
            result = GenerationResult(prompt_tokens=prompt_tokens, dropped_messages=dropped)
            started = time.perf_counter()
            pieces: list[str] = []
            for delta in _cut_at_stop_strings(
                self._stream_fn(
                    tokenizer=self.tokenizer,
                    decoder=self.decoder,
                    prompt=prompt,
                    max_new_tokens=active.max_tokens,
                    temperature=active.temperature,
                    top_k=active.top_k,
                    repetition_penalty=active.repetition_penalty,
                    stop_token_ids=stop_ids,
                    penalized_text=previous[:2000],
                ),
                active.stop,
            ):
                pieces.append(delta)
                yield delta
            result.seconds = time.perf_counter() - started
            result.text = "".join(pieces)
            result.completion_tokens = self.count_tokens(result.text)
            result.finish_reason = "length" if result.completion_tokens >= active.max_tokens else "stop"
            self.last_generation = result

    def complete(self, messages: Sequence[Message], settings: GenerationSettings | None = None) -> GenerationResult:
        text = "".join(self.stream(messages, settings))
        return replace(self.last_generation, text=text)


def _cut_at_stop_strings(deltas: Iterator[str], stop: Sequence[str]) -> Iterator[str]:
    if not stop:
        yield from deltas
        return
    hold = max(len(item) for item in stop) - 1
    buffer = ""
    for delta in deltas:
        buffer += delta
        cut = min((index for index in (buffer.find(item) for item in stop) if index >= 0), default=-1)
        if cut >= 0:
            if cut > 0:
                yield buffer[:cut]
            return
        if len(buffer) > hold:
            yield buffer[: len(buffer) - hold] if hold else buffer
            buffer = buffer[len(buffer) - hold :] if hold else ""
    if buffer:
        yield buffer
