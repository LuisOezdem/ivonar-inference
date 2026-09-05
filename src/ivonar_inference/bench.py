from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .engine import Engine, GenerationSettings

DEFAULT_PROMPT = "Explain how a car engine works and why oil matters."


@dataclass(frozen=True)
class BenchResult:
    prompt_tokens: int
    prefill_seconds: float
    step_tokens: int
    step_seconds: float
    completion_tokens: int
    completion_seconds: float
    first_token_seconds: float = 0.0

    @property
    def step_tokens_per_second(self) -> float:
        return self.step_tokens / self.step_seconds if self.step_seconds > 0 else 0.0

    @property
    def completion_tokens_per_second(self) -> float:
        return self.completion_tokens / self.completion_seconds if self.completion_seconds > 0 else 0.0

    def lines(self) -> list[str]:
        return [
            f"prompt: {self.prompt_tokens} tokens, prefill {self.prefill_seconds * 1000:.1f} ms",
            f"decode: {self.step_tokens} steps, {self.step_seconds / max(self.step_tokens, 1) * 1000:.3f} ms per token, "
            f"{self.step_tokens_per_second:.0f} tokens per second",
            f"end to end: {self.completion_tokens} tokens in {self.completion_seconds:.2f} s, "
            f"{self.completion_tokens_per_second:.0f} tokens per second, first token after "
            f"{self.first_token_seconds * 1000:.0f} ms",
        ]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(engine: Engine, prompt: str = DEFAULT_PROMPT, tokens: int = 256, system: str | None = None) -> BenchResult:
    decoder = engine.decoder
    if decoder is None:
        raise RuntimeError("the engine has no decoder to benchmark")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    settings = GenerationSettings(max_tokens=tokens).validated(engine.info.context_tokens)
    text, prompt_tokens = engine.build_prompt(messages, settings)
    ids = engine.tokenizer.encode(text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=decoder.device)
    budget = decoder.max_len - len(ids) - 2
    if budget < 4:
        raise ValueError("the prompt leaves no room to benchmark decoding")
    warmup = min(8, budget // 4)
    steps = max(1, min(tokens, budget - warmup))
    with engine._lock:
        decoder.prefill(input_ids)
        _synchronize(decoder.device)
        started = time.perf_counter()
        decoder.prefill(input_ids)
        _synchronize(decoder.device)
        prefill_seconds = time.perf_counter() - started
        decoder.configure_sampling(settings.temperature, settings.top_k, settings.repetition_penalty)
        decoder.sample()
        for _ in range(warmup):
            decoder.advance()
        _synchronize(decoder.device)
        started = time.perf_counter()
        for _ in range(steps):
            decoder.advance()
        _synchronize(decoder.device)
        step_seconds = time.perf_counter() - started
    result = engine.complete(messages, settings)
    return BenchResult(
        prompt_tokens=prompt_tokens,
        prefill_seconds=prefill_seconds,
        step_tokens=steps,
        step_seconds=step_seconds,
        completion_tokens=result.completion_tokens,
        completion_seconds=result.seconds,
        first_token_seconds=result.first_token_seconds,
    )
