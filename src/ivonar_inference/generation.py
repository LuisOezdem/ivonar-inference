from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import torch

from .decoder import StaticDecoder
from .tokenizer import TernaryTokenizer


def stream_text(
    tokenizer: TernaryTokenizer,
    decoder: StaticDecoder,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float = 1.0,
    stop_token_ids: Iterable[int] = (),
    penalized_text: str = "",
) -> Iterator[str]:
    """Yield the answer as text increments whose concatenation is the answer.

    The decoder samples every token on its device and hands them over in
    bursts, so the host reads a handful of ids per graph replay. Only the
    tokens around the newest one are decoded again: a byte-level tokenizer
    can leave an incomplete character at the end of a partial decode, so
    text is held back until the next token completes it.
    ``penalized_text`` starts the repetition penalty on the words it contains,
    which the caller uses to discourage repeating the previous answer.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    token_ids = tokenizer.encode(prompt) or [tokenizer.endoftext_id]
    max_context = decoder.max_len
    if len(token_ids) > max_context:
        token_ids = token_ids[-max_context:]
    stop_ids = {int(tokenizer.endoftext_id), *(int(token_id) for token_id in stop_token_ids)}
    decoder.prefill(torch.tensor([token_ids], device=decoder.device, dtype=torch.long))
    decoder.configure_sampling(
        temperature,
        top_k,
        repetition_penalty,
        penalized_ids=tokenizer.encode(penalized_text) if penalized_text else (),
    )
    generated: list[int] = []
    prefix_offset = read_offset = 0
    context_tokens = len(token_ids)
    pending = deque([decoder.sample()])
    for token_index in range(max_new_tokens):
        next_id = pending.popleft()
        if context_tokens >= max_context or next_id in stop_ids:
            break
        generated.append(next_id)
        context_tokens += 1
        shown = tokenizer.decode(generated[prefix_offset:read_offset])
        text = tokenizer.decode(generated[prefix_offset:])
        if len(text) > len(shown) and not text.endswith("�"):
            yield text[len(shown) :]
            prefix_offset, read_offset = read_offset, len(generated)
        remaining = min(max_new_tokens - token_index - 1, max_context - context_tokens)
        if remaining <= 0:
            break
        if not pending:
            pending.extend(decoder.advance_many(min(decoder.burst_size, remaining)))
    shown = tokenizer.decode(generated[prefix_offset:read_offset])
    text = tokenizer.decode(generated[prefix_offset:])
    if text.startswith(shown) and len(text) > len(shown):
        yield text[len(shown) :]
