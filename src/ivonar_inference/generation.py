from __future__ import annotations

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

    The decoder samples every token on its device, so a step is one graph
    replay and one id read. A byte-level tokenizer can leave an incomplete
    character at the end of a partial decode, so the trailing replacement
    characters are held back until the next token completes them.
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
    emitted = ""
    context_tokens = len(token_ids)
    next_id = decoder.sample()
    for token_index in range(max_new_tokens):
        if context_tokens >= max_context or next_id in stop_ids:
            break
        generated.append(next_id)
        context_tokens += 1
        stable = tokenizer.decode(generated).rstrip("�")
        if stable.startswith(emitted) and len(stable) > len(emitted):
            yield stable[len(emitted):]
            emitted = stable
        if token_index + 1 >= max_new_tokens or context_tokens >= max_context:
            break
        next_id = decoder.advance()
    final = tokenizer.decode(generated)
    if final.startswith(emitted) and len(final) > len(emitted):
        yield final[len(emitted):]
