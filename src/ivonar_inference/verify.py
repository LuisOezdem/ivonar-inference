from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .decoder import StaticDecoder
from .loader import load_model
from .model import IvonarModel
from .tokenizer import TernaryTokenizer, format_chat_messages, tokenizer_file_sha256

DEFAULT_QUESTION = "Explain how a car engine works and why oil matters."
DEFAULT_ANSWER = (
    " A car engine converts the chemical energy in fuel into motion. Air and fuel enter the cylinder, "
    "the piston compresses the mixture, a spark ignites it, and the expanding gas pushes the piston "
    "down. Oil reduces friction between moving parts, carries heat away and keeps deposits from "
    "forming, so without it the engine overheats and wears out quickly."
)


@dataclass(frozen=True)
class DecoderCheck:
    """How far the decoder's next-token predictions sit from the single-precision forward."""

    positions: int
    max_abs_diff: float
    mean_abs_diff: float
    top1_agreement: float
    reference_log_prob: float
    decoder_log_prob: float

    @property
    def passed(self) -> bool:
        return self.top1_agreement >= 0.9 and abs(self.decoder_log_prob - self.reference_log_prob) <= 0.1

    def lines(self) -> list[str]:
        return [
            f"positions compared: {self.positions}",
            f"max |logit diff|: {self.max_abs_diff:.4f}, mean: {self.mean_abs_diff:.5f}",
            f"top-1 agreement: {self.top1_agreement:.3f}",
            f"mean log-prob of the reference answer: reference {self.reference_log_prob:.4f}, decoder {self.decoder_log_prob:.4f}",
            "result: PASS" if self.passed else "result: FAIL",
        ]


@torch.no_grad()
def check_decoder(reference: IvonarModel, decoder: StaticDecoder, ids: list[int], split: int) -> DecoderCheck:
    """Teacher-force ``ids[split:]`` through both paths and compare their predictions.

    The reference is the model's one-pass forward on its own device; the
    decoder runs its prefill on ``ids[:split]`` and then one step per answer
    token, which exercises the graph replay path when graphs are recorded.
    """

    if not 0 < split < len(ids):
        raise ValueError("split must leave at least one token on each side")
    reference_logits = reference(torch.tensor([ids], device=reference.device))[0].float().cpu()[split - 1 : -1]
    predictions = [decoder.prefill(torch.tensor([ids[:split]], device=decoder.device))[0].float().cpu().clone()]
    for token_id in ids[split:-1]:
        step = decoder.step(torch.tensor([token_id], device=decoder.device))
        predictions.append(step[0].float().cpu().clone())
    decoded = torch.stack(predictions)
    count = min(len(decoded), len(reference_logits))
    decoded, reference_logits = decoded[:count], reference_logits[:count]
    targets = torch.tensor(ids[split : split + count])
    diff = (decoded - reference_logits).abs()
    reference_log_prob = torch.log_softmax(reference_logits, dim=-1).gather(1, targets[:, None]).mean()
    decoder_log_prob = torch.log_softmax(decoded, dim=-1).gather(1, targets[:, None]).mean()
    return DecoderCheck(
        positions=count,
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        top1_agreement=float((decoded.argmax(-1) == reference_logits.argmax(-1)).float().mean()),
        reference_log_prob=float(reference_log_prob),
        decoder_log_prob=float(decoder_log_prob),
    )


def verify_model(
    model_path: str | Path,
    tokenizer_path: str | Path | None = None,
    device: str = "cpu",
    graph: bool = True,
    question: str = DEFAULT_QUESTION,
    answer: str = DEFAULT_ANSWER,
) -> DecoderCheck:
    """Load the model twice, as the single-precision CPU reference and as the served decoder, and compare."""

    model_file = Path(model_path)
    tokenizer_file = Path(tokenizer_path) if tokenizer_path is not None else model_file.parent / "tokenizer.json"
    tokenizer = TernaryTokenizer.load(tokenizer_file)
    expected = tokenizer_file_sha256(tokenizer_file)
    prompt = format_chat_messages([{"role": "user", "content": question}])
    prompt_ids = tokenizer.encode(prompt)
    ids = prompt_ids + tokenizer.encode(answer)
    reference = load_model(model_file, device="cpu", expected_tokenizer_sha256=expected).model
    served = load_model(model_file, device=device, expected_tokenizer_sha256=expected).model
    decoder = StaticDecoder(served, device=device)
    if graph and torch.device(device).type == "cuda":
        decoder.capture()
    return check_decoder(reference, decoder, ids, len(prompt_ids))
