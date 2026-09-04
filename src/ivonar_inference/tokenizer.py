from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ROLE_SYSTEM = "<|system|>"
ROLE_USER = "<|user|>"
ROLE_ASSISTANT = "<|assistant|>"
TOOL_CALL = "<|tool_call|>"
THINK_OPEN = "<|think|>"
THINK_CLOSE = "<|/think|>"
# Conversation structure is carried by reserved atomic tokens rather than by
# free text, so a model can never be argued out of its role boundaries by
# content that merely looks like a role label.
CHAT_SPECIAL_TOKENS = (
    IM_START,
    IM_END,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    TOOL_CALL,
    THINK_OPEN,
    THINK_CLOSE,
)

_RESERVED_LITERAL_PATTERN = re.compile(
    "|".join(re.escape(token) for token in ("<|pad|>", "<|endoftext|>", *CHAT_SPECIAL_TOKENS))
)


def sanitize_content(text: str) -> str:
    """Neutralize reserved marker literals inside untrusted content.

    Reserved markers are atomic tokens, so content that contains the literal
    text of one would otherwise encode to the genuine structural id. Rewriting
    the literal to a bracketed form keeps the text readable while removing
    every byte sequence the tokenizer treats as a marker.
    """

    return _RESERVED_LITERAL_PATTERN.sub(lambda match: f"[{match.group(0)[2:-2]}]", str(text))


def format_chat_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """Render a conversation in the chat format the model trained on.

    An optional system message comes first, user and assistant turns follow,
    the last message must be from the user, and the prompt ends with the
    opening of the assistant turn the model is asked to write.
    """

    if not messages:
        raise ValueError("chat messages cannot be empty")
    parts: list[str] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip().lower()
        content = sanitize_content(str(message.get("content", ""))).strip()
        if role == "system":
            if index != 0:
                raise ValueError("a system message must come first")
            if content:
                parts.append(f"{IM_START}{ROLE_SYSTEM}{content}{IM_END}")
        elif role == "user":
            if not content:
                raise ValueError("a user message cannot be empty")
            parts.append(f"{IM_START}{ROLE_USER}{content}{IM_END}")
        elif role == "assistant":
            parts.append(f"{IM_START}{ROLE_ASSISTANT}{content}{IM_END}")
        else:
            raise ValueError(f"unsupported chat role: {role!r}")
    if str(messages[-1].get("role", "")).strip().lower() != "user":
        raise ValueError("the last chat message must come from the user")
    parts.append(f"{IM_START}{ROLE_ASSISTANT}")
    return "".join(parts)


def tokenizer_file_sha256(path: str | Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


class TernaryTokenizer:
    """The Ivonar byte-level BPE tokenizer, loaded from its tokenizer.json."""

    pad_token = "<|pad|>"
    endoftext_token = "<|endoftext|>"
    reserved_tokens = (pad_token, endoftext_token, *CHAT_SPECIAL_TOKENS)
    byte_base = len(reserved_tokens)
    minimum_vocab_size = byte_base + 256

    def __init__(self) -> None:
        self.special_tokens = {token: idx for idx, token in enumerate(self.reserved_tokens)}
        self.id_to_special = {idx: token for token, idx in self.special_tokens.items()}
        self.vocab: dict[int, bytes] = {idx + self.byte_base: bytes([idx]) for idx in range(256)}
        self.merges: dict[tuple[int, int], int] = {}
        self.merge_ranks: dict[tuple[int, int], int] = {}
        self.next_id = self.minimum_vocab_size
        self.backend = "python_byte_bpe"
        self._hf_tokenizer: object | None = None
        # Reserved markers lead the split so they always tokenize atomically.
        # The punctuation run additionally refuses to start on a marker, so a
        # marker that follows punctuation cannot be swallowed into that run.
        self._split_re = re.compile(
            r"<\|/?[a-z_]+\|>|[A-Za-z]+(?:'[A-Za-z]+)?|\d+"
            r"|(?:(?!<\|/?[a-z_]+\|>)[^\sA-Za-z\d])+|\s+",
            re.UNICODE,
        )

    @property
    def pad_id(self) -> int:
        return self.special_tokens[self.pad_token]

    @property
    def endoftext_id(self) -> int:
        return self.special_tokens[self.endoftext_token]

    def __len__(self) -> int:
        return self.next_id

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("TernaryTokenizer.encode expects a string")
        if self._hf_tokenizer is not None:
            encoded = self._hf_tokenizer.encode(text, add_special_tokens=False)
            return [int(idx) for idx in encoded.ids]
        ids: list[int] = []
        for piece in self._pretokenize(text):
            special_id = self.special_tokens.get(piece)
            if special_id is not None:
                ids.append(special_id)
                continue
            seq = [byte + self.byte_base for byte in piece.encode("utf-8")]
            ids.extend(self._apply_bpe(seq))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        values = [int(raw_id) for raw_id in ids]
        if self._hf_tokenizer is not None:
            return str(self._hf_tokenizer.decode(values, skip_special_tokens=False))
        out = bytearray()
        for idx in values:
            special = self.id_to_special.get(idx)
            if special is not None:
                out.extend(special.encode("utf-8"))
                continue
            token = self.vocab.get(idx)
            if token is None:
                raise ValueError(f"unknown token id {idx}")
            out.extend(token)
        return out.decode("utf-8", errors="replace")

    @classmethod
    def load(cls, path: str | Path) -> "TernaryTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.special_tokens = {str(key): int(value) for key, value in payload["special_tokens"].items()}
        tokenizer.id_to_special = {idx: token for token, idx in tokenizer.special_tokens.items()}
        if payload.get("backend") == "hf_tokenizers_bpe":
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RuntimeError("Loading this tokenizer requires the 'tokenizers' package.") from exc
            tokenizer.backend = "hf_tokenizers_bpe"
            tokenizer._hf_tokenizer = Tokenizer.from_str(json.dumps(payload["tokenizer_json"]))
            tokenizer.next_id = int(payload["next_id"])
            return tokenizer
        tokenizer.backend = "python_byte_bpe"
        tokenizer.vocab = {int(idx): bytes(values) for idx, values in payload["vocab"].items()}
        tokenizer.merges = {(int(left), int(right)): int(new_id) for left, right, new_id in payload["merges"]}
        tokenizer.merge_ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges.keys())}
        tokenizer.next_id = int(payload["next_id"])
        return tokenizer

    def _pretokenize(self, text: str) -> Iterator[str]:
        for match in self._split_re.finditer(text):
            piece = match.group(0)
            if piece:
                yield piece

    @staticmethod
    def _merge_sequence(seq: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        merged: list[int] = []
        idx = 0
        left, right = pair
        while idx < len(seq):
            if idx + 1 < len(seq) and seq[idx] == left and seq[idx + 1] == right:
                merged.append(new_id)
                idx += 2
            else:
                merged.append(seq[idx])
                idx += 1
        return tuple(merged)

    def _apply_bpe(self, seq: list[int]) -> list[int]:
        if len(seq) < 2 or not self.merge_ranks:
            return seq
        current = tuple(seq)
        while len(current) >= 2:
            ranked_pairs = [
                (self.merge_ranks[(left, right)], (left, right))
                for left, right in zip(current, current[1:])
                if (left, right) in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs, key=lambda item: item[0])
            current = self._merge_sequence(current, best_pair, self.merges[best_pair])
        return list(current)
