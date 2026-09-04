from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Chat:
    id: str
    title: str
    created: float
    updated: float
    messages: list[dict[str, str]] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "messages": len(self.messages),
        }


class ChatStore:
    """Conversations as one JSON file each under a directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, chat_id: str) -> Path:
        if not chat_id or not all(ch.isalnum() for ch in chat_id):
            raise KeyError(chat_id)
        return self.root / f"{chat_id}.json"

    def _read(self, chat_id: str) -> Chat:
        path = self._path(chat_id)
        if not path.is_file():
            raise KeyError(chat_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Chat(
            id=str(payload["id"]),
            title=str(payload.get("title", "")),
            created=float(payload.get("created", 0.0)),
            updated=float(payload.get("updated", 0.0)),
            messages=[{"role": str(m["role"]), "content": str(m["content"])} for m in payload.get("messages", [])],
        )

    def _write(self, chat: Chat) -> None:
        path = self._path(chat.id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(chat), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def list(self) -> list[dict[str, object]]:
        with self._lock:
            chats = [self._read(path.stem) for path in self.root.glob("*.json")]
        chats.sort(key=lambda chat: chat.updated, reverse=True)
        return [chat.summary() for chat in chats]

    def create(self, title: str = "New chat") -> Chat:
        now = time.time()
        chat = Chat(id=uuid.uuid4().hex[:12], title=title.strip() or "New chat", created=now, updated=now)
        with self._lock:
            self._write(chat)
        return chat

    def get(self, chat_id: str) -> Chat:
        with self._lock:
            return self._read(chat_id)

    def rename(self, chat_id: str, title: str) -> Chat:
        with self._lock:
            chat = self._read(chat_id)
            chat.title = title.strip() or chat.title
            chat.updated = time.time()
            self._write(chat)
            return chat

    def append(self, chat_id: str, role: str, content: str) -> Chat:
        with self._lock:
            chat = self._read(chat_id)
            chat.messages.append({"role": role, "content": content})
            if chat.title == "New chat" and role == "user":
                chat.title = content.strip().splitlines()[0][:48] if content.strip() else chat.title
            chat.updated = time.time()
            self._write(chat)
            return chat

    def delete(self, chat_id: str) -> None:
        with self._lock:
            path = self._path(chat_id)
            if not path.is_file():
                raise KeyError(chat_id)
            path.unlink()
