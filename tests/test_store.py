from __future__ import annotations

from pathlib import Path

import pytest

from ivonar_inference.store import ChatStore


def test_store_creates_lists_appends_renames_and_deletes(tmp_path: Path) -> None:
    store = ChatStore(tmp_path / "chats")
    first = store.create()
    second = store.create("Planning")
    assert first.title == "New chat"
    assert second.title == "Planning"
    store.append(first.id, "user", "Where is Paris?\nsecond line")
    store.append(first.id, "assistant", "In France.")
    chat = store.get(first.id)
    assert chat.title == "Where is Paris?"
    assert chat.messages == [
        {"role": "user", "content": "Where is Paris?\nsecond line"},
        {"role": "assistant", "content": "In France."},
    ]
    listed = store.list()
    assert [item["id"] for item in listed][0] == first.id
    assert listed[0]["messages"] == 2
    store.rename(first.id, "  Paris  ")
    assert store.get(first.id).title == "Paris"
    store.delete(second.id)
    assert [item["id"] for item in store.list()] == [first.id]


def test_store_persists_across_instances_and_rejects_bad_ids(tmp_path: Path) -> None:
    root = tmp_path / "chats"
    chat = ChatStore(root).create("Kept")
    again = ChatStore(root)
    assert again.get(chat.id).title == "Kept"
    with pytest.raises(KeyError):
        again.get("missing")
    with pytest.raises(KeyError):
        again.get("../escape")
    with pytest.raises(KeyError):
        again.delete("missing")
