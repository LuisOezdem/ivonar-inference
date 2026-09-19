from __future__ import annotations

from pathlib import Path

import pytest

from ivonar_inference import paths
from ivonar_inference.paths import MODEL_FILE, available_models, resolve_model_path

REAL_SOURCE_CHECKOUT = paths.source_checkout


def _release(root: Path, name: str, file_name: str = MODEL_FILE) -> Path:
    folder = root / "models" / name
    folder.mkdir(parents=True)
    (folder / file_name).write_bytes(b"x")
    (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
    return folder / file_name


def test_single_model_is_found_without_a_spec(tmp_path: Path) -> None:
    model_file = _release(tmp_path, "nano")
    assert available_models(tmp_path) == [model_file]
    assert resolve_model_path(None, root=tmp_path) == model_file
    assert resolve_model_path("nano", root=tmp_path) == model_file
    assert resolve_model_path(model_file.parent, root=tmp_path) == model_file
    assert resolve_model_path(model_file, root=tmp_path) == model_file


def test_several_models_start_with_the_first_and_stay_switchable(tmp_path: Path) -> None:
    nano = _release(tmp_path, "nano")
    other = _release(tmp_path, "mini", file_name="mini.pt")
    assert available_models(tmp_path) == [other, nano]
    assert resolve_model_path(None, root=tmp_path) == other
    assert resolve_model_path("nano", root=tmp_path) == nano


def test_missing_models_explain_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ivonar pull"):
        resolve_model_path(None, root=tmp_path)
    _release(tmp_path, "nano")
    with pytest.raises(FileNotFoundError, match="available: nano"):
        resolve_model_path("missing", root=tmp_path)
    empty = tmp_path / "models" / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="holds no model file"):
        resolve_model_path(empty, root=tmp_path)


def test_resolve_device_picks_a_gpu_when_there_is_one(monkeypatch) -> None:
    import torch

    from ivonar_inference.loader import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_the_user_folder_is_searched_after_the_local_one(tmp_path: Path, isolated_home: Path) -> None:
    user_copy = isolated_home / "models" / "nano" / MODEL_FILE
    user_copy.parent.mkdir(parents=True)
    user_copy.write_bytes(b"x")
    (user_copy.parent / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert available_models(tmp_path) == [user_copy]
    assert resolve_model_path("nano", root=tmp_path) == user_copy
    local_copy = _release(tmp_path, "nano")
    assert available_models(tmp_path) == [local_copy]
    assert resolve_model_path("nano", root=tmp_path) == local_copy


def test_hidden_folders_such_as_unfinished_downloads_are_ignored(tmp_path: Path, isolated_home: Path) -> None:
    staging = isolated_home / "models" / ".nano.partial"
    staging.mkdir(parents=True)
    (staging / MODEL_FILE).write_bytes(b"x")
    assert available_models(tmp_path) == []
    with pytest.raises(FileNotFoundError, match="ivonar pull"):
        resolve_model_path(None, root=tmp_path)


def test_ivonar_home_can_be_moved(tmp_path: Path, monkeypatch) -> None:
    from ivonar_inference.paths import ivonar_home, user_models_dir

    monkeypatch.setenv("IVONAR_HOME", str(tmp_path / "elsewhere"))
    assert ivonar_home() == tmp_path / "elsewhere"
    assert user_models_dir() == tmp_path / "elsewhere" / "models"


def test_a_model_still_being_copied_in_is_not_picked_up(tmp_path: Path) -> None:
    folder = tmp_path / "models" / "copied"
    folder.mkdir(parents=True)
    checkpoint = folder / MODEL_FILE
    tokenizer = folder / "tokenizer.json"
    checkpoint.write_bytes(bytes(4096))
    tokenizer.write_text('{"model": {', encoding="utf-8")
    assert available_models(tmp_path) == []
    checkpoint.write_bytes(b"PK\x03\x04" + bytes(4096))
    assert available_models(tmp_path) == []
    checkpoint.write_bytes(b"PK\x03\x04" + bytes(4096) + b"PK\x05\x06" + bytes(18))
    assert available_models(tmp_path) == []
    tokenizer.write_text('{"model": {}}', encoding="utf-8")
    assert available_models(tmp_path) == [checkpoint]
    assert resolve_model_path(None, root=tmp_path) == checkpoint


def test_a_folder_without_tokenizer_is_not_listed_but_can_be_named(tmp_path: Path) -> None:
    folder = tmp_path / "models" / "bare"
    folder.mkdir(parents=True)
    (folder / MODEL_FILE).write_bytes(b"x")
    assert available_models(tmp_path) == []
    assert resolve_model_path("bare", root=tmp_path) == folder / MODEL_FILE


def test_the_checkout_models_folder_is_searched_from_anywhere(tmp_path: Path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    model_file = _release(checkout, "nano")
    monkeypatch.setattr(paths, "source_checkout", lambda: checkout)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert available_models(elsewhere) == [model_file]
    assert paths.model_roots(checkout) == [checkout / "models", paths.user_models_dir()]


def test_this_repository_is_recognised_as_a_checkout() -> None:
    assert REAL_SOURCE_CHECKOUT() == Path(__file__).resolve().parents[1]
