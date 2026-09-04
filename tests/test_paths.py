from __future__ import annotations

from pathlib import Path

import pytest

from ivonar_inference.paths import MODEL_FILE, available_models, resolve_model_path


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
    with pytest.raises(FileNotFoundError, match="put a release folder"):
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
