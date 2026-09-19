from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from ivonar_inference import download
from ivonar_inference.paths import MODEL_FILE, TOKENIZER_FILE, available_models


class _Response(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _release(extra: dict[str, bytes] | None = None) -> dict[str, bytes]:
    files = {
        MODEL_FILE: b"packed weights" * 100,
        TOKENIZER_FILE: b'{"format": "tiny"}',
        "config.json": b"{}",
        "LICENSE": b"Apache-2.0",
        "README.md": b"# Tiny",
        "attribution_bundle.md": b"# sources",
    }
    files.update(extra or {})
    lines = [f"{hashlib.sha256(data).hexdigest()} *{name}" for name, data in files.items()]
    files[download.CHECKSUM_FILE] = ("\n".join(lines) + "\n").encode()
    return files


def _serve(files: dict[str, bytes], fetched: list[str], fail: set[str] | None = None):
    listing = json.dumps([{"type": "file", "path": name, "size": len(data)} for name, data in files.items()]).encode()

    def opener(url: str, timeout: float) -> _Response:
        if url.endswith("/tree/main"):
            return _Response(listing)
        name = url.rsplit("/", 1)[-1]
        if name not in files or name in (fail or set()):
            raise OSError(f"404 for {name}")
        fetched.append(name)
        return _Response(files[name])

    return opener


def test_pull_downloads_verifies_and_then_reuses(tmp_path: Path, monkeypatch) -> None:
    files = _release()
    fetched: list[str] = []
    monkeypatch.setattr(download, "_open", _serve(files, fetched))
    updates: list[tuple[str, int, int]] = []
    folder = download.pull_model("tiny", repo="owner/tiny", models_dir=tmp_path, progress=lambda *u: updates.append(u))
    assert folder == tmp_path / "tiny"
    for name in download.RELEASE_FILES:
        assert (folder / name).read_bytes() == files[name]
    total = sum(len(files[name]) for name in download.RELEASE_FILES)
    assert updates[-1][1:] == (total, total)
    assert not list(tmp_path.glob(".*"))

    fetched.clear()
    download.pull_model("tiny", repo="owner/tiny", models_dir=tmp_path)
    assert fetched == [download.CHECKSUM_FILE]

    (folder / TOKENIZER_FILE).write_bytes(b"damaged")
    fetched.clear()
    download.pull_model("tiny", repo="owner/tiny", models_dir=tmp_path)
    assert TOKENIZER_FILE in fetched and MODEL_FILE not in fetched
    assert (folder / TOKENIZER_FILE).read_bytes() == files[TOKENIZER_FILE]


def test_an_interrupted_download_is_not_installed_and_resumes(tmp_path: Path, monkeypatch) -> None:
    files = _release()
    fetched: list[str] = []
    models = tmp_path / "models"
    monkeypatch.setattr(download, "_open", _serve(files, fetched, fail={TOKENIZER_FILE}))
    with pytest.raises(RuntimeError, match="download failed"):
        download.pull_model("tiny", repo="owner/tiny", models_dir=models)
    assert not (models / "tiny").exists()
    assert (models / ".tiny.partial" / MODEL_FILE).is_file()
    assert available_models(tmp_path) == []

    fetched.clear()
    monkeypatch.setattr(download, "_open", _serve(files, fetched))
    folder = download.pull_model("tiny", repo="owner/tiny", models_dir=models)
    assert MODEL_FILE not in fetched and TOKENIZER_FILE in fetched
    assert (folder / TOKENIZER_FILE).read_bytes() == files[TOKENIZER_FILE]
    assert not (models / ".tiny.partial").exists()
    assert available_models(tmp_path) == [folder / MODEL_FILE]


def test_pull_refuses_content_that_fails_its_checksum(tmp_path: Path, monkeypatch) -> None:
    files = _release()
    files[MODEL_FILE] = b"replaced after the checksums were written"
    monkeypatch.setattr(download, "_open", _serve(files, []))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download.pull_model("tiny", repo="owner/tiny", models_dir=tmp_path)
    assert not (tmp_path / "tiny").exists()


def test_pull_reports_a_release_without_the_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(download, "_open", _serve({download.CHECKSUM_FILE: b""}, []))
    with pytest.raises(RuntimeError, match=f"download failed.*{MODEL_FILE}"):
        download.pull_model("tiny", repo="owner/tiny", models_dir=tmp_path)
    assert not (tmp_path / "tiny").exists()


def test_optional_files_may_be_absent(tmp_path: Path, monkeypatch) -> None:
    files = _release()
    del files["LICENSE"]
    monkeypatch.setattr(download, "_open", _serve(files, []))
    folder = download.pull_model("tiny", repo="owner/tiny", models_dir=tmp_path)
    assert (folder / MODEL_FILE).is_file()
    assert not (folder / "LICENSE").exists()


def test_pull_goes_to_the_user_model_folder_by_default(isolated_home: Path, monkeypatch) -> None:
    monkeypatch.setattr(download, "_open", _serve(_release(), []))
    folder = download.pull_model("tiny", repo="owner/tiny")
    assert folder == isolated_home / "models" / "tiny"


def test_file_url_points_at_the_published_release() -> None:
    assert download.file_url("Ivonar/ivonar-nano", MODEL_FILE) == (
        f"https://huggingface.co/Ivonar/ivonar-nano/resolve/main/{MODEL_FILE}"
    )


def test_pull_job_downloads_then_loads_and_reports_errors(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(name: str, repo: str | None = None, progress=None) -> Path:
        calls.append("fetch")
        progress(MODEL_FILE, 50, 100)
        progress(MODEL_FILE, 100, 100)
        return tmp_path / name

    job = download.PullJob(installed=lambda: False, load=lambda folder: calls.append(("load", folder)), fetch=fetch)
    assert job.snapshot()["phase"] == "idle"
    job.start()
    job.wait(5)
    state = job.snapshot()
    assert calls == ["fetch", ("load", tmp_path / download.DEFAULT_MODEL)]
    assert state["phase"] == "ready" and state["done"] == state["total"] == 100
    assert state["folder"] == str(tmp_path / download.DEFAULT_MODEL)

    calls.clear()
    installed = download.PullJob(installed=lambda: True, load=lambda folder: calls.append(("load", folder)), fetch=fetch)
    assert installed.start()["phase"] == "loading"
    installed.wait(5)
    assert calls == [("load", None)]

    def broken_fetch(name: str, repo: str | None = None, progress=None) -> Path:
        raise RuntimeError("network unreachable")

    failing = download.PullJob(installed=lambda: False, load=lambda folder: None, fetch=broken_fetch)
    failing.start()
    failing.wait(5)
    assert failing.snapshot()["phase"] == "error"
    assert failing.snapshot()["failed"] == "downloading"
    assert "network unreachable" in failing.snapshot()["error"]
