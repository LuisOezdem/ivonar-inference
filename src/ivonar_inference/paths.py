from __future__ import annotations

import json
import os
from pathlib import Path

MODEL_FILE = "packed_inference_checkpoint.pt"
TOKENIZER_FILE = "tokenizer.json"
MODELS_DIR = "models"
ZIP_START = b"PK\x03\x04"
ZIP_END = b"PK\x05\x06"
ZIP_TAIL = 65557

_tokenizer_checks: dict[Path, tuple[tuple[int, int], bool]] = {}


def ivonar_home() -> Path:
    """Where Ivonar keeps downloaded models, chats and compiled kernels; ``IVONAR_HOME`` moves it."""

    configured = os.environ.get("IVONAR_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".ivonar"


def user_models_dir() -> Path:
    return ivonar_home() / MODELS_DIR


def source_checkout() -> Path | None:
    """The repository this copy of Ivonar runs from, or None for an installed package."""

    root = Path(__file__).resolve().parents[2]
    try:
        project = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    return root if 'name = "ivonar-inference"' in project else None


def model_roots(root: Path | None = None) -> list[Path]:
    """The folders searched for models: ``models/`` in the working directory, in the checkout, then the user folder."""

    candidates = [(Path.cwd() if root is None else Path(root)) / MODELS_DIR]
    checkout = source_checkout()
    if checkout is not None:
        candidates.append(checkout / MODELS_DIR)
    candidates.append(user_models_dir())
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        key = candidate.resolve()
        if key not in seen:
            seen.add(key)
            roots.append(candidate)
    return roots


def _model_file_in(directory: Path) -> Path | None:
    candidate = directory / MODEL_FILE
    if candidate.is_file():
        return candidate
    files = sorted(path for path in directory.glob("*.pt") if path.is_file())
    return files[0] if len(files) == 1 else None


def _finished_checkpoint(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(len(ZIP_START))
            if head != ZIP_START:
                return bool(head.strip(b"\0"))
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - ZIP_TAIL))
            return ZIP_END in handle.read()
    except OSError:
        return False


def _finished_tokenizer(path: Path) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    signature = (stat.st_size, stat.st_mtime_ns)
    cached = _tokenizer_checks.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        json.loads(path.read_bytes())
        finished = True
    except (OSError, ValueError):
        finished = False
    _tokenizer_checks[path] = (signature, finished)
    return finished


def is_complete(model_file: Path) -> bool:
    """Whether a model is fully on disk, so a folder still being copied in is not picked up yet."""

    return _finished_checkpoint(model_file) and _finished_tokenizer(model_file.parent / TOKENIZER_FILE)


def _models_in(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    candidates = [_model_file_in(base)]
    try:
        children = sorted(base.iterdir())
    except OSError:
        children = []
    for child in children:
        if child.is_dir() and not child.name.startswith("."):
            candidates.append(_model_file_in(child))
    return [model_file for model_file in candidates if model_file is not None and is_complete(model_file)]


def available_models(root: Path | None = None) -> list[Path]:
    """Model files found in the model folders, one per subfolder; a name found twice keeps its first copy."""

    found: list[Path] = []
    names: set[str] = set()
    for base in model_roots(root):
        for model_file in _models_in(base):
            name = model_name(model_file, root)
            if name not in names:
                names.add(name)
                found.append(model_file)
    return found


def model_name(path: Path, root: Path | None = None) -> str:
    """The name ``--model`` accepts for a model file found in a model folder."""

    parent = Path(path).parent
    for base in model_roots(root):
        try:
            relative = parent.relative_to(base)
        except ValueError:
            continue
        return str(relative).replace("\\", "/")
    return parent.name or str(path)


def resolve_model_path(spec: str | Path | None, root: Path | None = None) -> Path:
    """Turn what the user gave into a model file.

    ``spec`` may be a model file, a folder holding one, or the name of a
    subfolder of a model folder. Without ``spec`` the first model found is
    taken; both the server and the terminal chat can switch to any of the
    others while they run, so several models are not an error.
    """

    if spec is None or str(spec).strip() == "":
        found = available_models(root)
        if found:
            return found[0]
        raise FileNotFoundError(
            "no model is installed yet; start 'ivonar serve' and click Download, "
            "type /download in 'ivonar chat', or run 'ivonar pull'"
        )
    path = Path(spec)
    if path.is_file():
        return path
    if path.is_dir():
        model_file = _model_file_in(path)
        if model_file is None:
            raise FileNotFoundError(f"folder holds no model file: {path}")
        return model_file
    for base in model_roots(root):
        named = base / str(spec)
        if named.is_dir():
            model_file = _model_file_in(named)
            if model_file is not None:
                return model_file
    available = available_models(root)
    hint = ""
    if available:
        hint = "; available: " + ", ".join(model_name(item, root) for item in available)
    raise FileNotFoundError(f"model not found: {spec}{hint}")
