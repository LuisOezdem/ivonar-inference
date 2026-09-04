from __future__ import annotations

from pathlib import Path

MODEL_FILE = "packed_inference_checkpoint.pt"
TOKENIZER_FILE = "tokenizer.json"
MODELS_DIR = "models"


def _model_file_in(directory: Path) -> Path | None:
    candidate = directory / MODEL_FILE
    if candidate.is_file():
        return candidate
    files = sorted(path for path in directory.glob("*.pt") if path.is_file())
    return files[0] if len(files) == 1 else None


def available_models(root: Path | None = None) -> list[Path]:
    """Model files found under ``models/`` in the working directory, one per subfolder."""

    base = (Path.cwd() if root is None else Path(root)) / MODELS_DIR
    if not base.is_dir():
        return []
    found: list[Path] = []
    direct = _model_file_in(base)
    if direct is not None:
        found.append(direct)
    for child in sorted(base.iterdir()):
        if child.is_dir():
            model_file = _model_file_in(child)
            if model_file is not None:
                found.append(model_file)
    return found


def model_name(path: Path, root: Path | None = None) -> str:
    """The name ``--model`` accepts for a model file found under ``models/``."""

    base = (Path.cwd() if root is None else Path(root)) / MODELS_DIR
    parent = Path(path).parent
    try:
        return str(parent.relative_to(base)).replace("\\", "/")
    except ValueError:
        return parent.name or str(path)


def resolve_model_path(spec: str | Path | None, root: Path | None = None) -> Path:
    """Turn what the user gave into a model file.

    ``spec`` may be a model file, a folder holding one, or the name of a
    subfolder of ``models/``. Without ``spec`` the single model under
    ``models/`` is used.
    """

    base = Path.cwd() if root is None else Path(root)
    if spec is None or str(spec).strip() == "":
        found = available_models(base)
        if len(found) == 1:
            return found[0]
        if not found:
            raise FileNotFoundError(
                f"no model found; put a release folder with {MODEL_FILE} and {TOKENIZER_FILE} under {base / MODELS_DIR}"
                " or pass --model"
            )
        names = ", ".join(model_name(path, base) for path in found)
        raise FileNotFoundError(f"several models found, pass --model with one of: {names}")
    path = Path(spec)
    if path.is_file():
        return path
    if path.is_dir():
        model_file = _model_file_in(path)
        if model_file is None:
            raise FileNotFoundError(f"folder holds no model file: {path}")
        return model_file
    named = base / MODELS_DIR / str(spec)
    if named.is_dir():
        model_file = _model_file_in(named)
        if model_file is not None:
            return model_file
    available = available_models(base)
    hint = ""
    if available:
        hint = "; available: " + ", ".join(model_name(item, base) for item in available)
    raise FileNotFoundError(f"model not found: {spec}{hint}")
