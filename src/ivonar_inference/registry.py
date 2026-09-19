from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from .engine import Engine, GenerationSettings
from .paths import available_models, model_name, resolve_model_path

DEFAULT_SYSTEM = "Your name is Ivonar. Give a helpful answer to what the user writes."


@dataclass(frozen=True)
class ModelEntry:
    """A model the registry can serve."""

    name: str
    path: Path
    loaded: bool

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "path": str(self.path), "loaded": self.loaded}


class ModelRegistry:
    """Every model found on disk, with at most one loaded at a time.

    Loading releases the previous model first, so several installed models
    cost disk space rather than memory. The registry may start empty and
    load a model once one has been downloaded.
    """

    def __init__(
        self,
        current: Engine | None = None,
        current_path: Path | None = None,
        root: Path | None = None,
        device: str = "cpu",
        graph: bool = True,
        defaults: GenerationSettings | None = None,
        system: str | None = None,
        kernels: bool = True,
        loader: Callable[..., Engine] | None = None,
    ) -> None:
        self.root = Path.cwd() if root is None else Path(root)
        self.device = device
        self.graph = graph
        self.kernels = kernels
        self.defaults = defaults or (current.defaults if current is not None else GenerationSettings())
        self.system = DEFAULT_SYSTEM if system is None else system
        self.notes: list[str] = []
        self._engine = current
        self._path = None if current_path is None else Path(current_path)
        self._tokenizer: Path | str | None = None
        self._loader = loader or Engine.load
        self._lock = threading.Lock()

    @property
    def engine(self) -> Engine | None:
        return self._engine

    @property
    def ready(self) -> bool:
        return self._engine is not None

    @property
    def current(self) -> str | None:
        return None if self._path is None else self._name_of(self._path)

    def _name_of(self, path: Path) -> str:
        return model_name(path, self.root)

    def installed(self) -> bool:
        return bool(available_models(self.root))

    def entries(self) -> list[ModelEntry]:
        found: dict[str, Path] = {}
        for path in available_models(self.root):
            found.setdefault(self._name_of(path), path)
        current = self.current
        if current is not None and self._path is not None:
            found.setdefault(current, self._path)
        return [ModelEntry(name, path, name == current) for name, path in sorted(found.items())]

    def load(self, target: Path, tokenizer_path: Path | str | None = None) -> Engine:
        """Load ``target`` in place of the model in memory and return its engine.

        The previous model is released first so two never share the memory;
        when the new one fails to load, the previous one is loaded again.
        """

        target = Path(target)
        with self._lock:
            if self._engine is not None and self._path == target:
                return self._engine
            previous, previous_tokenizer = self._path, self._tokenizer
            self._engine = None
            self._path = None
            _release_memory()
            try:
                engine = self._load_on_device(target, tokenizer_path)
            except Exception:
                _release_memory()
                if previous is not None:
                    try:
                        self._engine = self._load_on_device(previous, previous_tokenizer)
                        self._path = previous
                    except Exception:
                        _release_memory()
                raise
            self._engine = engine
            self._path = target
            self._tokenizer = tokenizer_path
            return engine

    def load_first(self, preferred: Path | str | None = None) -> Engine:
        """Load ``preferred`` when given, otherwise the first installed model."""

        return self.load(resolve_model_path(preferred, root=self.root))

    def switch(self, name: str) -> Engine:
        """Load the named model and release the one in memory; returns the new engine."""

        return self.load(resolve_model_path(name, root=self.root))

    def _load_on_device(self, target: Path, tokenizer_path: Path | str | None) -> Engine:
        options = {
            "tokenizer_path": tokenizer_path,
            "model_id": self._name_of(target),
            "defaults": self.defaults,
            "graph": self.graph,
            "kernels": self.kernels,
        }
        try:
            return self._loader(target, device=self.device, **options)
        except Exception as exc:
            if torch.device(self.device).type == "cpu":
                raise
            failure = exc
        _release_memory()
        engine = self._loader(target, device="cpu", **options)
        self.notes.append(f"{self.device} did not work ({type(failure).__name__}: {failure}); running on the CPU")
        self.device = "cpu"
        return engine


def _release_memory() -> None:
    gc.collect()
    try:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass
