from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from .engine import Engine, GenerationSettings
from .paths import available_models, model_name, resolve_model_path

# Naming the model in its own system message makes it answer about itself
# instead of the question, so the default stays plain.
DEFAULT_SYSTEM = "You are a helpful assistant."


@dataclass(frozen=True)
class ModelEntry:
    """A model the registry can serve."""

    name: str
    path: Path
    loaded: bool

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "path": str(self.path), "loaded": self.loaded}


class ModelRegistry:
    """Every model found on disk, with one loaded at a time.

    Switching releases the previous model before loading the next, so several
    installed models cost disk space rather than memory.
    """

    def __init__(
        self,
        current: Engine,
        current_path: Path,
        root: Path | None = None,
        device: str = "cpu",
        graph: bool = True,
        defaults: GenerationSettings | None = None,
        system: str | None = None,
    ) -> None:
        self.root = Path.cwd() if root is None else Path(root)
        self.device = device
        self.graph = graph
        self.defaults = defaults or current.defaults
        self.system = DEFAULT_SYSTEM if system is None else system
        self._engine = current
        self._path = Path(current_path)
        self._lock = threading.Lock()

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def current(self) -> str:
        return self._name_of(self._path)

    def _name_of(self, path: Path) -> str:
        return model_name(path, self.root)

    def entries(self) -> list[ModelEntry]:
        found = {self._name_of(path): path for path in available_models(self.root)}
        found.setdefault(self.current, self._path)
        current = self.current
        return [ModelEntry(name, path, name == current) for name, path in sorted(found.items())]

    def switch(self, name: str) -> Engine:
        """Load the named model and release the one in memory; returns the new engine."""

        target = resolve_model_path(name, root=self.root)
        with self._lock:
            if target == self._path:
                return self._engine
            previous, self._engine = self._engine, None
            del previous
            engine = Engine.load(
                target,
                device=self.device,
                model_id=self._name_of(target),
                defaults=self.defaults,
                graph=self.graph,
            )
            self._engine = engine
            self._path = target
            return engine
