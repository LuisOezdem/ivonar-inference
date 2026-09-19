from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import MODEL_FILE, TOKENIZER_FILE, user_models_dir

DEFAULT_MODEL = "ivonar-nano"
DEFAULT_TITLE = "Ivonar Nano"
DEFAULT_SIZE_MB = 100
DEFAULT_OWNER = "Ivonar"
CHECKSUM_FILE = "SHA256SUMS.txt"
REQUIRED_FILES = (MODEL_FILE, TOKENIZER_FILE)
RELEASE_FILES = (MODEL_FILE, TOKENIZER_FILE, "config.json", "LICENSE", "README.md", "attribution_bundle.md")
BLOCK = 1 << 20
NETWORK_ERRORS = (urllib.error.URLError, OSError, TimeoutError, ValueError)

Progress = Callable[[str, int, int], None]


def file_url(repo: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{name}"


def _open(url: str, timeout: float):
    request = urllib.request.Request(url, headers={"User-Agent": "ivonar-inference"})
    return urllib.request.urlopen(request, timeout=timeout)


def published_checksums(repo: str, timeout: float = 30.0) -> dict[str, str]:
    """The SHA-256 of every published file, empty when the release lists none."""

    try:
        with _open(file_url(repo, CHECKSUM_FILE), timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except NETWORK_ERRORS:
        return {}
    sums: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


def published_sizes(repo: str, timeout: float = 30.0) -> dict[str, int]:
    """The size of every published file, empty when the listing is unavailable."""

    try:
        with _open(f"https://huggingface.co/api/models/{repo}/tree/main", timeout) as response:
            entries = json.loads(response.read().decode("utf-8"))
    except NETWORK_ERRORS:
        return {}
    return {str(entry["path"]): int(entry.get("size") or 0) for entry in entries if isinstance(entry, dict) and "path" in entry}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected: str | None, advance: Callable[[int], None]) -> None:
    """Fetch ``url`` into ``destination``, refusing content that fails ``expected``."""

    temporary = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    try:
        with _open(url, 60.0) as response, temporary.open("wb") as handle:
            while block := response.read(BLOCK):
                handle.write(block)
                digest.update(block)
                advance(len(block))
    except NETWORK_ERRORS as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {url}: {exc}") from exc
    if expected and digest.hexdigest() != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {destination.name}; the download was discarded")
    temporary.replace(destination)


def pull_model(
    name: str = DEFAULT_MODEL,
    repo: str | None = None,
    models_dir: Path | None = None,
    progress: Progress | None = None,
) -> Path:
    """Download a release into ``<models_dir>/<name>`` and return that folder.

    A new release is assembled in a hidden folder and renamed into place once
    every required file matches its published checksum, so an interrupted
    download never looks installed; a repeated pull keeps verified files and
    fetches only what is missing. ``progress`` receives the file being fetched
    and the bytes done and expected across the whole release.
    """

    source = repo or f"{DEFAULT_OWNER}/{name}"
    base = Path(models_dir) if models_dir is not None else user_models_dir()
    target = base / name
    work = target if target.is_dir() else base / f".{name}.partial"
    work.mkdir(parents=True, exist_ok=True)
    sums = published_checksums(source)
    sizes = published_sizes(source)
    plan = []
    for item in RELEASE_FILES:
        destination = work / item
        expected = sums.get(item)
        if destination.is_file() and expected and file_sha256(destination) == expected:
            continue
        plan.append(item)
    total = sum(sizes.get(item, 0) for item in plan)
    done = 0
    for item in plan:

        def advance(count: int, item: str = item) -> None:
            nonlocal done
            done += count
            if progress is not None:
                progress(item, done, total)

        try:
            download_file(file_url(source, item), work / item, sums.get(item), advance)
        except RuntimeError:
            if item in REQUIRED_FILES:
                raise
    missing = [item for item in REQUIRED_FILES if not (work / item).is_file()]
    if missing:
        raise RuntimeError(f"{source} is missing {', '.join(missing)}")
    if work != target:
        work.replace(target)
    return target


@dataclass
class PullState:
    phase: str = "idle"
    name: str = DEFAULT_MODEL
    file: str = ""
    done: int = 0
    total: int = 0
    error: str = ""
    failed: str = ""
    folder: str = ""


class PullJob:
    """Downloads the default release and loads it in the background, one run at a time.

    ``installed`` says whether a model is already on disk, in which case only
    the load runs; ``load`` gets the downloaded folder, or None for a model
    that was already there, and makes it available to the caller.
    """

    ACTIVE = ("downloading", "loading")

    def __init__(
        self,
        installed: Callable[[], bool],
        load: Callable[[Path | None], object],
        fetch: Callable[..., Path] = pull_model,
        name: str = DEFAULT_MODEL,
        repo: str | None = None,
    ) -> None:
        self._installed = installed
        self._load = load
        self._fetch = fetch
        self._name = name
        self._repo = repo
        self._lock = threading.Lock()
        self._state = PullState(name=name)
        self._thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return self._name

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return asdict(self._state)

    def start(self) -> dict[str, object]:
        with self._lock:
            if self._state.phase not in self.ACTIVE:
                phase = "loading" if self._installed() else "downloading"
                self._state = PullState(phase=phase, name=self._name)
                self._thread = threading.Thread(target=self._run, name="ivonar-pull", daemon=True)
                self._thread.start()
            return asdict(self._state)

    def fail(self, error: str) -> None:
        """Report a model that is on disk but could not be loaded, so the page can explain why."""

        with self._lock:
            if self._state.phase not in self.ACTIVE:
                self._state = PullState(phase="error", name=self._name, error=error, failed="loading")

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _update(self, **changes: object) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._state, key, value)

    def _run(self) -> None:
        try:
            folder = None
            if not self._installed():
                folder = self._fetch(
                    self._name,
                    repo=self._repo,
                    progress=lambda file, done, total: self._update(file=file, done=done, total=total),
                )
                self._update(folder=str(folder))
            self._update(phase="loading", file="")
            self._load(folder)
            self._update(phase="ready")
        except Exception as exc:
            with self._lock:
                self._state.failed = self._state.phase
                self._state.phase = "error"
                self._state.error = str(exc) or type(exc).__name__
