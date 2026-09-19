from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from .bench import DEFAULT_PROMPT
from .download import DEFAULT_MODEL, DEFAULT_SIZE_MB, DEFAULT_TITLE, PullJob, pull_model
from .engine import Engine, GenerationSettings
from .loader import pick_device, resolve_device
from .paths import available_models, ivonar_home, resolve_model_path, user_models_dir
from .registry import DEFAULT_SYSTEM, ModelRegistry

PORT_ATTEMPTS = 20


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="Model file, a folder holding one, or the name of an installed model. "
        "Without it the first installed model is used.",
    )
    parser.add_argument("--tokenizer", help="tokenizer.json; defaults to the file next to the model.")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda.")
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Run the decode step eagerly instead of replaying it as CUDA graphs.",
    )
    parser.add_argument(
        "--no-kernels",
        action="store_true",
        help="Use the torch decoder instead of the ternary CUDA kernels.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="System message placed before every conversation. Pass an empty string to send none.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)


def _settings(args: argparse.Namespace) -> GenerationSettings:
    return GenerationSettings(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )


def _registry(args: argparse.Namespace) -> tuple[ModelRegistry, str | None]:
    """A registry with the requested model loaded, or empty when nothing is installed yet.

    When the first installed model cannot be loaded the registry stays empty
    and the reason comes back as the second value; a model named with
    ``--model`` that fails raises instead.
    """

    device, note = pick_device(args.device)
    registry = ModelRegistry(
        device=device,
        graph=not args.no_graph,
        defaults=_settings(args),
        system=args.system,
        kernels=not args.no_kernels,
    )
    if note:
        registry.notes.append(note)
    if not (args.model or registry.installed()):
        return registry, None
    try:
        registry.load(resolve_model_path(args.model), tokenizer_path=args.tokenizer)
    except Exception as exc:
        if args.model:
            raise
        return registry, str(exc) or type(exc).__name__
    _report(registry)
    return registry, None


def _report(registry: ModelRegistry) -> None:
    engine = registry.engine
    if engine is None:
        return
    info = engine.info
    for note in registry.notes:
        print(f"[WARN] {note}", flush=True)
    print(
        f"[INFO] {registry.current}: stage={info.stage} step={info.step} context={info.context_tokens} "
        f"device={info.device} backend={info.backend} graph={info.graph}",
        flush=True,
    )
    for note in info.notes:
        print(f"[WARN] {note}", flush=True)
    others = [entry.name for entry in registry.entries() if not entry.loaded]
    if others:
        print(f"[INFO] also installed: {', '.join(others)} (switch with /model, or in the app)", flush=True)


def _download(name: str = DEFAULT_MODEL, repo: str | None = None) -> Path:
    live = sys.stdout.isatty()
    print(f"[INFO] Downloading {name} from Hugging Face into {user_models_dir()}", flush=True)

    def show(file: str, done: int, total: int) -> None:
        if live:
            amount = f"{done / 1e6:5.1f} of {total / 1e6:.1f} MB" if total else f"{done / 1e6:5.1f} MB"
            print(f"\r  {amount}", end="", flush=True)

    folder = pull_model(name, repo=repo, progress=show)
    if live:
        print(flush=True)
    print(f"[INFO] Saved to {folder}", flush=True)
    return folder


def _free_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + PORT_ATTEMPTS):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port between {preferred} and {preferred + PORT_ATTEMPTS - 1}")


def _open_when_ready(url: str) -> None:
    def wait_and_open() -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url + "health", timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=wait_and_open, name="ivonar-browser", daemon=True).start()


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app
    from .store import ChatStore

    registry, failure = _registry(args)
    store = ChatStore(args.data_dir)
    pull = PullJob(installed=registry.installed, load=registry.load_first)
    if failure:
        print(f"[WARN] The installed model could not be loaded: {failure}", flush=True)
        pull.fail(failure)
    app = create_app(registry.engine, default_system=registry.system, store=store, registry=registry, pull=pull)
    port = _free_port(args.host, args.port)
    shown = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{shown}:{port}/"
    if port != args.port:
        print(f"[INFO] Port {args.port} is busy, using {port}", flush=True)
    if not registry.ready and not failure:
        print(
            f"[INFO] No model is installed yet; the page offers to download {DEFAULT_TITLE} "
            f"(about {DEFAULT_SIZE_MB} MB)",
            flush=True,
        )
    print(f"[INFO] Ivonar is running at {url} (API under /v1, chats in {store.root}); press Ctrl+C to stop", flush=True)
    if not args.no_browser:
        _open_when_ready(url)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


_HELP = """commands:
  /help              show this list
  /new               start a new conversation
  /system [text]     show or set the system message, /system off removes it
  /models            list the installed models
  /model <name>      switch to another model
  /set k=v ...       change temperature, top_k, max_tokens, repetition_penalty
  /stats             show the current settings
  /exit              quit"""

_SETUP_HELP = f"""commands:
  /download          download {DEFAULT_TITLE} (about {DEFAULT_SIZE_MB} MB) and start chatting
  /exit              quit"""


def _apply_set(engine: Engine, assignments: list[str]) -> str:
    fields = {"temperature": float, "top_k": int, "max_tokens": int, "repetition_penalty": float}
    values = {}
    for item in assignments:
        key, _, raw = item.partition("=")
        key = key.strip()
        if key not in fields or not raw:
            return f"unknown setting: {item}"
        try:
            values[key] = fields[key](raw)
        except ValueError:
            return f"{key} needs a {fields[key].__name__} value"
    try:
        engine.defaults = GenerationSettings(**{**vars_of(engine.defaults), **values}).validated(
            engine.info.context_tokens
        )
    except ValueError as exc:
        return str(exc)
    return _settings_line(engine)


def vars_of(settings: GenerationSettings) -> dict[str, object]:
    return {
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_k": settings.top_k,
        "repetition_penalty": settings.repetition_penalty,
        "stop": settings.stop,
    }


def _settings_line(engine: Engine) -> str:
    settings = engine.defaults
    return (
        f"max_tokens={settings.max_tokens} temperature={settings.temperature} "
        f"top_k={settings.top_k} repetition_penalty={settings.repetition_penalty}"
    )


def _read_line() -> str | None:
    try:
        return input("\nuser> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(flush=True)
        return None


def _load_model(registry: ModelRegistry, folder: Path | None = None) -> bool:
    print("[INFO] Preparing the model; the first start also compiles the GPU kernels", flush=True)
    try:
        registry.load_first(folder)
    except Exception as exc:
        print(f"[ERROR] {exc}", flush=True)
        return False
    _report(registry)
    return True


def _setup_chat(registry: ModelRegistry) -> tuple[bool, str | None]:
    """Wait for a model: downloaded with /download or copied into the model folder.

    Returns whether one is loaded, and a message typed meanwhile that should
    become the first question.
    """

    print(
        f"[INFO] No model is installed yet. Type /download to get {DEFAULT_TITLE} "
        f"(about {DEFAULT_SIZE_MB} MB), or copy a model folder into {user_models_dir()}.",
        flush=True,
    )
    while True:
        line = _read_line()
        if line is None or line in {"/exit", "/quit"}:
            return False, None
        if line == "/help":
            print(_SETUP_HELP, flush=True)
        elif line != "/download" and registry.installed():
            if _load_model(registry):
                return True, line if line and not line.startswith("/") else None
        elif line == "/download":
            try:
                folder = _download()
            except KeyboardInterrupt:
                print("\n[INFO] Download stopped; type /download to continue where it left off.", flush=True)
                continue
            except (RuntimeError, OSError, ValueError) as exc:
                print(f"\n[ERROR] {exc}", flush=True)
                continue
            if _load_model(registry, folder):
                return True, None
        elif line:
            print(f"[INFO] There is no model yet; type /download, or copy a model folder into {user_models_dir()}.", flush=True)


def _chat(args: argparse.Namespace) -> int:
    registry, failure = _registry(args)
    if failure:
        print(f"[ERROR] The installed model could not be loaded: {failure}", flush=True)
    pending = None
    if not registry.ready:
        loaded, pending = _setup_chat(registry)
        if not loaded:
            return 0
    system = registry.system
    turns: list[dict[str, str]] = []
    print("[INFO] Type a message, /help lists the commands.", flush=True)
    while True:
        line, pending = (pending, None) if pending else (_read_line(), None)
        if line is None:
            return 0
        if not line:
            continue
        if line.startswith("/"):
            command, _, rest = line.partition(" ")
            rest = rest.strip()
            if command in {"/exit", "/quit"}:
                return 0
            if command == "/help":
                print(_HELP, flush=True)
            elif command == "/new":
                turns = []
                print("[INFO] New conversation.", flush=True)
            elif command == "/system":
                if not rest:
                    print(f"[INFO] system: {system or '(none)'}", flush=True)
                elif rest == "off":
                    system = ""
                    print("[INFO] System message removed.", flush=True)
                else:
                    system = rest
                    print("[INFO] System message set.", flush=True)
            elif command == "/models":
                for entry in registry.entries():
                    print(f"  {'*' if entry.loaded else ' '} {entry.name}", flush=True)
            elif command == "/model":
                if not rest:
                    print("[INFO] usage: /model <name>", flush=True)
                    continue
                try:
                    registry.switch(rest)
                except Exception as exc:
                    print(f"[ERROR] {exc}", flush=True)
                    if not registry.ready:
                        return 1
                    print(f"[INFO] Still on {registry.current}.", flush=True)
                    continue
                turns = []
                _report(registry)
                print("[INFO] New conversation on the new model.", flush=True)
            elif command == "/set":
                print(f"[INFO] {_apply_set(registry.engine, rest.split())}", flush=True)
            elif command == "/stats":
                print(f"[INFO] {registry.current}: {_settings_line(registry.engine)}", flush=True)
            elif command == "/download":
                print("[INFO] A model is already installed; 'ivonar pull <name>' downloads others.", flush=True)
            else:
                print("[INFO] unknown command, /help lists them", flush=True)
            continue

        engine = registry.engine
        messages = ([{"role": "system", "content": system}] if system.strip() else []) + [
            *turns,
            {"role": "user", "content": line},
        ]
        print("\nassistant> ", end="", flush=True)
        try:
            pieces = [delta for delta in engine.stream(messages) if not print(delta, end="", flush=True)]
        except KeyboardInterrupt:
            print("\n[INFO] Answer stopped.", flush=True)
            continue
        except (RuntimeError, ValueError, MemoryError) as exc:
            print(f"\n[ERROR] {exc}", flush=True)
            continue
        answer = "".join(pieces)
        result = engine.last_generation
        turns.extend([{"role": "user", "content": line}, {"role": "assistant", "content": answer}])
        used = result.prompt_tokens + result.completion_tokens
        note = f" dropped_turns={result.dropped_messages}" if result.dropped_messages else ""
        print(
            f"\n[INFO] tokens={result.completion_tokens} seconds={result.seconds:.2f} "
            f"first_token_ms={result.first_token_seconds * 1000:.0f} decode_tokens_per_s={result.decode_tokens_per_second:.0f} "
            f"tokens_per_s={result.tokens_per_second:.0f} context={used}/{engine.info.context_tokens}{note}",
            flush=True,
        )


def _verify(args: argparse.Namespace) -> int:
    from .verify import verify_model

    check = verify_model(
        resolve_model_path(args.model),
        tokenizer_path=args.tokenizer,
        device=resolve_device(args.device),
        graph=not args.no_graph,
        kernels=not args.no_kernels,
    )
    for line in check.lines():
        print(f"[INFO] {line}", flush=True)
    return 0 if check.passed else 1


def _bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmark

    resolve_model_path(args.model)
    registry, failure = _registry(args)
    if failure:
        raise RuntimeError(failure)
    result = run_benchmark(registry.engine, prompt=args.prompt, tokens=args.tokens, system=registry.system or None)
    for line in result.lines():
        print(f"[INFO] {line}", flush=True)
    return 0


def _pull(args: argparse.Namespace) -> int:
    _download(args.model, repo=args.repo)
    print("[INFO] Run 'ivonar serve' for the chat page or 'ivonar chat' for the terminal.", flush=True)
    return 0


def _models(args: argparse.Namespace) -> int:
    found = available_models()
    if not found:
        print(
            "[INFO] No model is installed yet. Run 'ivonar pull', or start 'ivonar serve' and click Download.",
            flush=True,
        )
        return 1
    for path in found:
        print(f"  {path.parent.name}  ({path.parent})", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ivonar", description="Run Ivonar models locally.")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Start the chat page and the OpenAI-compatible API.")
    _add_model_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000, help="Preferred port; the next free one is used if busy.")
    serve.add_argument("--no-browser", action="store_true", help="Do not open the chat page in a browser.")
    serve.add_argument(
        "--data-dir",
        default=str(ivonar_home() / "chats"),
        help="Directory that keeps the chat history of the web page.",
    )
    serve.set_defaults(handler=_serve)

    chat = commands.add_parser("chat", help="Chat in the terminal.")
    _add_model_arguments(chat)
    chat.set_defaults(handler=_chat)

    listing = commands.add_parser("models", help="List the installed models.")
    listing.set_defaults(handler=_models)

    pull = commands.add_parser("pull", help="Download a model release from Hugging Face.")
    pull.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="Release name, by default ivonar-nano.")
    pull.add_argument("--repo", help="Hugging Face repository; by default Ivonar/<model>.")
    pull.set_defaults(handler=_pull)

    verify = commands.add_parser(
        "verify",
        help="Compare the served decoder against the single-precision reference on a fixed answer.",
    )
    verify.add_argument("--model", help="Model file, folder, or installed model name; optional with one model.")
    verify.add_argument("--tokenizer", help="tokenizer.json; defaults to the file next to the model.")
    verify.add_argument("--device", default="auto", help="auto, cpu or cuda.")
    verify.add_argument("--no-graph", action="store_true", help="Check the eager decoder instead of the graphs.")
    verify.add_argument("--no-kernels", action="store_true", help="Check the torch decoder instead of the ternary kernels.")
    verify.set_defaults(handler=_verify)

    bench = commands.add_parser("bench", help="Measure prefill and decode speed of the served decoder.")
    _add_model_arguments(bench)
    bench.add_argument("--tokens", type=int, default=256, help="Decode steps to time.")
    bench.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to time.")
    bench.set_defaults(handler=_bench)

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print(flush=True)
        return 130
    except (RuntimeError, OSError, ValueError, MemoryError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
