from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import Engine, GenerationSettings
from .paths import resolve_model_path
from .registry import DEFAULT_SYSTEM, ModelRegistry


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="Model file, a folder holding one, or the name of a folder under models/. "
        "Without it the single model under models/ is used.",
    )
    parser.add_argument("--tokenizer", help="tokenizer.json; defaults to the file next to the model.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda.")
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Run the decode step eagerly instead of replaying it as CUDA graphs.",
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


def _registry(args: argparse.Namespace) -> ModelRegistry:
    model_file = resolve_model_path(args.model)
    engine = Engine.load(
        model_file,
        tokenizer_path=args.tokenizer,
        device=args.device,
        model_id=model_file.parent.name,
        defaults=_settings(args),
        graph=not args.no_graph,
    )
    registry = ModelRegistry(
        engine,
        model_file,
        device=args.device,
        graph=not args.no_graph,
        defaults=engine.defaults,
        system=args.system,
    )
    _report(registry)
    return registry


def _report(registry: ModelRegistry) -> None:
    info = registry.engine.info
    print(
        f"[INFO] {registry.current}: stage={info.stage} step={info.step} context={info.context_tokens} "
        f"device={info.device} graph={info.graph}",
        flush=True,
    )


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app
    from .store import ChatStore

    registry = _registry(args)
    store = ChatStore(args.data_dir)
    app = create_app(registry.engine, default_system=registry.system, store=store, registry=registry)
    print(
        f"[INFO] Serving on http://{args.host}:{args.port} (chat UI at /, API under /v1, chats in {store.root})",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


_HELP = """commands:
  /help              show this list
  /new               start a new conversation
  /system [text]     show or set the system message, /system off removes it
  /models            list the models under models/
  /model <name>      switch to another model
  /set k=v ...       change temperature, top_k, max_tokens, repetition_penalty
  /stats             show the current settings
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


def _chat(args: argparse.Namespace) -> int:
    registry = _registry(args)
    system = registry.system
    turns: list[dict[str, str]] = []
    print("[INFO] Type a message, /help lists the commands.", flush=True)
    while True:
        try:
            line = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(flush=True)
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
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    print(f"[ERROR] {exc}", flush=True)
                    continue
                turns = []
                _report(registry)
                print("[INFO] New conversation on the new model.", flush=True)
            elif command == "/set":
                print(f"[INFO] {_apply_set(registry.engine, rest.split())}", flush=True)
            elif command == "/stats":
                print(f"[INFO] {registry.current}: {_settings_line(registry.engine)}", flush=True)
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
        except ValueError as exc:
            print(f"\n[ERROR] {exc}", flush=True)
            continue
        answer = "".join(pieces)
        result = engine.last_generation
        turns.extend([{"role": "user", "content": line}, {"role": "assistant", "content": answer}])
        used = result.prompt_tokens + result.completion_tokens
        note = f" dropped_turns={result.dropped_messages}" if result.dropped_messages else ""
        print(
            f"\n[INFO] tokens={result.completion_tokens} seconds={result.seconds:.2f} "
            f"tokens_per_s={result.tokens_per_second:.1f} context={used}/{engine.info.context_tokens}{note}",
            flush=True,
        )


def _verify(args: argparse.Namespace) -> int:
    from .verify import verify_model

    check = verify_model(
        resolve_model_path(args.model), tokenizer_path=args.tokenizer, device=args.device, graph=not args.no_graph
    )
    for line in check.lines():
        print(f"[INFO] {line}", flush=True)
    return 0 if check.passed else 1


def _models(args: argparse.Namespace) -> int:
    from .paths import available_models

    found = available_models()
    if not found:
        print("[INFO] no model under models/; put a release folder there", flush=True)
        return 1
    for path in found:
        print(f"  {path.parent.name}  ({path})", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ivonar", description="Run Ivonar models locally.")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Start the OpenAI-compatible server with a chat UI.")
    _add_model_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--data-dir",
        default=str(Path.home() / ".ivonar" / "chats"),
        help="Directory that keeps the chat history of the web page.",
    )
    serve.set_defaults(handler=_serve)

    chat = commands.add_parser("chat", help="Chat in the terminal.")
    _add_model_arguments(chat)
    chat.set_defaults(handler=_chat)

    listing = commands.add_parser("models", help="List the models under models/.")
    listing.set_defaults(handler=_models)

    verify = commands.add_parser(
        "verify",
        help="Compare the served decoder against the single-precision reference on a fixed answer.",
    )
    verify.add_argument("--model", help="Model file, folder, or name under models/; optional when only one model is there.")
    verify.add_argument("--tokenizer", help="tokenizer.json; defaults to the file next to the model.")
    verify.add_argument("--device", default="cpu", help="cpu or cuda.")
    verify.add_argument("--no-graph", action="store_true", help="Check the eager decoder instead of the graphs.")
    verify.set_defaults(handler=_verify)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
