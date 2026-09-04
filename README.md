# Ivonar Inference

Run Ivonar Nano locally: a terminal chat, an OpenAI-compatible server with a
chat page, and a check that the served model matches its reference.

## Install

```bash
pip install .
```

Python 3.12 or newer. A CUDA build of PyTorch is needed for GPU decoding.

## Add a model

Put a downloaded release into its own folder under `models/`:

```
models/
  ivonar-nano/
    packed_inference_checkpoint.pt
    tokenizer.json
```

That is the whole setup. `ivonar models` lists what it finds, the first one is
loaded, and `--model ivonar-nano` picks another; both the chat page and the
terminal switch between installed models while running. The folder name is the
model id the API reports.

`--device` defaults to `auto`: a CUDA GPU when present, otherwise the CPU, with
a fallback to the CPU if the GPU refuses the model.

## Run

```bash
ivonar serve
```

Opens a chat page on `http://127.0.0.1:8000/` with stored conversations,
search, rename, a model picker, a settings dialog, streaming answers, and light
and dark themes.

```bash
ivonar chat
```

Chats in the terminal. `/help` lists the commands: `/new`, `/system`, `/model`,
`/models`, `/set temperature=0.7`, `/stats`, `/exit`. Every answer reports its
tokens, speed and context use; when a conversation outgrows the context the
oldest turns are dropped automatically.

```bash
ivonar verify
```

Compares the served decoder against the single-precision reference.

## API

`POST /v1/chat/completions` follows the OpenAI protocol, including `stream` and
`stop`. `GET /v1/models` lists the models, and naming one in a request switches
to it. `GET /api/status`, `POST /api/settings` and `POST /api/model` read and
change the running configuration; the chat page's own conversation endpoints
live under `/api/chats`.

## Speed

| Setup | Typical |
|---|---|
| GPU, CUDA graph decoder (default) | 170 to 190 tokens per second on an RTX 4060 Ti |
| GPU, `--no-graph` | 15 to 20 tokens per second |
| CPU | 8 to 12 tokens per second |

The decode step is recorded as CUDA graphs at load time, so a token costs a
single launch. Switching models releases the previous one.

## Defaults

Temperature 0.5, top-k 40, repetition penalty 1.15, and the system message
`Your name is Ivonar. Give a helpful answer to what the user writes.` Change
them with `--system`, `--temperature`, `--top-k`, `--repetition-penalty` or in
the settings dialog. They were picked by scoring 83 system messages on a
66-case suite, so a change is a trade rather than an improvement.

Ivonar Nano is a 349M model: arithmetic is right about a third of the time and
facts about two thirds.

## Test

```bash
pip install -e ".[test]"
pytest
```

The tests build tiny models in memory and need no model file.

## Licence

Apache-2.0. Every Ivonar release ships an `attribution_bundle.md` for its
training data, which belongs next to the model file when you redistribute it.

Built by Luis Oezdem.
