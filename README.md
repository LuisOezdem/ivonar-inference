# Ivonar Inference

Run Ivonar models locally: a terminal chat, an OpenAI-compatible server with a
chat page, and a check that the served model matches its reference.

## Install

```bash
pip install .
```

Python 3.12 or newer. A CUDA build of PyTorch is needed for `--device cuda`.

## Add a model

Put a downloaded release into its own folder under `models/`:

```
models/
  nano/
    packed_inference_checkpoint.pt
    tokenizer.json
```

That is the whole setup. With one model there it is picked automatically; with
several, name the folder: `--model nano`. `ivonar models` lists what it finds.

## Run

```bash
ivonar serve --device cuda
```

Opens a chat page on `http://127.0.0.1:8000/` with stored conversations,
search, rename, a model picker, a settings dialog, streaming answers, and
light and dark themes.

```bash
ivonar chat --device cuda
```

Chats in the terminal. `/help` lists the commands: `/new`, `/system`,
`/model`, `/models`, `/set temperature=0.7`, `/stats`, `/exit`. Every answer
reports its tokens, speed and context use; when a conversation outgrows the
context the oldest turns are dropped automatically.

```bash
ivonar verify --device cuda
```

Compares the served decoder against the single-precision reference and prints
whether they agree.

## API

`POST /v1/chat/completions` follows the OpenAI protocol, including `stream`
and `stop`. `GET /v1/models` lists the models, and naming one in a request
switches to it. `GET /api/status`, `POST /api/settings` and `POST /api/model`
read and change the running configuration; the chat page's own conversation
endpoints live under `/api/chats`.

## Speed

| Setup | Typical |
|---|---|
| GPU, CUDA graph decoder (default) | 170 to 190 tokens per second on an RTX 4060 Ti |
| GPU, `--no-graph` | 15 to 20 tokens per second |
| CPU | 8 to 12 tokens per second |

The decode step is recorded as CUDA graphs at load time, so a token costs a
single launch. Switching models releases the previous one.

## Answer quality

Defaults: temperature 0.5, top-k 40, repetition penalty 1.15, system message
`You are a helpful assistant.` Two details matter more than they look. Naming
the model inside its own system message makes it answer about itself instead
of the question, so the default stays plain; write your own with `--system` or
in the settings dialog. The repetition penalty also covers the previous
answer, which keeps a follow-up like "why" from repeating it. Measured over
five runs of a five-turn conversation, repeated answers fell from 9 of 25 to
1 of 25.

## Test

```bash
pip install -e ".[test]"
pytest
```

The tests build tiny models in memory and need no model file.

## Licence and attribution

Apache-2.0; put your name on the copyright line in `LICENSE`. Every Ivonar
release ships an `attribution_bundle.md` for its training data, which belongs
next to the model file when you redistribute it.
