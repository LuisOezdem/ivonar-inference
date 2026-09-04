# Ivonar Inference

Run Ivonar Nano locally: a terminal chat, an OpenAI-compatible server with a
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
  ivonar-nano/
    packed_inference_checkpoint.pt
    tokenizer.json
```

That is the whole setup. `ivonar models` lists what it finds, the first one is
loaded, and `--model ivonar-nano` picks another. Both the chat page and the terminal
switch between installed models while running.

The released model is called Ivonar Nano; where a version is needed it is
v1.0. The folder name is what the API reports as the model id.

`--device` defaults to `auto`: a CUDA GPU when one is present, otherwise the
CPU. If the GPU refuses the model, the CPU takes over with a message rather
than an error.

## Run

```bash
ivonar serve
```

Opens a chat page on `http://127.0.0.1:8000/` with stored conversations,
search, rename, a model picker, a settings dialog, streaming answers, and
light and dark themes.

```bash
ivonar chat
```

Chats in the terminal. `/help` lists the commands: `/new`, `/system`,
`/model`, `/models`, `/set temperature=0.7`, `/stats`, `/exit`. Every answer
reports its tokens, speed and context use; when a conversation outgrows the
context the oldest turns are dropped automatically.

```bash
ivonar verify
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

Defaults: temperature 0.5, top-k 40, repetition penalty 1.15, and the system
message `Your name is Ivonar. Give a helpful answer to what the user writes.`
Set your own with `--system` or in the settings dialog.

It was chosen by search: 83 system messages were screened on a 45-case suite,
the best 11 ran the full 66-case suite over four seeds, and the last three ran
six seeds each. Answers were scored on eleven rates, including whether the
model echoes the question back, leaks its own name into unrelated answers,
greets normally, states its name when asked, invents a biography, answers
facts correctly, restates the question instead of answering it, refuses,
produces a usable task answer, and repeats itself across turns.

| System message | Score | Name leak | Facts |
|---|---|---|---|
| `Your name is Ivonar. Give a helpful answer to what the user writes.` | 9.05 | 14% | 0.62 |
| `Name: Ivonar Answer briefly and correctly.` | 9.17 | 21% | 0.74 |
| none | 8.53 | 0% | 0.72 |

The second line scores higher and answers worse. Its identity credit came from
sentences like "Ivonar is a fictional character created by the author of The
Hunger Games", which contain the name without meaning it, and it opens
unrelated answers with "Ivonar is a software that ...". The shipped message
loses a tenth of a point and answers "I'm Ivonar" when asked, greets normally,
and keeps the name out of everything else.

Two smaller findings: a label followed by a line break makes the model treat
the label as text to continue, so the parts belong on one line; and a role
sentence such as "You are a helpful AI assistant" makes it read "can you tell
me ..." as a question about its abilities and refuse.

What no system message fixes: arithmetic is right about a third of the time,
facts about two thirds, and the model sometimes invents a biography. Those are
limits of a 349M model.

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
