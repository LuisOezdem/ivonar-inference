# Ivonar Inference

Run Ivonar Nano locally: a terminal chat, an OpenAI-compatible server with a
chat page, ternary CUDA kernels that read the 2-bit weights as they are stored,
and a check that the served model matches its reference.

Model: [huggingface.co/Ivonar/ivonar-nano](https://huggingface.co/Ivonar/ivonar-nano) ·
Project: [ivonar.com](https://ivonar.com/)

## Install

```bash
pip install .
```

Python 3.12 or newer. A CUDA build of PyTorch is needed for GPU decoding; the
kernels compile themselves on first start, no CUDA toolkit is required.

## Add a model

Download the release from
[huggingface.co/Ivonar/ivonar-nano](https://huggingface.co/Ivonar/ivonar-nano) and put it into
its own folder under `models/`:

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
`/models`, `/set temperature=0.7`, `/stats`, `/exit`. When a conversation
outgrows the context the oldest turns are dropped automatically.

```bash
ivonar bench
ivonar verify
```

`bench` times the prompt prefill, the decode step and a full answer. `verify`
compares the served decoder against the single-precision reference.

## API

`POST /v1/chat/completions` follows the OpenAI protocol, including `stream` and
`stop`. `GET /v1/models` lists the models, and naming one in a request switches
to it. `GET /api/status`, `POST /api/settings` and `POST /api/model` read and
change the running configuration; the chat page's conversation endpoints live
under `/api/chats`. A client that stops reading a stream early never blocks the
next request.

## Ternary kernels

On a GPU the decode step runs through CUDA kernels written for the ternary
format. Every weight matrix stays packed, four ternary values per byte with one
float16 scale per 128 inputs, activations are quantized to int8 per token inside
the kernel, dot products run on `dp4a`, and results accumulate in fp32, so the
arithmetic follows the reference model's own quantization rule.

One token is a single CUDA graph of 129 launches: fused normalize, quantize and
matrix-vector kernels, one kernel for the Mamba recurrence, one for latent
attention, a fused top-k sampler, and the output head pinned in the persisting
part of the L2 cache. Prompts run through the same kernels in 32-token tiles,
so no fp16 weight copies exist; Ivonar Nano occupies about 275 MiB of GPU
memory. The tiles win up to a few hundred prompt tokens; past roughly a
thousand the torch prefill is faster.

The kernels compile with NVRTC from the torch installation the first time a
model loads and are cached in `~/.ivonar/kernels`. Where they cannot be built
the engine falls back to the torch decoder and says so in the status line.
`--no-kernels` forces the torch decoder, `--no-graph` runs either decoder
eagerly.

## Speed

Measured on an RTX 4060 Ti with Ivonar Nano, 4096-token context.

| Setup | Decode | Notes |
|---|---|---|
| Ternary kernels, CUDA graph (default) | 1,078 tokens per second | 36-token prompt in 9 ms, 465 tokens in 68 ms, 275 MiB of GPU memory |
| Ternary kernels, `--no-graph` | about 850 tokens per second | one launch per kernel from Python |
| Torch decoder, `--no-kernels` | 187 tokens per second | fp16 weights, 1.3 GB of GPU memory |
| CPU | 8 to 12 tokens per second | |

A 36-token prompt with a 256-token answer runs end to end at about 1000 tokens
per second. `ivonar verify` reports the same agreement with the reference for
both decoders: top-1 agreement above 0.95 and a mean logit difference of 0.047,
the noise floor of the model's own int8 activation quantization.

Every answer reports its time to the first token and its decoding speed
separately, in the terminal, on the chat page and in the chats API. The two
differ after a pause: Windows parks an idle GeForce in its lowest power state
after roughly ten seconds, and it takes a few hundred milliseconds of work to
return to full clocks, so a short answer typed after a break runs at a third of
the speed. To keep the clocks up while the server runs, set the NVIDIA Control
Panel power management mode to "Prefer maximum performance" for `python.exe`;
it costs idle power, which is why the program does not do it for you.

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

The tests build tiny models in memory and need no model file. The kernel tests
run when a GPU is present and compare every kernel path against the
single-precision forward.

## Licence

Apache-2.0. Every Ivonar release ships an `attribution_bundle.md` for its
training data, which belongs next to the model file when you redistribute it.

Built by Luis Oezdem.
