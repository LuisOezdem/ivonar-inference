# Ivonar Inference

Run Ivonar Nano on your own computer: a chat page, a terminal chat, an
OpenAI-compatible API, and CUDA kernels that read the ternary 2-bit weights as
they are stored.

Model: [huggingface.co/Ivonar/ivonar-nano](https://huggingface.co/Ivonar/ivonar-nano) ·
Project: [ivonar.com](https://ivonar.com/)

## Install

macOS and Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/LuisCode28/ivonar-inference/main/install.sh | sh
```

Windows, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/LuisCode28/ivonar-inference/main/install.ps1 | iex
```

The command installs [uv](https://docs.astral.sh/uv/), which brings its own
Python, then Ivonar with the matching PyTorch build, CUDA when an NVIDIA GPU is
present, and opens the chat page in your browser. Click **Download Ivonar
Nano** once: the model, about 100 MB, is saved to `~/.ivonar/models` and the
chat starts as soon as it is ready. Without a usable NVIDIA GPU Ivonar runs on
the CPU; the install takes about 3 GB of disk, mostly PyTorch.

Next time run `ivonar serve`, or `ivonar chat` for the terminal, where
`/download` fetches the model the same way. Running the install command again
updates Ivonar. To remove it, run `uv tool uninstall ivonar-inference` and
delete `~/.ivonar`.

## Commands

| Command | What it does |
|---|---|
| `ivonar serve` | Chat page and API on `http://127.0.0.1:8000/`, opened in the browser. The next free port is used when 8000 is busy; `--no-browser` skips opening it. |
| `ivonar chat` | Terminal chat. `/help` lists `/new`, `/system`, `/model`, `/set temperature=0.7`, `/exit`. |
| `ivonar pull [name]` | Download a release from Hugging Face, verified against its published checksums. |
| `ivonar models` | List what is installed. |
| `ivonar bench` | Time prefill, decode and a full answer. |
| `ivonar verify` | Compare the served decoder against the single-precision reference. |

`--device` defaults to `auto`: the CUDA GPU when this PyTorch build runs on it,
otherwise the CPU. `--no-kernels` forces the torch decoder, `--no-graph` runs
either eagerly.

## Models

Every model lives in its own folder:

```
~/.ivonar/models/
  ivonar-nano/
    packed_inference_checkpoint.pt
    tokenizer.json
    config.json, LICENSE, README.md, attribution_bundle.md
```

Downloads are checked against the published checksums and resume where they
stopped. A model folder dragged in by hand works too, as soon as its checkpoint
and `tokenizer.json` are complete; a page waiting for a model loads it by
itself. `ivonar pull <name>` fetches further releases.

`models/` in the current directory and in a clone of this repository is
searched too. `--model <name>` picks a model; `IVONAR_HOME` moves models, chat
history and the kernel cache away from `~/.ivonar`.

## API

`POST /v1/chat/completions` follows the OpenAI protocol, including `stream` and
`stop`. `GET /v1/models` lists the models, and naming one in a request switches
to it. `GET /api/status`, `POST /api/settings` and `POST /api/model` read and
change the running configuration; the chat page's conversation endpoints live
under `/api/chats`. A client that stops reading a stream early never blocks the
next request.

## Ternary kernels

Every weight matrix stays packed: four ternary values per byte with one float16
scale per 128 inputs. Activations are quantized to int8 per token inside the
kernel, dot products run on `dp4a`, and results accumulate in fp32, so the
arithmetic follows the reference model's own quantization rule.

One token is a single CUDA graph of 131 launches: fused normalize, quantize and
matrix-vector kernels, one kernel for the Mamba recurrence, one for latent
attention, a two-stage top-k sampler that scans only the logit groups which can
hold a candidate, and the output head pinned in the persisting part of the L2
cache. Prompts run through the same kernels in 32-token tiles, so no fp16 weight
copies exist; Ivonar Nano occupies 242 MiB of GPU memory. The tiles win up to a
few hundred prompt tokens; past roughly a thousand the torch prefill is faster.

Kernels compile with NVRTC from the torch installation on first load and are
cached in `~/.ivonar/kernels`. Before the first chat a short prompt runs through
the tile and the step kernels, which must agree. When the kernels cannot be
built or fail that check, the torch decoder takes over, and when the GPU fails
altogether, the CPU; the status line says which.

## Speed

Measured with Ivonar Nano on an RTX 4060 Ti and a Ryzen 9 3900X, 4096-token
context.

| Setup | Decode | Notes |
|---|---|---|
| Ternary kernels, CUDA graph (default) | about 1,200 tokens per second | 36-token prompt in 7 ms, 465 tokens in 57 ms, 242 MiB |
| Ternary kernels, `--no-graph` | 900 tokens per second | one launch per kernel from Python |
| Torch decoder, `--no-kernels` | 187 tokens per second | fp16 weights, 0.9 GB |
| CPU | 24 tokens per second | float16 weights where the CPU is faster with them, 36-token prompt in 0.27 s |

A 36-token prompt with a 256-token answer runs end to end at about 1,100 tokens
per second, with the same agreement to the reference as the torch decoder in
`ivonar verify`. One token reads 92.7 MB of weights and this card streams at
most 269 GB/s, so no decoder on it can beat about 2,900 tokens per second.

Answers report time to the first token and decoding speed separately. They
differ after a pause: Windows parks an idle GeForce in its lowest power state
after about ten seconds, and waking it takes a few hundred milliseconds. The
NVIDIA Control Panel setting "Prefer maximum performance" avoids that.

## Defaults

Temperature 0.5, top-k 40, repetition penalty 1.15, and the system message
`Your name is Ivonar. Give a helpful answer to what the user writes.` Change
them with `--system`, `--temperature`, `--top-k`, `--repetition-penalty` or in
the settings dialog. They were picked by scoring 83 system messages on a
66-case suite, so a change is a trade rather than an improvement.

Ivonar Nano is a 349M model: arithmetic is right about a third of the time and
facts about two thirds.

## Development

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
