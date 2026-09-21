# Ivonar Inference

Run Ivonar Nano on your own computer: a chat page, a terminal chat, an
OpenAI-compatible API, and kernels that compute directly on the ternary 2-bit
weights, on NVIDIA GPUs and on the CPU.

Model: [huggingface.co/Ivonar/ivonar-nano](https://huggingface.co/Ivonar/ivonar-nano) ·
Project: [ivonar.com](https://ivonar.com/)

## Install

macOS and Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/LuisOezdem/ivonar-inference/main/install.sh | sh
```

Windows, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/LuisOezdem/ivonar-inference/main/install.ps1 | iex
```

The command installs [uv](https://docs.astral.sh/uv/) with its own Python, then
Ivonar with the matching PyTorch build, and opens the chat page. Click
**Download Ivonar Nano** once: the model, about 100 MB, goes to
`~/.ivonar/models` and the chat starts. An NVIDIA GPU is used when present,
otherwise the CPU. The install takes about 3 GB, mostly PyTorch.

Next time run `ivonar serve`, or `ivonar chat` for the terminal, where
`/download` fetches the model. Run the install command again to update; remove
Ivonar with `uv tool uninstall ivonar-inference` and by deleting `~/.ivonar`.

## Commands

| Command | What it does |
|---|---|
| `ivonar serve` | Chat page and API on `http://127.0.0.1:8000/`, or the next free port. `--no-browser` keeps the browser closed. |
| `ivonar chat` | Terminal chat. `/help` lists `/new`, `/system`, `/model`, `/set temperature=0.7`, `/exit`. |
| `ivonar pull [name]` | Download a release from Hugging Face, checked against its published checksums. |
| `ivonar models` | List the installed models. |
| `ivonar bench` | Time prefill, decode and a full answer. |
| `ivonar verify` | Compare the decoder with the single-precision reference. |

`--device` is `auto` by default: the NVIDIA GPU when this PyTorch build runs on
it, otherwise the CPU. `--model <name>` picks a model, `--no-kernels` forces the
torch decoder and `--no-graph` turns off CUDA graphs.

## Models

Each model is a folder with its checkpoint and `tokenizer.json`; a release also
carries `config.json`, `LICENSE`, `README.md` and `attribution_bundle.md`:

```
~/.ivonar/models/ivonar-nano/packed_inference_checkpoint.pt
~/.ivonar/models/ivonar-nano/tokenizer.json
```

Downloads are checked against the published checksums and resume where they
stopped. A model folder copied in by hand works as soon as both files are
complete, and a page waiting for a model loads it by itself. `models/` in the
current directory and in a clone of this repository is searched too;
`IVONAR_HOME` moves models, chats and the kernel caches away from `~/.ivonar`.

## API

`POST /v1/chat/completions` follows the OpenAI protocol, including `stream` and
`stop`. `GET /v1/models` lists the models, and naming one in a request switches
to it. `GET /api/status`, `POST /api/settings` and `POST /api/model` read and
change the running setup; the chat page's conversations live under
`/api/chats`.

## How it runs

The weights stay packed: four ternary values per byte and one float16 scale per
128 inputs. Activations are quantized to int8 per token, as in the reference
model, and every dot product is exact integer arithmetic with fp32 scales. On
the GPU a token is 131 fused kernel launches and sixteen tokens run as one CUDA
graph; prompts use the same kernels in 32-token tiles, and Ivonar Nano needs
242 MiB of GPU memory. Past about a thousand prompt tokens the torch prefill of
`--no-kernels` is faster. On the CPU, numba kernels read the same packed
weights, 93 MB per token instead of 700 MB in float16; Intel Macs and Windows on
ARM have no numba and use the torch decoder.

Kernels compile on first use and are cached in `~/.ivonar`. When the model
loads they run a test prompt and must agree with a second path, and on the CPU
also beat the torch decoder; otherwise the torch decoder takes over, and a
failing GPU hands over to the CPU. The status line says which one runs.

## Speed

Ivonar Nano on an RTX 4060 Ti and a Ryzen 9 3900X:

| Setup | Decode | Notes |
|---|---|---|
| GPU, ternary kernels (default) | about 1,430 tokens/s | 36-token prompt in 7 ms |
| GPU, `--no-graph` | 950 tokens/s | |
| GPU, `--no-kernels` | 190 tokens/s | float16 weights, 0.9 GB |
| CPU, ternary kernels (default) | about 130 tokens/s | 36-token prompt in 0.25 s |
| CPU, `--no-kernels` | 24 tokens/s | float16 weights |

A 256-token answer runs end to end at about 1,350 tokens/s on the GPU and 100 on
the CPU, with the same agreement to the reference in `ivonar verify` on every
path. The card's memory bandwidth caps any decoder at about 2,900 tokens/s.
Windows parks an idle GPU after about ten seconds, so the next answer starts a
few hundred milliseconds later unless the NVIDIA Control Panel is set to
"Prefer maximum performance".

## Defaults

Temperature 0.5, top-k 40, repetition penalty 1.15 and the system message `Your
name is Ivonar. Give a helpful answer to what the user writes.`, chosen by
scoring 83 system messages on a 66-case suite. Change them with `--system`,
`--temperature`, `--top-k`, `--repetition-penalty` or in the settings dialog.
Ivonar Nano is a 349M model: arithmetic is right about a third of the time and
facts about two thirds.

## Development

```bash
pip install -e ".[test]"
pytest
```

The tests build tiny models in memory and need no model file; the GPU kernel
tests run when a GPU is present.

## Licence

Apache-2.0. Every Ivonar release ships an `attribution_bundle.md` for its
training data, which belongs next to the model when you redistribute it.

Built by Luis Oezdem.
