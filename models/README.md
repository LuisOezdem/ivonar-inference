# Models

Downloads go to `~/.ivonar/models`: `ivonar pull`, the chat page's download
button and `/download` in the terminal chat put them there.

A release copied into this folder is found as well, from any directory. Give
it its own subfolder:

```
models/
  ivonar-nano/
    packed_inference_checkpoint.pt
    tokenizer.json
    attribution_bundle.md
```

It appears once the checkpoint and `tokenizer.json` are complete. With several
models, name the folder: `ivonar chat --model ivonar-nano`. `ivonar models`
lists what was found.

The model files themselves stay out of version control.
