# Models

Put every downloaded Ivonar release into its own folder here:

```
models/
  nano/
    packed_inference_checkpoint.pt
    tokenizer.json
    attribution_bundle.md
```

Then run `ivonar chat` or `ivonar serve` from the repository root. With one
model in this folder it is picked automatically; with several, name the folder:
`ivonar chat --model nano`. `ivonar models` lists what was found.

The model files themselves stay out of version control.
