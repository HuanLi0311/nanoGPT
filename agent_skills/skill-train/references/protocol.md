# Checkpoint protocol

Before loading a model, run:

```bash
python3 scripts/checkpoint_info.py --checkpoint <file> \
  --output runs/<run>/checkpoint.json
```

The result records the safetensors header, tensor names/shapes, parameter
count, file size, and SHA-256. The benchmark must use the same checkpoint path
and hash. Missing or unexpected state-dict keys are load failures, not silent
compatibility warnings.
