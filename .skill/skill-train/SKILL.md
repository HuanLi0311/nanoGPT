---
name: skill-train
description: Inspect, load, and manage DLLM training runs and checkpoints with reproducible manifests. Use for SMDM/MDM weight loading, checkpoint validation, fine-tuning setup, or checkpoint handoff.
---

# DLLM training and checkpoint operations

Use the existing SMDM `TransEncoder`/`Config` implementation for the supplied `mdm-170M` and `mdm-1028M` weights. Keep training code in the source tree; this skill owns run metadata and validation, not a second trainer.

## Workflow

1. Run `scripts/checkpoint_info.py --checkpoint <file> --output <run-dir>/checkpoint.json` before loading. Confirm format, tensor names/shapes, parameter count, byte size, and SHA-256.
2. Match checkpoint size to `Config.from_name("Diff_LLaMA_<size>M")`; fail closed on missing or unexpected tensors.
3. Use a fresh run directory containing `config.json`, `env.json`, `data_manifest.json`, `checkpoint.json`, stdout/stderr logs, and benchmark JSON.
4. Load weights with the repository's `safetensors.torch.load_file` and `model.load_state_dict`; do not silently allow missing or unexpected keys.
5. For a smoke run, use a small sample count/sequence length and assert one finite loss before scaling up. Record OOM or import failures as failed attempts.

Checkpoint inspection is dependency-free and does not deserialize tensor payloads. See [references/protocol.md](references/protocol.md).
