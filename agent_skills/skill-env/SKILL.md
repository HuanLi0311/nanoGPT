---
name: skill-env
description: Probe and prepare reproducible DLLM environments, checkpoints, and remote GPU nodes. Use for air-node-03 connection checks, dependency audits, run manifests, or explicitly authorized project deployment.
---

# Environment and node operations

Use the smallest reproducible environment that can run the selected model. Treat `/home/JJ_Group/lih2511/test/dllm` as the project root and `air-node-03` as the fixed compute target.

## Workflow

1. Run `scripts/probe.py --host air-node-03 --output <run-dir>/env.json` before changing anything. Capture hostname, GPU, Python, relevant packages, free disk, and UTC time.
2. Prefer an already-installed environment. For the SMDM checkpoint use the existing SMDM source tree and its `environment.yaml`; do not install packages during an experiment unless the run manifest records the approval and exact command.
3. Verify checkpoint paths and SHA-256 before loading. Do not copy the large checkpoints unless the target node lacks them.
4. Keep deployment explicit: use `scp`/`rsync` only with a narrow source list and a user-owned destination. Do not sync caches, credentials, or whole home directories.
5. Save the probe JSON beside every run log. A failed SSH, missing CUDA, or missing package is an environment failure, not a model result.

`probe.py` is read-only. It validates host syntax, uses batch SSH, and emits JSON even when individual probes fail.

## Required run fields

Record `host`, `source_root`, `checkpoint`, `checkpoint_sha256`, `python`, `torch`, `cuda`, `gpu`, `git_revision` if available, `seed`, and the exact command. See [references/protocol.md](references/protocol.md).
