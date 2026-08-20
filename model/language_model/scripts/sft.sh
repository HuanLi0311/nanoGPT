#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
workers=$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count() or 1)')
PYTHONPYCACHEPREFIX=/tmp/nanogpt_pycache .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$workers" -m language_model.scripts.sft
