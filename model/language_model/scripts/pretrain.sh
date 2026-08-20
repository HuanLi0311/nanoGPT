#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
workers=$(.venv/bin/python -c 'import torch; print(torch.cuda.device_count() or 1)')
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29501}
PYTHONPYCACHEPREFIX=/tmp/nanogpt_pycache .venv/bin/python -m torch.distributed.run --master_addr="$master_addr" --master_port="$master_port" --nproc_per_node="$workers" -m model.language_model.scripts.pretrain "${1:-model/language_model/config/pretrain.yaml}"
